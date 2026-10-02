#!/usr/bin/env python3
"""Check that a fresh clone contains every local asset needed by the evaluators."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PROFILE_COUNT = 40
LFS_HEADER = b"version https://git-lfs.github.com/spec/v1"

REQUIRED_FILES = (
    "assets/Factory/franka_mimic.usd",
    "assets/Factory/factory_hole_8mm.usd",
    "assets/Factory/factory_peg_8mm.usd",
    "franka_env_background_edit/franka_env.usd",
    "franka_env_background_edit/Simple_Room/simple_room.usd",
    "franka_env_background_edit/panda_long_finger.usd",
    "franka_new/background.usdz",
    "franka_new/green-screen-workspace.usdz",
    "sim_side_test_bundle/src/openpi/force_features.py",
    "sim_side_test_bundle/scripts/sim_force_trend_preprocessor.py",
    "sim_side_test_bundle/configs/tavla_sim_force_trend_affine.json",
    "sim_side_test_bundle/assets/wrench_adapters/sim_aligned_to_real_affine.pt",
    "sim_side_test_bundle/assets/wrench_adapters/sim_aligned_to_real_affine.json",
)

EXPECTED_PACKAGES = {
    "isaacsim": "6.0.0.0",
    "isaaclab": "6.1.14",
    "torch": "2.10.0+cu128",
    "openpi-client": "0.1.2",
    "websockets": "16.0",
}


def is_lfs_pointer(path: Path) -> bool:
    if path.is_symlink() or not path.is_file():
        return False
    with path.open("rb") as handle:
        return handle.read(len(LFS_HEADER)) == LFS_HEADER


def check_environment(errors: list[str], warnings: list[str]) -> None:
    if sys.version_info[:2] != (3, 12):
        errors.append(f"Python 3.12 required, found {sys.version.split()[0]}")
    for package, expected in EXPECTED_PACKAGES.items():
        try:
            actual = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            errors.append(f"missing Python package: {package}=={expected}")
            continue
        if actual != expected:
            warnings.append(f"{package}: expected {expected}, found {actual}")


def check_portable_usd_paths(errors: list[str]) -> None:
    try:
        from pxr import Sdf
    except ImportError:
        errors.append("missing Python package: pxr (provided by Isaac Sim)")
        return

    forbidden = ("file:/home/", "@/home/", "@../Factory/", "@../NVIDIA/")
    for relative in (
        "franka_env_background_edit/franka_env.usd",
        "franka_env_background_edit/Simple_Room/simple_room.usd",
    ):
        path = ROOT / relative
        layer = Sdf.Layer.FindOrOpen(str(path))
        if layer is None:
            errors.append(f"invalid USD layer: {relative}")
            continue
        text = layer.ExportToString()
        for token in forbidden:
            if token in text:
                errors.append(f"non-portable USD reference in {relative}: {token}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-environment", action="store_true")
    args = parser.parse_args()

    errors: list[str] = []
    warnings: list[str] = []
    files: list[Path] = [ROOT / relative for relative in REQUIRED_FILES]

    for profile_id in range(PROFILE_COUNT):
        sim = ROOT / "sim-data" / f"traj_{profile_id}"
        aligned = ROOT / "sim-data-aligned" / f"traj_{profile_id}"
        files.extend(
            (
                sim / "data.h5",
                sim / "replay_metadata.json",
                sim / "front_camera.mp4",
                sim / "wrist_camera.mp4",
                aligned / "data.h5",
            )
        )
        metadata_path = sim / "replay_metadata.json"
        if metadata_path.is_file() and not is_lfs_pointer(metadata_path):
            try:
                metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                errors.append(f"invalid metadata {metadata_path.relative_to(ROOT)}: {exc}")
            else:
                if int(metadata.get("frame_count", 0)) <= 0:
                    errors.append(f"invalid frame_count: {metadata_path.relative_to(ROOT)}")
                source = str(metadata.get("real_source", ""))
                if source and os.path.isabs(source):
                    errors.append(f"absolute real_source: {metadata_path.relative_to(ROOT)}")

    for path in files:
        relative = path.relative_to(ROOT)
        if not path.exists():
            errors.append(f"missing: {relative}")
        elif path.is_file() and path.stat().st_size == 0:
            errors.append(f"empty: {relative}")
        elif is_lfs_pointer(path):
            errors.append(f"Git LFS object not downloaded: {relative}")

    for data_root in (ROOT / "sim-data-aligned", ROOT / "real-data-aligned"):
        if not data_root.is_dir():
            continue
        for link in data_root.glob("traj_*/*.mp4"):
            if link.is_symlink() and not link.exists():
                errors.append(f"broken symlink: {link.relative_to(ROOT)} -> {os.readlink(link)}")
            if link.is_symlink() and os.path.isabs(os.readlink(link)):
                errors.append(f"absolute symlink: {link.relative_to(ROOT)}")

    try:
        import h5py
    except ImportError:
        if args.check_environment:
            errors.append("missing Python package: h5py")
    else:
        for profile_id in range(PROFILE_COUNT):
            shapes = []
            for folder in ("sim-data", "sim-data-aligned"):
                path = ROOT / folder / f"traj_{profile_id}" / "data.h5"
                if not path.is_file() or is_lfs_pointer(path):
                    continue
                try:
                    with h5py.File(path, "r") as handle:
                        q = handle["obs/state/joint_pos"]
                        wrench = handle["obs/state/ee_wrench_base"]
                        shapes.append((int(q.shape[0]), tuple(q.shape[1:]), tuple(wrench.shape[1:])))
                except (OSError, KeyError) as exc:
                    errors.append(f"invalid H5 {path.relative_to(ROOT)}: {exc}")
            if len(shapes) == 2 and shapes[0] != shapes[1]:
                errors.append(f"profile {profile_id}: sim/aligned H5 shapes differ: {shapes}")

    if args.check_environment:
        check_environment(errors, warnings)
        check_portable_usd_paths(errors)

    for warning in warnings:
        print(f"[repository-check] WARNING: {warning}")
    if errors:
        for error in errors:
            print(f"[repository-check] ERROR: {error}", file=sys.stderr)
        print(f"[repository-check] failed: {len(errors)} error(s)", file=sys.stderr)
        return 1
    print(
        f"[repository-check] OK: {PROFILE_COUNT} sim profiles, "
        f"{PROFILE_COUNT} aligned force profiles, assets and adapters are present"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
