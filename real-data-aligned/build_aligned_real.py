#!/usr/bin/env python3
"""Rebuild real-data-aligned with the shared online-causal force contract."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import h5py
import numpy as np

from tacex_tasks.real2sim.force_alignment import (
    FORCE_ALIGNMENT_VERSION,
    ForceAlignmentConfig,
    force_to_zero_torque_wrench,
    process_force_series,
)


ROOT = Path(__file__).resolve().parents[1]
REAL = ROOT / "real_data"
OUT = ROOT / "real-data-aligned"

BASE_WRENCH = "obs/state/ee_wrench_base"
BASE_FORCE = "obs/state/ee_force_base"
BASE_TORQUE = "obs/state/ee_torque_base"


def build_one(
    trajectory: int,
    out_dir: Path,
    config: ForceAlignmentConfig,
    free_space: dict,
) -> dict:
    src = REAL / f"traj_{trajectory}"
    a, b = free_space[f"traj_{trajectory}"]["free_space_frames"][0]
    out_dir.mkdir(parents=True, exist_ok=True)
    dst = out_dir / "data.h5"
    with h5py.File(src / "data.h5", "r") as source, h5py.File(dst, "w") as output:
        for name in source.attrs:
            output.attrs[name] = source.attrs[name]
        for key in source.keys():
            source.copy(source[key], output, name=key)

        raw_wrench = output[BASE_WRENCH][:].astype(np.float64)
        joint_pos = output["obs/state/joint_pos"][:].astype(np.float64)
        joint_vel = output["obs/state/joint_vel"][:].astype(np.float64)
        timestamps = output["timestamps"][:].astype(np.float64)
        aligned = process_force_series(
            timestamps - timestamps[0],
            raw_wrench[:, :3],
            joint_pos,
            joint_vel=joint_vel,
            config=config,
            domain="real",
            add_model_noise=False,
        )
        cleaned = force_to_zero_torque_wrench(aligned["force_clean"])
        output[BASE_WRENCH][...] = cleaned
        output[BASE_FORCE][...] = cleaned[:, :3]
        output[BASE_TORQUE][...] = 0.0

        for path, role in (
            (BASE_WRENCH, "model_input"),
            (BASE_FORCE, "model_input"),
            (BASE_TORQUE, "zero_compatibility_channel"),
        ):
            dataset = output[path]
            for key in list(dataset.attrs):
                del dataset.attrs[key]
            dataset.attrs["source"] = "real_robot_O_F_ext_hat_K_causal_aligned"
            dataset.attrs["frame"] = "robot_base_O"
            dataset.attrs["role"] = role
            dataset.attrs["online_causal_alignment"] = True
            dataset.attrs["torque_zeroed"] = True

        force_group = output.require_group("real").require_group("force")
        for name, values in aligned.items():
            force_group.create_dataset(name, data=values, compression="gzip")
        force_group.attrs["alignment_version"] = FORCE_ALIGNMENT_VERSION
        force_group.attrs["alignment_config"] = json.dumps(
            config.to_dict(), sort_keys=True
        )
        force_group.attrs["model_equals_clean"] = True

        output.attrs["force_alignment"] = FORCE_ALIGNMENT_VERSION
        output.attrs["force_source"] = "real/force/force_model"
        output.attrs["force_clean_source"] = "real/force/force_clean"
        output.attrs["force_frame"] = "robot_base_O"
        output.attrs["force_torque_mode"] = "zero"
        output.attrs["force_amplitude_mapping"] = "identity"
        output.attrs["force_fixed_delay_frames"] = 0
        output.attrs["force_baseline_frames"] = [a, b]

        raw_free_space = float(np.linalg.norm(raw_wrench[a:b, :3], axis=1).mean())
        aligned_free_space = float(np.linalg.norm(cleaned[a:b, :3], axis=1).mean())
        peak = float(np.linalg.norm(cleaned[:, :3], axis=1).max())

    for name in ("front_camera.mp4", "wrist_camera.mp4"):
        link = out_dir / name
        if link.is_symlink() or link.exists():
            link.unlink()
        source_video = (src / name).resolve()
        link.symlink_to(Path(os.path.relpath(source_video, start=out_dir.resolve())))
    return {
        "trajectory": trajectory,
        "frames": int(raw_wrench.shape[0]),
        "free_space_frames": [a, b],
        "alignment": FORCE_ALIGNMENT_VERSION,
        "free_space_F_raw_n": raw_free_space,
        "free_space_F_aligned_n": aligned_free_space,
        "contact_peak_F_n": peak,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--force-alignment-config",
        type=Path,
        default=ROOT / "outputs/force_alignment/force_alignment_config.json",
    )
    args = parser.parse_args()
    if not args.force_alignment_config.is_file():
        raise FileNotFoundError(
            "fit the shared online force configuration first: "
            f"{args.force_alignment_config}"
        )
    config = ForceAlignmentConfig.load(args.force_alignment_config)
    free_space = json.loads(
        (REAL / "free_space_intervals.json").read_text(encoding="utf-8")
    )["trajectories"]

    OUT.mkdir(parents=True, exist_ok=True)
    manifest = []
    for trajectory in range(40):
        info = build_one(
            trajectory,
            OUT / f"traj_{trajectory}",
            config,
            free_space,
        )
        manifest.append(info)
        print(
            f"[{trajectory:02d}/39] frames={info['frames']:4d} "
            f"free-space |F| {info['free_space_F_raw_n']:.2f}->"
            f"{info['free_space_F_aligned_n']:.2f}N "
            f"peak={info['contact_peak_F_n']:.2f}N",
            flush=True,
        )
    (OUT / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    raw = np.mean([item["free_space_F_raw_n"] for item in manifest])
    aligned = np.mean([item["free_space_F_aligned_n"] for item in manifest])
    print(
        f"[DONE] {len(manifest)} trajectories | mean free-space |F| "
        f"{raw:.2f}->{aligned:.2f}N (alignment={FORCE_ALIGNMENT_VERSION}) -> {OUT}"
    )


if __name__ == "__main__":
    main()
