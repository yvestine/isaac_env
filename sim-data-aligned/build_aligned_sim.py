#!/usr/bin/env python3
"""Build sim-data-aligned from the causal robot-base force contract.

The source replay must provide ``causal_force_alignment_v1`` streams.  No
post-hoc sign flip or global magnitude gain is permitted here: frame
calibration, motion-baseline removal, residual tare and temporal filtering are
all performed by the shared online force pipeline before packaging.

``obs/state/ee_wrench_base`` stores force_model XYZ and exactly zero torque.
The deterministic force_clean and every diagnostic stage are retained under
``sim/force``.  Videos remain symlinks to the visually aligned sim-data set.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE_ROOT = ROOT / "outputs/causal_force_replay"
SIM_BUILT = ROOT / "sim-data"          # existing built folder (source of videos to symlink)
OUT_ROOT = ROOT / "sim-data-aligned"

REAL_STATE_FIELDS = [
    "ee_force", "ee_force_base", "ee_force_stiffness",
    "ee_pose",
    "ee_torque", "ee_torque_base", "ee_torque_stiffness",
    "ee_wrench", "ee_wrench_base", "ee_wrench_stiffness",
    "gripper_pos", "gripper_width_m",
    "joint_pos", "joint_torque", "joint_torque_external", "joint_vel",
]
FORCE_MODEL_SOURCE = "sim/force/panda_link7_force_model_base"
FORCE_CLEAN_SOURCE = "sim/force/panda_link7_force_clean_base"
FORCE_ALIGNMENT_VERSION = "causal_force_alignment_v1"

GRIPPER_MAX_WIDTH_M = 0.08
GRIPPER_GRASP_POS = 0.0865
GRIPPER_GRASP_WIDTH_M = GRIPPER_GRASP_POS * GRIPPER_MAX_WIDTH_M


def sources(source_root: Path, trajectory: int) -> Path:
    return source_root / f"traj_{trajectory}" / "data.h5"


def build_one(
    trajectory: int,
    out_dir: Path,
    source_root: Path,
    visual_root: Path,
) -> dict:
    src_h5 = sources(source_root, trajectory)
    if not src_h5.is_file():
        raise FileNotFoundError(f"missing source h5: {src_h5}")
    out_dir.mkdir(parents=True, exist_ok=True)
    dst_h5 = out_dir / "data.h5"
    with h5py.File(src_h5, "r") as fs, h5py.File(dst_h5, "w") as fd:
        if str(fs.attrs.get("force_alignment_version", "")) != FORCE_ALIGNMENT_VERSION:
            raise RuntimeError(
                f"{src_h5}: expected {FORCE_ALIGNMENT_VERSION} source force contract"
            )
        for key in (FORCE_MODEL_SOURCE, FORCE_CLEAN_SOURCE):
            if key not in fs:
                raise KeyError(f"{src_h5}: missing required aligned force stream {key}")
        fd.attrs["success"] = True
        fd.attrs["data_source"] = "isaac_sim_replay"
        fd.attrs["force_field"] = "obs/state/ee_wrench_base"
        fd.attrs["force_source"] = FORCE_MODEL_SOURCE
        fd.attrs["force_clean_source"] = FORCE_CLEAN_SOURCE
        fd.attrs["force_alignment"] = FORCE_ALIGNMENT_VERSION
        fd.attrs["force_frame"] = "robot_base_O"
        fd.attrs["force_torque_mode"] = "zero"
        fd.attrs["force_amplitude_mapping"] = "identity"
        fd.attrs["force_fixed_delay_frames"] = 0
        fd.attrs["paired_real_source"] = f"real-data-aligned/traj_{trajectory}/data.h5"
        state = fd.create_group("obs/state")
        for key in REAL_STATE_FIELDS:
            fs.copy(fs[f"obs/state/{key}"], state, name=key)
        fs.copy(fs["action"], fd, name="action")
        fs.copy(fs["timestamps"], fd, name="timestamps")
        fs.copy(fs["info"], fd, name="info")
        fs.copy(fs["meta"], fd, name="meta")
        fd["obs/state/gripper_pos"][...] = GRIPPER_GRASP_POS
        fd["obs/state/gripper_width_m"][...] = GRIPPER_GRASP_WIDTH_M
        fd["action/actual/gripper"][...] = GRIPPER_GRASP_POS
        fd["action/policy/gripper"][...] = GRIPPER_GRASP_POS
        fd.attrs["gripper_unified_pos"] = GRIPPER_GRASP_POS
        force = fs[FORCE_MODEL_SOURCE][:, :3].astype(np.float64)
        force_clean = fs[FORCE_CLEAN_SOURCE][:, :3].astype(np.float64)
        if force.shape != force_clean.shape or not np.isfinite(force).all():
            raise ValueError(f"{src_h5}: invalid clean/model force streams")
        wrench = np.zeros((force.shape[0], 6), dtype=np.float64)
        wrench[:, :3] = force
        fd["obs/state/ee_wrench_base"][...] = wrench
        fd["obs/state/ee_force_base"][...] = force
        fd["obs/state/ee_torque_base"][...] = 0.0
        for path, role in (
            ("obs/state/ee_wrench_base", "model_input"),
            ("obs/state/ee_force_base", "model_input"),
            ("obs/state/ee_torque_base", "zero_compatibility_channel"),
        ):
            dataset = fd[path]
            for attr in list(dataset.attrs):
                del dataset.attrs[attr]
            dataset.attrs["source"] = FORCE_MODEL_SOURCE
            dataset.attrs["clean_source"] = FORCE_CLEAN_SOURCE
            dataset.attrs["frame"] = "robot_base_O"
            dataset.attrs["role"] = role
            dataset.attrs["online_causal_alignment"] = True
            dataset.attrs["torque_zeroed"] = True
        sim_force = fd.require_group("sim").require_group("force")
        source_force = fs["sim/force"]
        for name in source_force:
            if name.startswith("panda_link7_"):
                source_force.copy(source_force[name], sim_force, name=name)
        for name, value in source_force.attrs.items():
            sim_force.attrs[name] = value
        sim_force.attrs["training_contract_source"] = "panda_link7_force_model_base"
        sim_force.attrs["online_causal_alignment"] = True
        sim_force.attrs["torque_zeroed"] = True
        frames = int(force.shape[0])
        norm = np.linalg.norm(force, axis=1)
    for name in ("front_camera.mp4", "wrist_camera.mp4"):
        src_v = visual_root / f"traj_{trajectory}" / name
        if not src_v.is_file():
            raise FileNotFoundError(f"missing video to symlink: {src_v}")
        link = out_dir / name
        if link.is_symlink() or link.exists():
            link.unlink()
        link.symlink_to(Path(os.path.relpath(src_v.resolve(), start=out_dir.resolve())))
    return {
        "trajectory": trajectory, "frames": frames,
        "force_source": FORCE_MODEL_SOURCE,
        "force_clean_source": FORCE_CLEAN_SOURCE,
        "force_alignment": FORCE_ALIGNMENT_VERSION,
        "amplitude_mapping": "identity",
        "force_norm_p95_n": float(np.quantile(norm, 0.95)),
        "force_norm_max_n": float(norm.max()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--visual-root", type=Path, default=SIM_BUILT)
    parser.add_argument("--output-root", type=Path, default=OUT_ROOT)
    args = parser.parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    manifest = []
    for trajectory in range(40):
        info = build_one(
            trajectory,
            args.output_root / f"traj_{trajectory}",
            args.source_root,
            args.visual_root,
        )
        manifest.append(info)
        print(f"[{trajectory:02d}/39] frames={info['frames']:4d} "
              f"|F|p95={info['force_norm_p95_n']:.2f}N |F|max={info['force_norm_max_n']:.2f}N", flush=True)
    (args.output_root / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    p95 = np.mean([m["force_norm_p95_n"] for m in manifest])
    print(f"[DONE] {len(manifest)} trajectories | mean |F|p95={p95:.2f}N "
          f"(source={FORCE_MODEL_SOURCE}, alignment={FORCE_ALIGNMENT_VERSION}, "
          f"amplitude=identity, torque=0) -> {args.output_root}")


if __name__ == "__main__":
    main()
