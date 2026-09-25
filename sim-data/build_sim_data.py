#!/usr/bin/env python3
"""Build a clean, real-format-aligned sim-data folder for TAVLA co-training.

Per trajectory traj_0..traj_39 we emit  data.h5 + front_camera.mp4 + wrist_camera.mp4,
mirroring the real_data/ layout so the SAME convert_user_hdf5_to_lerobot.py works.

Force policy (decided from the correct-vs-crooked A/B analysis):
  obs/state/ee_wrench_base  <-  panda_link7_wrench_contact_isolated_base_at_K
  (real-robot-available 6-axis EE wrench, contact-isolated + filtered; ~7.5x
   separation between correct and hole-offset insertion, quiet in free space).

Dropped as useless: the whole sim/force/* debug group (wrench_raw/model/final,
by-body wrenches, 120Hz arrays, contact_total_base 300-800N spikes) and the
insertion_* proxy channels. Only the real-schema fields are kept.
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import h5py
import numpy as np

ROOT = Path("/home/sutai/workspace/gujiawei/isaac_env")
PAIR_ROOT = ROOT / "outputs/link7_success_force_pairs"
VIDEO_ROOT = ROOT / "outputs/sim_force"
TRAJ0_DIR = ROOT / "outputs/traj0_verified_visual_link7_force_smooth/traj_0"
OUT_ROOT = ROOT / "sim-data"

# The exact obs/state field set present in real_data/traj_N/data.h5.
REAL_STATE_FIELDS = [
    "ee_force", "ee_force_base", "ee_force_stiffness",
    "ee_pose",
    "ee_torque", "ee_torque_base", "ee_torque_stiffness",
    "ee_wrench", "ee_wrench_base", "ee_wrench_stiffness",
    "gripper_pos", "gripper_width_m",
    "joint_pos", "joint_torque", "joint_torque_external", "joint_vel",
]
FORCE_SOURCE = "panda_link7_wrench_contact_isolated_base_at_K"

# The peg is rigidly grasped for the whole episode, but the raw data recorded two
# inconsistent constants (traj_0-19 -> ~0.0865, traj_20-39 -> 0.0). Unify all 40 to a
# single grasp value (state + action). gripper_pos is normalized (0=closed, 1=open),
# width = pos * max_width, Franka max width = 0.08 m.
GRIPPER_MAX_WIDTH_M = 0.08
GRIPPER_GRASP_POS = 0.0865
GRIPPER_GRASP_WIDTH_M = GRIPPER_GRASP_POS * GRIPPER_MAX_WIDTH_M


def sources(trajectory: int) -> tuple[Path, Path]:
    if trajectory == 0:
        return TRAJ0_DIR / "data.h5", TRAJ0_DIR
    return PAIR_ROOT / f"traj_{trajectory}/data.h5", VIDEO_ROOT / f"traj_{trajectory}"


def build_one(trajectory: int, out_dir: Path) -> dict:
    src_h5, video_dir = sources(trajectory)
    if not src_h5.is_file():
        raise FileNotFoundError(f"missing source h5: {src_h5}")
    out_dir.mkdir(parents=True, exist_ok=True)
    dst_h5 = out_dir / "data.h5"
    with h5py.File(src_h5, "r") as fs, h5py.File(dst_h5, "w") as fd:
        fd.attrs["success"] = True
        fd.attrs["data_source"] = "isaac_sim_replay"
        fd.attrs["force_field"] = "obs/state/ee_wrench_base"
        fd.attrs["force_source"] = FORCE_SOURCE
        fd.attrs["paired_real_source"] = f"real_data/traj_{trajectory}/data.h5"
        state = fd.create_group("obs/state")
        for key in REAL_STATE_FIELDS:
            fs.copy(fs[f"obs/state/{key}"], state, name=key)
        fs.copy(fs["action"], fd, name="action")
        fs.copy(fs["timestamps"], fd, name="timestamps")
        fs.copy(fs["info"], fd, name="info")
        fs.copy(fs["meta"], fd, name="meta")
        # unify gripper (state + action) to one grasp value across all 40 trajectories
        fd["obs/state/gripper_pos"][...] = GRIPPER_GRASP_POS
        fd["obs/state/gripper_width_m"][...] = GRIPPER_GRASP_WIDTH_M
        fd["action/actual/gripper"][...] = GRIPPER_GRASP_POS
        fd["action/policy/gripper"][...] = GRIPPER_GRASP_POS
        fd.attrs["gripper_unified_pos"] = GRIPPER_GRASP_POS
        wrench = fd["obs/state/ee_wrench_base"][:]
        frames = int(wrench.shape[0])
    for name in ("front_camera.mp4", "wrist_camera.mp4"):
        src_v = video_dir / name
        if not src_v.is_file():
            raise FileNotFoundError(f"missing video: {src_v}")
        shutil.copy2(src_v, out_dir / name)
    norm = np.linalg.norm(wrench[:, :3], axis=1)
    torque = np.linalg.norm(wrench[:, 3:], axis=1)
    return {
        "trajectory": trajectory,
        "frames": frames,
        "force_source": FORCE_SOURCE,
        "force_norm_p95_n": float(np.quantile(norm, 0.95)),
        "force_norm_max_n": float(norm.max()),
        "torque_norm_p95_nm": float(np.quantile(torque, 0.95)),
        "src_h5": str(src_h5.relative_to(ROOT)),
        "video_dir": str(video_dir.relative_to(ROOT)),
    }


def main() -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    manifest = []
    for trajectory in range(40):
        info = build_one(trajectory, OUT_ROOT / f"traj_{trajectory}")
        manifest.append(info)
        print(f"[{trajectory:02d}/39] frames={info['frames']:4d} "
              f"|F|p95={info['force_norm_p95_n']:.3f}N |F|max={info['force_norm_max_n']:.3f}N",
              flush=True)
    (OUT_ROOT / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    total = sum(m["frames"] for m in manifest)
    print(f"[DONE] {len(manifest)} trajectories, {total} frames -> {OUT_ROOT}")


if __name__ == "__main__":
    main()
