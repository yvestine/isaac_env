#!/usr/bin/env python3
"""Build sim-data-aligned/: put the SIM force on the SAME definition/sign/scale as REAL.

The prior sim effort channel (panda_link7_wrench_contact_isolated_base_at_K) did NOT
physically match real: force direction was ~orthogonal (dir-cos ~0), the seating axis Fz
was *anti*-correlated with real, and torque was ~0. We re-derive the sim effort force from
`panda_link7_wrench_calibrated` (a calibrated net contact wrench, free-space ~0), which is
the best real-tracking channel available:
  * Fz per-frame Pearson vs real = -0.48 on 38/40 trajs  -> a consistent CONVENTION flip,
    fixed by SIGN = -1  (after flip: +0.48, best of all channels)
  * force-direction cosine vs real (after flip) ~ +0.56  (vs ~0 for the old channel)
  * magnitude real/sim p95 ~ 3.4x  -> isotropic SCALE = 3.4 to match real magnitude

Torque is dropped (Tx,Ty,Tz := 0): no sim channel carries meaningful contact torque
(~0.1 vs real ~2 N.m), so per the aligned design the 6-axis effort is force-only on both
sides. SCALE is a single documented constant (attrs/manifest); set SCALE=1.0 to defer
magnitude to norm_stats. Everything else mirrors build_sim_data.py; videos are symlinked
to sim-data/ (no duplication). Output mirrors real_data/ so the SAME
convert_user_hdf5_to_lerobot.py works on --raw-dir sim-data-aligned.
"""
from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np

ROOT = Path("/home/sutai/workspace/gujiawei/isaac_env")
PAIR_ROOT = ROOT / "outputs/link7_success_force_pairs"
TRAJ0_DIR = ROOT / "outputs/traj0_verified_visual_link7_force_smooth/traj_0"
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
FORCE_SOURCE = "sim/force/panda_link7_wrench_calibrated"
SIGN = -1.0        # calibrated Fz is anti-correlated with real (38/40 trajs) -> flip
SCALE = 1.0        # physically-correct magnitude; NO artificial scaling (norm_stats handles it)

GRIPPER_MAX_WIDTH_M = 0.08
GRIPPER_GRASP_POS = 0.0865
GRIPPER_GRASP_WIDTH_M = GRIPPER_GRASP_POS * GRIPPER_MAX_WIDTH_M


def sources(trajectory: int) -> Path:
    if trajectory == 0:
        return TRAJ0_DIR / "data.h5"
    return PAIR_ROOT / f"traj_{trajectory}/data.h5"


def build_one(trajectory: int, out_dir: Path) -> dict:
    src_h5 = sources(trajectory)
    if not src_h5.is_file():
        raise FileNotFoundError(f"missing source h5: {src_h5}")
    out_dir.mkdir(parents=True, exist_ok=True)
    dst_h5 = out_dir / "data.h5"
    with h5py.File(src_h5, "r") as fs, h5py.File(dst_h5, "w") as fd:
        fd.attrs["success"] = True
        fd.attrs["data_source"] = "isaac_sim_replay"
        fd.attrs["force_field"] = "obs/state/ee_wrench_base"
        fd.attrs["force_source"] = FORCE_SOURCE
        fd.attrs["force_alignment"] = f"real_aligned:sign={SIGN:+.0f},scale={SCALE:.3f};torque_zeroed"
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
        # re-derive the effort force: sign-flip + isotropic scale onto real's definition
        force = SIGN * SCALE * fs[FORCE_SOURCE][:, :3].astype(np.float64)
        wrench = np.zeros((force.shape[0], 6), dtype=np.float64)
        wrench[:, :3] = force                      # torque stays 0 (dropped)
        fd["obs/state/ee_wrench_base"][...] = wrench
        fd["obs/state/ee_force_base"][...] = force
        fd["obs/state/ee_torque_base"][...] = 0.0
        frames = int(force.shape[0])
        norm = np.linalg.norm(force, axis=1)
    for name in ("front_camera.mp4", "wrist_camera.mp4"):
        src_v = SIM_BUILT / f"traj_{trajectory}" / name
        if not src_v.is_file():
            raise FileNotFoundError(f"missing video to symlink: {src_v}")
        link = out_dir / name
        if link.is_symlink() or link.exists():
            link.unlink()
        link.symlink_to(src_v)
    return {
        "trajectory": trajectory, "frames": frames,
        "force_source": FORCE_SOURCE, "sign": SIGN, "scale": SCALE,
        "force_norm_p95_n": float(np.quantile(norm, 0.95)),
        "force_norm_max_n": float(norm.max()),
    }


def main() -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    manifest = []
    for trajectory in range(40):
        info = build_one(trajectory, OUT_ROOT / f"traj_{trajectory}")
        manifest.append(info)
        print(f"[{trajectory:02d}/39] frames={info['frames']:4d} "
              f"|F|p95={info['force_norm_p95_n']:.2f}N |F|max={info['force_norm_max_n']:.2f}N", flush=True)
    (OUT_ROOT / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    p95 = np.mean([m["force_norm_p95_n"] for m in manifest])
    print(f"[DONE] {len(manifest)} trajectories | mean |F|p95={p95:.2f}N "
          f"(source={FORCE_SOURCE}, sign={SIGN:+.0f}, scale={SCALE:.3f}, torque=0) -> {OUT_ROOT}")


if __name__ == "__main__":
    main()
