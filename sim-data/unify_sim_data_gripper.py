#!/usr/bin/env python3
"""Unify the gripper channel in the regenerated sim-data/ replay H5s.

The visual re-render forced a constant grasp pose in the RENDERED pixels
(--gripper-constant 0.0865), but the replay output H5 still stored the raw
recorded gripper (traj_20-39 ~= 0 -> looks closed in the data channel while the
video shows grasping). This unifies the data channel to the same single grasp
value used by build_sim_data.py / build_aligned_sim.py, so every consumer of
sim-data/*.h5 sees a consistent grasp (state + action), matching the pixels.

gripper_pos is normalized (0=closed, 1=open); width = pos * Franka max width 0.08 m.
"""
from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SIM = ROOT / "sim-data"
GRIPPER_MAX_WIDTH_M = 0.08
GRIPPER_GRASP_POS = 0.0865
GRIPPER_GRASP_WIDTH_M = GRIPPER_GRASP_POS * GRIPPER_MAX_WIDTH_M
FIELDS = {
    "obs/state/gripper_pos": GRIPPER_GRASP_POS,
    "obs/state/gripper_width_m": GRIPPER_GRASP_WIDTH_M,
    "action/actual/gripper": GRIPPER_GRASP_POS,
    "action/policy/gripper": GRIPPER_GRASP_POS,
}


def unify_one(path: Path) -> dict:
    before = {}
    with h5py.File(path, "r+") as f:
        for key, value in FIELDS.items():
            if key not in f:
                continue
            before[key] = float(np.asarray(f[key][:]).reshape(-1)[0])
            f[key][...] = value
        f.attrs["gripper_unified_pos"] = GRIPPER_GRASP_POS
    return before


def main() -> None:
    changed = 0
    for t in range(40):
        p = SIM / f"traj_{t}" / "data.h5"
        if not p.is_file():
            raise FileNotFoundError(p)
        before = unify_one(p)
        gp = before.get("obs/state/gripper_pos", float("nan"))
        changed += 1
        print(f"[{t:02d}/39] gripper_pos {gp:.5f} -> {GRIPPER_GRASP_POS:.4f}", flush=True)
    print(f"[DONE] unified gripper on {changed} sim-data trajectories "
          f"(pos={GRIPPER_GRASP_POS}, width={GRIPPER_GRASP_WIDTH_M:.5f} m)")


if __name__ == "__main__":
    main()
