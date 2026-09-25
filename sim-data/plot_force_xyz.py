#!/usr/bin/env python3
"""Per-axis Fx/Fy/Fz overlay: REAL vs SIM on the aligned datasets.

For each selected trajectory it draws three stacked panels (Fx, Fy, Fz) with the
REAL force (solid red) and the SIM force (dashed blue) on one time axis, and shades
the labelled free-space (no-contact) interval. Both datasets are frame-aligned, so a
single time axis is exact. Force is read from obs/state/ee_wrench_base[:, :3] (the
training channel), which after alignment is contact-isolated on both sides.

Defaults to real-data-aligned / sim-data-aligned; free-space spans come from
real_data/free_space_intervals.json (same intervals the real baseline fit used).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path("/home/sutai/workspace/gujiawei/isaac_env")
REAL_COLOR = "#d62728"   # red  = real robot
SIM_COLOR = "#1f77b4"    # blue = simulation


def read_force(path: Path):
    with h5py.File(path, "r") as f:
        w = f["obs/state/ee_wrench_base"][:, :3].astype(np.float64)
        t = f["timestamps"][:].astype(np.float64)
    return w, t - t[0]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--real-root", type=Path, default=ROOT / "real-data-aligned")
    ap.add_argument("--sim-root", type=Path, default=ROOT / "sim-data-aligned")
    ap.add_argument("--trajs", type=str, default="0,5,12,27,38")
    ap.add_argument("--output", type=Path,
                    default=ROOT / "outputs/real_sim_aligned_compare/force_xyz_real_vs_sim.png")
    args = ap.parse_args()
    trajs = [int(s) for s in args.trajs.split(",") if s.strip()]

    fs = json.loads((ROOT / "real_data/free_space_intervals.json").read_text())["trajectories"]
    ncol, axis_names = len(trajs), ("Fx", "Fy", "Fz")
    fig, axes = plt.subplots(3, ncol, figsize=(3.4 * ncol, 7.2), dpi=110, squeeze=False)
    for c, t in enumerate(trajs):
        rw, tr = read_force(args.real_root / f"traj_{t}/data.h5")
        sw, _ = read_force(args.sim_root / f"traj_{t}/data.h5")
        try:
            a, b = fs[f"traj_{t}"]["free_space_frames"][0]
        except (KeyError, IndexError):
            a, b = None, None
        for r in range(3):
            ax = axes[r][c]
            ax.plot(tr, rw[:, r], color=REAL_COLOR, lw=1.5, label="REAL")
            ax.plot(tr, sw[:, r], color=SIM_COLOR, lw=1.5, ls="--", label="SIM")
            if a is not None and b is not None and b <= len(tr):
                ax.axvspan(tr[a], tr[min(b, len(tr)) - 1], color="0.6", alpha=0.15,
                           label="free-space")
            ax.grid(alpha=0.25)
            ax.set_xlim(tr[0], tr[-1])
            if c == 0:
                ax.set_ylabel(f"{axis_names[r]} [N]")
            if r == 0:
                ax.set_title(f"traj_{t}", fontsize=10)
                if c == 0:
                    ax.legend(loc="upper left", fontsize=7)
            if r == 2:
                ax.set_xlabel("time [s]")
    fig.suptitle("REAL vs SIM per-axis force  obs/state/ee_wrench_base  (aligned, frame-synced)",
                 fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output)
    plt.close(fig)
    print(f"[DONE] per-axis force figure -> {args.output}")


if __name__ == "__main__":
    main()
