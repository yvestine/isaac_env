#!/usr/bin/env python3
"""Prepare minimal paired-data files and a success cutoff from one real H5.

The hole reference used by the verified visual rollout is the final peg pose.
Therefore the first frame within 3 mm laterally and 1 mm vertically of the
final recorded end-effector position reproduces the traj0 success cutoff
(frame 253) without launching a third simulation pass.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import h5py
import numpy as np


def write_csv(path: Path, header: list[str], values: np.ndarray) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(np.asarray(values).tolist())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--real-h5", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--trajectory", type=int, required=True)
    parser.add_argument("--xy-threshold-m", type=float, default=0.003)
    parser.add_argument("--z-threshold-m", type=float, default=0.001)
    args = parser.parse_args()
    if not 0 <= args.trajectory <= 39:
        raise ValueError("the successful real dataset contains traj_0..traj_39")

    with h5py.File(args.real_h5, "r") as data:
        q = np.asarray(data["obs/state/joint_pos"][:], dtype=np.float64)
        pose = np.asarray(data["obs/state/ee_pose"][:], dtype=np.float64)
        timestamps = np.asarray(data["timestamps"][:], dtype=np.float64).reshape(-1)
    if q.shape != (len(timestamps), 7) or pose.shape[0] != len(timestamps) or pose.shape[1] < 3:
        raise ValueError("real H5 has invalid joint/pose/timestamp shapes")
    relative_timestamps = timestamps - timestamps[0]
    if not np.isfinite(relative_timestamps).all() or np.any(np.diff(relative_timestamps) <= 0.0):
        raise ValueError("real H5 timestamps must be finite and strictly increasing")
    relative = pose[:, :3] - pose[-1, :3]
    xy_error = np.linalg.norm(relative[:, :2], axis=1)
    z_error = relative[:, 2]
    success = (xy_error <= args.xy_threshold_m) & (np.abs(z_error) <= args.z_threshold_m)
    indices = np.flatnonzero(success)
    if not len(indices):
        raise RuntimeError(f"traj_{args.trajectory} never reaches the declared success region")
    first_success = int(indices[0])

    output = args.output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    write_csv(output / "joint_pos_sim.csv", [f"joint_{index}" for index in range(7)], q)
    # Keep the CSV near zero. package_continuous_ppo_traj0 historically reads
    # numeric CSV values as float32; absolute Unix timestamps around 1.8e9
    # would collapse 0.1 s frame intervals into duplicate values.
    write_csv(output / "timestamps.csv", ["timestamp"], relative_timestamps[:, None])
    write_csv(
        output / "gt_replay_validation.csv",
        ["frame", "timestamp_s", "xy_error_m", "z_disp_m", "success"],
        np.column_stack(
            (
                np.arange(len(q)),
                relative_timestamps,
                xy_error,
                z_error,
                success,
            )
        ),
    )
    metadata = {
        "trajectory": int(args.trajectory),
        "real_source": str(args.real_h5.expanduser().resolve()),
        "replay_mode": "direct",
        "geometric_success": True,
        "first_success_frame": first_success,
        "successful_frame_count": int(np.count_nonzero(success)),
        "success_xy_threshold_m": float(args.xy_threshold_m),
        "success_z_threshold_m": float(args.z_threshold_m),
        "success_reference": "final recorded EE position; equivalent to gt-final peg reference",
        "dataset_semantics": "successful insertion",
    }
    (output / "replay_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print(
        f"[SUCCESS] traj_{args.trajectory} first_success_frame={first_success} "
        f"time={relative_timestamps[first_success]:.3f}s"
    )


if __name__ == "__main__":
    main()
