#!/usr/bin/env python3
"""Fit the deployable q/qd baselines and real-sensor AR(1) residual model."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np

from tacex_tasks.real2sim.force_alignment import (
    FORCE_ALIGNMENT_VERSION,
    ForceAlignmentConfig,
    fit_ar1_noise,
    fit_ridge_motion_baseline,
    process_force_series,
)


TRAIN_IDS = tuple(range(1, 32))
VALIDATION_IDS = tuple(range(32, 40))
HOLDOUT_IDS = (0,)


def read_named_csv(path: Path) -> tuple[list[str], np.ndarray]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        header = next(reader, None)
        rows = [[float(value) for value in row] for row in reader if row]
    values = np.asarray(rows, dtype=np.float64)
    if not header or values.ndim != 2 or values.shape[1] != len(header):
        raise ValueError(f"{path}: invalid named CSV")
    if not np.isfinite(values).all():
        raise ValueError(f"{path}: contains NaN or Inf")
    return header, values


def columns(header: list[str], values: np.ndarray, names: list[str]) -> np.ndarray:
    lookup = {name: index for index, name in enumerate(header)}
    missing = [name for name in names if name not in lookup]
    if missing:
        raise KeyError(f"missing CSV columns: {missing}")
    return values[:, [lookup[name] for name in names]]


def read_real(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    with h5py.File(path, "r") as data:
        time = np.asarray(data["timestamps"][:], dtype=np.float64)
        q = np.asarray(data["obs/state/joint_pos"][:], dtype=np.float64)
        qd = np.asarray(data["obs/state/joint_vel"][:], dtype=np.float64)
        force = np.asarray(data["obs/state/ee_wrench_base"][:, :3], dtype=np.float64)
    time -= time[0]
    if q.shape != qd.shape or q.shape != (len(time), 7) or force.shape != (len(time), 3):
        raise ValueError(f"{path}: unexpected real force/joint shapes")
    return time, q, qd, force


def read_sim_no_contact(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    force_header, force_values = read_named_csv(
        path / "panda_link7_force_alignment_120hz.csv"
    )
    joint_header, joint_values = read_named_csv(
        path / "panda_link7_joint_state_120hz.csv"
    )
    time = columns(force_header, force_values, ["timestamp"])[:, 0]
    joint_time = columns(joint_header, joint_values, ["timestamp"])[:, 0]
    tolerance = 1.0e-8 + 1.0e-5 * max(float(np.max(np.abs(time))), 1.0)
    if time.shape != joint_time.shape or not np.allclose(time, joint_time, atol=tolerance):
        raise ValueError(f"{path}: high-rate force and joint timestamps differ")
    force = columns(
        force_header,
        force_values,
        [f"raw_base_{axis}" for axis in ("Fx", "Fy", "Fz")],
    )
    q = columns(joint_header, joint_values, [f"q{index}" for index in range(7)])
    qd = columns(joint_header, joint_values, [f"qd{index}" for index in range(7)])
    return time - time[0], q, qd, force


def validate_direction_test(path: Path) -> dict:
    result = json.loads(path.read_text(encoding="utf-8"))
    summary = result.get("summary", {})
    if summary.get("passed") is not True or summary.get("sign_consistent") is not True:
        raise RuntimeError("directed panda_link7 +/-XYZ test did not pass")
    if result.get("incoming_wrench_read_body") != "panda_link7":
        raise ValueError("direction test was not read at panda_link7")
    directions = result.get("directions")
    if not isinstance(directions, list) or len(directions) != 6:
        raise ValueError("direction test must contain six +/-XYZ entries")
    for item in directions:
        leakage = float(item.get("cross_axis_leakage_ratio", np.inf))
        if item.get("passed") is not True or leakage > 0.15 + 1.0e-12:
            raise RuntimeError(
                f"direction test exceeds 15% cross-axis leakage: "
                f"{item.get('label', '?')}={leakage:.6f}"
            )
    return result


def rmse(target: np.ndarray, estimate: np.ndarray) -> float:
    return float(np.sqrt(np.mean((target - estimate) ** 2)))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--real-data-dir", type=Path, default=Path("real_data"))
    parser.add_argument("--free-space-intervals", type=Path, default=Path("real_data/free_space_intervals.json"))
    parser.add_argument("--sim-no-contact-root", type=Path, required=True)
    parser.add_argument("--direction-test", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/force_alignment/force_alignment_config.json"),
    )
    parser.add_argument("--ridge", type=float, default=1.0e-3)
    parser.add_argument("--cutoff-hz", type=float, default=0.35)
    parser.add_argument("--tare-seconds", type=float, default=3.0)
    parser.add_argument("--output-hz", type=float, default=10.0)
    parser.add_argument("--noise-seed", type=int, default=0)
    args = parser.parse_args()
    if args.ridge < 0.0:
        raise ValueError("--ridge must be non-negative")

    direction_result = validate_direction_test(args.direction_test)
    intervals = json.loads(args.free_space_intervals.read_text(encoding="utf-8"))[
        "trajectories"
    ]
    real_blocks = {}
    sim_blocks = {}
    for trajectory in range(40):
        real_blocks[trajectory] = read_real(
            args.real_data_dir / f"traj_{trajectory}" / "data.h5"
        )
        sim_blocks[trajectory] = read_sim_no_contact(
            args.sim_no_contact_root / f"traj_{trajectory}"
        )

    real_q, real_qd, real_force = [], [], []
    sim_q, sim_qd, sim_force = [], [], []
    for trajectory in TRAIN_IDS:
        _time, q, qd, force = real_blocks[trajectory]
        start, stop = intervals[f"traj_{trajectory}"]["free_space_frames"][0]
        real_q.append(q[start:stop])
        real_qd.append(qd[start:stop])
        real_force.append(force[start:stop])
        _time, q, qd, force = sim_blocks[trajectory]
        sim_q.append(q)
        sim_qd.append(qd)
        sim_force.append(force)

    real_baseline = fit_ridge_motion_baseline(
        np.concatenate(real_q),
        np.concatenate(real_qd),
        np.concatenate(real_force),
        ridge=args.ridge,
    )
    sim_baseline = fit_ridge_motion_baseline(
        np.concatenate(sim_q),
        np.concatenate(sim_qd),
        np.concatenate(sim_force),
        ridge=args.ridge,
    )
    provisional = ForceAlignmentConfig(
        cutoff_hz=float(args.cutoff_hz),
        tare_seconds=float(args.tare_seconds),
        output_hz=float(args.output_hz),
        sim_baseline=sim_baseline,
        real_baseline=real_baseline,
        noise_seed=int(args.noise_seed),
    )
    real_residual_blocks = []
    for trajectory in TRAIN_IDS:
        time, q, qd, force = real_blocks[trajectory]
        aligned = process_force_series(
            time,
            force,
            q,
            joint_vel=qd,
            config=provisional,
            domain="real",
        )
        start, stop = intervals[f"traj_{trajectory}"]["free_space_frames"][0]
        block = aligned["force_clean"][start:stop]
        real_residual_blocks.append(block - np.median(block, axis=0, keepdims=True))
    noise = fit_ar1_noise(real_residual_blocks)
    config = ForceAlignmentConfig(
        cutoff_hz=float(args.cutoff_hz),
        tare_seconds=float(args.tare_seconds),
        output_hz=float(args.output_hz),
        sim_baseline=sim_baseline,
        real_baseline=real_baseline,
        sim_noise=noise,
        noise_seed=int(args.noise_seed),
    )

    def split_metrics(ids: tuple[int, ...]) -> dict[str, float]:
        real_target, real_estimate, sim_target, sim_estimate = [], [], [], []
        for trajectory in ids:
            _time, q, qd, force = real_blocks[trajectory]
            start, stop = intervals[f"traj_{trajectory}"]["free_space_frames"][0]
            real_target.append(force[start:stop])
            real_estimate.append(real_baseline.predict(q[start:stop], qd[start:stop]))
            _time, q, qd, force = sim_blocks[trajectory]
            sim_target.append(force)
            sim_estimate.append(sim_baseline.predict(q, qd))
        return {
            "real_no_contact_baseline_rmse_n": rmse(
                np.concatenate(real_target), np.concatenate(real_estimate)
            ),
            "sim_no_contact_baseline_rmse_n": rmse(
                np.concatenate(sim_target), np.concatenate(sim_estimate)
            ),
        }

    result = config.to_dict()
    result["fit"] = {
        "train_trajectories": list(TRAIN_IDS),
        "validation_trajectories": list(VALIDATION_IDS),
        "holdout_trajectories": list(HOLDOUT_IDS),
        "holdout_used_for_fit": False,
        "ridge": float(args.ridge),
        "amplitude_mapping": "identity",
        "direction_test": {
            "path": str(args.direction_test.resolve()),
            "sha256": hashlib.sha256(args.direction_test.read_bytes()).hexdigest(),
            "summary": direction_result.get("summary", {}),
        },
        "train_metrics": split_metrics(TRAIN_IDS),
        "validation_metrics": split_metrics(VALIDATION_IDS),
        "holdout_metrics": split_metrics(HOLDOUT_IDS),
    }
    if result["version"] != FORCE_ALIGNMENT_VERSION:
        raise RuntimeError("internal force-alignment version mismatch")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
