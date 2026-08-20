#!/usr/bin/env python3
"""Fit a small empirical Franka wrench sensor model from real trajectories."""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np


def load_trajectory(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with h5py.File(path, "r") as h5:
        q = np.asarray(h5["obs/state/joint_pos"], dtype=np.float64)
        wrench = np.asarray(h5["obs/state/ee_wrench_base"], dtype=np.float64)
        timestamps = np.asarray(h5["timestamps"], dtype=np.float64)
    if q.ndim != 2 or q.shape[1] != 7 or wrench.shape != (len(q), 6):
        raise ValueError(f"{path}: expected q=(N,7), wrench=(N,6), got {q.shape}, {wrench.shape}")
    if len(q) < 3 or len(timestamps) != len(q) or not np.isfinite(q).all() or not np.isfinite(wrench).all():
        raise ValueError(f"{path}: invalid or too-short trajectory")
    dt = float(np.median(np.diff(timestamps)))
    if not np.isfinite(dt) or dt <= 0.0:
        raise ValueError(f"{path}: timestamps are not strictly increasing")
    qd = np.gradient(q, dt, axis=0)
    qdd = np.gradient(qd, dt, axis=0)
    features = np.concatenate((q, qd, qdd), axis=1)
    return features, wrench, timestamps


def fit_model(features: np.ndarray, targets: np.ndarray, ridge: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = features.mean(axis=0)
    scale = features.std(axis=0)
    scale[scale < 1.0e-8] = 1.0
    normalized = (features - mean) / scale
    design = np.concatenate((np.ones((len(normalized), 1)), normalized), axis=1)
    regularizer = np.eye(design.shape[1], dtype=np.float64) * float(ridge)
    regularizer[0, 0] = 0.0
    coef = np.linalg.solve(design.T @ design + regularizer, design.T @ targets)
    return mean, scale, coef


def predict(features: np.ndarray, mean: np.ndarray, scale: np.ndarray, coef: np.ndarray) -> np.ndarray:
    normalized = (features - mean) / scale
    design = np.concatenate((np.ones((len(normalized), 1)), normalized), axis=1)
    return design @ coef


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--holdout", type=int, default=None, help="exclude traj_N and print its validation metrics")
    parser.add_argument("--ridge", type=float, default=1.0e-3)
    args = parser.parse_args()

    paths = sorted(
        args.data_dir.glob("traj_*/data.h5"),
        key=lambda path: int(path.parent.name.removeprefix("traj_")),
    )
    if not paths:
        raise FileNotFoundError(f"no traj_*/data.h5 found under {args.data_dir}")
    train_features = []
    train_targets = []
    holdout_data = None
    train_ids = []
    for path in paths:
        trajectory_id = int(path.parent.name.removeprefix("traj_"))
        features, wrench, timestamps = load_trajectory(path)
        if args.holdout is not None and trajectory_id == args.holdout:
            holdout_data = (features, wrench)
            continue
        train_ids.append(trajectory_id)
        train_features.append(features)
        train_targets.append(wrench)
    if not train_features:
        raise ValueError("no training trajectories remain")

    features = np.concatenate(train_features, axis=0)
    targets = np.concatenate(train_targets, axis=0)
    mean, scale, coef = fit_model(features, targets, args.ridge)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        feature_mean=mean.astype(np.float32),
        feature_scale=scale.astype(np.float32),
        coef=coef.astype(np.float32),
        feature_definition=np.asarray("[joint_pos, joint_vel, joint_accel]"),
        train_trajectory_ids=np.asarray(train_ids, dtype=np.int32),
        ridge=np.asarray(args.ridge, dtype=np.float32),
    )
    print(f"[WrenchModel] fitted {len(train_ids)} trajectories / {len(features)} frames -> {args.output}")
    if holdout_data is not None:
        predicted = predict(holdout_data[0], mean, scale, coef)
        target = holdout_data[1]
        r2 = 1.0 - np.sum((target - predicted) ** 2, axis=0) / np.sum((target - target.mean(axis=0)) ** 2, axis=0)
        corr = [np.corrcoef(target[:, i], predicted[:, i])[0, 1] for i in range(6)]
        norm_corr = np.corrcoef(np.linalg.norm(target[:, :3], axis=1), np.linalg.norm(predicted[:, :3], axis=1))[0, 1]
        print(f"[WrenchModel] holdout traj_{args.holdout}: r2=" + ",".join(f"{x:.3f}" for x in r2))
        print(f"[WrenchModel] holdout axis_corr=" + ",".join(f"{x:.3f}" for x in corr) + f" norm_corr={norm_corr:.3f}")


if __name__ == "__main__":
    main()
