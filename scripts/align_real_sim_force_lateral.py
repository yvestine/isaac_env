#!/usr/bin/env python3
"""Compare real and paired Isaac Sim wrench trends with motion compensation.

The real Franka wrench is a model-based estimate, so a moving, non-contact
robot can still report a pose- and velocity-dependent wrench.  This script
fits that free-space component from the first few seconds of the other real
trajectories, subtracts it from traj0, and then compares the resulting contact
residual with the paired contact/no-hole simulation difference.

This is an analysis/deployment-candidate script.  It does not modify the
replay or TAVLA observation path.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import butter, sosfiltfilt


NAMES = ("Fx", "Fy", "Fz", "Tx", "Ty", "Tz")


def load_real(path: Path) -> dict[str, np.ndarray]:
    with h5py.File(path, "r") as h5:
        q = np.asarray(h5["obs/state/joint_pos"], dtype=np.float64)
        wrench = np.asarray(h5["obs/state/ee_wrench_base"], dtype=np.float64)
        timestamps = np.asarray(h5["timestamps"], dtype=np.float64)
    timestamps = timestamps - timestamps[0]
    if q.ndim != 2 or q.shape[1] != 7 or wrench.shape != (len(q), 6):
        raise ValueError(f"{path}: invalid q/wrench shapes {q.shape}, {wrench.shape}")
    dt = float(np.median(np.diff(timestamps)))
    if not np.isfinite(dt) or dt <= 0.0:
        raise ValueError(f"{path}: invalid timestamps")
    qd = np.gradient(q, dt, axis=0, edge_order=1)
    qdd = np.gradient(qd, dt, axis=0, edge_order=1)
    # sin/cos terms are needed because gravity loading is nonlinear in q.
    features = np.concatenate((q, qd, qdd, np.sin(q), np.cos(q)), axis=1)
    return {"q": q, "wrench": wrench, "timestamps": timestamps, "features": features}


def standardize_features(features: np.ndarray, mean: np.ndarray | None = None, scale: np.ndarray | None = None):
    if mean is None:
        mean = features.mean(axis=0)
    if scale is None:
        scale = features.std(axis=0)
    scale = np.asarray(scale, dtype=np.float64).copy()
    scale[scale < 1.0e-8] = 1.0
    return (features - mean) / scale, mean, scale


def fit_free_space_model(
    trajectories: list[dict[str, np.ndarray]],
    holdout: int,
    free_seconds: float,
    ridge: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    train_features = []
    train_targets = []
    for index, data in enumerate(trajectories):
        if index == holdout:
            continue
        mask = data["timestamps"] <= free_seconds
        train_features.append(data["features"][mask])
        train_targets.append(data["wrench"][mask])
    features = np.concatenate(train_features, axis=0)
    targets = np.concatenate(train_targets, axis=0)
    normalized, mean, scale = standardize_features(features)
    design = np.concatenate((np.ones((len(normalized), 1)), normalized), axis=1)
    regularizer = np.eye(design.shape[1], dtype=np.float64) * float(ridge)
    regularizer[0, 0] = 0.0
    coef = np.linalg.solve(design.T @ design + regularizer, design.T @ targets)
    return mean, scale, coef


def predict_free_space(data: dict[str, np.ndarray], mean: np.ndarray, scale: np.ndarray, coef: np.ndarray) -> np.ndarray:
    normalized, _, _ = standardize_features(data["features"], mean, scale)
    design = np.concatenate((np.ones((len(normalized), 1)), normalized), axis=1)
    return design @ coef


def lowpass(values: np.ndarray, sample_hz: float, cutoff_hz: float) -> np.ndarray:
    if cutoff_hz <= 0.0 or cutoff_hz >= sample_hz / 2.0:
        return values.copy()
    sos = butter(2, cutoff_hz, btype="low", fs=sample_hz, output="sos")
    return sosfiltfilt(sos, values, axis=0)


def read_csv(path: Path) -> np.ndarray:
    values = np.loadtxt(path, delimiter=",", skiprows=1, ndmin=2).astype(np.float64)
    if values.shape[1] != 6 or not np.isfinite(values).all():
        raise ValueError(f"{path}: expected finite Nx6 CSV")
    return values


def read_time(path: Path) -> np.ndarray:
    return np.loadtxt(path, delimiter=",", skiprows=1, ndmin=1).astype(np.float64).reshape(-1)


def paired(values_a: np.ndarray, values_b: np.ndarray, shift: int) -> tuple[np.ndarray, np.ndarray]:
    if shift > 0:
        return values_a[shift:], values_b[:-shift]
    if shift < 0:
        return values_a[:shift], values_b[-shift:]
    return values_a, values_b


def corr(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 3 or np.std(a) < 1.0e-10 or np.std(b) < 1.0e-10:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def metrics(real: np.ndarray, sim: np.ndarray) -> dict[str, float]:
    real_xy = np.linalg.norm(real[:, :2], axis=1)
    sim_xy = np.linalg.norm(sim[:, :2], axis=1)
    real_norm = np.linalg.norm(real[:, :3], axis=1)
    sim_norm = np.linalg.norm(sim[:, :3], axis=1)
    return {
        **{f"{name}_pearson": corr(real[:, i], sim[:, i]) for i, name in enumerate(NAMES)},
        "Fxy_norm_pearson": corr(real_xy, sim_xy),
        "F_norm_pearson": corr(real_norm, sim_norm),
        "dFxy_norm_pearson": corr(np.diff(real_xy), np.diff(sim_xy)),
        "dF_norm_pearson": corr(np.diff(real_norm), np.diff(sim_norm)),
    }


def write_csv(path: Path, time: np.ndarray, real: np.ndarray, sim: np.ndarray) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["timestamp", *[f"real_{n}" for n in NAMES], *[f"sim_{n}" for n in NAMES]])
        for t, r, s in zip(time, real, sim):
            writer.writerow([float(t), *r.tolist(), *s.tolist()])


def sim_to_real_force(sim_wrench: np.ndarray, rotation: np.ndarray) -> np.ndarray:
    # The position translation changes torque reference, not the force vector.
    # For the lateral-force diagnosis we intentionally transform forces only.
    return (rotation.T @ sim_wrench[:, :3].T).T


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("real_data"))
    parser.add_argument("--trajectory", type=int, default=0)
    parser.add_argument("--sim-contact", type=Path, default=Path("outputs/traj0"))
    parser.add_argument("--sim-no-contact", type=Path, default=Path("outputs/wrench_alignment/traj0_no_hole"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/wrench_alignment/traj0_lateral_fix"))
    parser.add_argument("--free-seconds", type=float, default=3.0)
    parser.add_argument("--offset-seconds", type=float, default=0.5)
    parser.add_argument("--cutoff-hz", type=float, default=1.0)
    parser.add_argument("--max-lag-seconds", type=float, default=1.0)
    parser.add_argument("--ridge", type=float, default=1.0e-2)
    args = parser.parse_args()

    paths = sorted(args.data_dir.glob("traj_*/data.h5"), key=lambda p: int(p.parent.name.removeprefix("traj_")))
    if not paths:
        raise FileNotFoundError(f"No real trajectories under {args.data_dir}")
    trajectory_ids = [int(path.parent.name.removeprefix("traj_")) for path in paths]
    if args.trajectory not in trajectory_ids:
        raise ValueError(f"trajectory {args.trajectory} not found; available={trajectory_ids}")
    holdout = trajectory_ids.index(args.trajectory)
    trajectories = [load_real(path) for path in paths]
    target = trajectories[holdout]

    feature_mean, feature_scale, coef = fit_free_space_model(
        trajectories, holdout, args.free_seconds, args.ridge
    )
    predicted_free = predict_free_space(target, feature_mean, feature_scale, coef)
    offset_mask = target["timestamps"] <= args.offset_seconds
    target_offset = np.median(target["wrench"][offset_mask] - predicted_free[offset_mask], axis=0)
    real_contact = target["wrench"] - predicted_free - target_offset
    real_rate = 1.0 / float(np.median(np.diff(target["timestamps"])))
    real_contact = lowpass(real_contact, real_rate, args.cutoff_hz)

    contact = read_csv(args.sim_contact / "wrench_base.csv")
    no_contact = read_csv(args.sim_no_contact / "wrench_base.csv")
    sim_t = read_time(args.sim_contact / "timestamps.csv")
    no_contact_t = read_time(args.sim_no_contact / "timestamps.csv")
    if contact.shape != no_contact.shape or not np.allclose(sim_t, no_contact_t):
        raise ValueError("paired simulation files are not frame aligned")
    metadata = json.loads((args.sim_contact / "replay_metadata.json").read_text(encoding="utf-8"))
    rotation = np.asarray(metadata["real_to_sim_xyz"]["real_to_sim_rotation"], dtype=np.float64)
    sim_contact = sim_to_real_force(contact - no_contact, rotation)
    sim_rate = 1.0 / float(np.median(np.diff(sim_t)))
    sim_contact = lowpass(sim_contact, sim_rate, args.cutoff_hz)
    sim_on_real_grid = np.column_stack(
        [np.interp(target["timestamps"], sim_t, sim_contact[:, axis]) for axis in range(3)]
    )

    max_shift = int(round(args.max_lag_seconds * real_rate))
    best = None
    for shift in range(-max_shift, max_shift + 1):
        real_pair, sim_pair = paired(real_contact[:, :3], sim_on_real_grid, shift)
        values = metrics(np.column_stack((real_pair, np.zeros((len(real_pair), 3)))), np.column_stack((sim_pair, np.zeros((len(sim_pair), 3)))))
        score_values = [values["Fz_pearson"], values["Fxy_norm_pearson"], values["dFxy_norm_pearson"]]
        finite = [value for value in score_values if np.isfinite(value)]
        score = float(np.mean(finite)) if finite else float("nan")
        if best is None or score > best[0]:
            best = (score, shift)
    _, best_shift = best
    real_pair, sim_pair = paired(real_contact[:, :3], sim_on_real_grid, best_shift)
    result_metrics = metrics(
        np.column_stack((real_pair, np.zeros((len(real_pair), 3)))),
        np.column_stack((sim_pair, np.zeros((len(sim_pair), 3)))),
    )

    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    write_csv(output / "traj0_lateral_force_comparison.csv", target["timestamps"][best_shift:] if best_shift >= 0 else target["timestamps"][:best_shift], real_pair, sim_pair)
    (output / "lateral_force_metrics.json").write_text(
        json.dumps(
            {
                "method": "cross-trajectory free-space q/qd/qdd/sin(q)/cos(q) ridge model + initial offset + paired sim residual",
                "trajectory": args.trajectory,
                "free_seconds": args.free_seconds,
                "offset_seconds": args.offset_seconds,
                "cutoff_hz": args.cutoff_hz,
                "sim_lead_seconds": best_shift / real_rate,
                "metrics": result_metrics,
                "real_contact_std_N": real_contact[:, :3].std(axis=0).tolist(),
                "sim_contact_std_N": sim_on_real_grid.std(axis=0).tolist(),
                "real_to_sim_rotation": rotation.tolist(),
                "note": "No scalar Fz display scale is applied to Fx/Fy; forces are compared in the transformed real base frame.",
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    time = target["timestamps"]
    fig, axes = plt.subplots(3, 2, figsize=(15, 11), sharex=True)
    for index, name in enumerate(("Fx", "Fy", "Fz")):
        axis = axes[index, 0]
        axis.plot(time, real_contact[:, index], color="#d62728", label="real motion-compensated")
        axis.plot(time, sim_on_real_grid[:, index], color="#1f77b4", linestyle="--", label="sim contact residual")
        axis.set_ylabel(f"{name} [N]")
        axis.set_title(name)
        axis.grid(alpha=0.25)
        if index == 0:
            axis.legend(fontsize=9)
        axis = axes[index, 1]
        real_std = max(float(np.std(real_contact[:, index])), 1.0e-9)
        sim_std = max(float(np.std(sim_on_real_grid[:, index])), 1.0e-9)
        axis.plot(time, real_contact[:, index] / real_std, color="#d62728", label="real z-score")
        axis.plot(time, sim_on_real_grid[:, index] / sim_std, color="#1f77b4", linestyle="--", label="sim z-score")
        axis.set_ylabel("standardized")
        axis.set_title(f"{name} shape")
        axis.grid(alpha=0.25)
        if index == 0:
            axis.legend(fontsize=9)

    real_xy = np.linalg.norm(real_contact[:, :2], axis=1)
    sim_xy = np.linalg.norm(sim_on_real_grid[:, :2], axis=1)
    axis = axes[0, 1]
    axis.clear()
    axis.plot(time, real_xy / max(float(real_xy.std()), 1.0e-9), color="#d62728", label="real |Fxy| z-score")
    axis.plot(time, sim_xy / max(float(sim_xy.std()), 1.0e-9), color="#1f77b4", linestyle="--", label="sim |Fxy| z-score")
    axis.set_title(f"|Fxy| trend r={result_metrics['Fxy_norm_pearson']:.3f}")
    axis.set_ylabel("standardized")
    axis.grid(alpha=0.25)
    axis.legend(fontsize=9)

    axis = axes[1, 1]
    axis.axis("off")
    summary = "\n".join(
        (
            "Lateral-force diagnostic",
            f"free-space model: {args.free_seconds:.1f}s / 39 trajectories",
            f"sim leads real: {best_shift / real_rate:.2f}s",
            f"Fx r: {result_metrics['Fx_pearson']:.3f}",
            f"Fy r: {result_metrics['Fy_pearson']:.3f}",
            f"Fz r: {result_metrics['Fz_pearson']:.3f}",
            f"|Fxy| r: {result_metrics['Fxy_norm_pearson']:.3f}",
            f"d|Fxy| r: {result_metrics['dFxy_norm_pearson']:.3f}",
        )
    )
    axis.text(0.02, 0.95, summary, va="top", family="monospace", fontsize=12)

    axis = axes[2, 0]
    axis.plot(time, real_xy, color="#d62728", label="real |Fxy|")
    axis.plot(time, sim_xy, color="#1f77b4", linestyle="--", label="sim |Fxy|")
    axis.set_xlabel("Time [s]")
    axis.set_ylabel("|Fxy| [N]")
    axis.set_title("Lateral force magnitude")
    axis.grid(alpha=0.25)
    axis.legend(fontsize=9)

    axis = axes[2, 1]
    axis.scatter(sim_pair[:, 0], real_pair[:, 0], s=12, alpha=0.5, label="Fx")
    axis.scatter(sim_pair[:, 1], real_pair[:, 1], s=12, alpha=0.5, label="Fy")
    limits = np.array([min(axis.get_xlim()[0], axis.get_ylim()[0]), max(axis.get_xlim()[1], axis.get_ylim()[1])])
    axis.plot(limits, limits, "k:")
    axis.set_xlabel("simulation force [N]")
    axis.set_ylabel("real residual force [N]")
    axis.set_title("Lateral force scatter")
    axis.grid(alpha=0.25)
    axis.legend(fontsize=9)

    fig.suptitle("traj0 lateral force: motion-compensated real vs paired Isaac Sim", fontsize=16)
    fig.tight_layout()
    fig.savefig(output / "lateral_force_alignment.png", dpi=180)
    plt.close(fig)
    print(f"[DONE] {output / 'lateral_force_alignment.png'}")
    print(json.dumps(result_metrics, indent=2))


if __name__ == "__main__":
    main()
