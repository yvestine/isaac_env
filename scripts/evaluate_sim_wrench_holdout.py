#!/usr/bin/env python3
"""Holdout report for traj0, truncated at successful insertion."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


NAMES = ("Fx", "Fy", "Fz", "Tx", "Ty", "Tz")


def _time(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    if len(values) < 4 or np.any(np.diff(values) <= 0.0):
        raise ValueError("timestamps must be strictly increasing")
    return values - values[0]


def _features(q: np.ndarray, t: np.ndarray) -> np.ndarray:
    return np.column_stack((np.ones(len(q)), q, np.gradient(q, t, axis=0, edge_order=1)))


def _corr(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 3 or np.std(a) < 1.0e-12 or np.std(b) < 1.0e-12:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def _resample(t: np.ndarray, values: np.ndarray, target: np.ndarray) -> np.ndarray:
    return np.column_stack([np.interp(target, t, values[:, axis]) for axis in range(values.shape[1])])


def _first_contact(t: np.ndarray, force: np.ndarray) -> float | None:
    norm = np.linalg.norm(force[:, :3], axis=1)
    threshold = max(1.0, 0.20 * float(np.quantile(norm, 0.95)))
    indices = np.flatnonzero(norm >= threshold)
    return float(t[indices[0]]) if len(indices) else None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--real-h5", type=Path, default=Path("real_data/traj_0/data.h5"))
    parser.add_argument("--sim-h5", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    calibration = json.loads(args.calibration.expanduser().read_text(encoding="utf-8"))
    semantics = calibration.get("dataset_semantics", {})
    if semantics.get("all_40_real_trajectories_are_successful_insertions") is not True:
        raise RuntimeError("calibration does not declare all 40 real trajectories as successful")
    if 0 in calibration.get("fitted_trajectories", []):
        raise RuntimeError("traj_0 is listed in fitted_trajectories and cannot be a holdout")
    baseline = np.asarray(calibration["baseline_model_weights"], dtype=np.float64)
    if baseline.shape != (15, 6):
        raise ValueError("baseline_model_weights must have shape (15,6)")
    with h5py.File(args.real_h5, "r") as data:
        real_t = _time(data["timestamps"][:])
        real_q = np.asarray(data["obs/state/joint_pos"][:], dtype=np.float64)
        real = np.asarray(data["obs/state/ee_wrench_base"][:], dtype=np.float64)
    with h5py.File(args.sim_h5, "r") as data:
        sim_t = _time(data["timestamps"][:])
        if "sim/force/wrench_calibrated" not in data:
            raise KeyError("sim H5 is missing sim/force/wrench_calibrated")
        sim = np.asarray(data["sim/force/wrench_calibrated"][:], dtype=np.float64)
        contact_count = np.asarray(data.get("sim/force/contact_count", np.zeros(len(sim))), dtype=np.float64)
    if real.shape != (len(real_t), 6) or real_q.shape != (len(real_t), 7) or sim.shape != (len(sim_t), 6):
        raise ValueError("invalid real/sim wrench shapes")
    metadata_path = args.sim_h5.parent / "replay_metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("geometric_success") is not True:
        raise RuntimeError("traj0 simulation is not marked as a successful insertion")
    success_frame = int(metadata["first_success_frame"])
    if not 0 <= success_frame < len(sim_t):
        raise ValueError(f"invalid first_success_frame={success_frame}")
    success_time = float(sim_t[success_frame])
    target_t = real_t[
        (real_t >= sim_t[0])
        & (real_t <= sim_t[-1])
        & (real_t <= success_time + 1.0e-9)
    ]
    if len(target_t) < 4:
        raise ValueError("real and sim timestamps do not overlap")
    real_contact = real - _features(real_q, real_t) @ baseline
    real_contact = _resample(real_t, real_contact, target_t)
    sim = _resample(sim_t, sim, target_t)
    contact_count = np.interp(target_t, sim_t, contact_count)

    fz_corr = _corr(real_contact[:, 2], sim[:, 2])
    norm_corr = _corr(np.linalg.norm(real_contact[:, :3], axis=1), np.linalg.norm(sim[:, :3], axis=1))
    real_contact_t = _first_contact(target_t, real_contact)
    sim_indices = np.flatnonzero(contact_count > 0.0)
    sim_contact_t = float(target_t[sim_indices[0]]) if len(sim_indices) else _first_contact(target_t, sim)
    timing_error = (
        abs(real_contact_t - sim_contact_t)
        if real_contact_t is not None and sim_contact_t is not None
        else float("inf")
    )
    real_p95 = float(np.quantile(np.linalg.norm(real_contact[:, :3], axis=1), 0.95))
    sim_p95 = float(np.quantile(np.linalg.norm(sim[:, :3], axis=1), 0.95))
    p95_ratio = sim_p95 / max(real_p95, 1.0e-12)
    acceptance = {
        "fz_pearson_gte_0_6": bool(np.isfinite(fz_corr) and fz_corr >= 0.6),
        "force_norm_pearson_gte_0_6": bool(np.isfinite(norm_corr) and norm_corr >= 0.6),
        "contact_timing_error_lte_0_3_s": bool(timing_error <= 0.3),
        "p95_ratio_in_0_5_to_2": bool(0.5 <= p95_ratio <= 2.0),
    }
    metrics = {
        "trajectory": 0,
        "trajectory_semantics": "successful insertion holdout",
        "success_frame": success_frame,
        "success_time_s": success_time,
        "post_success_samples_included": False,
        "coordinate_contract": calibration.get("coordinate_contract"),
        "real_is_baseline_compensated": True,
        "fz_pearson": fz_corr,
        "force_norm_pearson": norm_corr,
        "real_contact_time_s": real_contact_t,
        "sim_contact_time_s": sim_contact_t,
        "contact_timing_error_s": timing_error,
        "real_force_norm_p95_n": real_p95,
        "sim_force_norm_p95_n": sim_p95,
        "p95_ratio": p95_ratio,
        "acceptance": acceptance,
        "passed": bool(all(acceptance.values())),
    }
    output = args.output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    (output / "traj0_wrench_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    figure, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
    for index, axis in enumerate(axes):
        axis.plot(target_t, real_contact[:, index], label=f"real {NAMES[index]}", color="#ff7f0e")
        axis.plot(target_t, sim[:, index], label=f"sim {NAMES[index]}", color="#1f77b4")
        axis.set_ylabel(f"{NAMES[index]} [N]")
        axis.grid(alpha=0.3)
        axis.legend(loc="upper right")
    axes[-1].set_xlabel("time [s]")
    figure.suptitle("traj_0 holdout: baseline-compensated physical wrench")
    figure.tight_layout()
    figure.savefig(output / "traj0_force_axes.png", dpi=160)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(12, 4))
    axis.plot(target_t, np.linalg.norm(real_contact[:, :3], axis=1), label="real |F|", color="#ff7f0e")
    axis.plot(target_t, np.linalg.norm(sim[:, :3], axis=1), label="sim |F|", color="#1f77b4")
    axis.set(xlabel="time [s]", ylabel="force norm [N]", title=f"|F| corr={norm_corr:.3f}, p95 ratio={p95_ratio:.3f}")
    axis.grid(alpha=0.3)
    axis.legend(loc="upper right")
    figure.tight_layout()
    figure.savefig(output / "traj0_force_norm.png", dpi=160)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(12, 3))
    axis.step(target_t, contact_count > 0.0, where="post", label="sim contact_count > 0", color="#1f77b4")
    if real_contact_t is not None:
        axis.axvline(real_contact_t, color="#ff7f0e", linestyle="--", label="real first contact")
    if sim_contact_t is not None:
        axis.axvline(sim_contact_t, color="#1f77b4", linestyle=":", label="sim first contact")
    axis.set(xlabel="time [s]", ylabel="contact", ylim=(-0.1, 1.1), title=f"contact timing error={timing_error:.3f}s")
    axis.grid(alpha=0.3)
    axis.legend(loc="upper right")
    figure.tight_layout()
    figure.savefig(output / "traj0_contact_timeline.png", dpi=160)
    plt.close(figure)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
