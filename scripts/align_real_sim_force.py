#!/usr/bin/env python3
"""Repair real Franka wrench bias and compare traj0 with a paired Isaac Sim rollout."""

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
from scipy.ndimage import median_filter
from scipy.signal import butter, sosfiltfilt
from scipy.stats import pearsonr, spearmanr

NAMES = ("Fx", "Fy", "Fz", "Tx", "Ty", "Tz")


def rotation_xyzw(quaternion: np.ndarray) -> np.ndarray:
    x, y, z, w = np.moveaxis(np.asarray(quaternion, dtype=np.float64), -1, 0)
    return np.stack(
        [
            1 - 2 * (y * y + z * z),
            2 * (x * y - z * w),
            2 * (x * z + y * w),
            2 * (x * y + z * w),
            1 - 2 * (x * x + z * z),
            2 * (y * z - x * w),
            2 * (x * z - y * w),
            2 * (y * z + x * w),
            1 - 2 * (x * x + y * y),
        ],
        axis=-1,
    ).reshape(-1, 3, 3)


def lowpass(values: np.ndarray, sample_hz: float, cutoff_hz: float) -> np.ndarray:
    sos = butter(2, cutoff_hz, btype="low", fs=sample_hz, output="sos")
    return sosfiltfilt(sos, np.asarray(values, dtype=np.float64), axis=0)


def load_real(path: Path) -> dict[str, np.ndarray]:
    with h5py.File(path, "r") as data:
        result = {
            "timestamps": np.asarray(data["timestamps"], dtype=np.float64),
            "base": np.asarray(data["obs/state/ee_wrench_base"], dtype=np.float64),
            "stiffness": np.asarray(data["obs/state/ee_wrench_stiffness"], dtype=np.float64),
            "pose": np.asarray(data["obs/state/ee_pose"], dtype=np.float64),
        }
    result["timestamps"] -= result["timestamps"][0]
    return result


def stiffness_to_base(wrench: np.ndarray, pose: np.ndarray) -> np.ndarray:
    rotation = rotation_xyzw(pose[:, 3:7])
    force = np.einsum("nij,nj->ni", rotation, wrench[:, :3])
    torque = np.einsum("nij,nj->ni", rotation, wrench[:, 3:]) + np.cross(pose[:, :3], force)
    return np.concatenate((force, torque), axis=1)


def repair_real(data: dict[str, np.ndarray], window: int, cutoff_hz: float) -> np.ndarray:
    sample_hz = 1.0 / float(np.median(np.diff(data["timestamps"])))
    baseline_k = median_filter(data["stiffness"], size=(window, 1), mode="nearest")
    contact_k = lowpass(data["stiffness"] - baseline_k, sample_hz, cutoff_hz)
    return stiffness_to_base(contact_k, data["pose"])


def read_csv(path: Path) -> np.ndarray:
    values = np.loadtxt(path, delimiter=",", skiprows=1, ndmin=2).astype(np.float64)
    if values.ndim != 2 or values.shape[1] != 6 or not np.isfinite(values).all():
        raise ValueError(f"Expected finite Nx6 CSV: {path}")
    return values


def read_time(path: Path) -> np.ndarray:
    return np.loadtxt(path, delimiter=",", skiprows=1, ndmin=1).astype(np.float64).reshape(-1)


def sim_to_real_base(wrench: np.ndarray, rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    force_sim = wrench[:, :3]
    force_real = (rotation.T @ force_sim.T).T
    torque_real = (rotation.T @ (wrench[:, 3:] - np.cross(translation, force_sim)).T).T
    return np.concatenate((force_real, torque_real), axis=1)


def paired(values_a: np.ndarray, values_b: np.ndarray, shift: int) -> tuple[np.ndarray, np.ndarray]:
    """Positive shift means simulation leads real data by ``shift`` samples."""
    if shift > 0:
        return values_a[shift:], values_b[:-shift]
    if shift < 0:
        return values_a[:shift], values_b[-shift:]
    return values_a, values_b


def safe_corr(function, a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 3 or np.std(a) < 1.0e-12 or np.std(b) < 1.0e-12:
        return float("nan")
    return float(function(a, b).statistic)


def write_comparison_csv(path: Path, timestamps: np.ndarray, real: np.ndarray, sim: np.ndarray) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["timestamp", *[f"real_{x}" for x in NAMES], *[f"sim_{x}" for x in NAMES]])
        for index in range(len(timestamps)):
            writer.writerow([float(timestamps[index]), *real[index].tolist(), *sim[index].tolist()])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("real_data"))
    parser.add_argument("--trajectory", type=int, default=0)
    parser.add_argument("--sim-contact", type=Path, default=Path("outputs/traj0"))
    parser.add_argument("--sim-no-contact", type=Path, default=Path("outputs/wrench_alignment/traj0_no_hole"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/wrench_alignment/traj0_final"))
    parser.add_argument("--baseline-window", type=int, default=15)
    parser.add_argument("--cutoff-hz", type=float, default=1.0)
    parser.add_argument("--no-contact-seconds", type=float, default=3.0)
    parser.add_argument("--max-lag-seconds", type=float, default=1.0)
    args = parser.parse_args()

    if args.baseline_window < 3 or args.baseline_window % 2 == 0:
        raise ValueError("baseline-window must be an odd integer >= 3")
    paths = sorted(
        args.data_dir.glob("traj_*/data.h5"),
        key=lambda path: int(path.parent.name.removeprefix("traj_")),
    )
    if len(paths) != 40:
        raise ValueError(f"Expected 40 real trajectories, found {len(paths)}")

    target_path = args.data_dir / f"traj_{args.trajectory}" / "data.h5"
    target = load_real(target_path)
    real_corrected = repair_real(target, args.baseline_window, args.cutoff_hz)

    raw_no_contact = []
    corrected_no_contact = []
    reconstruction_errors = []
    for path in paths:
        data = load_real(path)
        corrected = repair_real(data, args.baseline_window, args.cutoff_hz)
        reconstructed = stiffness_to_base(data["stiffness"], data["pose"])
        reconstruction_errors.append(float(np.max(np.abs(reconstructed - data["base"]))))
        mask = data["timestamps"] < args.no_contact_seconds
        raw_no_contact.append(data["base"][mask])
        corrected_no_contact.append(corrected[mask])
    raw_no_contact = np.concatenate(raw_no_contact)
    corrected_no_contact = np.concatenate(corrected_no_contact)

    contact = read_csv(args.sim_contact / "wrench_base.csv")
    no_contact = read_csv(args.sim_no_contact / "wrench_base.csv")
    sim_t = read_time(args.sim_contact / "timestamps.csv")
    no_contact_t = read_time(args.sim_no_contact / "timestamps.csv")
    if contact.shape != no_contact.shape or len(sim_t) != len(contact) or not np.allclose(sim_t, no_contact_t):
        raise ValueError("Paired simulation rollouts are not frame aligned")
    contact_q = np.loadtxt(args.sim_contact / "joint_pos_sim.csv", delimiter=",", skiprows=1)
    no_contact_q = np.loadtxt(args.sim_no_contact / "joint_pos_sim.csv", delimiter=",", skiprows=1)
    max_joint_difference = float(np.max(np.abs(contact_q - no_contact_q)))

    metadata = json.loads((args.sim_contact / "replay_metadata.json").read_text(encoding="utf-8"))
    calibration = metadata["real_to_sim_xyz"]
    real_to_sim_rotation = np.asarray(calibration["real_to_sim_rotation"], dtype=np.float64)
    real_to_sim_translation = np.asarray(calibration["real_to_sim_translation_m"], dtype=np.float64)
    sim_contact_real_base = sim_to_real_base(
        contact - no_contact,
        real_to_sim_rotation,
        real_to_sim_translation,
    )
    sim_sample_hz = 1.0 / float(np.median(np.diff(sim_t)))
    sim_contact_real_base = lowpass(sim_contact_real_base, sim_sample_hz, args.cutoff_hz)
    sim_on_real_grid = np.column_stack(
        [np.interp(target["timestamps"], sim_t, sim_contact_real_base[:, axis]) for axis in range(6)]
    )

    max_shift = int(round(args.max_lag_seconds * sim_sample_hz))
    best = None
    for shift in range(-max_shift, max_shift + 1):
        real_pair, sim_pair = paired(real_corrected, sim_on_real_grid, shift)
        fz_corr = safe_corr(pearsonr, real_pair[:, 2], sim_pair[:, 2])
        force_norm_corr = safe_corr(
            pearsonr,
            np.linalg.norm(real_pair[:, :3], axis=1),
            np.linalg.norm(sim_pair[:, :3], axis=1),
        )
        score = float(np.nanmean((fz_corr, force_norm_corr)))
        if best is None or score > best[0]:
            best = (score, shift)
    _, best_shift = best
    real_pair, sim_pair = paired(real_corrected, sim_on_real_grid, best_shift)

    pearson = [safe_corr(pearsonr, real_pair[:, axis], sim_pair[:, axis]) for axis in range(6)]
    spearman = [safe_corr(spearmanr, real_pair[:, axis], sim_pair[:, axis]) for axis in range(6)]
    real_force_norm = np.linalg.norm(real_pair[:, :3], axis=1)
    sim_force_norm = np.linalg.norm(sim_pair[:, :3], axis=1)
    real_torque_norm = np.linalg.norm(real_pair[:, 3:], axis=1)
    sim_torque_norm = np.linalg.norm(sim_pair[:, 3:], axis=1)
    force_norm_pearson = safe_corr(pearsonr, real_force_norm, sim_force_norm)
    force_norm_spearman = safe_corr(spearmanr, real_force_norm, sim_force_norm)
    torque_norm_pearson = safe_corr(pearsonr, real_torque_norm, sim_torque_norm)
    torque_norm_spearman = safe_corr(spearmanr, real_torque_norm, sim_torque_norm)
    axial_scale = float(np.dot(real_pair[:, 2], sim_pair[:, 2]) / np.dot(sim_pair[:, 2], sim_pair[:, 2]))
    norm_scale = float(np.dot(real_force_norm, sim_force_norm) / np.dot(sim_force_norm, sim_force_norm))

    aligned_sim = np.full_like(real_corrected, np.nan)
    if best_shift > 0:
        aligned_sim[best_shift:] = sim_on_real_grid[:-best_shift]
    elif best_shift < 0:
        aligned_sim[:best_shift] = sim_on_real_grid[-best_shift:]
    else:
        aligned_sim[:] = sim_on_real_grid

    def distribution(values: np.ndarray) -> dict[str, float]:
        force_norm = np.linalg.norm(values[:, :3], axis=1)
        torque_norm = np.linalg.norm(values[:, 3:], axis=1)
        return {
            "force_norm_median": float(np.median(force_norm)),
            "force_norm_p95": float(np.percentile(force_norm, 95)),
            "force_norm_p99": float(np.percentile(force_norm, 99)),
            "torque_norm_median": float(np.median(torque_norm)),
            "torque_norm_p95": float(np.percentile(torque_norm, 95)),
            "torque_norm_p99": float(np.percentile(torque_norm, 99)),
        }

    raw_stats = distribution(raw_no_contact)
    corrected_stats = distribution(corrected_no_contact)
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    write_comparison_csv(output / "traj0_force_comparison.csv", target["timestamps"], real_corrected, aligned_sim)

    metrics = {
        "trajectory": args.trajectory,
        "real_repair": {
            "frame": "stiffness/K -> full spatial wrench transform -> real base/O",
            "baseline": f"rolling median, {args.baseline_window} frames",
            "filter": f"second-order zero-phase Butterworth, {args.cutoff_hz} Hz",
            "validated_trajectories": len(paths),
            "assumed_no_contact_seconds": args.no_contact_seconds,
            "raw_no_contact": raw_stats,
            "corrected_no_contact": corrected_stats,
            "max_h5_frame_reconstruction_error": max(reconstruction_errors),
        },
        "simulation": {
            "definition": "wrench_base(hole collision enabled) - wrench_base(hole collision disabled)",
            "coordinate_transform": "sim base/origin -> real base/origin using full wrench adjoint",
            "max_paired_joint_difference_rad": max_joint_difference,
        },
        "alignment": {
            "sim_leads_real_seconds": float(best_shift / sim_sample_hz),
            "pearson": dict(zip(NAMES, pearson)),
            "spearman": dict(zip(NAMES, spearman)),
            "force_norm_pearson": force_norm_pearson,
            "force_norm_spearman": force_norm_spearman,
            "torque_norm_pearson": torque_norm_pearson,
            "torque_norm_spearman": torque_norm_spearman,
            "axial_display_scale": axial_scale,
            "force_norm_display_scale": norm_scale,
            "note": "display scaling changes amplitude only and does not change correlation",
        },
    }
    (output / "force_alignment_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    raw_force_norm = np.linalg.norm(raw_no_contact[:, :3], axis=1)
    corrected_force_norm = np.linalg.norm(corrected_no_contact[:, :3], axis=1)
    figure = plt.figure(figsize=(17, 13), constrained_layout=True)
    grid = figure.add_gridspec(3, 3, height_ratios=(0.9, 1.15, 1.15))

    axis = figure.add_subplot(grid[0, 0])
    for values, label, color in (
        (raw_force_norm, "raw real", "#777777"),
        (corrected_force_norm, "corrected real", "#d62728"),
    ):
        sorted_values = np.sort(values)
        axis.plot(sorted_values, np.linspace(0, 1, len(sorted_values)), label=label, color=color, linewidth=2)
    axis.set_xlim(0, min(8.0, float(np.percentile(raw_force_norm, 99.9))))
    axis.set_xlabel("Force norm during first 3 s [N]")
    axis.set_ylabel("Empirical CDF")
    axis.set_title("40-trajectory no-contact validation")
    axis.grid(alpha=0.25)
    axis.legend()

    axis = figure.add_subplot(grid[0, 1])
    labels = ["median", "p95", "p99"]
    raw_values = [raw_stats[f"force_norm_{label}"] for label in labels]
    corrected_values = [corrected_stats[f"force_norm_{label}"] for label in labels]
    positions = np.arange(len(labels))
    axis.bar(positions - 0.18, raw_values, width=0.36, label="raw", color="#888888")
    axis.bar(positions + 0.18, corrected_values, width=0.36, label="corrected", color="#d62728")
    axis.set_xticks(positions, labels)
    axis.set_ylabel("Force norm [N]")
    axis.set_title("No-contact residual reduction")
    axis.grid(axis="y", alpha=0.25)
    axis.legend()

    axis = figure.add_subplot(grid[0, 2])
    axis.axis("off")
    summary = (
        f"traj{args.trajectory} real/sim force alignment\n\n"
        f"Sim leads real: {best_shift / sim_sample_hz:.2f} s\n"
        f"Pearson Fz: {pearson[2]:.3f}\n"
        f"Pearson |F|: {force_norm_pearson:.3f}\n"
        f"Spearman |F|: {force_norm_spearman:.3f}\n"
        f"Pearson |T|: {torque_norm_pearson:.3f}\n\n"
        f"Fz display scale: {axial_scale:.3f}x\n"
        f"Paired max joint delta: {max_joint_difference:.2e} rad"
    )
    axis.text(0.03, 0.97, summary, va="top", ha="left", fontsize=13, family="monospace")

    plot_t = target["timestamps"]
    for component, cell in zip(range(3), ((1, 0), (1, 1), (1, 2))):
        axis = figure.add_subplot(grid[cell])
        axis.plot(plot_t, target["base"][:, component], color="#aaaaaa", alpha=0.55, linewidth=0.9, label="real raw")
        axis.plot(plot_t, real_corrected[:, component], color="#d62728", linewidth=1.7, label="real corrected")
        axis.plot(
            plot_t,
            aligned_sim[:, component] * axial_scale,
            color="#1f77b4",
            linewidth=1.4,
            linestyle="--",
            label=f"sim contact x{axial_scale:.2f}",
        )
        axis.set_title(f"{NAMES[component]}  Pearson r={pearson[component]:.3f}")
        axis.set_xlabel("Time [s]")
        axis.set_ylabel("Force [N]")
        axis.grid(alpha=0.25)
        if component == 0:
            axis.legend(fontsize=9)

    axis = figure.add_subplot(grid[2, 0])
    full_real_norm = np.linalg.norm(real_corrected[:, :3], axis=1)
    full_sim_norm = np.linalg.norm(aligned_sim[:, :3], axis=1)
    axis.plot(plot_t, full_real_norm, color="#d62728", linewidth=1.7, label="real corrected |F|")
    axis.plot(plot_t, full_sim_norm * norm_scale, color="#1f77b4", linestyle="--", linewidth=1.4, label=f"sim |F| x{norm_scale:.2f}")
    axis.set_title(f"Force norm  Pearson r={force_norm_pearson:.3f}")
    axis.set_xlabel("Time [s]")
    axis.set_ylabel("|F| [N]")
    axis.grid(alpha=0.25)
    axis.legend(fontsize=9)

    axis = figure.add_subplot(grid[2, 1])
    axis.scatter(sim_pair[:, 2] * axial_scale, real_pair[:, 2], s=18, alpha=0.55, color="#6a3d9a")
    limits = np.asarray(
        [min(np.min(sim_pair[:, 2] * axial_scale), np.min(real_pair[:, 2])), max(np.max(sim_pair[:, 2] * axial_scale), np.max(real_pair[:, 2]))]
    )
    axis.plot(limits, limits, color="black", linewidth=1, linestyle=":")
    axis.set_xlabel("Aligned simulation Fz [N]")
    axis.set_ylabel("Corrected real Fz [N]")
    axis.set_title(f"Axial-force agreement  r={pearson[2]:.3f}")
    axis.grid(alpha=0.25)

    axis = figure.add_subplot(grid[2, 2])
    correlation_labels = ["Fx", "Fy", "Fz", "|F|", "|T|"]
    correlation_values = [pearson[0], pearson[1], pearson[2], force_norm_pearson, torque_norm_pearson]
    colors = ["#2ca02c" if value >= 0 else "#ff7f0e" for value in correlation_values]
    bars = axis.bar(correlation_labels, correlation_values, color=colors)
    axis.axhline(0.0, color="black", linewidth=0.8)
    axis.set_ylim(-1.0, 1.0)
    axis.set_ylabel("Pearson correlation")
    axis.set_title("Aligned correlation summary")
    axis.grid(axis="y", alpha=0.25)
    for bar, value in zip(bars, correlation_values):
        axis.text(bar.get_x() + bar.get_width() / 2, value + (0.04 if value >= 0 else -0.08), f"{value:.2f}", ha="center")

    figure.suptitle("Franka real-wrench repair and paired Isaac Sim contact-force alignment", fontsize=17)
    figure.savefig(output / "force_alignment.png", dpi=180)
    plt.close(figure)
    print(f"[DONE] {output / 'force_alignment.png'}")
    print(json.dumps(metrics["alignment"], indent=2))


if __name__ == "__main__":
    main()
