#!/usr/bin/env python3
"""Compare paired sim-data-aligned and real-data-aligned force trajectories."""

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


FORCE_KEY = "obs/state/ee_wrench_base"
AXES = ("Fx", "Fy", "Fz")


def trajectory_id(path: Path) -> int:
    return int(path.parent.name.removeprefix("traj_"))


def load_force(path: Path) -> tuple[np.ndarray, np.ndarray, dict[str, str]]:
    with h5py.File(path, "r") as data:
        if FORCE_KEY not in data:
            raise KeyError(f"{path}: missing {FORCE_KEY}")
        wrench = np.asarray(data[FORCE_KEY], dtype=np.float64)
        timestamps = np.asarray(data["timestamps"], dtype=np.float64).reshape(-1)
        attrs = {str(key): str(value) for key, value in data.attrs.items()}
    if wrench.ndim != 2 or wrench.shape[1] != 6:
        raise ValueError(f"{path}: expected Nx6 wrench, got {wrench.shape}")
    if len(timestamps) != len(wrench) or not np.isfinite(wrench).all():
        raise ValueError(f"{path}: invalid timestamps/wrench")
    return timestamps - timestamps[0], wrench[:, :3], attrs


def remove_initial_bias(force: np.ndarray, frames: int) -> tuple[np.ndarray, np.ndarray]:
    count = min(max(1, frames), len(force))
    bias = np.median(force[:count], axis=0)
    return force - bias, bias


def correlation(left: np.ndarray, right: np.ndarray) -> float:
    left = np.asarray(left, dtype=np.float64).reshape(-1)
    right = np.asarray(right, dtype=np.float64).reshape(-1)
    if len(left) < 2 or np.std(left) < 1.0e-12 or np.std(right) < 1.0e-12:
        return float("nan")
    return float(np.corrcoef(left, right)[0, 1])


def positive_ls_gain(sim: np.ndarray, real: np.ndarray) -> float:
    denominator = float(np.sum(sim * sim))
    if denominator < 1.0e-12:
        return 1.0
    return max(0.0, float(np.sum(sim * real)) / denominator)


def rmse(left: np.ndarray, right: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(left - right))))


def percentile_norm(force: np.ndarray, percentile: float = 95.0) -> float:
    return float(np.percentile(np.linalg.norm(force, axis=1), percentile))


def save_selected_plot(
    records: list[dict], selected_ids: list[int], global_gain: float, output: Path
) -> None:
    selected = [record for record in records if record["trajectory"] in selected_ids]
    if not selected:
        return
    figure, axes = plt.subplots(
        len(selected), 4, figsize=(17, 3.4 * len(selected)), squeeze=False
    )
    for row, record in enumerate(selected):
        time = record["time"]
        real = record["real"]
        sim = record["sim"]
        aligned = sim * global_gain
        series = [real[:, 0], real[:, 1], real[:, 2], np.linalg.norm(real, axis=1)]
        sim_series = [sim[:, 0], sim[:, 1], sim[:, 2], np.linalg.norm(sim, axis=1)]
        aligned_series = [
            aligned[:, 0],
            aligned[:, 1],
            aligned[:, 2],
            np.linalg.norm(aligned, axis=1),
        ]
        for column, (real_y, sim_y, aligned_y) in enumerate(
            zip(series, sim_series, aligned_series)
        ):
            axis = axes[row, column]
            axis.plot(time, real_y, color="black", linewidth=1.5, label="real")
            axis.plot(time, sim_y, color="#f59e0b", linewidth=1.0, alpha=0.65, label="sim raw")
            axis.plot(time, aligned_y, color="#2563eb", linewidth=1.15, label="sim amplitude-aligned")
            axis.grid(alpha=0.25)
            axis.set_xlabel("time [s]")
            axis.set_ylabel("force [N]")
            axis.set_title(
                f"traj_{record['trajectory']} "
                + (AXES[column] if column < 3 else "|F|")
            )
            if row == 0 and column == 0:
                axis.legend(fontsize=8)
    figure.suptitle(
        "Paired force trends after initial-bias removal; "
        f"robust force-norm amplitude gain = {global_gain:.4f}"
    )
    figure.tight_layout()
    figure.savefig(output, dpi=180)
    plt.close(figure)


def save_all_norms_plot(records: list[dict], global_gain: float, output: Path) -> None:
    figure, axes = plt.subplots(8, 5, figsize=(18, 21), squeeze=False)
    for axis, record in zip(axes.flat, records):
        time = record["time"]
        real_norm = np.linalg.norm(record["real"], axis=1)
        sim_norm = np.linalg.norm(record["sim"] * global_gain, axis=1)
        corr = correlation(real_norm, sim_norm)
        axis.plot(time, real_norm, color="black", linewidth=1.0, label="real")
        axis.plot(time, sim_norm, color="#2563eb", linewidth=0.9, label="sim aligned")
        axis.set_title(f"traj_{record['trajectory']}  r={corr:.2f}", fontsize=9)
        axis.grid(alpha=0.2)
        axis.tick_params(labelsize=7)
    axes.flat[0].legend(fontsize=7)
    figure.supxlabel("time [s]")
    figure.supylabel("bias-removed force norm [N]")
    figure.suptitle(
        f"All 40 paired trajectories; robust force-norm amplitude gain = {global_gain:.4f}"
    )
    figure.tight_layout()
    figure.savefig(output, dpi=170)
    plt.close(figure)


def save_overview_plot(rows: list[dict], output: Path) -> None:
    ids = np.asarray([row["trajectory"] for row in rows])
    correlations = np.asarray([row["force_norm_correlation"] for row in rows])
    gains = np.asarray([row["trajectory_positive_ls_gain"] for row in rows])
    real_p95 = np.asarray([row["real_force_norm_p95_n"] for row in rows])
    raw_p95 = np.asarray([row["sim_raw_force_norm_p95_n"] for row in rows])
    aligned_p95 = np.asarray([row["sim_aligned_force_norm_p95_n"] for row in rows])

    figure, axes = plt.subplots(1, 3, figsize=(17, 4.8))
    axes[0].bar(ids, correlations, color=np.where(correlations >= 0.0, "#2563eb", "#dc2626"))
    axes[0].axhline(0.0, color="black", linewidth=0.8)
    axes[0].set_ylim(-1.0, 1.0)
    axes[0].set_title("Per-trajectory |F| trend correlation")
    axes[0].set_xlabel("trajectory")
    axes[0].set_ylabel("Pearson r")
    axes[0].grid(axis="y", alpha=0.25)

    limit = max(float(np.max(real_p95)), float(np.max(raw_p95)), float(np.max(aligned_p95)), 1.0)
    axes[1].scatter(real_p95, raw_p95, s=22, alpha=0.7, label="sim raw")
    axes[1].scatter(real_p95, aligned_p95, s=22, alpha=0.7, label="sim aligned")
    axes[1].plot([0.0, limit], [0.0, limit], "k--", linewidth=0.8, label="y=x")
    axes[1].set_xlim(0.0, limit)
    axes[1].set_ylim(0.0, limit)
    axes[1].set_title("Force-norm P95 amplitude")
    axes[1].set_xlabel("real P95 [N]")
    axes[1].set_ylabel("sim P95 [N]")
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.25)

    axes[2].bar(ids, gains, color="#0f766e")
    axes[2].set_title("Independent positive gain per trajectory")
    axes[2].set_xlabel("trajectory")
    axes[2].set_ylabel("least-squares gain")
    axes[2].grid(axis="y", alpha=0.25)

    figure.tight_layout()
    figure.savefig(output, dpi=180)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sim-dir", type=Path, default=Path("sim-data-aligned"))
    parser.add_argument("--real-dir", type=Path, default=Path("real-data-aligned"))
    parser.add_argument(
        "--output-dir", type=Path, default=Path("outputs/sim_data_aligned_force_comparison")
    )
    parser.add_argument("--baseline-frames", type=int, default=30)
    parser.add_argument("--selected", type=int, nargs="*", default=[0, 9, 39])
    args = parser.parse_args()
    if args.baseline_frames <= 0:
        raise ValueError("--baseline-frames must be positive")

    sim_paths = sorted(args.sim_dir.glob("traj_*/data.h5"), key=trajectory_id)
    real_paths = sorted(args.real_dir.glob("traj_*/data.h5"), key=trajectory_id)
    if len(sim_paths) != 40 or len(real_paths) != 40:
        raise ValueError(
            f"Expected 40 paired trajectories, got sim={len(sim_paths)}, real={len(real_paths)}"
        )

    records: list[dict] = []
    sim_all: list[np.ndarray] = []
    real_all: list[np.ndarray] = []
    provenance: dict[str, dict[str, str]] = {}
    for sim_path, real_path in zip(sim_paths, real_paths):
        sim_id = trajectory_id(sim_path)
        real_id = trajectory_id(real_path)
        if sim_id != real_id:
            raise ValueError(f"Trajectory mismatch: {sim_path} vs {real_path}")
        sim_time, sim_force_raw, sim_attrs = load_force(sim_path)
        real_time, real_force_raw, _ = load_force(real_path)
        if sim_force_raw.shape != real_force_raw.shape:
            raise ValueError(
                f"traj_{sim_id}: shape mismatch sim={sim_force_raw.shape}, real={real_force_raw.shape}"
            )
        if not np.allclose(sim_time, real_time, atol=1.0e-5, rtol=0.0):
            raise ValueError(f"traj_{sim_id}: timestamps are not frame aligned")
        sim_force, sim_bias = remove_initial_bias(sim_force_raw, args.baseline_frames)
        real_force, real_bias = remove_initial_bias(real_force_raw, args.baseline_frames)
        records.append(
            {
                "trajectory": sim_id,
                "time": real_time,
                "sim": sim_force,
                "real": real_force,
                "sim_bias": sim_bias,
                "real_bias": real_bias,
            }
        )
        sim_all.append(sim_force)
        real_all.append(real_force)
        provenance[str(sim_id)] = sim_attrs

    sim_concat = np.concatenate(sim_all, axis=0)
    real_concat = np.concatenate(real_all, axis=0)
    signed_component_ls_gain = positive_ls_gain(sim_concat, real_concat)
    amplitude_ratios = np.asarray(
        [
            percentile_norm(record["real"]) / max(percentile_norm(record["sim"]), 1.0e-12)
            for record in records
        ],
        dtype=np.float64,
    )
    # A single signed-component fit is unstable when force directions differ.
    # The median trajectory-level P95 ratio aligns amplitude while preserving
    # every sign, relative component and timestamp in the simulation stream.
    global_gain = float(np.median(amplitude_ratios))
    aligned_concat = sim_concat * global_gain

    rows: list[dict] = []
    for record in records:
        sim = record["sim"]
        real = record["real"]
        aligned = sim * global_gain
        real_norm = np.linalg.norm(real, axis=1)
        sim_norm = np.linalg.norm(sim, axis=1)
        aligned_norm = np.linalg.norm(aligned, axis=1)
        rows.append(
            {
                "trajectory": record["trajectory"],
                "frames": len(real),
                "duration_s": float(record["time"][-1]),
                "force_norm_correlation": correlation(real_norm, sim_norm),
                "fx_correlation": correlation(real[:, 0], sim[:, 0]),
                "fy_correlation": correlation(real[:, 1], sim[:, 1]),
                "fz_correlation": correlation(real[:, 2], sim[:, 2]),
                "trajectory_positive_ls_gain": positive_ls_gain(sim, real),
                "real_force_norm_p95_n": percentile_norm(real),
                "sim_raw_force_norm_p95_n": percentile_norm(sim),
                "sim_aligned_force_norm_p95_n": percentile_norm(aligned),
                "component_rmse_raw_n": rmse(sim, real),
                "component_rmse_aligned_n": rmse(aligned, real),
                "sim_initial_bias_n": record["sim_bias"].tolist(),
                "real_initial_bias_n": record["real_bias"].tolist(),
            }
        )

    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    save_selected_plot(records, args.selected, global_gain, output / "selected_force_trends.png")
    save_all_norms_plot(records, global_gain, output / "all_40_force_norm_trends.png")
    save_overview_plot(rows, output / "force_alignment_overview.png")

    with (output / "per_trajectory_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        fieldnames = [key for key in rows[0] if not key.endswith("_bias_n")]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows({key: row[key] for key in fieldnames} for row in rows)

    summary = {
        "sim_dir": str(args.sim_dir.resolve()),
        "real_dir": str(args.real_dir.resolve()),
        "force_key": FORCE_KEY,
        "trajectories": len(records),
        "frames": int(len(sim_concat)),
        "pairing": "same trajectory id, frame index, and timestamp",
        "preprocessing": {
            "initial_bias": f"per-axis median of first {args.baseline_frames} frames, removed separately",
            "amplitude_alignment": (
                "median across trajectories of real/sim force-norm P95; "
                "one positive scalar applied to all XYZ channels"
            ),
            "sign_flip": False,
            "time_warp": False,
        },
        "global_positive_gain": global_gain,
        "signed_component_positive_ls_gain_diagnostic": signed_component_ls_gain,
        "global_component_rmse_raw_n": rmse(sim_concat, real_concat),
        "global_component_rmse_aligned_n": rmse(aligned_concat, real_concat),
        "global_correlations": {
            **{
                AXES[index]: correlation(real_concat[:, index], sim_concat[:, index])
                for index in range(3)
            },
            "force_norm": correlation(
                np.linalg.norm(real_concat, axis=1), np.linalg.norm(sim_concat, axis=1)
            ),
        },
        "trajectory_force_norm_correlation": {
            "median": float(np.nanmedian([row["force_norm_correlation"] for row in rows])),
            "mean": float(np.nanmean([row["force_norm_correlation"] for row in rows])),
            "positive_count": int(
                sum(row["force_norm_correlation"] > 0.0 for row in rows)
            ),
        },
        "torque_note": "sim-data-aligned torque channels are zero and are intentionally excluded",
        "sim_root_attributes": provenance,
        "outputs": {
            "selected": "selected_force_trends.png",
            "all_40": "all_40_force_norm_trends.png",
            "overview": "force_alignment_overview.png",
            "per_trajectory": "per_trajectory_metrics.csv",
        },
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps({key: value for key, value in summary.items() if key != "sim_root_attributes"}, indent=2))
    print(f"[DONE] {output}")


if __name__ == "__main__":
    main()
