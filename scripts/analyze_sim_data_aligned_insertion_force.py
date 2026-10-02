#!/usr/bin/env python3
"""Analyze whether derived simulation-force features track insertion geometry."""

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
from scipy.stats import rankdata, spearmanr


FORCE_KEY = "obs/state/ee_wrench_base"
FEATURES = ("lateral_force_n", "axial_force_n", "lateral_axial_ratio", "force_norm_n")


def read_validation(path: Path) -> dict[str, np.ndarray]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    return {
        "xy_error_m": np.asarray([float(row["xy_error_m"]) for row in rows]),
        "z_disp_m": np.asarray([float(row["z_disp_m"]) for row in rows]),
        "success": np.asarray(
            [str(row["success"]).strip().lower() in {"true", "1", "yes"} for row in rows],
            dtype=bool,
        ),
    }


def auc(labels: np.ndarray, scores: np.ndarray) -> float:
    labels = np.asarray(labels, dtype=bool)
    scores = np.asarray(scores, dtype=np.float64)
    positive = int(np.count_nonzero(labels))
    negative = int(len(labels) - positive)
    if positive == 0 or negative == 0:
        return float("nan")
    ranks = rankdata(scores, method="average")
    return float(
        (np.sum(ranks[labels]) - positive * (positive + 1) / 2.0)
        / (positive * negative)
    )


def corr(left: np.ndarray, right: np.ndarray) -> float:
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    if len(left) < 3 or np.std(left) < 1.0e-12 or np.std(right) < 1.0e-12:
        return float("nan")
    return float(spearmanr(left, right).statistic)


def finite_stats(values: list[float]) -> dict[str, float | int]:
    array = np.asarray(values, dtype=np.float64)
    array = array[np.isfinite(array)]
    if len(array) == 0:
        return {"count": 0, "median": float("nan"), "mean": float("nan")}
    return {
        "count": int(len(array)),
        "median": float(np.median(array)),
        "mean": float(np.mean(array)),
        "q25": float(np.percentile(array, 25)),
        "q75": float(np.percentile(array, 75)),
    }


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sim-aligned-dir", type=Path, default=Path("sim-data-aligned"))
    parser.add_argument("--rollout-dir", type=Path, default=Path("sim-data"))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/sim_data_aligned_insertion_force_analysis"),
    )
    parser.add_argument("--near-hole-z-m", type=float, default=0.010)
    parser.add_argument("--aligned-xy-m", type=float, default=0.003)
    parser.add_argument("--ratio-epsilon-n", type=float, default=0.05)
    args = parser.parse_args()

    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    records: list[dict] = []
    trajectory_rows: list[dict] = []
    frame_rows: list[dict] = []

    for trajectory in range(40):
        h5_path = args.sim_aligned_dir / f"traj_{trajectory}" / "data.h5"
        validation_path = args.rollout_dir / f"traj_{trajectory}" / "gt_replay_validation.csv"
        with h5py.File(h5_path, "r") as data:
            wrench = np.asarray(data[FORCE_KEY], dtype=np.float64)
            time = np.asarray(data["timestamps"], dtype=np.float64).reshape(-1)
        validation = read_validation(validation_path)
        count = len(wrench)
        if any(len(value) != count for value in validation.values()) or len(time) != count:
            raise ValueError(f"traj_{trajectory}: force/geometry length mismatch")
        time = time - time[0]
        force = wrench[:, :3]
        lateral = np.linalg.norm(force[:, :2], axis=1)
        axial = np.abs(force[:, 2])
        norm = np.linalg.norm(force, axis=1)
        ratio = lateral / (axial + float(args.ratio_epsilon_n))
        xy_error = validation["xy_error_m"]
        z_disp = validation["z_disp_m"]
        success = validation["success"]
        near = z_disp <= float(args.near_hole_z_m)
        aligned = xy_error <= float(args.aligned_xy_m)
        approach = ~near
        misaligned_near = near & ~aligned
        aligned_near = near & aligned
        depth_progress = z_disp[0] - z_disp
        downward_speed = np.gradient(depth_progress, time)
        feature_values = {
            "lateral_force_n": lateral,
            "axial_force_n": axial,
            "lateral_axial_ratio": ratio,
            "force_norm_n": norm,
        }

        row: dict[str, float | int | bool] = {
            "trajectory": trajectory,
            "frames": count,
            "approach_frames": int(np.count_nonzero(approach)),
            "near_hole_frames": int(np.count_nonzero(near)),
            "aligned_near_frames": int(np.count_nonzero(aligned_near)),
            "misaligned_near_frames": int(np.count_nonzero(misaligned_near)),
            "success_frames": int(np.count_nonzero(success)),
            "geometric_success": bool(np.any(success)),
            "lateral_vs_xy_spearman_near": corr(lateral[near], xy_error[near]),
            "axial_vs_depth_spearman": corr(axial, depth_progress),
            "norm_vs_depth_spearman": corr(norm, depth_progress),
            "axial_vs_down_speed_spearman": corr(axial, downward_speed),
        }
        for name, values in feature_values.items():
            approach_median = float(np.median(values[approach])) if np.any(approach) else float("nan")
            near_median = float(np.median(values[near])) if np.any(near) else float("nan")
            row[f"{name}_approach_median"] = approach_median
            row[f"{name}_near_median"] = near_median
            row[f"{name}_near_over_approach"] = near_median / max(approach_median, 1.0e-12)
            row[f"{name}_near_auc"] = auc(near, values)
            row[f"{name}_success_auc"] = auc(success, values)
        trajectory_rows.append(row)

        normalizers = {
            name: max(float(np.percentile(values, 95)), 1.0e-12)
            for name, values in feature_values.items()
        }
        for index in range(count):
            frame_row: dict[str, float | int | bool] = {
                "trajectory": trajectory,
                "frame": index,
                "time_s": float(time[index]),
                "xy_error_m": float(xy_error[index]),
                "z_disp_m": float(z_disp[index]),
                "depth_progress_m": float(depth_progress[index]),
                "downward_speed_m_s": float(downward_speed[index]),
                "near_hole": bool(near[index]),
                "aligned": bool(aligned[index]),
                "success": bool(success[index]),
            }
            for name, values in feature_values.items():
                frame_row[name] = float(values[index])
                frame_row[f"{name}_normalized"] = float(values[index] / normalizers[name])
            frame_rows.append(frame_row)

        records.append(
            {
                "trajectory": trajectory,
                "time": time,
                "lateral": lateral,
                "axial": axial,
                "ratio": ratio,
                "z_disp": z_disp,
                "xy_error": xy_error,
                "success": success,
            }
        )

    write_csv(output / "per_trajectory_statistics.csv", trajectory_rows)
    write_csv(output / "per_frame_features.csv", frame_rows)

    phase_summary: dict[str, dict] = {}
    for name in FEATURES:
        ratios = [float(row[f"{name}_near_over_approach"]) for row in trajectory_rows]
        near_aucs = [float(row[f"{name}_near_auc"]) for row in trajectory_rows]
        success_aucs = [float(row[f"{name}_success_auc"]) for row in trajectory_rows]
        phase_summary[name] = {
            "near_over_approach": finite_stats(ratios),
            "near_hole_auc": finite_stats(near_aucs),
            "success_auc": finite_stats(success_aucs),
            "near_increase_trajectories": int(sum(value > 1.0 for value in ratios)),
        }

    near_rows = [row for row in frame_rows if bool(row["near_hole"])]
    aligned_labels = np.asarray([not bool(row["aligned"]) for row in near_rows], dtype=bool)
    misalignment_auc = {}
    for name in FEATURES:
        score = np.asarray([float(row[f"{name}_normalized"]) for row in near_rows])
        misalignment_auc[name] = auc(aligned_labels, score)

    both_class_trajectories = [
        int(row["trajectory"])
        for row in trajectory_rows
        if int(row["aligned_near_frames"]) > 0 and int(row["misaligned_near_frames"]) > 0
    ]
    summary = {
        "thresholds": {
            "near_hole": f"z_disp_m <= {args.near_hole_z_m}",
            "aligned": f"xy_error_m <= {args.aligned_xy_m}",
            "lateral_axial_ratio": f"F_lat / (|Fz| + {args.ratio_epsilon_n} N)",
        },
        "data_contract": {
            "force": "sim-data-aligned obs/state/ee_wrench_base XYZ",
            "geometry": "sim-data gt_replay_validation.csv",
            "geometry_note": "GT replay geometry paired by trajectory/frame; not physical force-replay measured pose",
        },
        "counts": {
            "trajectories": len(trajectory_rows),
            "frames": len(frame_rows),
            "successful_trajectories": int(sum(bool(row["geometric_success"]) for row in trajectory_rows)),
            "failed_trajectories": int(sum(not bool(row["geometric_success"]) for row in trajectory_rows)),
            "near_hole_frames": len(near_rows),
            "aligned_near_frames": int(np.count_nonzero(~aligned_labels)),
            "misaligned_near_frames": int(np.count_nonzero(aligned_labels)),
            "trajectories_with_both_near_classes": both_class_trajectories,
        },
        "phase_discrimination": phase_summary,
        "misalignment_auc_exploratory": misalignment_auc,
        "correlations": {
            "lateral_force_vs_xy_error_near_median": finite_stats(
                [float(row["lateral_vs_xy_spearman_near"]) for row in trajectory_rows]
            ),
            "axial_force_vs_depth_progress_median": finite_stats(
                [float(row["axial_vs_depth_spearman"]) for row in trajectory_rows]
            ),
            "force_norm_vs_depth_progress_median": finite_stats(
                [float(row["norm_vs_depth_spearman"]) for row in trajectory_rows]
            ),
            "axial_force_vs_down_speed_median": finite_stats(
                [float(row["axial_vs_down_speed_spearman"]) for row in trajectory_rows]
            ),
        },
        "limitations": [
            "Only one trajectory is geometrically unsuccessful.",
            "Only 25 near-hole frames are misaligned, from three trajectories.",
            "Only two trajectories contain both aligned and misaligned near-hole frames.",
            "Torque is zero in sim-data-aligned.",
            "Force-frame semantics are not fully resolved.",
        ],
    }
    failed_rows = [row for row in trajectory_rows if not bool(row["geometric_success"])]
    successful_rows = [row for row in trajectory_rows if bool(row["geometric_success"])]
    if len(failed_rows) == 1:
        failed = failed_rows[0]
        summary["single_failure_case"] = {
            "trajectory": int(failed["trajectory"]),
            "warning": "case study only; one failed trajectory is insufficient for a threshold",
        }
        for name in ("lateral_force_n", "axial_force_n", "force_norm_n"):
            key = f"{name}_near_median"
            successful_values = np.asarray(
                [float(row[key]) for row in successful_rows], dtype=np.float64
            )
            summary["single_failure_case"][name] = {
                "failed_near_median": float(failed[key]),
                "successful_near_median": float(np.median(successful_values)),
                "successful_near_q25": float(np.percentile(successful_values, 25)),
                "successful_near_q75": float(np.percentile(successful_values, 75)),
                "failed_over_success_median": float(
                    float(failed[key]) / max(float(np.median(successful_values)), 1.0e-12)
                ),
            }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    selected = [record for record in records if record["trajectory"] in {0, 9, 12, 39}]
    figure, axes = plt.subplots(len(selected), 5, figsize=(20, 3.4 * len(selected)), squeeze=False)
    for row_index, record in enumerate(selected):
        values = (
            record["lateral"],
            record["axial"],
            record["ratio"],
            record["z_disp"] * 1000.0,
            record["xy_error"] * 1000.0,
        )
        labels = ("F_lat [N]", "|Fz| [N]", "F_lat/(|Fz|+0.05)", "z_disp [mm]", "xy error [mm]")
        for column, (value, label) in enumerate(zip(values, labels)):
            axis = axes[row_index, column]
            axis.plot(record["time"], value, linewidth=1.15)
            if np.any(record["success"]):
                start = int(np.flatnonzero(record["success"])[0])
                axis.axvline(record["time"][start], color="#16a34a", linestyle="--", linewidth=1.0)
            axis.grid(alpha=0.25)
            axis.set_title(f"traj_{record['trajectory']} {label}")
            axis.set_xlabel("time [s]")
    figure.suptitle("Derived simulation-force features and GT replay insertion geometry")
    figure.tight_layout()
    figure.savefig(output / "selected_derived_features.png", dpi=180)
    plt.close(figure)

    figure, axes = plt.subplots(1, 3, figsize=(16, 4.6))
    x = np.arange(len(FEATURES))
    near_auc_medians = [phase_summary[name]["near_hole_auc"]["median"] for name in FEATURES]
    success_auc_medians = [phase_summary[name]["success_auc"]["median"] for name in FEATURES]
    misalignment_values = [misalignment_auc[name] for name in FEATURES]
    axes[0].bar(x, near_auc_medians, color="#2563eb")
    axes[0].axhline(0.5, color="black", linestyle="--", linewidth=0.8)
    axes[0].set_title("Near-hole detection: median trajectory AUC")
    axes[1].bar(x, success_auc_medians, color="#16a34a")
    axes[1].axhline(0.5, color="black", linestyle="--", linewidth=0.8)
    axes[1].set_title("Success-phase detection: median trajectory AUC")
    axes[2].bar(x, misalignment_values, color="#dc2626")
    axes[2].axhline(0.5, color="black", linestyle="--", linewidth=0.8)
    axes[2].set_title("Misalignment AUC (exploratory, insufficient negatives)")
    short_labels = ("F_lat", "|Fz|", "ratio", "|F|")
    for axis in axes:
        axis.set_xticks(x, short_labels, rotation=20)
        axis.set_ylim(0.0, 1.0)
        axis.set_ylabel("AUC")
        axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    figure.savefig(output / "feature_discrimination_auc.png", dpi=180)
    plt.close(figure)

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"[DONE] {output}")


if __name__ == "__main__":
    main()
