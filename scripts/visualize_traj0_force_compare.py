#!/usr/bin/env python3
"""Visualize calibrated panda_link7 force and insertion-contact feedback."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import cv2
import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


CHANNELS = ("Fx", "Fy", "Fz", "Tx", "Ty", "Tz")
STREAM_FILES = {
    "contact_wrench": "contact_wrench_base_at_K.csv",
}
COLORS = {
    "panda_link7_calibrated": "#d62728",
    "contact_wrench": "#1f77b4",
}


def read_csv_matrix(path: Path, columns: int) -> np.ndarray:
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        next(reader, None)
        rows = [[float(value) for value in row] for row in reader if row]
    values = np.asarray(rows, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != columns:
        raise ValueError(f"Expected {columns} columns in {path}, got {values.shape}")
    if not np.isfinite(values).all():
        raise FloatingPointError(f"NaN or Inf in {path}")
    return values


def read_real_wrench(path: Path) -> np.ndarray:
    with h5py.File(path, "r") as data:
        key = "obs/state/ee_wrench_base"
        if key not in data:
            raise KeyError(f"{path}: missing {key} (Franka O_F_ext_hat_K)")
        values = np.asarray(data[key], dtype=np.float64).reshape(-1, 6)
    if not np.isfinite(values).all():
        raise FloatingPointError(f"NaN or Inf in {path}:{key}")
    return values


def read_times(sim_dir: Path, count: int, fallback_fps: float) -> np.ndarray:
    path = sim_dir / "timestamps.csv"
    if path.is_file():
        values = read_csv_matrix(path, 1)[:count, 0]
        if len(values) == count and np.all(np.diff(values) > 0.0):
            return values - values[0]
    return np.arange(count, dtype=np.float64) / fallback_fps


def baseline_correct(values: np.ndarray, times: np.ndarray, seconds: float) -> tuple[np.ndarray, np.ndarray]:
    mask = times <= seconds
    if not np.any(mask):
        mask = np.arange(len(times)) < min(len(times), 30)
    baseline = np.median(values[mask], axis=0)
    return values - baseline, baseline


def pearson(a: np.ndarray, b: np.ndarray) -> float | None:
    if len(a) < 2 or np.std(a) < 1.0e-12 or np.std(b) < 1.0e-12:
        return None
    value = float(np.corrcoef(a, b)[0, 1])
    return value if np.isfinite(value) else None


def best_nonnegative_lag(real: np.ndarray, sim: np.ndarray, max_lag: int) -> dict[str, float | int | None]:
    candidates = []
    for lag in range(max_lag + 1):
        if lag == 0:
            corr = pearson(real, sim)
        else:
            corr = pearson(real[:-lag], sim[lag:])
        if corr is not None:
            candidates.append((corr, lag))
    if not candidates:
        return {"correlation": None, "lag_frames": 0}
    correlation, lag = max(candidates, key=lambda item: item[0])
    return {"correlation": correlation, "lag_frames": int(lag)}


def wrench_metrics(real: np.ndarray, simulated: np.ndarray, max_lag: int) -> dict[str, object]:
    real_force_norm = np.linalg.norm(real[:, :3], axis=1)
    sim_force_norm = np.linalg.norm(simulated[:, :3], axis=1)
    real_torque_norm = np.linalg.norm(real[:, 3:], axis=1)
    sim_torque_norm = np.linalg.norm(simulated[:, 3:], axis=1)
    real_p95 = float(np.percentile(real_force_norm, 95))
    sim_p95 = float(np.percentile(sim_force_norm, 95))
    return {
        "component_pearson": {
            CHANNELS[index]: pearson(real[:, index], simulated[:, index])
            for index in range(6)
        },
        "force_norm_pearson": pearson(real_force_norm, sim_force_norm),
        "torque_norm_pearson": pearson(real_torque_norm, sim_torque_norm),
        "force_norm_best_nonnegative_lag": best_nonnegative_lag(
            real_force_norm, sim_force_norm, max_lag
        ),
        "force_norm_rmse_n": float(np.sqrt(np.mean((sim_force_norm - real_force_norm) ** 2))),
        "torque_norm_rmse_nm": float(np.sqrt(np.mean((sim_torque_norm - real_torque_norm) ** 2))),
        "real_force_norm_p95_n": real_p95,
        "sim_force_norm_p95_n": sim_p95,
        "force_p95_ratio_sim_over_real": sim_p95 / real_p95 if real_p95 > 1.0e-12 else None,
    }


def plot_components(
    path: Path,
    times: np.ndarray,
    real: np.ndarray,
    streams: dict[str, np.ndarray],
    start: int,
    title: str,
) -> None:
    figure, axes = plt.subplots(3, 1, figsize=(15, 10), sharex=True)
    for row, component in enumerate(range(start, start + 3)):
        axes[row].plot(times, real[:, component], color="black", linewidth=1.8, label="real O_F_ext_hat_K")
        for name, values in streams.items():
            axes[row].plot(
                times,
                values[:, component],
                color=COLORS[name],
                linewidth=1.1,
                label=f"sim {name}",
            )
        axes[row].set_ylabel(CHANNELS[component] + (" [N]" if start == 0 else " [N*m]"))
        axes[row].grid(alpha=0.25)
        axes[row].legend(loc="upper right", ncol=2, fontsize=8)
    axes[-1].set_xlabel("trajectory time [s]")
    figure.suptitle(title)
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def plot_norms_and_contact(
    path: Path,
    times: np.ndarray,
    real: np.ndarray,
    streams: dict[str, np.ndarray],
    contact_count: np.ndarray,
    insertion_feedback: np.ndarray,
    link7_feedback: np.ndarray,
) -> None:
    figure, axes = plt.subplots(3, 1, figsize=(15, 11), sharex=True)
    axes[0].plot(
        times,
        np.linalg.norm(real[:, :3], axis=1),
        color="black",
        linewidth=1.8,
        label="real |O_F_ext_hat_K force|",
    )
    for name, values in streams.items():
        axes[0].plot(times, np.linalg.norm(values[:, :3], axis=1), color=COLORS[name], label=name)
    axes[1].plot(
        times,
        link7_feedback[:, 0],
        color=COLORS["panda_link7_calibrated"],
        linewidth=1.6,
        label="panda_link7 calibrated |F|",
    )
    axes[1].plot(
        times,
        insertion_feedback[:, 1],
        color="#2ca02c",
        linewidth=1.4,
        label="peg-hole contact EMA",
    )
    axes[1].axhline(
        2.0,
        color="#ff7f0e",
        linestyle="--",
        linewidth=1.2,
        label="2 N jam threshold",
    )
    axes[2].step(times, contact_count, where="post", color="#9467bd", label="PhysX contact")
    axes[2].step(
        times,
        link7_feedback[:, 2],
        where="post",
        color="#d62728",
        label="jam: >2 N for 0.5 s",
    )
    axes[0].set_ylabel("|F| [N]")
    axes[1].set_ylabel("filtered feedback [N]")
    axes[2].set_ylabel("state")
    axes[2].set_xlabel("trajectory time [s]")
    for axis in axes:
        axis.grid(alpha=0.25)
        axis.legend(loc="upper right", fontsize=8)
    figure.suptitle("traj0 insertion feedback: calibrated panda_link7 and peg-hole contact")
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def write_wrist_comparison_video(
    path: Path,
    wrist_path: Path,
    times: np.ndarray,
    real: np.ndarray,
    simulated: np.ndarray,
    insertion_feedback: np.ndarray,
    link7_feedback: np.ndarray,
    fallback_fps: float,
) -> None:
    capture = cv2.VideoCapture(str(wrist_path))
    if not capture.isOpened():
        raise RuntimeError(f"Cannot open wrist video: {wrist_path}")
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    if not np.isfinite(fps) or fps <= 0.0:
        fps = fallback_fps

    figure, axes = plt.subplots(3, 1, figsize=(8.0, 4.8), dpi=100, sharex=True)
    cursors = []
    axes[0].plot(
        times,
        np.linalg.norm(real[:, :3], axis=1),
        color="black",
        linewidth=1.5,
        label="real |F|",
    )
    axes[0].plot(
        times,
        np.linalg.norm(simulated[:, :3], axis=1),
        color=COLORS["panda_link7_calibrated"],
        linewidth=1.3,
        label="panda_link7 |F|",
    )
    axes[1].plot(
        times,
        insertion_feedback[:, 1],
        color="#2ca02c",
        linewidth=1.4,
        label="contact EMA",
    )
    axes[1].axhline(2.0, color="#ff7f0e", linestyle="--", linewidth=1.0, label="2 N")
    axes[2].step(
        times,
        link7_feedback[:, 2],
        where="post",
        color="#d62728",
        label="jam flag",
    )
    axes[0].set_ylabel("|F| [N]")
    axes[1].set_ylabel("feedback [N]")
    axes[2].set_ylabel("jam")
    for axis in axes:
        cursors.append(axis.axvline(times[0], color="#ff7f0e", linewidth=1.2))
        axis.grid(alpha=0.2)
        axis.legend(loc="upper right", fontsize=7)
    axes[-1].set_xlabel("time [s]")
    figure.suptitle("traj0 wrist + filtered insertion feedback")
    figure.tight_layout()
    figure.canvas.draw()
    chart_rgb = np.asarray(figure.canvas.buffer_rgba())[..., :3]
    chart_height, chart_width = chart_rgb.shape[:2]

    ok, wrist = capture.read()
    if not ok:
        capture.release()
        plt.close(figure)
        raise RuntimeError(f"Wrist video has no frames: {wrist_path}")
    wrist_target_width = int(round(wrist.shape[1] * chart_height / wrist.shape[0]))
    wrist = cv2.resize(wrist, (wrist_target_width, chart_height))
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (wrist.shape[1] + chart_width, chart_height),
    )
    if not writer.isOpened():
        capture.release()
        plt.close(figure)
        raise RuntimeError(f"Cannot create video: {path}")

    for index in range(len(times)):
        if index > 0:
            ok, wrist = capture.read()
            if not ok:
                break
            wrist = cv2.resize(wrist, (wrist_target_width, chart_height))
        for cursor in cursors:
            cursor.set_xdata([times[index], times[index]])
        figure.canvas.draw()
        chart = cv2.cvtColor(np.asarray(figure.canvas.buffer_rgba())[..., :3], cv2.COLOR_RGB2BGR)
        cv2.putText(
            wrist,
            f"frame {index:03d}  t={times[index]:.2f}s",
            (16, 32),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        writer.write(np.concatenate((wrist, chart), axis=1))
    writer.release()
    capture.release()
    plt.close(figure)


def first_persistent_index(mask: np.ndarray, minimum_frames: int) -> int | None:
    """Return the first index belonging to a persistent true run."""

    run_start = None
    run_length = 0
    for index, active in enumerate(np.asarray(mask, dtype=bool)):
        if active:
            if run_start is None:
                run_start = index
            run_length += 1
            if run_length >= minimum_frames:
                return int(run_start)
        else:
            run_start = None
            run_length = 0
    return None


def contact_onset(
    times: np.ndarray,
    real: np.ndarray,
    contact_count: np.ndarray,
    baseline_seconds: float,
    minimum_frames: int,
) -> dict[str, float | int | None]:
    baseline_mask = times <= baseline_seconds
    real_norm = np.linalg.norm(real[:, :3], axis=1)
    baseline = real_norm[baseline_mask]
    median = float(np.median(baseline))
    mad = float(np.median(np.abs(baseline - median)))
    threshold = max(1.0, median + 6.0 * 1.4826 * mad)
    search_mask = times >= baseline_seconds
    real_index = first_persistent_index(
        search_mask & (real_norm > threshold), minimum_frames
    )
    sim_index = first_persistent_index(contact_count > 0.0, minimum_frames)
    real_time = float(times[real_index]) if real_index is not None else None
    sim_time = float(times[sim_index]) if sim_index is not None else None
    return {
        "real_force_threshold_n": threshold,
        "minimum_consecutive_frames": int(minimum_frames),
        "real_onset_s": real_time,
        "sim_contact_onset_s": sim_time,
        "absolute_difference_s": (
            abs(real_time - sim_time)
            if real_time is not None and sim_time is not None
            else None
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--real-h5", type=Path, required=True)
    parser.add_argument("--sim-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--fps", type=float, default=10.0)
    parser.add_argument("--baseline-seconds", type=float, default=3.0)
    parser.add_argument("--max-lag-seconds", type=float, default=0.3)
    parser.add_argument("--onset-min-consecutive", type=int, default=3)
    args = parser.parse_args()
    if args.onset_min_consecutive <= 0:
        raise ValueError("--onset-min-consecutive must be positive")

    real_raw = read_real_wrench(args.real_h5)
    calibrated_path = args.sim_dir / "panda_link7_wrench_calibrated.csv"
    primary_path = (
        calibrated_path
        if calibrated_path.is_file()
        else args.sim_dir / "panda_link7_wrench_filtered.csv"
    )
    stream_files = {
        "panda_link7_calibrated": primary_path,
        **{
            name: args.sim_dir / filename
            for name, filename in STREAM_FILES.items()
        },
    }
    streams_raw = {
        name: read_csv_matrix(path, 6)
        for name, path in stream_files.items()
    }
    contact_count = read_csv_matrix(args.sim_dir / "contact_count.csv", 1)[:, 0]
    insertion_feedback = read_csv_matrix(args.sim_dir / "insertion_feedback.csv", 3)
    link7_feedback = read_csv_matrix(args.sim_dir / "panda_link7_force_feedback.csv", 3)
    count = min(
        len(real_raw),
        len(contact_count),
        len(insertion_feedback),
        len(link7_feedback),
        *[len(values) for values in streams_raw.values()],
    )
    if count < 2:
        raise RuntimeError("Not enough aligned traj0 frames")
    real_raw = real_raw[:count]
    streams_raw = {name: values[:count] for name, values in streams_raw.items()}
    contact_count = contact_count[:count]
    insertion_feedback = insertion_feedback[:count]
    link7_feedback = link7_feedback[:count]
    times = read_times(args.sim_dir, count, args.fps)

    real, real_baseline = baseline_correct(real_raw, times, args.baseline_seconds)
    streams = {}
    stream_baselines = {}
    for name, values in streams_raw.items():
        streams[name], stream_baselines[name] = baseline_correct(values, times, args.baseline_seconds)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    plot_components(
        args.output_dir / "force_components.png",
        times,
        real,
        streams,
        0,
        "traj0 force components: real base frame vs calibrated panda_link7 native frame",
    )
    plot_components(
        args.output_dir / "torque_components.png",
        times,
        real,
        streams,
        3,
        "traj0 torque components: real base frame vs calibrated panda_link7 native frame",
    )
    plot_norms_and_contact(
        args.output_dir / "wrench_norms_contact.png",
        times,
        real,
        streams,
        contact_count,
        insertion_feedback,
        link7_feedback,
    )
    plot_components(
        args.output_dir / "force_components_raw.png",
        times,
        real_raw,
        streams_raw,
        0,
        "traj0 force components before visualization baseline correction",
    )
    write_wrist_comparison_video(
        args.output_dir / "traj0_wrist_force_compare.mp4",
        args.sim_dir / "wrist_camera.mp4",
        times,
        real,
        streams["panda_link7_calibrated"],
        insertion_feedback,
        link7_feedback,
        args.fps,
    )

    max_lag_frames = max(0, int(round(args.max_lag_seconds * args.fps)))
    metrics = {
        "real_source": str(args.real_h5.resolve()),
        "sim_source": str(args.sim_dir.resolve()),
        "frame_count": int(count),
        "baseline_seconds": float(args.baseline_seconds),
        "frame_contract_note": (
            "panda_link7 components remain in the PhysX native incoming-joint frame; "
            "force-norm trend is valid but per-axis comparison is provisional"
        ),
        "real_baseline": real_baseline.tolist(),
        "sim_baselines": {name: value.tolist() for name, value in stream_baselines.items()},
        "contact_onset": contact_onset(
            times,
            real,
            contact_count,
            args.baseline_seconds,
            args.onset_min_consecutive,
        ),
        "baseline_corrected_metrics": {
            name: wrench_metrics(real, values, max_lag_frames)
            for name, values in streams.items()
        },
        "raw_metrics": {
            name: wrench_metrics(real_raw, values, max_lag_frames)
            for name, values in streams_raw.items()
        },
    }
    metrics_path = args.output_dir / "comparison_metrics.json"
    metrics_path.write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )
    estimate = metrics["baseline_corrected_metrics"]["panda_link7_calibrated"]
    print(f"[DONE] visualization: {args.output_dir.resolve()}")
    print(f"[DONE] video: {(args.output_dir / 'traj0_wrist_force_compare.mp4').resolve()}")
    print(f"[DONE] metrics: {metrics_path.resolve()}")
    print(
        "[TREND] calibrated panda_link7 native-frame wrench: "
        f"|F|_r={estimate['force_norm_pearson']} "
        f"p95_ratio={estimate['force_p95_ratio_sim_over_real']}"
    )


if __name__ == "__main__":
    main()
