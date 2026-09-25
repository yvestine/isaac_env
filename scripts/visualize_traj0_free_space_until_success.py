#!/usr/bin/env python3
"""Free-space-compensated real-vs-sim force visualization for traj0.

This is an offline analysis tool.  It does not modify the rollout or HDF5
files.  The real Franka wrench is corrected with a leave-one-trajectory-out
free-space model fitted on the first few seconds of the other real
trajectories.  The comparison is truncated at the first ``success=True`` row
in the simulation replay validation CSV.

For the simulation stream, ``auto`` prefers the 120 Hz baseline-corrected
``panda_link7`` stream and resamples it once to the trajectory timestamps.
This avoids comparing a twice-filtered 10 Hz stream and makes the displayed
amplitude reflect the recorded source more faithfully.  This remains an
offline visualization change; it does not modify rollout or HDF5 data.
"""

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
from scipy.signal import butter, sosfiltfilt


CHANNELS = ("Fx", "Fy", "Fz", "Tx", "Ty", "Tz")


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
    features = np.concatenate((q, qd, qdd, np.sin(q), np.cos(q)), axis=1)
    return {
        "q": q,
        "wrench": wrench,
        "timestamps": timestamps,
        "features": features,
    }


def standardize(
    features: np.ndarray,
    mean: np.ndarray | None = None,
    scale: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if mean is None:
        mean = features.mean(axis=0)
    if scale is None:
        scale = features.std(axis=0)
    scale = np.asarray(scale, dtype=np.float64).copy()
    scale[scale < 1.0e-8] = 1.0
    return (features - mean) / scale, mean, scale


def fit_free_space_model(
    trajectories: list[tuple[int, dict[str, np.ndarray]]],
    holdout_id: int,
    free_seconds: float,
    ridge: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    train_features = []
    train_targets = []
    train_frames = 0
    for trajectory_id, data in trajectories:
        if trajectory_id == holdout_id:
            continue
        mask = data["timestamps"] <= free_seconds
        train_features.append(data["features"][mask])
        train_targets.append(data["wrench"][mask])
        train_frames += int(np.count_nonzero(mask))
    if not train_features:
        raise RuntimeError("No training trajectories were found for free-space fitting")
    features = np.concatenate(train_features, axis=0)
    targets = np.concatenate(train_targets, axis=0)
    normalized, mean, scale = standardize(features)
    design = np.concatenate((np.ones((len(normalized), 1)), normalized), axis=1)
    regularizer = np.eye(design.shape[1], dtype=np.float64) * float(ridge)
    regularizer[0, 0] = 0.0
    coef = np.linalg.solve(
        design.T @ design + regularizer,
        design.T @ targets,
    )
    return mean, scale, coef, train_frames


def predict_free_space(
    data: dict[str, np.ndarray],
    mean: np.ndarray,
    scale: np.ndarray,
    coef: np.ndarray,
) -> np.ndarray:
    normalized, _, _ = standardize(data["features"], mean, scale)
    design = np.concatenate((np.ones((len(normalized), 1)), normalized), axis=1)
    return design @ coef


def lowpass(values: np.ndarray, sample_hz: float, cutoff_hz: float) -> np.ndarray:
    if cutoff_hz <= 0.0 or cutoff_hz >= sample_hz / 2.0:
        return values.copy()
    sos = butter(2, cutoff_hz, btype="low", fs=sample_hz, output="sos")
    return sosfiltfilt(sos, values, axis=0)


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


def read_named_wrench_csv(
    path: Path,
    prefix: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Read timestamp plus one of the named 120 Hz wrench streams."""
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames or []
        required = [f"{prefix}_{name}" for name in CHANNELS]
        missing = [name for name in ["timestamp", *required] if name not in fieldnames]
        if missing:
            raise ValueError(f"Missing columns {missing} in {path}")
        rows = []
        for row in reader:
            if not row:
                continue
            rows.append(
                [float(row["timestamp"]), *[float(row[name]) for name in required]]
            )
    values = np.asarray(rows, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 7 or not np.isfinite(values).all():
        raise ValueError(f"Invalid timestamp/wrench data in {path}: {values.shape}")
    sample_time = values[:, 0] - values[0, 0]
    wrench = values[:, 1:]
    if len(sample_time) < 2 or np.any(np.diff(sample_time) <= 0.0):
        raise ValueError(f"Invalid sample timestamps in {path}")
    return sample_time, wrench


def load_sim_stream(
    sim_dir: Path,
    stream: str,
    fps: float,
) -> tuple[np.ndarray, np.ndarray, Path, str]:
    """Load the selected recorded simulation stream before frame resampling."""
    if stream == "auto":
        if (
            sim_dir
            / "panda_link7_wrench_contact_isolated_base_at_K_120hz.csv"
        ).is_file():
            stream = "isolated-base-k-120hz"
        elif (sim_dir / "panda_link7_wrench_120hz.csv").is_file():
            stream = "corrected-120hz"
        elif (sim_dir / "panda_link7_wrench_filtered.csv").is_file():
            stream = "filtered-10hz"
        else:
            stream = "calibrated-10hz"

    stream_config = {
        "isolated-base-k-120hz": (
            "panda_link7_wrench_contact_isolated_base_at_K_120hz.csv",
            "filtered",
        ),
        "raw-120hz": ("panda_link7_wrench_120hz.csv", "raw"),
        "corrected-120hz": ("panda_link7_wrench_120hz.csv", "corrected"),
        "filtered-120hz": ("panda_link7_wrench_120hz.csv", "filtered"),
    }
    if stream in stream_config:
        filename, prefix = stream_config[stream]
        path = sim_dir / filename
        if not path.is_file():
            raise FileNotFoundError(f"Requested simulation stream is missing: {path}")
        sample_time, wrench = read_named_wrench_csv(path, prefix)
        return sample_time, wrench, path, stream

    if stream == "filtered-10hz":
        path = sim_dir / "panda_link7_wrench_filtered.csv"
    elif stream == "calibrated-10hz":
        path = sim_dir / "panda_link7_wrench_calibrated.csv"
    else:
        raise ValueError(f"Unknown --sim-stream={stream}")
    if not path.is_file():
        raise FileNotFoundError(f"Simulation wrench stream is missing: {path}")
    wrench = read_csv_matrix(path, 6)
    timestamps_path = sim_dir / "timestamps.csv"
    if timestamps_path.is_file():
        sample_time = read_csv_matrix(timestamps_path, 1)[:, 0]
        sample_time -= sample_time[0]
    else:
        sample_time = np.arange(len(wrench), dtype=np.float64) / float(fps)
    if len(sample_time) != len(wrench):
        raise ValueError(f"Timestamp/stream length mismatch for {path}")
    return sample_time, wrench, path, stream


def resample_wrench(
    sample_time: np.ndarray,
    wrench: np.ndarray,
    target_time: np.ndarray,
) -> np.ndarray:
    """Linearly resample a recorded stream to the paired 10 Hz timestamps."""
    tolerance = max(2.0 * float(np.median(np.diff(sample_time))), 1.0e-6)
    if target_time[0] < sample_time[0] - tolerance or target_time[-1] > sample_time[-1] + tolerance:
        raise ValueError(
            "Simulation wrench stream does not cover the paired trajectory: "
            f"stream=[{sample_time[0]:.3f}, {sample_time[-1]:.3f}] "
            f"target=[{target_time[0]:.3f}, {target_time[-1]:.3f}]"
        )
    return np.column_stack(
        [np.interp(target_time, sample_time, wrench[:, index]) for index in range(6)]
    )


def first_success_frame(path: Path) -> int:
    with path.open("r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        value = str(row.get("success", "")).strip().lower()
        if value in {"true", "1", "yes"}:
            return int(row["frame"])
    raise RuntimeError(f"No success=True row found in {path}")


def pearson(first: np.ndarray, second: np.ndarray) -> float | None:
    if len(first) < 2 or np.std(first) < 1.0e-12 or np.std(second) < 1.0e-12:
        return None
    value = float(np.corrcoef(first, second)[0, 1])
    return value if np.isfinite(value) else None


def best_lag(real_norm: np.ndarray, sim_norm: np.ndarray, max_lag: int) -> dict[str, float | int | None]:
    candidates: list[tuple[float, int]] = []
    for lag in range(max_lag + 1):
        if lag == 0:
            value = pearson(real_norm, sim_norm)
        else:
            value = pearson(real_norm[:-lag], sim_norm[lag:])
        if value is not None:
            candidates.append((value, lag))
    if not candidates:
        return {"correlation": None, "lag_frames": 0}
    value, lag = max(candidates, key=lambda item: item[0])
    return {"correlation": value, "lag_frames": lag}


def wrench_metrics(real: np.ndarray, sim: np.ndarray, max_lag: int) -> dict[str, object]:
    real_force = np.linalg.norm(real[:, :3], axis=1)
    sim_force = np.linalg.norm(sim[:, :3], axis=1)
    return {
        "component_pearson": {
            name: pearson(real[:, index], sim[:, index])
            for index, name in enumerate(CHANNELS)
        },
        "force_norm_pearson": pearson(real_force, sim_force),
        "force_norm_best_nonnegative_lag": best_lag(real_force, sim_force, max_lag),
        "real_force_norm_p95_n": float(np.percentile(real_force, 95)),
        "sim_force_norm_p95_n": float(np.percentile(sim_force, 95)),
        "force_p95_ratio_sim_over_real": float(
            np.percentile(sim_force, 95) / max(np.percentile(real_force, 95), 1.0e-12)
        ),
    }


def plot_components(
    path: Path,
    time: np.ndarray,
    real: np.ndarray,
    sim: np.ndarray,
    success_time: float,
    real_label: str,
    sim_label: str,
) -> None:
    figure, axes = plt.subplots(3, 1, figsize=(15, 10), sharex=True)
    for row, index in enumerate(range(3)):
        axes[row].plot(time, real[:, index], color="#d62728", linewidth=1.8, label=f"real {real_label}")
        axes[row].plot(time, sim[:, index], color="#1f77b4", linewidth=1.3, label=f"sim {sim_label}")
        axes[row].axvline(success_time, color="#2ca02c", linestyle="--", linewidth=1.2, label="sim success")
        axes[row].set_ylabel(f"{CHANNELS[index]} [N]")
        axes[row].grid(alpha=0.25)
        axes[row].legend(loc="upper right", fontsize=8)
    axes[-1].set_xlabel("time [s]")
    figure.suptitle("traj0 real-vs-sim force until simulation success")
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def plot_norm(
    path: Path,
    time: np.ndarray,
    real: np.ndarray,
    sim: np.ndarray,
    success_time: float,
    real_label: str,
    sim_label: str,
) -> None:
    real_norm = np.linalg.norm(real[:, :3], axis=1)
    sim_norm = np.linalg.norm(sim[:, :3], axis=1)
    figure, axis = plt.subplots(figsize=(15, 5))
    axis.plot(time, real_norm, color="#d62728", linewidth=1.8, label=f"real {real_label} |F|")
    axis.plot(time, sim_norm, color="#1f77b4", linewidth=1.3, label=f"sim {sim_label} |F|")
    axis.axvline(success_time, color="#2ca02c", linestyle="--", linewidth=1.2, label="sim success")
    axis.set_xlabel("time [s]")
    axis.set_ylabel("force norm [N]")
    axis.set_title("traj0 force norm until simulation success")
    axis.grid(alpha=0.25)
    axis.legend(loc="upper right")
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def write_csv(path: Path, time: np.ndarray, real: np.ndarray, sim: np.ndarray) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["timestamp", *[f"real_{name}" for name in CHANNELS], *[f"sim_{name}" for name in CHANNELS]])
        for timestamp, real_row, sim_row in zip(time, real, sim):
            writer.writerow([float(timestamp), *real_row.tolist(), *sim_row.tolist()])


def write_video(
    output: Path,
    wrist_video: Path,
    time: np.ndarray,
    real: np.ndarray,
    sim: np.ndarray,
    success_time: float,
    fps: float,
    real_label: str,
    sim_label: str,
) -> bool:
    if not wrist_video.is_file():
        return False
    capture = cv2.VideoCapture(str(wrist_video))
    if not capture.isOpened():
        return False
    input_fps = float(capture.get(cv2.CAP_PROP_FPS))
    if not np.isfinite(input_fps) or input_fps <= 0.0:
        input_fps = fps

    figure, axis = plt.subplots(figsize=(8.0, 4.5), dpi=100)
    real_norm = np.linalg.norm(real[:, :3], axis=1)
    sim_norm = np.linalg.norm(sim[:, :3], axis=1)
    axis.plot(time, real_norm, color="#d62728", linewidth=1.5, label=f"real {real_label} |F|")
    axis.plot(time, sim_norm, color="#1f77b4", linewidth=1.2, label=f"sim {sim_label} |F|")
    axis.axvline(success_time, color="#2ca02c", linestyle="--", linewidth=1.0, label="success")
    cursor = axis.axvline(time[0], color="#ff7f0e", linewidth=1.2)
    axis.set_xlim(time[0], time[-1])
    axis.set_xlabel("time [s]")
    axis.set_ylabel("|F| [N]")
    axis.grid(alpha=0.2)
    axis.legend(loc="upper right", fontsize=7)
    figure.tight_layout()
    figure.canvas.draw()
    chart = np.asarray(figure.canvas.buffer_rgba())[..., :3]
    chart_height, chart_width = chart.shape[:2]

    ok, frame = capture.read()
    if not ok:
        capture.release()
        plt.close(figure)
        return False
    wrist_width = int(round(frame.shape[1] * chart_height / frame.shape[0]))
    writer = cv2.VideoWriter(
        str(output),
        cv2.VideoWriter_fourcc(*"mp4v"),
        input_fps,
        (wrist_width + chart_width, chart_height),
    )
    if not writer.isOpened():
        capture.release()
        plt.close(figure)
        return False

    for index in range(len(time)):
        if index > 0:
            ok, frame = capture.read()
            if not ok:
                break
        wrist = cv2.resize(frame, (wrist_width, chart_height))
        cursor.set_xdata([time[index], time[index]])
        figure.canvas.draw()
        chart_bgr = cv2.cvtColor(
            np.asarray(figure.canvas.buffer_rgba())[..., :3],
            cv2.COLOR_RGB2BGR,
        )
        cv2.putText(
            wrist,
            f"frame {index:03d}  t={time[index]:.2f}s",
            (16, 32),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        writer.write(np.concatenate((wrist, chart_bgr), axis=1))
    writer.release()
    capture.release()
    plt.close(figure)
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--real-data-dir", type=Path, default=Path("real_data"))
    parser.add_argument("--trajectory", type=int, default=0)
    parser.add_argument("--real-h5", type=Path, default=None)
    parser.add_argument("--sim-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--free-seconds", type=float, default=3.0)
    parser.add_argument("--offset-seconds", type=float, default=0.5)
    parser.add_argument("--cutoff-hz", type=float, default=1.0)
    parser.add_argument("--ridge", type=float, default=1.0e-2)
    parser.add_argument("--max-lag-seconds", type=float, default=0.3)
    parser.add_argument("--fps", type=float, default=10.0)
    parser.add_argument(
        "--real-compensation",
        choices=("static", "free-space-model"),
        default="static",
        help=(
            "Real-wrench preprocessing used in the primary plots. static subtracts the "
            "first free-space interval; free-space-model uses the leave-traj0-out model."
        ),
    )
    parser.add_argument(
        "--sim-stream",
        choices=(
            "auto",
            "isolated-base-k-120hz",
            "raw-120hz",
            "corrected-120hz",
            "filtered-120hz",
            "filtered-10hz",
            "calibrated-10hz",
        ),
        default="auto",
        help=(
            "Simulation stream used for visualization. auto first prefers the base/K "
            "contact-isolated 120 Hz panda_link7 stream, then the native baseline-corrected "
            "120 Hz stream, and resamples it once to 10 Hz."
        ),
    )
    args = parser.parse_args()

    real_h5 = args.real_h5 or (args.real_data_dir / f"traj_{args.trajectory}" / "data.h5")
    paths = sorted(
        args.real_data_dir.glob("traj_*/data.h5"),
        key=lambda path: int(path.parent.name.removeprefix("traj_")),
    )
    if not paths:
        raise FileNotFoundError(f"No real trajectories under {args.real_data_dir}")
    trajectories = [
        (int(path.parent.name.removeprefix("traj_")), load_real(path))
        for path in paths
    ]
    target_id = args.trajectory
    target = next((data for trajectory_id, data in trajectories if trajectory_id == target_id), None)
    if target is None:
        raise ValueError(f"Trajectory {target_id} was not found under {args.real_data_dir}")

    feature_mean, feature_scale, coef, train_frames = fit_free_space_model(
        trajectories,
        target_id,
        args.free_seconds,
        args.ridge,
    )
    predicted_free = predict_free_space(target, feature_mean, feature_scale, coef)
    initial_mask = target["timestamps"] <= args.offset_seconds
    if not np.any(initial_mask):
        raise RuntimeError("No frames available for the initial free-space offset")
    real_residual = target["wrench"] - predicted_free
    real_residual -= np.median(real_residual[initial_mask], axis=0, keepdims=True)

    sim_dir = args.sim_dir.resolve()
    sim_sample_time, sim_samples, sim_wrench_path, sim_stream = load_sim_stream(
        sim_dir, args.sim_stream, args.fps
    )
    sim_time_path = sim_dir / "timestamps.csv"
    if sim_time_path.is_file():
        sim_time = read_csv_matrix(sim_time_path, 1)[:, 0]
        sim_time -= sim_time[0]
    else:
        sim_time = np.arange(len(sim_samples), dtype=np.float64) / args.fps

    count = min(len(target["wrench"]), len(target["timestamps"]), len(sim_time))
    if count < 2:
        raise RuntimeError("Not enough aligned frames")
    target_time = target["timestamps"][:count]
    real_residual = real_residual[:count]
    sim_time = sim_time[:count]
    sim_wrench = resample_wrench(sim_sample_time, sim_samples, sim_time)

    stop_frame = min(first_success_frame(sim_dir / "gt_replay_validation.csv"), count - 1)
    keep = stop_frame + 1
    time = target_time[:keep]
    real_raw_static = target["wrench"][:count] - np.median(
        target["wrench"][:count][target_time <= args.free_seconds],
        axis=0,
        keepdims=True,
    )
    sim_baseline = np.median(sim_wrench[sim_time <= args.free_seconds], axis=0, keepdims=True)
    sim_corrected = sim_wrench - sim_baseline

    real_free_space_plot = lowpass(
        real_residual[:keep],
        1.0 / np.median(np.diff(target_time)),
        args.cutoff_hz,
    )
    real_static_plot = lowpass(
        real_raw_static[:keep],
        1.0 / np.median(np.diff(target_time)),
        args.cutoff_hz,
    )
    if args.real_compensation == "free-space-model":
        real_plot = real_free_space_plot
        real_label = "free-space model"
    else:
        real_plot = real_static_plot
        real_label = "static baseline"
    if sim_stream == "isolated-base-k-120hz":
        # This stream has already received the rollout contract's one causal
        # low-pass after counterfactual subtraction. Do not attenuate it twice.
        sim_plot = sim_corrected[:keep].copy()
    else:
        sim_plot = lowpass(
            sim_corrected[:keep],
            1.0 / np.median(np.diff(sim_time)),
            args.cutoff_hz,
        )
    success_time = float(time[-1])

    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    plot_components(
        output / "force_components_free_space_until_success.png",
        time,
        real_plot,
        sim_plot,
        success_time,
        real_label,
        sim_stream,
    )
    plot_norm(
        output / "force_norm_free_space_until_success.png",
        time,
        real_plot,
        sim_plot,
        success_time,
        real_label,
        sim_stream,
    )
    write_csv(output / "force_streams_free_space_until_success.csv", time, real_plot, sim_plot)

    max_lag = max(0, int(round(args.max_lag_seconds * args.fps)))
    metrics = {
        "trajectory": args.trajectory,
        "success_frame": int(stop_frame),
        "success_time_s": success_time,
        "frame_count_until_success": int(keep),
        "free_space_fit": {
            "holdout_trajectory": args.trajectory,
            "training_trajectory_count": len(trajectories) - 1,
            "training_free_space_frames": train_frames,
            "free_seconds": args.free_seconds,
            "offset_seconds": args.offset_seconds,
            "ridge": args.ridge,
        },
        "filter": {
            "offline_lowpass_cutoff_hz": args.cutoff_hz,
            "sim_additional_lowpass_applied": sim_stream != "isolated-base-k-120hz",
            "note": "sosfiltfilt is for offline visualization only; use a causal filter online",
        },
        "real_compensation": args.real_compensation,
        "metrics_until_success": wrench_metrics(real_plot, sim_plot, max_lag),
        "static_baseline_metrics_until_success": wrench_metrics(
            real_static_plot, sim_plot, max_lag
        ),
        "free_space_model_metrics_until_success": wrench_metrics(
            real_free_space_plot, sim_plot, max_lag
        ),
        "sim_stream": sim_stream,
        "sim_source": str(sim_wrench_path.resolve()),
        "real_source": str(real_h5.resolve()),
        "success_source": str((sim_dir / "gt_replay_validation.csv").resolve()),
    }
    (output / "free_space_comparison_metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )

    video_path = output / "wrist_force_free_space_until_success.mp4"
    video_ok = write_video(
        video_path,
        sim_dir / "wrist_camera.mp4",
        time,
        real_plot,
        sim_plot,
        success_time,
        args.fps,
        real_label,
        sim_stream,
    )
    print(f"[DONE] success frame={stop_frame} time={success_time:.3f}s")
    print(f"[DONE] components={output / 'force_components_free_space_until_success.png'}")
    print(f"[DONE] norm={output / 'force_norm_free_space_until_success.png'}")
    print(f"[DONE] metrics={output / 'free_space_comparison_metrics.json'}")
    if video_ok:
        print(f"[DONE] video={video_path}")
    else:
        print("[WARN] wrist video was not available; PNG/CSV/JSON were still written")


if __name__ == "__main__":
    main()
