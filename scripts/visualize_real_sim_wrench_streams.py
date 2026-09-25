#!/usr/bin/env python3
"""Visualize real traj0 wrench against all saved simulation wrench streams."""

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


CHANNELS = ("Fx", "Fy", "Fz", "Tx", "Ty", "Tz")
STREAM_FILES = {
    "wrench_raw": "wrench_raw.csv",
    "wrench_base": "wrench_base.csv",
    "wrench_final": "wrench_final.csv",
    "wrench_model_clean": "wrench_model_clean.csv",
    "wrench_model": "wrench_model.csv",
    "wrench_tool_smooth": "wrench_tool_smooth.csv",
}


def read_six_csv(path: Path) -> np.ndarray:
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        next(reader, None)
        rows = [[float(value) for value in row] for row in reader if row]
    values = np.asarray(rows, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 6:
        raise ValueError(f"Expected six columns in {path}, got {values.shape}")
    return values


def read_real_wrench(path: Path) -> np.ndarray:
    with h5py.File(path, "r") as data:
        for key in (
            "obs/state/ee_wrench_base",
            "observations/ee_wrench_base",
            "state/ee_wrench_base",
        ):
            if key in data:
                values = np.asarray(data[key], dtype=np.float64).reshape(-1, 6)
                break
        else:
            raise KeyError(f"{path}: ee_wrench_base not found")
    return values


def read_sim_training_wrench(sim_dir: Path) -> np.ndarray:
    """Read the exact simulated wrench written into the training contract."""
    path = sim_dir / "data.h5"
    if not path.is_file():
        raise FileNotFoundError(f"Simulation training H5 not found: {path}")
    with h5py.File(path, "r") as data:
        key = "obs/state/ee_wrench_base"
        if key not in data:
            raise KeyError(f"{path}: {key} not found")
        values = np.asarray(data[key], dtype=np.float64).reshape(-1, 6)
    return values


def correlation(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 2 or np.std(a) < 1.0e-12 or np.std(b) < 1.0e-12:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def stream_metrics(real: np.ndarray, simulated: np.ndarray) -> dict[str, object]:
    force_norm_real = np.linalg.norm(real[:, :3], axis=1)
    force_norm_sim = np.linalg.norm(simulated[:, :3], axis=1)
    torque_norm_real = np.linalg.norm(real[:, 3:], axis=1)
    torque_norm_sim = np.linalg.norm(simulated[:, 3:], axis=1)
    error = simulated - real
    return {
        "force_norm_correlation": correlation(force_norm_real, force_norm_sim),
        "torque_norm_correlation": correlation(torque_norm_real, torque_norm_sim),
        "force_norm_rmse": float(np.sqrt(np.mean((force_norm_sim - force_norm_real) ** 2))),
        "torque_norm_rmse": float(np.sqrt(np.mean((torque_norm_sim - torque_norm_real) ** 2))),
        "component_mae": np.mean(np.abs(error), axis=0).tolist(),
        "component_rmse": np.sqrt(np.mean(error**2, axis=0)).tolist(),
        "sim_force_norm_max": float(np.max(force_norm_sim)),
        "sim_torque_norm_max": float(np.max(torque_norm_sim)),
    }


def plot_norms(output: Path, times: np.ndarray, real: np.ndarray, streams: dict[str, np.ndarray]) -> None:
    figure, axes = plt.subplots(2, 1, figsize=(15, 9), sharex=True)
    real_force = np.linalg.norm(real[:, :3], axis=1)
    real_torque = np.linalg.norm(real[:, 3:], axis=1)
    axes[0].plot(times, real_force, color="black", linewidth=1.8, label="real ee_wrench_base")
    axes[1].plot(times, real_torque, color="black", linewidth=1.8, label="real ee_wrench_base")
    for name, values in streams.items():
        axes[0].plot(times, np.linalg.norm(values[:, :3], axis=1), linewidth=1.0, label=name)
        axes[1].plot(times, np.linalg.norm(values[:, 3:], axis=1), linewidth=1.0, label=name)
    axes[0].set_ylabel("force norm [N]")
    axes[1].set_ylabel("torque norm [N*m]")
    axes[1].set_xlabel("trajectory time [s]")
    axes[0].set_title("traj0 real vs simulated wrench norms")
    for axis in axes:
        axis.grid(alpha=0.25)
        axis.legend(loc="upper right", ncol=2, fontsize=8)
    figure.tight_layout()
    figure.savefig(output, dpi=160)
    plt.close(figure)


def plot_components(
    output: Path,
    times: np.ndarray,
    real: np.ndarray,
    simulated: np.ndarray,
    sim_name: str,
    start: int,
    title: str,
) -> None:
    figure, axes = plt.subplots(3, 1, figsize=(15, 10), sharex=True)
    for row, channel in enumerate(range(start, start + 3)):
        axes[row].plot(times, real[:, channel], color="black", linewidth=1.6, label="real")
        axes[row].plot(times, simulated[:, channel], linewidth=1.2, label=f"sim {sim_name}")
        axes[row].set_ylabel(CHANNELS[channel])
        axes[row].grid(alpha=0.25)
        axes[row].legend(loc="upper right")
    axes[-1].set_xlabel("trajectory time [s]")
    figure.suptitle(title)
    figure.tight_layout()
    figure.savefig(output, dpi=160)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--real-h5", type=Path, required=True)
    parser.add_argument("--sim-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    real = read_real_wrench(args.real_h5)
    # This is the force actually consumed by training.  In the restored
    # sim_force rollout it is built from contact_total_base; wrench_final is a
    # separate incoming-joint-wrench diagnostic and is expected to be zero
    # when joints are replayed by direct state writes.
    streams = {"training_wrench": read_sim_training_wrench(args.sim_dir)}
    streams.update({
        name: read_six_csv(args.sim_dir / filename)
        for name, filename in STREAM_FILES.items()
        if (args.sim_dir / filename).is_file()
    })
    if not streams:
        raise FileNotFoundError(f"No simulation wrench CSV files found in {args.sim_dir}")

    count = min([len(real), *[len(values) for values in streams.values()]])
    if count < 2:
        raise RuntimeError("Not enough aligned wrench frames")
    real = real[:count]
    streams = {name: values[:count] for name, values in streams.items()}

    timestamp_path = args.sim_dir / "timestamps.csv"
    if timestamp_path.is_file():
        with timestamp_path.open("r", newline="", encoding="utf-8") as handle:
            reader = csv.reader(handle)
            next(reader, None)
            times = np.asarray([float(row[0]) for row in reader if row], dtype=np.float64)[:count]
        if len(times) != count or len(times) < 2 or np.any(np.diff(times) <= 0.0):
            times = np.arange(count, dtype=np.float64) / 10.0
    else:
        times = np.arange(count, dtype=np.float64) / 10.0

    args.output_dir.mkdir(parents=True, exist_ok=True)
    plot_norms(args.output_dir / "wrench_norms.png", times, real, streams)
    training = streams["training_wrench"]
    final = streams.get("wrench_final", training)
    model = streams.get("wrench_model", final)
    plot_components(
        args.output_dir / "force_components_training.png",
        times,
        real,
        training,
        "training_wrench",
        0,
        "traj0 force components: real vs simulated training wrench",
    )
    plot_components(
        args.output_dir / "torque_components_training.png",
        times,
        real,
        training,
        "training_wrench",
        3,
        "traj0 torque components: real vs simulated training wrench",
    )
    plot_components(
        args.output_dir / "force_components_incoming_wrench.png",
        times,
        real,
        final,
        "wrench_final",
        0,
        "traj0 diagnostic: real vs incoming-joint wrench_final",
    )
    plot_components(
        args.output_dir / "torque_components_incoming_wrench.png",
        times,
        real,
        final,
        "wrench_final",
        3,
        "traj0 diagnostic: real vs incoming-joint wrench_final",
    )
    # Keep the historical output filenames, but make them show the force that
    # is actually written to obs/state/ee_wrench_base.  This prevents users
    # from accidentally opening the all-zero direct-replay diagnostic again.
    plot_components(
        args.output_dir / "force_components_final.png",
        times,
        real,
        training,
        "training_wrench",
        0,
        "traj0 force components: real vs simulated training wrench",
    )
    plot_components(
        args.output_dir / "torque_components_final.png",
        times,
        real,
        training,
        "training_wrench",
        3,
        "traj0 torque components: real vs simulated training wrench",
    )
    plot_components(
        args.output_dir / "force_components_model.png",
        times,
        real,
        model,
        "wrench_model",
        0,
        "traj0 force components: real vs simulated wrench_model",
    )

    metrics = {
        "real_source": str(args.real_h5.resolve()),
        "sim_source": str(args.sim_dir.resolve()),
        "frame_count": int(count),
        "streams": {name: stream_metrics(real, values) for name, values in streams.items()},
    }
    metrics_path = args.output_dir / "wrench_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[DONE] output_dir: {args.output_dir.resolve()}")
    print(f"[DONE] metrics: {metrics_path.resolve()}")
    for name, values in metrics["streams"].items():
        print(
            f"[METRIC] {name}: "
            f"force_corr={values['force_norm_correlation']:.4f}, "
            f"torque_corr={values['torque_norm_correlation']:.4f}"
        )


if __name__ == "__main__":
    main()
