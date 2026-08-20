#!/usr/bin/env python3
"""Compare traj0 real force with PPO, raw incoming, and Peg-Hole contact force."""
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


def read_csv(path: Path) -> np.ndarray:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.reader(handle))
    return np.asarray([[float(value) for value in row] for row in rows[1:]], dtype=np.float64)


def read_real(path: Path) -> np.ndarray:
    with h5py.File(path, "r") as data:
        for key in ("obs/state/ee_wrench_base", "observations/ee_wrench_base", "state/ee_wrench_base"):
            if key in data:
                return np.asarray(data[key], dtype=np.float64).reshape(-1, 6)
    raise KeyError(f"{path}: ee_wrench_base not found")


def correlation(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 2 or np.std(a) < 1.0e-12 or np.std(b) < 1.0e-12:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--real-h5", type=Path, required=True)
    parser.add_argument("--sim-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--metrics-output", type=Path, required=True)
    args = parser.parse_args()

    real = read_real(args.real_h5)
    ppo = read_csv(args.sim_dir / "wrench_ppo.csv")
    raw = read_csv(args.sim_dir / "wrench_raw.csv")
    contact_table = read_csv(args.sim_dir / "peg_hole_contact.csv")
    contact = contact_table[:, 2:5]
    count = min(len(real), len(ppo), len(raw), len(contact))
    if count < 2:
        raise RuntimeError("traj0 does not contain enough aligned frames")
    real, ppo, raw, contact = real[:count], ppo[:count], raw[:count], contact[:count]
    t = np.arange(count)
    real_norm = np.linalg.norm(real[:, :3], axis=1)
    signals = {
        "PPO wrench_model": ppo,
        "raw incoming wrench": raw,
        "Peg-Hole ContactSensor": contact,
    }
    metrics = {}
    for name, signal in signals.items():
        signal_norm = np.linalg.norm(signal[:, :3], axis=1)
        metrics[name] = {
            "force_norm_correlation": correlation(real_norm, signal_norm),
            "real_force_norm_std": float(np.std(real_norm)),
            "signal_force_norm_std": float(np.std(signal_norm)),
            "signal_force_norm_max": float(np.max(signal_norm)),
        }

    fig, axes = plt.subplots(3, 2, figsize=(15, 11), sharex=True)
    for row, (name, signal) in enumerate(signals.items()):
        signal_norm = np.linalg.norm(signal[:, :3], axis=1)
        axes[row, 0].plot(t, real_norm, label="real", linewidth=1.4)
        axes[row, 0].plot(t, signal_norm, label="sim", linewidth=1.1)
        axes[row, 0].set_ylabel("force norm (N)")
        axes[row, 0].set_title(f"{name} | corr={metrics[name]['force_norm_correlation']:.3f}")
        axes[row, 0].grid(alpha=0.25)
        axes[row, 0].legend(loc="upper right")
        for index, label in enumerate(("Fx", "Fy", "Fz")):
            axes[row, 1].plot(t, real[:, index], label=f"real {label}", linewidth=1.1)
            axes[row, 1].plot(t, signal[:, index], "--", label=f"sim {label}", linewidth=0.9)
        axes[row, 1].set_title(f"{name} components")
        axes[row, 1].grid(alpha=0.25)
        axes[row, 1].legend(loc="upper right", ncol=2, fontsize=8)
    axes[-1, 0].set_xlabel("traj0 frame")
    axes[-1, 1].set_xlabel("traj0 frame")
    fig.suptitle("traj0: real force vs three simulator force interfaces")
    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=160)
    plt.close(fig)
    args.metrics_output.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"[DONE] visualization: {args.output}")
    print(f"[DONE] metrics: {args.metrics_output}")
    for name, values in metrics.items():
        print(f"[METRIC] {name}: corr={values['force_norm_correlation']:.4f}")


if __name__ == "__main__":
    main()
