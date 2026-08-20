#!/usr/bin/env python3
"""Plot all saved simulation wrench layers for one episode."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

CHANNELS = ("Fx", "Fy", "Fz", "Tx", "Ty", "Tz")
LAYERS = (
    ("wrench_raw.csv", "raw"),
    ("wrench_anchor.csv", "anchor"),
    ("wrench_base.csv", "base"),
    ("wrench_corrected.csv", "corrected"),
)


def load_csv(path: Path) -> np.ndarray:
    with path.open("r", newline="") as file:
        reader = csv.reader(file)
        next(reader, None)
        rows = [[float(value) for value in row] for row in reader if row]
    array = np.asarray(rows, dtype=np.float32)
    if array.ndim != 2 or array.shape[1] < 6:
        raise ValueError(f"Expected six wrench columns in {path}, got {array.shape}")
    return array[:, :6]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("episode_dir", type=Path)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--fps", type=float, default=10.0)
    args = parser.parse_args()
    episode_dir = args.episode_dir.resolve()
    series = {}
    for filename, label in LAYERS:
        path = episode_dir / filename
        if path.is_file():
            series[label] = load_csv(path)
    if not series:
        raise FileNotFoundError(f"No wrench layer CSV found in {episode_dir}")
    count = min(len(values) for values in series.values())
    if count <= 0:
        raise ValueError("Wrench layer files contain no rows")
    times = np.arange(count, dtype=np.float32) / max(args.fps, 1e-6)
    figure, axes = plt.subplots(2, 3, figsize=(15, 7), sharex=True)
    axes = axes.reshape(-1)
    for channel, axis in enumerate(axes):
        for label, values in series.items():
            axis.plot(times, values[:count, channel], linewidth=1.0, label=label)
        axis.set_title(CHANNELS[channel])
        axis.grid(True, alpha=0.25)
        axis.set_xlabel("time [s]")
        axis.set_ylabel("N" if channel < 3 else "N*m")
    axes[0].legend(loc="best")
    figure.suptitle("Simulation wrench layer comparison")
    figure.tight_layout()
    output = args.output.resolve() if args.output is not None else episode_dir / "wrench_layers.png"
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=160)
    plt.close(figure)
    print(output)


if __name__ == "__main__":
    main()
