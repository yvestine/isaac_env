"""Plot before/after residuals for replay time alignment."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replay-dir", type=Path, required=True)
    args = parser.parse_args()
    before = np.load(args.replay_dir / "replay_alignment_arrays.npz")
    after_dir = args.replay_dir / "time_aligned"
    after = np.load(after_dir / "replay_alignment_arrays.npz")
    t_before = before["timestamps"] - before["timestamps"][0]
    t_after = after["timestamps"] - after["timestamps"][0]
    output = args.replay_dir / "visualizations" / "07_time_alignment_before_after.png"
    output.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=False)
    axes[0].plot(t_before, before["position_error_mm"], label="before time alignment", color="#d62728")
    axes[0].plot(t_after, after["position_error_mm"], label="after +0.145 s", color="#2ca02c")
    axes[0].set_ylabel("position error [mm]")
    axes[0].set_title("Position residual before / after time alignment")
    axes[0].grid(alpha=0.25)
    axes[0].legend()
    axes[1].plot(t_before, before["orientation_error_deg"], label="before", color="#d62728")
    axes[1].plot(t_after, after["orientation_error_deg"], label="after +0.145 s", color="#2ca02c")
    axes[1].set_xlabel("time [s]")
    axes[1].set_ylabel("orientation error [deg]")
    axes[1].set_title("Orientation residual before / after time alignment")
    axes[1].grid(alpha=0.25)
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] Wrote {output.resolve()}")


if __name__ == "__main__":
    main()
