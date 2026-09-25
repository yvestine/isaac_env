#!/usr/bin/env python3
"""Create a side-by-side wrist-camera and contact-force diagnostic video."""

from __future__ import annotations

import argparse
import csv
import subprocess
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def read_csv_matrix(path: Path, columns: int) -> np.ndarray:
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        next(reader, None)
        rows = [[float(value) for value in row] for row in reader if row]
    values = np.asarray(rows, dtype=np.float32)
    if values.ndim != 2 or values.shape[1] != columns:
        raise ValueError(f"Expected {columns} columns in {path}, got {values.shape}")
    return values


def figure_to_bgr(figure, width: int, height: int) -> np.ndarray:
    figure.canvas.draw()
    rgba = np.asarray(figure.canvas.buffer_rgba())
    rgb = cv2.resize(rgba[..., :3], (width, height), interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--video",
        type=Path,
        default=Path("outputs/sim_force/traj_0/wrist_camera.mp4"),
    )
    parser.add_argument(
        "--force-csv",
        type=Path,
        default=Path("outputs/sim_force/traj_0/contact_total_base.csv"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/sim_force/traj_0/wrist_force_overlay.mp4"),
    )
    parser.add_argument("--fps", type=float, default=None)
    args = parser.parse_args()

    force = read_csv_matrix(args.force_csv, 3)
    force_norm = np.linalg.norm(force, axis=1)
    time_s = np.arange(len(force), dtype=np.float32) / 10.0

    capture = cv2.VideoCapture(str(args.video))
    if not capture.isOpened():
        raise RuntimeError(f"Cannot open wrist video: {args.video}")
    video_fps = float(capture.get(cv2.CAP_PROP_FPS))
    fps = float(args.fps or video_fps or 10.0)
    frame_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if frame_width <= 0 or frame_height <= 0:
        raise RuntimeError("Wrist video has invalid dimensions")

    output_height = frame_height
    output_width = frame_width * 2
    args.output.parent.mkdir(parents=True, exist_ok=True)
    raw_output = args.output.with_name(f"{args.output.stem}.mp4v{args.output.suffix}")
    writer = cv2.VideoWriter(
        str(raw_output),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (output_width, output_height),
    )
    if not writer.isOpened():
        capture.release()
        raise RuntimeError(f"Cannot open temporary output video: {raw_output}")

    figure, axes = plt.subplots(2, 1, figsize=(8.0, 6.0), dpi=120, sharex=True)
    components = ("Fx", "Fy", "Fz")
    colors = ("#d62728", "#2ca02c", "#1f77b4")
    frame_index = 0
    try:
        while frame_index < len(force):
            ok, wrist_bgr = capture.read()
            if not ok:
                break
            wrist_bgr = cv2.resize(
                wrist_bgr, (frame_width, frame_height), interpolation=cv2.INTER_AREA
            )
            current_time = float(time_s[frame_index])
            for axis in axes:
                axis.clear()

            axes[0].plot(time_s[: frame_index + 1], force[: frame_index + 1, 0], color=colors[0], label="Fx")
            axes[0].plot(time_s[: frame_index + 1], force[: frame_index + 1, 1], color=colors[1], label="Fy")
            axes[0].plot(time_s[: frame_index + 1], force[: frame_index + 1, 2], color=colors[2], label="Fz")
            axes[0].set_ylabel("force [N]")
            axes[0].set_title("Peg-hole contact force in robot base frame")
            axes[0].legend(loc="upper right", fontsize=8, ncol=3)

            axes[1].plot(time_s[: frame_index + 1], force_norm[: frame_index + 1], color="black", label="||F||")
            axes[1].axhline(0.0, color="#888888", linewidth=0.6)
            axes[1].set_ylabel("norm [N]")
            axes[1].set_xlabel("trajectory time [s]")
            axes[1].legend(loc="upper right", fontsize=8)

            for axis in axes:
                axis.axvline(current_time, color="#ff7f0e", linewidth=1.5)
                axis.set_xlim(0.0, max(float(time_s[-1]), 1.0))
                axis.grid(alpha=0.25)
            axes[1].text(
                0.02,
                0.92,
                f"frame={frame_index}  t={current_time:.2f}s  |F|={force_norm[frame_index]:.2f} N",
                transform=axes[1].transAxes,
                fontsize=9,
                verticalalignment="top",
                bbox={"facecolor": "white", "alpha": 0.75, "edgecolor": "none"},
            )

            plot_bgr = figure_to_bgr(figure, frame_width, output_height)
            combined = np.hstack((wrist_bgr, plot_bgr))
            writer.write(combined)
            frame_index += 1
    finally:
        figure.clear()
        plt.close(figure)
        writer.release()
        capture.release()

    encoded_output = args.output.with_name(f"{args.output.stem}.h264.tmp{args.output.suffix}")
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-i",
            str(raw_output),
            "-an",
            "-c:v",
            "libx264",
            "-preset",
            "medium",
            "-crf",
            "18",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(encoded_output),
        ],
        check=True,
    )
    encoded_output.replace(args.output)
    raw_output.unlink(missing_ok=True)

    print(f"[DONE] frames={frame_index} fps={fps:.3f}")
    print(f"[DONE] output={args.output.resolve()} codec=H.264")


if __name__ == "__main__":
    main()
