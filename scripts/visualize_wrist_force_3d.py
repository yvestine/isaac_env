#!/usr/bin/env python3
"""Render a wrist-camera video beside synchronized force-vs-time curves."""

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


def read_matrix(path: Path) -> np.ndarray:
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        next(reader, None)
        rows = [[float(value) for value in row] for row in reader if row]
    values = np.asarray(rows, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] < 3:
        raise ValueError(f"Expected at least 3 columns in {path}, got {values.shape}")
    if not np.isfinite(values).all():
        raise ValueError(f"Non-finite values in {path}")
    return values


def read_time(path: Path, count: int) -> np.ndarray:
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        next(reader, None)
        values = np.asarray([float(row[0]) for row in reader if row], dtype=np.float64)
    if values.ndim != 1 or not np.isfinite(values).all():
        raise ValueError(f"Invalid timestamps in {path}")
    if len(values) < count:
        raise ValueError(f"Not enough timestamps: {len(values)} < {count}")
    values = values[:count]
    return values - values[0]


def figure_to_bgr(figure: plt.Figure, width: int, height: int) -> np.ndarray:
    figure.canvas.draw()
    rgb = np.asarray(figure.canvas.buffer_rgba())[..., :3]
    rgb = cv2.resize(rgb, (width, height), interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--force-csv", type=Path, required=True)
    parser.add_argument("--timestamps", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--snapshot", type=Path, required=True)
    args = parser.parse_args()

    force_all = read_matrix(args.force_csv)[:, :3]
    capture = cv2.VideoCapture(str(args.video))
    if not capture.isOpened():
        raise RuntimeError(f"Cannot open wrist video: {args.video}")
    video_fps = float(capture.get(cv2.CAP_PROP_FPS))
    video_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    video_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    video_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    if video_fps <= 0 or video_width <= 0 or video_height <= 0:
        raise RuntimeError("Invalid wrist video metadata")

    count = min(len(force_all), video_frames)
    if count <= 0:
        raise RuntimeError("No synchronized frames available")
    force = force_all[:count]
    time_s = read_time(args.timestamps, count)
    force_norm = np.linalg.norm(force, axis=1)
    series = (force[:, 0], force[:, 1], force[:, 2], force_norm)
    labels = ("Fx [N]", "Fy [N]", "Fz [N]", "|F| [N]")
    colors = ("#dc2626", "#16a34a", "#2563eb", "#111827")

    output_height = video_height
    output_width = video_width * 2
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.snapshot.parent.mkdir(parents=True, exist_ok=True)
    raw_output = args.output.with_name(f".{args.output.stem}.mp4v.mp4")
    writer = cv2.VideoWriter(
        str(raw_output),
        cv2.VideoWriter_fourcc(*"mp4v"),
        video_fps,
        (output_width, output_height),
    )
    if not writer.isOpened():
        capture.release()
        raise RuntimeError(f"Cannot open output writer: {raw_output}")

    figure, axes = plt.subplots(
        4, 1, figsize=(6.4, 4.8), dpi=100, sharex=True,
        gridspec_kw={"height_ratios": [1, 1, 1, 1.1]},
    )
    cursors = []
    markers = []
    for axis, label, color, values in zip(axes, labels, colors, series):
        axis.plot(time_s, values, color=color, linewidth=1.0)
        cursor = axis.axvline(time_s[0], color="#f59e0b", linewidth=1.2)
        marker, = axis.plot([time_s[0]], [values[0]], "o", color="#f59e0b", markersize=4)
        cursors.append(cursor)
        markers.append(marker)
        span = max(float(np.ptp(values)), 1.0e-3)
        padding = max(0.08 * span, 0.02)
        axis.set_ylim(float(np.min(values) - padding), float(np.max(values) + padding))
        axis.set_ylabel(label, fontsize=8)
        axis.grid(alpha=0.25)
        axis.tick_params(labelsize=7)
    axes[0].set_title("Wrist force components vs time (robot base frame)", fontsize=10)
    axes[-1].set_xlabel("time [s]", fontsize=8)
    axes[-1].set_xlim(float(time_s[0]), float(time_s[-1]))
    status = figure.text(
        0.02, 0.01, "", fontsize=7,
        bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "none"},
    )
    figure.subplots_adjust(left=0.12, right=0.98, top=0.91, bottom=0.10, hspace=0.16)

    written = 0
    try:
        for index in range(count):
            ok, wrist_bgr = capture.read()
            if not ok:
                break
            current = force[index]
            for cursor, marker, values in zip(cursors, markers, series):
                cursor.set_xdata([time_s[index], time_s[index]])
                marker.set_data([time_s[index]], [values[index]])
            status.set_text(
                f"t={time_s[index]:.2f}s    Fx={current[0]:+.3f} N    "
                f"Fy={current[1]:+.3f} N    Fz={current[2]:+.3f} N    "
                f"|F|={force_norm[index]:.3f} N"
            )

            wrist_bgr = cv2.resize(wrist_bgr, (video_width, video_height), interpolation=cv2.INTER_AREA)
            cv2.rectangle(wrist_bgr, (0, 0), (video_width, 54), (0, 0, 0), thickness=-1)
            cv2.putText(
                wrist_bgr,
                f"profile 0  wrist view  frame {index + 1}/{count}  t={time_s[index]:.2f}s",
                (12, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 1, cv2.LINE_AA,
            )
            cv2.putText(
                wrist_bgr,
                f"Fx={current[0]:+.3f}  Fy={current[1]:+.3f}  Fz={current[2]:+.3f} N  |F|={force_norm[index]:.3f} N",
                (12, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.43, (255, 255, 255), 1, cv2.LINE_AA,
            )
            combined = np.hstack((wrist_bgr, figure_to_bgr(figure, video_width, video_height)))
            writer.write(combined)
            written += 1
            if index == count - 1:
                cv2.imwrite(str(args.snapshot), combined)
    finally:
        writer.release()
        capture.release()
        plt.close(figure)

    if written != count:
        raw_output.unlink(missing_ok=True)
        raise RuntimeError(f"Only wrote {written}/{count} synchronized frames")

    subprocess.run(
        [
            "ffmpeg", "-y", "-loglevel", "error", "-i", str(raw_output),
            "-an", "-c:v", "libx264", "-preset", "medium", "-crf", "18",
            "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(args.output),
        ],
        check=True,
    )
    raw_output.unlink(missing_ok=True)
    print(f"[DONE] frames={written} fps={video_fps:.3f} duration={written / video_fps:.3f}s")
    print(f"[DONE] video={args.output.resolve()}")
    print(f"[DONE] snapshot={args.snapshot.resolve()}")


if __name__ == "__main__":
    main()
