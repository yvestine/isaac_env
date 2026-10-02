#!/usr/bin/env python3
"""Render wrist video beside the synchronized sim-data-aligned force curves."""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

import cv2
import h5py
import imageio_ffmpeg
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


FORCE_KEY = "obs/state/ee_wrench_base"


def load_force(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with h5py.File(path, "r") as data:
        force = np.asarray(data[FORCE_KEY], dtype=np.float64)[:, :3]
        time = np.asarray(data["timestamps"], dtype=np.float64).reshape(-1)
    if force.ndim != 2 or force.shape[1] != 3 or len(time) != len(force):
        raise ValueError(f"Invalid force/timestamp shapes: {force.shape}, {time.shape}")
    if not np.isfinite(force).all() or not np.isfinite(time).all():
        raise ValueError("Force/timestamps contain NaN or Inf")
    return time - time[0], force


def transcode_h264(source: Path, output: Path) -> None:
    command = [
        imageio_ffmpeg.get_ffmpeg_exe(),
        "-y",
        "-loglevel",
        "error",
        "-i",
        str(source),
        "-an",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(output),
    ]
    subprocess.run(command, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trajectory", type=int, default=0)
    parser.add_argument("--sim-data-dir", type=Path, default=Path("sim-data-aligned"))
    parser.add_argument("--video-dir", type=Path, default=Path("sim-data"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/sim_data_aligned_force_video/traj_0_wrist_force.mp4"),
    )
    args = parser.parse_args()

    force_path = args.sim_data_dir / f"traj_{args.trajectory}" / "data.h5"
    wrist_path = args.video_dir / f"traj_{args.trajectory}" / "wrist_camera.mp4"
    if not force_path.is_file():
        raise FileNotFoundError(force_path)
    if not wrist_path.is_file():
        raise FileNotFoundError(wrist_path)

    time, force = load_force(force_path)
    force_norm = np.linalg.norm(force, axis=1)
    capture = cv2.VideoCapture(str(wrist_path))
    if not capture.isOpened():
        raise RuntimeError(f"Cannot open {wrist_path}")
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    video_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    if video_frames != len(force):
        capture.release()
        raise ValueError(
            f"Frame mismatch: wrist={video_frames}, force={len(force)}; refusing unsynchronized video"
        )
    if not np.isfinite(fps) or fps <= 0.0:
        capture.release()
        raise ValueError(f"Invalid video FPS: {fps}")

    figure, axes = plt.subplots(4, 1, figsize=(7.6, 4.8), dpi=100, sharex=True)
    labels = ("Fx [N]", "Fy [N]", "Fz [N]", "|F| [N]")
    colors = ("#2563eb", "#dc2626", "#16a34a", "#111827")
    values = (force[:, 0], force[:, 1], force[:, 2], force_norm)
    cursors = []
    markers = []
    for axis, label, color, value in zip(axes, labels, colors, values):
        axis.plot(time, value, color=color, linewidth=1.25)
        cursor = axis.axvline(time[0], color="#f59e0b", linewidth=1.2)
        marker, = axis.plot([time[0]], [value[0]], "o", color="#f59e0b", markersize=4)
        cursors.append(cursor)
        markers.append(marker)
        span = float(np.ptp(value))
        padding = max(0.05 * span, 0.02)
        axis.set_ylim(float(np.min(value) - padding), float(np.max(value) + padding))
        axis.set_ylabel(label, fontsize=8)
        axis.grid(alpha=0.25)
        axis.tick_params(labelsize=7)
    axes[-1].set_xlabel("time [s]", fontsize=8)
    axes[0].set_title(
        f"traj_{args.trajectory} simulated force (no extra scaling or bias removal)",
        fontsize=10,
    )
    axes[0].set_xlim(float(time[0]), float(time[-1]))
    figure.tight_layout()
    figure.canvas.draw()
    chart = np.asarray(figure.canvas.buffer_rgba())[..., :3]
    chart_height, chart_width = chart.shape[:2]

    ok, first_frame = capture.read()
    if not ok:
        capture.release()
        plt.close(figure)
        raise RuntimeError(f"Cannot read first frame from {wrist_path}")
    wrist_width = int(round(first_frame.shape[1] * chart_height / first_frame.shape[0]))
    output_width = wrist_width + chart_width
    if output_width % 2:
        output_width += 1

    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(f".{args.output.stem}.mp4v.mp4")
    writer = cv2.VideoWriter(
        str(temporary),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (output_width, chart_height),
    )
    if not writer.isOpened():
        capture.release()
        plt.close(figure)
        raise RuntimeError(f"Cannot open video writer: {temporary}")

    frame = first_frame
    written = 0
    for index in range(len(force)):
        if index > 0:
            ok, frame = capture.read()
            if not ok:
                break
        wrist = cv2.resize(frame, (wrist_width, chart_height), interpolation=cv2.INTER_AREA)
        cv2.rectangle(wrist, (0, 0), (wrist_width, 58), (0, 0, 0), thickness=-1)
        cv2.putText(
            wrist,
            f"traj_{args.trajectory}  frame {index}/{len(force) - 1}  t={time[index]:.2f}s",
            (12, 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            wrist,
            (
                f"Fx={force[index, 0]:+.3f}  Fy={force[index, 1]:+.3f}  "
                f"Fz={force[index, 2]:+.3f}  |F|={force_norm[index]:.3f} N"
            ),
            (12, 48),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        for cursor, marker, value in zip(cursors, markers, values):
            cursor.set_xdata([time[index], time[index]])
            marker.set_data([time[index]], [value[index]])
        figure.canvas.draw()
        chart_rgb = np.asarray(figure.canvas.buffer_rgba())[..., :3]
        chart_bgr = cv2.cvtColor(chart_rgb, cv2.COLOR_RGB2BGR)
        combined = np.zeros((chart_height, output_width, 3), dtype=np.uint8)
        combined[:, :wrist_width] = wrist
        combined[:, wrist_width : wrist_width + chart_width] = chart_bgr
        writer.write(combined)
        written += 1

    writer.release()
    capture.release()
    plt.close(figure)
    if written != len(force):
        temporary.unlink(missing_ok=True)
        raise RuntimeError(f"Only wrote {written}/{len(force)} synchronized frames")
    transcode_h264(temporary, args.output)
    temporary.unlink(missing_ok=True)
    print(
        f"[DONE] {args.output.resolve()} frames={written} fps={fps:.3f} "
        f"duration={written / fps:.3f}s force={force_path} wrist={wrist_path}"
    )


if __name__ == "__main__":
    main()
