#!/usr/bin/env python3
"""Create one H.264 wrist-video/force-curve visualization for traj_0..traj_39."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
from pathlib import Path

import cv2
import imageio_ffmpeg
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


CHANNELS = ("Fx", "Fy", "Fz")
FORCE_FILE = "panda_link7_wrench_contact_isolated_base_at_K.csv"


def read_csv(path: Path, columns: int) -> np.ndarray:
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        next(reader, None)
        rows = [[float(value) for value in row] for row in reader if row]
    values = np.asarray(rows, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != columns or not np.isfinite(values).all():
        raise ValueError(f"{path}: expected finite Nx{columns}, got {values.shape}")
    return values


def success_frame(path: Path) -> int | None:
    if not path.is_file():
        return None
    with path.open("r", newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if str(row.get("success", "")).strip().lower() in {"true", "1", "yes"}:
                return int(row["frame"])
    return None


def metadata_success(path: Path) -> int | None:
    if not path.is_file():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    value = data.get("first_success_frame")
    return None if value is None else int(value)


def resolve_dirs(
    trajectory: int,
    pair_root: Path,
    video_root: Path,
    traj0_dir: Path,
) -> tuple[Path, Path]:
    if trajectory == 0:
        return traj0_dir.resolve(), traj0_dir.resolve()
    return (pair_root / f"traj_{trajectory}").resolve(), (video_root / f"traj_{trajectory}").resolve()


def draw_plot(
    figure: plt.Figure,
    axes: np.ndarray,
    time: np.ndarray,
    force: np.ndarray,
    contact: np.ndarray,
    success: int | None,
    trajectory: int,
) -> np.ndarray:
    norm = np.linalg.norm(force[:, :3], axis=1)
    values = [force[:, 0], force[:, 1], force[:, 2], norm]
    labels = ["Fx [N]", "Fy [N]", "Fz [N]", "|F| [N]"]
    colors = ["#1f77b4", "#d62728", "#2ca02c", "#111111"]
    for axis, value, label, color in zip(axes, values, labels, colors):
        axis.clear()
        axis.plot(time, value, color=color, linewidth=1.35, label=label)
        if np.any(contact):
            contact_index = int(np.flatnonzero(contact)[0])
            axis.axvline(
                time[contact_index],
                color="#9467bd",
                linestyle="--",
                linewidth=0.9,
                label="contact start" if axis is axes[0] else None,
            )
        if success is not None and success < len(time):
            axis.axvline(
                time[success],
                color="#ff7f0e",
                linestyle=":",
                linewidth=0.9,
                label="success" if axis is axes[0] else None,
            )
        axis.set_ylabel(label)
        axis.grid(alpha=0.25)
        axis.set_xlim(time[0], time[-1])
        if axis is axes[0]:
            axis.legend(loc="upper right", fontsize=7)
    axes[-1].set_xlabel("time [s]")
    axes[0].set_title(
        f"traj_{trajectory}: panda_link7 contact-isolated wrench | "
        "purple=contact, orange=success"
    )
    figure.tight_layout()
    figure.canvas.draw()
    return np.asarray(figure.canvas.buffer_rgba())[..., :3].copy()


def transcode_h264(source: Path, output: Path) -> None:
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    temporary = output.with_name(f".{output.stem}.mp4v.mp4")
    command = [
        ffmpeg,
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
        str(temporary),
    ]
    subprocess.run(command, check=True)
    temporary.replace(output)


def render_one(
    trajectory: int,
    force_dir: Path,
    video_dir: Path,
    output_dir: Path,
    fps_fallback: float,
) -> dict[str, object]:
    wrist_path = video_dir / "wrist_camera.mp4"
    force_path = force_dir / FORCE_FILE
    timestamps_path = force_dir / "timestamps.csv"
    contact_path = force_dir / "contact_count.csv"
    if not wrist_path.is_file():
        raise FileNotFoundError(f"missing wrist video: {wrist_path}")
    if not force_path.is_file() or not timestamps_path.is_file():
        raise FileNotFoundError(f"missing force/timestamp stream in {force_dir}")

    force = read_csv(force_path, 6)
    time = read_csv(timestamps_path, 1)[:, 0]
    time -= time[0]
    contact = (
        read_csv(contact_path, 1)[:, 0] > 0.0
        if contact_path.is_file()
        else np.zeros(len(force), dtype=bool)
    )
    capture = cv2.VideoCapture(str(wrist_path))
    if not capture.isOpened():
        raise RuntimeError(f"cannot open wrist video: {wrist_path}")
    input_fps = float(capture.get(cv2.CAP_PROP_FPS))
    if not np.isfinite(input_fps) or input_fps <= 0.0:
        input_fps = fps_fallback
    video_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    count = min(len(force), len(time), len(contact), video_frames)
    if count < 2:
        capture.release()
        raise RuntimeError(f"{trajectory}: not enough aligned frames")
    force = force[:count]
    time = time[:count]
    contact = contact[:count]
    success = success_frame(force_dir / "gt_replay_validation.csv")
    if success is None:
        success = metadata_success(force_dir / "replay_metadata.json")
    if success is not None:
        success = min(success, count - 1)

    figure, axes = plt.subplots(4, 1, figsize=(7.4, 5.0), dpi=100, sharex=True)
    chart = draw_plot(figure, axes, time, force, contact, success, trajectory)
    chart_height, chart_width = chart.shape[:2]
    ok, frame = capture.read()
    if not ok:
        capture.release()
        plt.close(figure)
        raise RuntimeError(f"cannot read first frame: {wrist_path}")
    wrist_width = int(round(frame.shape[1] * chart_height / frame.shape[0]))
    temporary = output_dir / f".traj_{trajectory:02d}_wrist_force_raw.mp4"
    output = output_dir / f"traj_{trajectory:02d}_wrist_force.mp4"
    writer = cv2.VideoWriter(
        str(temporary),
        cv2.VideoWriter_fourcc(*"mp4v"),
        input_fps,
        (wrist_width + chart_width, chart_height),
    )
    if not writer.isOpened():
        capture.release()
        plt.close(figure)
        raise RuntimeError(f"cannot open video writer: {temporary}")

    norm = np.linalg.norm(force[:, :3], axis=1)
    contact_index = int(np.flatnonzero(contact)[0]) if np.any(contact) else None
    for index in range(count):
        if index > 0:
            ok, frame = capture.read()
            if not ok:
                break
        wrist = cv2.resize(frame, (wrist_width, chart_height))
        cursor_time = time[index]
        cursor_lines = [
            axis.axvline(cursor_time, color="#ff7f0e", linewidth=0.8)
            for axis in axes
        ]
        figure.canvas.draw()
        chart = np.asarray(figure.canvas.buffer_rgba())[..., :3]
        chart_bgr = cv2.cvtColor(chart, cv2.COLOR_RGB2BGR)
        cv2.putText(
            wrist,
            f"traj_{trajectory}  frame={index}  t={cursor_time:.2f}s  |F|={norm[index]:.3f}N",
            (12, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.58,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        writer.write(np.concatenate((wrist, chart_bgr), axis=1))
        for line in cursor_lines:
            line.remove()
    writer.release()
    capture.release()
    plt.close(figure)
    transcode_h264(temporary, output)
    temporary.unlink(missing_ok=True)
    return {
        "trajectory": trajectory,
        "frames": count,
        "fps": input_fps,
        "force_source": str(force_path),
        "wrist_source": str(wrist_path),
        "output": str(output.resolve()),
        "first_contact_frame": contact_index,
        "first_contact_time_s": None if contact_index is None else float(time[contact_index]),
        "first_success_frame": success,
        "first_success_time_s": None if success is None else float(time[success]),
        "force_norm_p95_n": float(np.quantile(norm, 0.95)),
        "force_norm_max_n": float(np.max(norm)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pair-root", type=Path, default=Path("outputs/link7_success_force_pairs"))
    parser.add_argument("--video-root", type=Path, default=Path("outputs/sim_force"))
    parser.add_argument(
        "--traj0-dir",
        type=Path,
        default=Path("outputs/traj0_verified_visual_link7_force_smooth/traj_0"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/all_success_wrist_force_videos"),
    )
    parser.add_argument("--fps-fallback", type=float, default=10.0)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=39)
    args = parser.parse_args()
    if not 0 <= args.start <= args.end <= 39:
        raise ValueError("--start/--end must select traj_0..traj_39")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = []
    for trajectory in range(args.start, args.end + 1):
        force_dir, video_dir = resolve_dirs(
            trajectory, args.pair_root, args.video_root, args.traj0_dir
        )
        print(f"[VIDEO {trajectory}/39] force={force_dir} wrist={video_dir}", flush=True)
        manifest.append(
            render_one(
                trajectory,
                force_dir,
                video_dir,
                args.output_dir.resolve(),
                args.fps_fallback,
            )
        )
    manifest_path = args.output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"[DONE] manifest: {manifest_path.resolve()}")


if __name__ == "__main__":
    main()
