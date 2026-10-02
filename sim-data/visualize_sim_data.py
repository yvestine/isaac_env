#!/usr/bin/env python3
"""Composite front + wrist + force-trend video for sim-data/traj_0..traj_39.

Layout per frame:  [ front_camera | wrist_camera | Fx/Fy/Fz/|F| plots + moving cursor ]
Force is read from data.h5 obs/state/ee_wrench_base (the contact-isolated 6-axis wrench).
Output: H.264 mp4 per trajectory + manifest.json.
"""
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import cv2
import h5py
import imageio_ffmpeg
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def read_h5(path: Path):
    with h5py.File(path, "r") as f:
        wrench = f["obs/state/ee_wrench_base"][:].astype(np.float64)
        time = f["timestamps"][:].astype(np.float64)
    return wrench, time - time[0]


def build_chart(time, force, trajectory, dpi=100):
    norm = np.linalg.norm(force[:, :3], axis=1)
    onset = None
    if norm.max() > 0.15:
        thr = max(0.1, 0.25 * norm.max())
        hits = np.flatnonzero(norm > thr)
        onset = int(hits[0]) if len(hits) else None
    values = [force[:, 0], force[:, 1], force[:, 2], norm]
    labels = ["Fx [N]", "Fy [N]", "Fz [N]", "|F| [N]"]
    colors = ["#1f77b4", "#d62728", "#2ca02c", "#111111"]
    fig, axes = plt.subplots(4, 1, figsize=(6.6, 5.2), dpi=dpi, sharex=True)
    for ax, val, lab, col in zip(axes, values, labels, colors):
        ax.plot(time, val, color=col, linewidth=1.3, label=lab)
        if onset is not None:
            ax.axvline(time[onset], color="#9467bd", linestyle="--", linewidth=0.9,
                       label="contact" if ax is axes[0] else None)
        ax.set_ylabel(lab); ax.grid(alpha=0.25); ax.set_xlim(time[0], time[-1])
        if ax is axes[0]:
            ax.legend(loc="upper right", fontsize=7)
    axes[-1].set_xlabel("time [s]")
    axes[0].set_title(f"sim traj_{trajectory}: ee_wrench_base (contact-isolated)", fontsize=10)
    fig.tight_layout()
    fig.canvas.draw()
    rgb = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
    height = rgb.shape[0]
    cols = axes[-1].transData.transform(np.c_[time, np.zeros_like(time)])[:, 0]
    cols = np.clip(np.round(cols).astype(int), 0, rgb.shape[1] - 1)
    y_top = int(height - axes[0].get_window_extent().y1)
    y_bot = int(height - axes[-1].get_window_extent().y0)
    plt.close(fig)
    chart_bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    return chart_bgr, cols, (max(0, y_top), min(height, y_bot)), norm, onset


def transcode_h264(source: Path, output: Path) -> None:
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    tmp = output.with_name(f".{output.stem}.mp4v.mp4")
    subprocess.run([ffmpeg, "-y", "-loglevel", "error", "-i", str(source), "-an",
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", "-movflags", "+faststart",
                    str(tmp)], check=True)
    tmp.replace(output)


def render_one(trajectory: int, traj_dir: Path, out_dir: Path, fps_fallback: float) -> dict:
    front_path = traj_dir / "front_camera.mp4"
    wrist_path = traj_dir / "wrist_camera.mp4"
    wrench, time = read_h5(traj_dir / "data.h5")
    chart, cols, (y0, y1), norm, onset = build_chart(time, wrench, trajectory)
    ch_h, ch_w = chart.shape[:2]

    front = cv2.VideoCapture(str(front_path))
    wrist = cv2.VideoCapture(str(wrist_path))
    fps = float(front.get(cv2.CAP_PROP_FPS)) or fps_fallback
    if not np.isfinite(fps) or fps <= 0:
        fps = fps_fallback
    count = min(len(time), int(front.get(cv2.CAP_PROP_FRAME_COUNT)),
                int(wrist.get(cv2.CAP_PROP_FRAME_COUNT)))

    ok_f, f0 = front.read()
    ok_w, w0 = wrist.read()
    if not (ok_f and ok_w):
        front.release(); wrist.release()
        raise RuntimeError(f"traj_{trajectory}: cannot read first video frame")
    fw = int(round(f0.shape[1] * ch_h / f0.shape[0]))
    ww = int(round(w0.shape[1] * ch_h / w0.shape[0]))
    tmp = out_dir / f".traj_{trajectory:02d}_raw.mp4"
    out = out_dir / f"traj_{trajectory:02d}_front_wrist_force.mp4"
    writer = cv2.VideoWriter(str(tmp), cv2.VideoWriter_fourcc(*"mp4v"), fps,
                             (fw + ww + ch_w, ch_h))
    frame_f, frame_w = f0, w0
    for i in range(count):
        if i > 0:
            ok_f, frame_f = front.read(); ok_w, frame_w = wrist.read()
            if not (ok_f and ok_w):
                break
        left = cv2.resize(frame_f, (fw, ch_h))
        mid = cv2.resize(frame_w, (ww, ch_h))
        panel = chart.copy()
        panel[y0:y1, max(0, cols[i] - 1):cols[i] + 1] = (14, 127, 255)
        cv2.putText(left, f"FRONT", (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(mid, f"WRIST", (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(left, f"traj_{trajectory} f={i} t={time[i]:.2f}s |F|={norm[i]:.2f}N",
                    (10, ch_h - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2, cv2.LINE_AA)
        writer.write(np.concatenate((left, mid, panel), axis=1))
    writer.release(); front.release(); wrist.release()
    transcode_h264(tmp, out)
    tmp.unlink(missing_ok=True)
    return {
        "trajectory": trajectory, "frames": count, "fps": fps,
        "force_norm_p95_n": float(np.quantile(norm, 0.95)),
        "force_norm_max_n": float(norm.max()),
        "contact_onset_frame": onset,
        "output": str(out.resolve()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sim-root", type=Path, default=ROOT / "sim-data")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs/sim_data_viz")
    parser.add_argument("--fps-fallback", type=float, default=10.0)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=39)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = []
    for trajectory in range(args.start, args.end + 1):
        traj_dir = args.sim_root / f"traj_{trajectory}"
        info = render_one(trajectory, traj_dir, args.output_dir, args.fps_fallback)
        manifest.append(info)
        print(f"[{trajectory:02d}/{args.end}] frames={info['frames']:4d} "
              f"|F|max={info['force_norm_max_n']:.2f}N -> {Path(info['output']).name}", flush=True)
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"[DONE] {len(manifest)} videos -> {args.output_dir}")


if __name__ == "__main__":
    main()
