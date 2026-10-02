#!/usr/bin/env python3
"""Side-by-side REAL vs SIM comparison for TAVLA co-training data.

Per trajectory it renders a video:
  [ REAL front | REAL wrist ]   [ force chart: |F| and Fz, REAL vs SIM overlaid ]
  [ SIM  front | SIM  wrist ]   [ + moving time cursor                          ]
and one aggregate PNG: |F| p95 / max per trajectory, REAL vs SIM (the co-training
magnitude gap that drives the effort norm_stats).

REAL and SIM are frame-aligned (sim is a faithful replay: jointmaxdiff=0, equal frame
counts, both 10 fps), so overlaying their force on a single time axis is exact.
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
REAL_ROOT = ROOT / "real_data"
SIM_ROOT = ROOT / "sim-data"
REAL_COLOR = "#d62728"   # red   = real robot
SIM_COLOR = "#1f77b4"    # blue  = simulation


def read_force(path: Path):
    with h5py.File(path, "r") as f:
        wrench = f["obs/state/ee_wrench_base"][:].astype(np.float64)
        time = f["timestamps"][:].astype(np.float64)
    return wrench, time - time[0]


def label(img, text, color):
    cv2.putText(img, text, (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(img, text, (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)
    return img


def build_chart(time, real_w, sim_w, trajectory, dpi=100):
    rnorm = np.linalg.norm(real_w[:, :3], axis=1)
    snorm = np.linalg.norm(sim_w[:, :3], axis=1)
    fig, axes = plt.subplots(2, 1, figsize=(6.4, 5.4), dpi=dpi, sharex=True)
    axes[0].plot(time, rnorm, color=REAL_COLOR, linewidth=1.6, label="REAL |F|")
    axes[0].plot(time, snorm, color=SIM_COLOR, linewidth=1.6, linestyle="--", label="SIM |F|")
    axes[0].set_ylabel("|F| [N]")
    axes[0].set_title(f"traj_{trajectory}: REAL vs SIM  ee_wrench_base", fontsize=10)
    axes[1].plot(time, real_w[:, 2], color=REAL_COLOR, linewidth=1.6, label="REAL Fz")
    axes[1].plot(time, sim_w[:, 2], color=SIM_COLOR, linewidth=1.6, linestyle="--", label="SIM Fz")
    axes[1].set_ylabel("Fz [N]")
    axes[1].set_xlabel("time [s]")
    for ax in axes:
        ax.grid(alpha=0.25)
        ax.set_xlim(time[0], time[-1])
        ax.legend(loc="upper left", fontsize=8)
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
    return chart_bgr, cols, (max(0, y_top), min(height, y_bot)), rnorm, snorm


def transcode_h264(source: Path, output: Path) -> None:
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    tmp = output.with_name(f".{output.stem}.mp4v.mp4")
    subprocess.run([ffmpeg, "-y", "-loglevel", "error", "-i", str(source), "-an",
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", "-movflags", "+faststart",
                    str(tmp)], check=True)
    tmp.replace(output)


def render_one(trajectory: int, out_dir: Path, fps_fallback: float) -> dict:
    rdir = REAL_ROOT / f"traj_{trajectory}"
    sdir = SIM_ROOT / f"traj_{trajectory}"
    real_w, time = read_force(sdir / "data.h5")  # time axis from sim (== real, frame-aligned)
    real_w, _ = read_force(rdir / "data.h5")
    sim_w, _ = read_force(sdir / "data.h5")
    chart, cols, (y0, y1), rnorm, snorm = build_chart(time, real_w, sim_w, trajectory)
    ch_h, ch_w = chart.shape[:2]
    tile_h = ch_h // 2
    tile_w = int(round(tile_h * 640 / 480))
    caps = {
        "rf": cv2.VideoCapture(str(rdir / "front_camera.mp4")),
        "rw": cv2.VideoCapture(str(rdir / "wrist_camera.mp4")),
        "sf": cv2.VideoCapture(str(sdir / "front_camera.mp4")),
        "sw": cv2.VideoCapture(str(sdir / "wrist_camera.mp4")),
    }
    fps = float(caps["rf"].get(cv2.CAP_PROP_FPS)) or fps_fallback
    if not np.isfinite(fps) or fps <= 0:
        fps = fps_fallback
    count = min(len(time), *[int(c.get(cv2.CAP_PROP_FRAME_COUNT)) for c in caps.values()])
    grid_w = tile_w * 2
    width = grid_w + ch_w
    width += width % 2
    height = ch_h + ch_h % 2
    tmp = out_dir / f".traj_{trajectory:02d}_raw.mp4"
    out = out_dir / f"traj_{trajectory:02d}_real_vs_sim.mp4"
    writer = cv2.VideoWriter(str(tmp), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    for i in range(count):
        frames, ok = {}, True
        for key, cap in caps.items():
            good, frame = cap.read()
            ok = ok and good
            frames[key] = frame
        if not ok:
            break
        top = np.concatenate((
            label(cv2.resize(frames["rf"], (tile_w, tile_h)), "REAL front", (0, 255, 0)),
            label(cv2.resize(frames["rw"], (tile_w, tile_h)), "REAL wrist", (0, 255, 0))), axis=1)
        bot = np.concatenate((
            label(cv2.resize(frames["sf"], (tile_w, tile_h)), "SIM front", (0, 200, 255)),
            label(cv2.resize(frames["sw"], (tile_w, tile_h)), "SIM wrist", (0, 200, 255))), axis=1)
        grid = np.concatenate((top, bot), axis=0)
        cv2.putText(grid, f"t={time[i]:.2f}s  |F|_real={rnorm[i]:.2f}N  |F|_sim={snorm[i]:.2f}N",
                    (10, grid.shape[0] - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)
        panel = chart.copy()
        panel[y0:y1, max(0, cols[i] - 1):cols[i] + 1] = (14, 127, 255)
        canvas = np.zeros((height, width, 3), dtype=np.uint8)
        canvas[:, grid_w:grid_w + ch_w] = 255
        canvas[:grid.shape[0], :grid.shape[1]] = grid
        canvas[:panel.shape[0], grid_w:grid_w + ch_w] = panel
        writer.write(canvas)
    writer.release()
    for cap in caps.values():
        cap.release()
    transcode_h264(tmp, out)
    tmp.unlink(missing_ok=True)
    return {
        "trajectory": trajectory, "frames": count, "fps": fps,
        "real_Fp95": float(np.quantile(rnorm, 0.95)), "real_Fmax": float(rnorm.max()),
        "sim_Fp95": float(np.quantile(snorm, 0.95)), "sim_Fmax": float(snorm.max()),
        "output": str(out.resolve()),
    }


def build_aggregate(out_dir: Path) -> Path:
    trajs, r_p95, r_max, s_p95, s_max = [], [], [], [], []
    for t in range(40):
        try:
            rw, _ = read_force(REAL_ROOT / f"traj_{t}/data.h5")
            sw, _ = read_force(SIM_ROOT / f"traj_{t}/data.h5")
        except (OSError, KeyError):
            continue
        rn = np.linalg.norm(rw[:, :3], axis=1)
        sn = np.linalg.norm(sw[:, :3], axis=1)
        trajs.append(t)
        r_p95.append(np.quantile(rn, 0.95)); r_max.append(rn.max())
        s_p95.append(np.quantile(sn, 0.95)); s_max.append(sn.max())
    x = np.arange(len(trajs))
    fig, axes = plt.subplots(2, 1, figsize=(13, 6.5), dpi=110, sharex=True)
    for ax, rv, sv, title in ((axes[0], r_p95, s_p95, "|F| p95 per trajectory  (co-training norm_stats driver)"),
                              (axes[1], r_max, s_max, "|F| max per trajectory  (seating-contact peak)")):
        ax.bar(x - 0.2, rv, width=0.4, color=REAL_COLOR, label="REAL")
        ax.bar(x + 0.2, sv, width=0.4, color=SIM_COLOR, label="SIM")
        ax.set_ylabel("N"); ax.set_title(title, fontsize=11); ax.grid(alpha=0.25, axis="y")
        ax.legend(loc="upper right")
    axes[1].set_xticks(x); axes[1].set_xticklabels([str(t) for t in trajs], fontsize=7)
    axes[1].set_xlabel("trajectory")
    fig.suptitle("REAL vs SIM  ee_wrench_base force magnitude  (40 trajectories, frame-aligned)", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    out = out_dir / "real_vs_sim_force_magnitude.png"
    fig.savefig(out); plt.close(fig)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs/real_sim_compare")
    parser.add_argument("--real-root", type=Path, default=ROOT / "real-data-aligned",
                        help="real dataset root (default: aligned)")
    parser.add_argument("--sim-root", type=Path, default=ROOT / "sim-data-aligned",
                        help="sim dataset root (default: aligned)")
    parser.add_argument("--fps-fallback", type=float, default=10.0)
    parser.add_argument("--trajs", type=str, default="0,5,12,27,38",
                        help="comma list, or 'all' for 0..39")
    parser.add_argument("--no-video", action="store_true", help="only the aggregate PNG")
    args = parser.parse_args()
    global REAL_ROOT, SIM_ROOT
    REAL_ROOT, SIM_ROOT = args.real_root, args.sim_root
    args.output_dir.mkdir(parents=True, exist_ok=True)
    agg = build_aggregate(args.output_dir)
    print(f"[AGG] {agg}", flush=True)
    if args.no_video:
        return
    trajs = range(40) if args.trajs == "all" else [int(s) for s in args.trajs.split(",") if s.strip()]
    manifest = []
    for t in trajs:
        info = render_one(t, args.output_dir, args.fps_fallback)
        manifest.append(info)
        print(f"[{t:02d}] frames={info['frames']:4d} "
              f"|F|max real={info['real_Fmax']:5.2f}N sim={info['sim_Fmax']:5.2f}N "
              f"-> {Path(info['output']).name}", flush=True)
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"[DONE] {len(manifest)} comparison videos + aggregate -> {args.output_dir}")


if __name__ == "__main__":
    main()
