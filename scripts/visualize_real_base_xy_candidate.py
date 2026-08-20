"""Visualize a real trajectory in candidate simulator base frames.

This is a static diagnostic only.  It does not launch Isaac Sim, call an
environment reset, run PPO, or contact a TAVLA server.  The real EE position
is treated as a point in the real Franka base frame and is transformed with
candidate +/-90 degree yaw rotations before being compared with the known
simulator hole position.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def read_first_frame(path: Path) -> np.ndarray:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {path}")
    ok, frame = capture.read()
    capture.release()
    if not ok:
        raise RuntimeError(f"Could not read first frame: {path}")
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def transform_xy(points: np.ndarray, angle_deg: float) -> np.ndarray:
    angle = np.deg2rad(angle_deg)
    rotation = np.array(
        [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]],
        dtype=np.float64,
    )
    return points @ rotation.T


def draw_base_axes(axis: plt.Axes, scale: float = 0.12) -> None:
    axis.arrow(0.0, 0.0, scale, 0.0, color="tab:red", width=0.002, head_width=0.018)
    axis.arrow(0.0, 0.0, 0.0, scale, color="tab:blue", width=0.002, head_width=0.018)
    axis.text(scale * 1.05, 0.0, "+X base", color="tab:red", fontsize=9)
    axis.text(0.0, scale * 1.05, "+Y base", color="tab:blue", fontsize=9)


def setup_xy_axis(axis: plt.Axes, title: str, hole_xy: np.ndarray) -> None:
    axis.scatter(0.0, 0.0, marker="s", s=90, color="black", label="robot base origin")
    axis.scatter(
        hole_xy[0],
        hole_xy[1],
        marker="*",
        s=180,
        color="tab:green",
        edgecolors="black",
        linewidths=0.6,
        label="sim hole root",
    )
    draw_base_axes(axis)
    axis.set_title(title)
    axis.set_xlabel("X [m]")
    axis.set_ylabel("Y [m]")
    axis.grid(alpha=0.25)
    axis.set_aspect("equal", adjustable="box")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trajectory-dir", type=Path, default=Path("real_data/traj_0"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/real_data_visualizations/real_base_xy_candidate.png"),
    )
    parser.add_argument("--hole", type=float, nargs=3, default=(-0.07, -0.42, 0.12))
    parser.add_argument("--stable-frames", type=int, default=10)
    args = parser.parse_args()

    with h5py.File(args.trajectory_dir / "data.h5", "r") as h5:
        pose = np.asarray(h5["obs/state/ee_pose"][:], dtype=np.float64)
        joints = np.asarray(h5["obs/state/joint_pos"][:], dtype=np.float64)
        timestamps = np.asarray(h5["timestamps"][:], dtype=np.float64)

    if pose.ndim != 2 or pose.shape[1] != 7:
        raise ValueError(f"Expected ee_pose shape (N,7), got {pose.shape}")
    if joints.shape != (len(pose), 7):
        raise ValueError(f"Expected joint_pos shape {(len(pose), 7)}, got {joints.shape}")
    if len(pose) < args.stable_frames:
        raise ValueError("stable-frames exceeds trajectory length")
    if not all(np.isfinite(value).all() for value in (pose, joints, timestamps)):
        raise ValueError("Non-finite values found in real trajectory")

    real_xy = pose[:, :2]
    real_z = pose[:, 2]
    stable_slice = slice(-args.stable_frames, None)
    stable_real_xy = real_xy[stable_slice].mean(axis=0)
    stable_real_z = float(real_z[stable_slice].mean())
    hole = np.asarray(args.hole, dtype=np.float64)
    minus90_xy = transform_xy(real_xy, -90.0)
    plus90_xy = transform_xy(real_xy, 90.0)
    minus90_stable = minus90_xy[stable_slice].mean(axis=0)
    plus90_stable = plus90_xy[stable_slice].mean(axis=0)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    front_path = args.trajectory_dir / "front_camera.mp4"
    first_frame = read_first_frame(front_path) if front_path.is_file() else None

    figure = plt.figure(figsize=(18, 11))
    grid = figure.add_gridspec(2, 3, width_ratios=(1.0, 1.0, 1.0), height_ratios=(1.0, 1.05))
    if first_frame is not None:
        image_axis = figure.add_subplot(grid[0, 0])
        image_axis.imshow(first_frame)
        image_axis.set_title("Real front camera: first frame")
        image_axis.axis("off")
    else:
        image_axis = figure.add_subplot(grid[0, 0])
        image_axis.axis("off")
        image_axis.text(0.5, 0.5, "front_camera.mp4 not found", ha="center", va="center")

    raw_axis = figure.add_subplot(grid[0, 1])
    raw_axis.plot(real_xy[:, 0], real_xy[:, 1], color="0.45", linewidth=1.8, label="real EE path (raw O frame)")
    raw_axis.scatter(*real_xy[0], color="tab:orange", s=65, label="real first EE")
    raw_axis.scatter(*stable_real_xy, color="tab:purple", s=65, label="real stable EE mean")
    raw_axis.scatter(0.0, 0.0, marker="s", s=90, color="black", label="real base origin")
    raw_axis.set_title("Raw real Franka-base XY\n(not directly comparable to sim world)")
    raw_axis.set_xlabel("X_real [m]")
    raw_axis.set_ylabel("Y_real [m]")
    raw_axis.grid(alpha=0.25)
    raw_axis.set_aspect("equal", adjustable="box")
    raw_axis.legend(fontsize=7, loc="best")

    hole_xy = hole[:2]
    minus_axis = figure.add_subplot(grid[0, 2])
    setup_xy_axis(minus_axis, "Candidate sim base: real → Rz(-90°)", hole_xy)
    minus_axis.plot(minus90_xy[:, 0], minus90_xy[:, 1], color="tab:orange", linewidth=2.0, label="transformed EE path")
    minus_axis.scatter(*minus90_xy[0], color="tab:red", s=65, label="transformed first EE")
    minus_axis.scatter(*minus90_stable, color="tab:purple", s=65, label="transformed stable EE")
    minus_axis.plot([minus90_stable[0], hole_xy[0]], [minus90_stable[1], hole_xy[1]], "k--", linewidth=1.0, label="stable EE → hole")
    minus_axis.legend(fontsize=7, loc="best")

    plus_axis = figure.add_subplot(grid[1, 0])
    setup_xy_axis(plus_axis, "Candidate sim base: real → Rz(+90°)", hole_xy)
    plus_axis.plot(plus90_xy[:, 0], plus90_xy[:, 1], color="tab:orange", linewidth=2.0, label="transformed EE path")
    plus_axis.scatter(*plus90_xy[0], color="tab:red", s=65, label="transformed first EE")
    plus_axis.scatter(*plus90_stable, color="tab:purple", s=65, label="transformed stable EE")
    plus_axis.plot([plus90_stable[0], hole_xy[0]], [plus90_stable[1], hole_xy[1]], "k--", linewidth=1.0, label="stable EE → hole")
    plus_axis.legend(fontsize=7, loc="best")

    three_d = figure.add_subplot(grid[1, 1], projection="3d")
    three_d.plot(minus90_xy[:, 0], minus90_xy[:, 1], real_z, color="tab:orange", linewidth=2.0, label="EE path, Rz(-90°)")
    three_d.scatter(0.0, 0.0, 0.0, marker="s", s=60, color="black", label="robot base origin")
    three_d.scatter(hole[0], hole[1], hole[2], marker="*", s=150, color="tab:green", label="sim hole root")
    three_d.scatter(*np.r_[minus90_stable, stable_real_z], color="tab:purple", s=60, label="stable EE")
    three_d.set_title("3D candidate initialization\nRz(-90°), Z unchanged")
    three_d.set_xlabel("X_sim [m]")
    three_d.set_ylabel("Y_sim [m]")
    three_d.set_zlabel("Z [m]")
    three_d.legend(fontsize=7, loc="best")

    joint_axis = figure.add_subplot(grid[1, 2])
    joint_axis.bar(np.arange(7) - 0.18, joints[0], width=0.36, label="real first q")
    joint_axis.bar(np.arange(7) + 0.18, joints[-1], width=0.36, label="real last q")
    joint_axis.set_xticks(np.arange(7), [f"q{i}" for i in range(7)])
    joint_axis.set_ylabel("rad")
    joint_axis.set_title("Initialization joint state\n(no PPO reset used)")
    joint_axis.grid(axis="y", alpha=0.25)
    joint_axis.legend(fontsize=8)

    minus_distance = float(np.linalg.norm(minus90_stable - hole_xy))
    plus_distance = float(np.linalg.norm(plus90_stable - hole_xy))
    figure.suptitle(
        "Real trajectory in candidate simulator base frames | "
        f"hole={hole.tolist()} | stable distance: Rz(-90°)={minus_distance:.3f} m, Rz(+90°)={plus_distance:.3f} m",
        fontsize=14,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.95))
    figure.savefig(args.output, dpi=180)
    plt.close(figure)

    metadata = {
        "trajectory": str(args.trajectory_dir),
        "ppo_reset_used": False,
        "sim_robot_base_origin": [0.0, 0.0, 0.0],
        "sim_hole_root": hole.tolist(),
        "real_first_ee_xyz": pose[0, :3].tolist(),
        "real_stable_ee_mean_xyz": [*stable_real_xy.tolist(), stable_real_z],
        "real_to_sim_candidate": {
            "minus_90_deg": {
                "stable_ee_xy": minus90_stable.tolist(),
                "distance_to_hole_xy_m": minus_distance,
            },
            "plus_90_deg": {
                "stable_ee_xy": plus90_stable.tolist(),
                "distance_to_hole_xy_m": plus_distance,
            },
        },
        "note": "EE origin is not automatically the peg tip; distances are geometric diagnostics, not success labels.",
    }
    args.output.with_suffix(".json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(metadata, indent=2, ensure_ascii=False))
    print(f"[OK] wrote {args.output}")


if __name__ == "__main__":
    main()

