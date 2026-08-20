#!/usr/bin/env python3
"""Run a robot-only +X external-force sign test for the RealSim wrench pipeline."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
from pathlib import Path

import numpy as np
import torch
from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task", type=str, default="TacEx-RealSim-PegInsert-Direct-v0")
parser.add_argument("--force", type=float, default=10.0, help="signed X force in robot-base frame, in N")
parser.add_argument("--settle-steps", type=int, default=240)
parser.add_argument("--measure-steps", type=int, default=240)
parser.add_argument("--arm-kp", type=float, default=4000.0)
parser.add_argument("--arm-kd", type=float, default=400.0)
parser.add_argument("--output-dir", type=Path, default=Path("outputs/wrench_sign_test_plus_x"))
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

simulation_app = AppLauncher(args).app

import gymnasium as gym  # noqa: E402
import isaacsim.core.utils.torch as torch_utils  # noqa: E402
import tacex_tasks  # noqa: E402,F401
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402


def _write_csv(path: Path, rows: list[list[float]]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "timestamp_s",
                "applied_force_base_x_N",
                "applied_force_base_y_N",
                "applied_force_base_z_N",
                *[f"wrench_raw_{i}" for i in range(6)],
                *[f"wrench_base_{i}" for i in range(6)],
            ]
        )
        writer.writerows(rows)


def _write_h264_mp4(path: Path, frames: list[np.ndarray], fps: int) -> None:
    if not frames:
        return
    import imageio_ffmpeg

    first = np.ascontiguousarray(frames[0], dtype=np.uint8)
    height, width = first.shape[:2]
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    process = subprocess.Popen(
        [
            ffmpeg,
            "-loglevel",
            "error",
            "-y",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "bgr24",
            "-s:v",
            f"{width}x{height}",
            "-r",
            str(fps),
            "-i",
            "-",
            "-an",
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            "23",
            "-profile:v",
            "baseline",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(path),
        ],
        stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )
    assert process.stdin is not None
    try:
        for frame in frames:
            process.stdin.write(np.ascontiguousarray(frame, dtype=np.uint8).tobytes())
        process.stdin.close()
        stderr = process.stderr.read()
        return_code = process.wait()
    except Exception:
        process.kill()
        process.wait()
        raise
    if return_code != 0:
        raise RuntimeError(f"ffmpeg failed with code {return_code}: {stderr.decode(errors='replace')}")


def _make_visualization(path_png: Path, path_mp4: Path, records: np.ndarray, applied_force: float) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    t = records[:, 0]
    raw = records[:, 4:10]
    base = records[:, 10:16]
    stable_start = int(np.argmax(t >= t[-1] - (t[-1] - t[0]) / 2.0))
    stable_start = max(stable_start, 0)

    figure, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    axes[0].plot(t, raw[:, 0], label="wrench_raw Fx", color="#1f77b4")
    axes[0].plot(t, base[:, 0], label="wrench_base Fx", color="#d62728")
    axes[0].axhline(applied_force, color="black", linestyle="--", label="applied +X")
    axes[0].axvline(t[stable_start], color="#555555", linestyle=":", label="stable window")
    axes[0].set_ylabel("Fx [N]")
    axes[0].legend(loc="best")
    axes[0].grid(alpha=0.3)

    for index, label in enumerate(("Fy", "Fz", "Tx", "Ty", "Tz"), start=1):
        axes[1].plot(t, base[:, index], label=f"wrench_base {label}")
    axes[1].axvline(t[stable_start], color="#555555", linestyle=":")
    axes[1].set_xlabel("time [s]")
    axes[1].set_ylabel("wrench components")
    axes[1].legend(ncol=3, loc="best", fontsize=8)
    axes[1].grid(alpha=0.3)
    figure.suptitle(f"RealSim wrench sign test: applied {applied_force:+.1f} N in robot-base X")
    figure.tight_layout()
    figure.savefig(path_png, dpi=160)

    from matplotlib.backends.backend_agg import FigureCanvasAgg

    frames: list[np.ndarray] = []
    frame_indices = np.linspace(0, len(t) - 1, min(180, len(t)), dtype=int)
    for end in frame_indices:
        frame_figure, frame_axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
        frame_axes[0].plot(t[: end + 1], raw[: end + 1, 0], label="wrench_raw Fx", color="#1f77b4")
        frame_axes[0].plot(t[: end + 1], base[: end + 1, 0], label="wrench_base Fx", color="#d62728")
        frame_axes[0].axhline(applied_force, color="black", linestyle="--", label="applied +X")
        frame_axes[0].set_xlim(t[0], t[-1])
        frame_axes[0].set_ylabel("Fx [N]")
        frame_axes[0].legend(loc="best")
        frame_axes[0].grid(alpha=0.3)
        for index, label in enumerate(("Fy", "Fz", "Tx", "Ty", "Tz"), start=1):
            frame_axes[1].plot(t[: end + 1], base[: end + 1, index], label=label)
        frame_axes[1].set_xlim(t[0], t[-1])
        frame_axes[1].set_xlabel("time [s]")
        frame_axes[1].set_ylabel("base wrench")
        frame_axes[1].legend(ncol=3, loc="best", fontsize=8)
        frame_axes[1].grid(alpha=0.3)
        frame_figure.suptitle(f"signed X wrench test | applied={applied_force:+.1f} N | t={t[end]:.3f}s")
        frame_figure.tight_layout()
        canvas = FigureCanvasAgg(frame_figure)
        canvas.draw()
        rgba = np.asarray(canvas.buffer_rgba())
        frames.append(np.ascontiguousarray(rgba[:, :, :3][:, :, ::-1]))
        plt.close(frame_figure)
    _write_h264_mp4(path_mp4, frames, fps=30)
    plt.close(figure)


def main() -> None:
    if args.force == 0.0:
        raise ValueError("--force must be non-zero; use +10 for +X or -10 for -X")
    if args.settle_steps < 1 or args.measure_steps < 1:
        raise ValueError("--settle-steps and --measure-steps must be positive")

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    cfg = parse_env_cfg(
        args.task,
        device=args.device,
        num_envs=1,
        use_fabric=not getattr(args, "disable_fabric", False),
    )
    cfg.scene.num_envs = 1
    cfg.data_collect_cfg["collect_data"] = False
    cfg.data_collect_cfg["save_tavla_hdf5"] = False
    cfg.data_collect_cfg["num_trajectories"] = 0
    cfg.robot.actuators["panda_arm1"].stiffness = args.arm_kp
    cfg.robot.actuators["panda_arm1"].damping = args.arm_kd
    cfg.robot.actuators["panda_arm2"].stiffness = args.arm_kp
    cfg.robot.actuators["panda_arm2"].damping = args.arm_kd
    if hasattr(cfg, "wait_for_textures"):
        cfg.wait_for_textures = False
    if hasattr(cfg, "num_rerenders_on_reset"):
        cfg.num_rerenders_on_reset = 0

    env = None
    try:
        env = gym.make(args.task, cfg=cfg)
        env.reset()
        base_env = env.unwrapped
        robot = base_env._robot
        sim_dt = float(base_env.sim.get_physics_dt())
        force_sensor_idx = robot.body_names.index("force_sensor")
        hold_q = robot.data.joint_pos.clone()
        hold_qd = torch.zeros_like(hold_q)

        # The configured base pose is identity by default. Convert +X in the
        # robot-base frame to world explicitly so the test remains valid if the
        # base is later rotated.
        applied_force_base = torch.tensor(
            [[args.force, 0.0, 0.0]], dtype=torch.float32, device=base_env.device
        )
        root_quat_w = robot.data.root_quat_w
        applied_force_world = torch_utils.quat_apply(root_quat_w, applied_force_base)
        applied_force_world = applied_force_world.unsqueeze(1)
        applied_torque_world = torch.zeros_like(applied_force_world)
        body_ids = torch.tensor([force_sensor_idx], dtype=torch.long, device=base_env.device)

        def step_once(apply_force: bool) -> tuple[np.ndarray, np.ndarray]:
            robot.write_joint_state_to_sim(hold_q, hold_qd)
            robot.set_joint_position_target(hold_q)
            if apply_force:
                robot.permanent_wrench_composer.set_forces_and_torques(
                    forces=applied_force_world,
                    torques=applied_torque_world,
                    body_ids=body_ids,
                    is_global=True,
                )
            else:
                robot.permanent_wrench_composer.reset()
            base_env.scene.write_data_to_sim()
            base_env.sim.step()
            base_env.scene.update(sim_dt)
            base_env._update_wrench()
            raw = base_env.wrench_raw[0].detach().cpu().numpy().astype(np.float64)
            base = base_env.wrench_base[0].detach().cpu().numpy().astype(np.float64)
            return raw, base

        # First settle without force to remove stale pre-reset wrench values.
        for _ in range(10):
            step_once(False)

        records: list[list[float]] = []
        total_steps = args.settle_steps + args.measure_steps
        for index in range(total_steps):
            raw, base = step_once(True)
            records.append(
                [
                    index * sim_dt,
                    args.force,
                    0.0,
                    0.0,
                    *raw.tolist(),
                    *base.tolist(),
                ]
            )

        records_array = np.asarray(records, dtype=np.float64)
        stable = records_array[args.settle_steps :, 10:16]
        stable_raw = records_array[args.settle_steps :, 4:10]
        base_mean = stable.mean(axis=0)
        base_std = stable.std(axis=0)
        raw_mean = stable_raw.mean(axis=0)
        raw_std = stable_raw.std(axis=0)
        applied_force_world_flat = applied_force_world[0, 0].detach().cpu().numpy().astype(float)
        result = {
            "task": args.task,
            "applied_force_frame": "robot_base",
            "applied_force": [float(args.force), 0.0, 0.0],
            "applied_force_units": "N",
            "applied_force_world": applied_force_world_flat.tolist(),
            "force_application_body": "force_sensor",
            "force_application_body_index": int(force_sensor_idx),
            "robot_pose_fixed": True,
            "settle_steps": int(args.settle_steps),
            "measure_steps": int(args.measure_steps),
            "physics_dt_s": sim_dt,
            "stable_window_start_s": float(records_array[args.settle_steps, 0]),
            "stable_window_end_s": float(records_array[-1, 0]),
            "wrench_component_order": ["Fx", "Fy", "Fz", "Tx", "Ty", "Tz"],
            "wrench_units": ["N", "N", "N", "N*m", "N*m", "N*m"],
            "wrench_raw_mean": raw_mean.tolist(),
            "wrench_raw_std": raw_std.tolist(),
            "wrench_base_mean": base_mean.tolist(),
            "wrench_base_std": base_std.tolist(),
            "wrench_base_fx_sign": "positive" if base_mean[0] > 0.0 else "negative" if base_mean[0] < 0.0 else "near_zero",
            "recommendation": "do_not_flip" if base_mean[0] > 0.0 else "flip_all_six_components" if base_mean[0] < 0.0 else "repeat_with_larger_force",
            "note": "This is a robot-only diagnostic. No TAVLA Server, HDF5 export, training, or checkpoint was used.",
        }
        (output_dir / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        _write_csv(output_dir / "wrench_sign_test.csv", records)
        _make_visualization(
            output_dir / "wrench_sign_test.png",
            output_dir / "wrench_sign_test.mp4",
            records_array,
            args.force,
        )
        print(json.dumps(result, indent=2))
        print(f"outputs: {output_dir}")
    finally:
        if env is not None:
            env.close()
        simulation_app.close()


if __name__ == "__main__":
    main()
