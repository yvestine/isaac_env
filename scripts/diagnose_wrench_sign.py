#!/usr/bin/env python3
"""Verify +/-XYZ robot-wrench axes before accepting simulation-force data."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import traceback
from pathlib import Path

import numpy as np
import torch
from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task", type=str, default="TacEx-RealSim-PegInsert-Direct-v0")
parser.add_argument("--force", type=float, default=10.0, help="positive test-force magnitude in N")
parser.add_argument(
    "--axis",
    choices=("all", "x", "y", "z"),
    default="all",
    help="Test both signs of one base axis, or all six +/-XYZ directions.",
)
parser.add_argument("--settle-steps", type=int, default=240)
parser.add_argument("--measure-steps", type=int, default=240)
parser.add_argument(
    "--baseline-steps",
    type=int,
    default=120,
    help="zero-load steps collected before every directed-force test",
)
parser.add_argument(
    "--apply-body",
    type=str,
    default="force_sensor",
    help="robot body receiving the known external force",
)
parser.add_argument(
    "--read-body",
    type=str,
    default="force_sensor",
    help="robot body whose incoming-joint wrench is read",
)
parser.add_argument(
    "--raw-wrench-frame",
    choices=("auto_child_joint", "parent_body", "sensor_body"),
    default="auto_child_joint",
    help="frame hypothesis used to rotate the PhysX child-joint wrench into the robot base",
)
parser.add_argument("--skip-video", action="store_true", help="write PNG/CSV/JSON only")
parser.add_argument("--arm-kp", type=float, default=4000.0)
parser.add_argument("--arm-kd", type=float, default=400.0)
parser.add_argument(
    "--max-cross-axis-leakage-ratio",
    type=float,
    default=0.15,
    help=(
        "Maximum cross-axis response divided by principal response. The default "
        "keeps the strict 15%% test; panda_link7 may require a separately recorded tolerance."
    ),
)
parser.add_argument("--output-dir", type=Path, default=Path("outputs/wrench_source_goal1"))
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if not 0.0 < args.max_cross_axis_leakage_ratio < 1.0:
    raise ValueError("--max-cross-axis-leakage-ratio must be in (0, 1)")
if hasattr(args, "enable_cameras"):
    args.enable_cameras = True

simulation_app = AppLauncher(args).app

# Isaac Sim 6 keeps these compatibility modules in extension directories
# outside ``isaacsim.__path__`` unless they are added after AppLauncher.
import isaacsim  # noqa: E402

_ISAACSIM_EXTS = Path(isaacsim.__file__).resolve().parent / "exts"
for _extension_name in ("isaacsim.core.prims", "isaacsim.core.utils"):
    _extension_pkg = _ISAACSIM_EXTS / _extension_name / "isaacsim"
    if _extension_pkg.is_dir() and str(_extension_pkg) not in isaacsim.__path__:
        isaacsim.__path__.append(str(_extension_pkg))
import isaacsim.core.utils.torch as torch_utils  # noqa: E402
import tacex_tasks  # noqa: E402,F401
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402
from tacex_tasks.real2sim.realsim_env import RealSimEnv  # noqa: E402


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
                *[f"wrench_final_{i}" for i in range(6)],
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


def _make_visualization(
    path_png: Path,
    path_mp4: Path,
    records: np.ndarray,
    applied_force: float,
    axis_index: int,
    axis_name: str,
    baseline_base: np.ndarray,
    baseline_final: np.ndarray,
    skip_video: bool,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    t = records[:, 0]
    base = records[:, 10:16] - baseline_base[None, :]
    final = records[:, 16:22] - baseline_final[None, :]
    stable_start = int(np.argmax(t >= t[-1] - (t[-1] - t[0]) / 2.0))
    stable_start = max(stable_start, 0)

    figure, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    axes[0].plot(t, base[:, axis_index], label=f"wrench_base F{axis_name}", color="#1f77b4")
    axes[0].plot(t, final[:, axis_index], label=f"wrench_final F{axis_name}", color="#d62728")
    axes[0].axhline(applied_force, color="black", linestyle="--", label=f"applied {axis_name.upper()}")
    axes[0].axvline(t[stable_start], color="#555555", linestyle=":", label="stable window")
    axes[0].set_ylabel(f"F{axis_name} response [N]")
    axes[0].legend(loc="best")
    axes[0].grid(alpha=0.3)

    for index, label in enumerate(("Fx", "Fy", "Fz")):
        if index != axis_index:
            axes[1].plot(t, final[:, index], label=f"wrench_final {label}")
    axes[1].axvline(t[stable_start], color="#555555", linestyle=":")
    axes[1].set_xlabel("time [s]")
    axes[1].set_ylabel("cross-axis response [N]")
    axes[1].legend(ncol=3, loc="best", fontsize=8)
    axes[1].grid(alpha=0.3)
    figure.suptitle(
        f"Incoming-joint wrench test: applied {applied_force:+.1f} N in robot-base {axis_name.upper()}"
    )
    figure.tight_layout()
    figure.savefig(path_png, dpi=160)

    if skip_video:
        plt.close(figure)
        return

    from matplotlib.backends.backend_agg import FigureCanvasAgg

    frames: list[np.ndarray] = []
    frame_indices = np.linspace(0, len(t) - 1, min(180, len(t)), dtype=int)
    for end in frame_indices:
        frame_figure, frame_axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
        frame_axes[0].plot(
            t[: end + 1], base[: end + 1, axis_index], label=f"wrench_base F{axis_name}", color="#1f77b4"
        )
        frame_axes[0].plot(
            t[: end + 1], final[: end + 1, axis_index], label=f"wrench_final F{axis_name}", color="#d62728"
        )
        frame_axes[0].axhline(applied_force, color="black", linestyle="--", label="applied force")
        frame_axes[0].set_xlim(t[0], t[-1])
        frame_axes[0].set_ylabel(f"F{axis_name} response [N]")
        frame_axes[0].legend(loc="best")
        frame_axes[0].grid(alpha=0.3)
        for index, label in enumerate(("Fx", "Fy", "Fz")):
            if index != axis_index:
                frame_axes[1].plot(t[: end + 1], final[: end + 1, index], label=label)
        frame_axes[1].set_xlim(t[0], t[-1])
        frame_axes[1].set_xlabel("time [s]")
        frame_axes[1].set_ylabel("cross-axis response [N]")
        frame_axes[1].legend(ncol=3, loc="best", fontsize=8)
        frame_axes[1].grid(alpha=0.3)
        frame_figure.suptitle(
            f"signed {axis_name.upper()} wrench test | applied={applied_force:+.1f} N | t={t[end]:.3f}s"
        )
        frame_figure.tight_layout()
        canvas = FigureCanvasAgg(frame_figure)
        canvas.draw()
        rgba = np.asarray(canvas.buffer_rgba())
        frames.append(np.ascontiguousarray(rgba[:, :, :3][:, :, ::-1]))
        plt.close(frame_figure)
    _write_h264_mp4(path_mp4, frames, fps=30)
    plt.close(figure)


def _make_frame_calibration_plot(path: Path, directions: list[dict]) -> None:
    """Plot known base-frame loads against child-joint-frame corrected responses."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = [item["label"].replace("1", "") for item in directions]
    applied = np.asarray([item["applied_force_base"] for item in directions], dtype=np.float64)
    corrected = np.asarray(
        [item["wrench_child_joint_corrected_base_response"] for item in directions], dtype=np.float64
    )
    x_values = np.arange(len(labels))
    figure, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    for axis_index, axis_name in enumerate(("Fx", "Fy", "Fz")):
        axes[axis_index].plot(x_values, applied[:, axis_index], "o--", label="applied", color="black")
        axes[axis_index].plot(
            x_values,
            corrected[:, axis_index],
            "s-",
            label="incoming wrench corrected",
            color="#d62728",
        )
        axes[axis_index].axhline(0.0, color="#777777", linewidth=0.8)
        axes[axis_index].set_ylabel(f"{axis_name} [N]")
        axes[axis_index].grid(alpha=0.3)
        axes[axis_index].legend(loc="best")
    axes[-1].set_xticks(x_values, labels)
    axes[-1].set_xlabel("known load direction")
    figure.suptitle("PhysX child-joint wrench frame identification")
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def main() -> None:
    if args.force <= 0.0:
        raise ValueError("--force must be a positive magnitude")
    if args.baseline_steps < 2 or args.settle_steps < 1 or args.measure_steps < 1:
        raise ValueError("--baseline-steps must be >= 2; settle/measure steps must be positive")

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    # Never leave a previous run's summary looking like the current result.
    for stale_summary in (output_dir / "result.json", output_dir / "failure.json"):
        stale_summary.unlink(missing_ok=True)
    if args.raw_wrench_frame == "auto_child_joint":
        for pattern in ("wrench_direction_*.csv", "wrench_direction_*.png", "wrench_direction_*.mp4"):
            for stale_artifact in output_dir.glob(pattern):
                stale_artifact.unlink()
        (output_dir / "frame_calibration.png").unlink(missing_ok=True)

    cfg = parse_env_cfg(
        args.task,
        device=args.device,
        num_envs=1,
        use_fabric=not getattr(args, "disable_fabric", False),
    )
    cfg.scene.num_envs = 1
    cfg.enable_cameras = True
    cfg.data_collect_cfg["collect_data"] = False
    cfg.data_collect_cfg["save_tavla_hdf5"] = False
    cfg.data_collect_cfg["num_trajectories"] = 1_000_000
    # The environment only supports its two legacy hypotheses.  The diagnostic
    # auto mode reads wrench_raw directly and identifies the missing fixed
    # child-joint-frame rotation from known loads.
    cfg.ft_raw_wrench_frame = (
        "parent_body" if args.raw_wrench_frame == "auto_child_joint" else args.raw_wrench_frame
    )
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
        print("[DirectedWrenchTest] constructing environment", flush=True)
        env = RealSimEnv(cfg, render_mode=None, output_dir=str(output_dir))
        print("[DirectedWrenchTest] resetting environment", flush=True)
        env.reset()
        print("[DirectedWrenchTest] reset complete", flush=True)
        base_env = env
        print("[DirectedWrenchTest] resolving robot", flush=True)
        robot = base_env._robot
        print("[DirectedWrenchTest] resolving physics dt", flush=True)
        sim_dt = float(base_env.sim.get_physics_dt())
        print(f"[DirectedWrenchTest] physics_dt={sim_dt:.8f}s", flush=True)
        print("[DirectedWrenchTest] resolving robot body names", flush=True)
        body_names = list(robot.body_names)
        print(f"[DirectedWrenchTest] robot_bodies={body_names}", flush=True)
        for option_name, body_name in (("--apply-body", args.apply_body), ("--read-body", args.read_body)):
            if body_name not in body_names:
                raise ValueError(f"{option_name}={body_name!r} not found; available bodies: {body_names}")
        apply_body_idx = body_names.index(args.apply_body)
        read_body_idx = body_names.index(args.read_body)
        base_env._incoming_wrench_body_idx = read_body_idx
        print(
            "[DirectedWrenchTest] "
            f"apply_body={args.apply_body}[{apply_body_idx}] "
            f"read_body={args.read_body}[{read_body_idx}]",
            flush=True,
        )
        hold_q = robot.data.joint_pos.clone()
        hold_qd = torch.zeros_like(hold_q)
        root_quat_w = base_env._as_torch(robot.data.root_quat_w).clone()
        parent_quat_w = base_env._as_torch(
            robot.data.body_quat_w[:, base_env.force_sensor_parent_body_idx]
        ).clone()

        def rotate_base_to_parent(vector_base: np.ndarray) -> np.ndarray:
            vector = torch.as_tensor(vector_base, dtype=torch.float32, device=base_env.device).view(1, 3)
            vector_world = torch_utils.quat_apply(root_quat_w, vector)
            vector_parent = torch_utils.quat_apply(torch_utils.quat_conjugate(parent_quat_w), vector_world)
            return vector_parent[0].detach().cpu().numpy().astype(np.float64)

        parent_basis = torch.eye(3, dtype=torch.float32, device=base_env.device)
        parent_quat_batch = parent_quat_w.expand(3, -1)
        root_quat_batch = root_quat_w.expand(3, -1)
        parent_basis_world = torch_utils.quat_apply(parent_quat_batch, parent_basis)
        parent_basis_base = torch_utils.quat_apply(
            torch_utils.quat_conjugate(root_quat_batch), parent_basis_world
        )
        parent_to_base_row = parent_basis_base.detach().cpu().numpy().astype(np.float64)

        body_ids = torch.tensor([apply_body_idx], dtype=torch.long, device=base_env.device)

        def step_once(applied_force_base: np.ndarray | None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
            try:
                robot.write_joint_state_to_sim(hold_q, hold_qd)
                robot.set_joint_position_target(hold_q)
                if applied_force_base is not None:
                    force_base = torch.as_tensor(
                        applied_force_base, dtype=torch.float32, device=base_env.device
                    ).view(1, 3)
                    root_quat_w = base_env._as_torch(robot.data.root_quat_w)
                    force_world = torch_utils.quat_apply(root_quat_w, force_base).unsqueeze(1)
                    robot.permanent_wrench_composer.set_forces_and_torques(
                        forces=force_world,
                        torques=torch.zeros_like(force_world),
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
                final = base_env.wrench_final[0].detach().cpu().numpy().astype(np.float64)
                return raw, base, final
            except BaseException:
                traceback.print_exc()
                raise

        # First settle without force to remove stale pre-reset wrench values.
        for _ in range(10):
            step_once(None)

        axes = {"x": 0, "y": 1, "z": 2}
        selected_axes = axes.items() if args.axis == "all" else ((args.axis, axes[args.axis]),)
        directions = []
        for axis_name, axis_index in selected_axes:
            for sign in (1.0, -1.0):
                baseline_records = [step_once(None) for _ in range(args.baseline_steps)]
                baseline_start = args.baseline_steps // 2
                baseline_raw = np.mean([item[0] for item in baseline_records[baseline_start:]], axis=0)
                baseline_base = np.mean([item[1] for item in baseline_records[baseline_start:]], axis=0)
                baseline_final = np.mean([item[2] for item in baseline_records[baseline_start:]], axis=0)
                applied = np.zeros(3, dtype=np.float64)
                applied[axis_index] = sign * args.force
                records: list[list[float]] = []
                for index in range(args.settle_steps + args.measure_steps):
                    raw, base, final = step_once(applied)
                    records.append([index * sim_dt, *applied.tolist(), *raw.tolist(), *base.tolist(), *final.tolist()])
                records_array = np.asarray(records, dtype=np.float64)
                stable_raw = records_array[args.settle_steps :, 4:10]
                stable_base = records_array[args.settle_steps :, 10:16]
                stable_final = records_array[args.settle_steps :, 16:22]
                raw_response = stable_raw.mean(axis=0) - baseline_raw
                base_response = stable_base.mean(axis=0) - baseline_base
                final_response = stable_final.mean(axis=0) - baseline_final
                target_parent = rotate_base_to_parent(applied)
                if args.raw_wrench_frame == "auto_child_joint":
                    signed_base_principal = None
                    principal_magnitude = None
                    leakage = None
                    recommended_sign = None
                    passed = False
                else:
                    # Validate the coordinate transform independently of the legacy
                    # global sign in wrench_final.  The known applied load tells us
                    # which overall sign O_F_ext_hat_K must use.
                    signed_base_principal = float(sign * base_response[axis_index])
                    principal_magnitude = abs(signed_base_principal)
                    leakage = float(np.max(np.abs(np.delete(base_response[:3], axis_index))))
                    recommended_sign = 1.0 if signed_base_principal >= 0.0 else -1.0
                    passed = bool(
                        principal_magnitude >= 0.2 * args.force
                        and leakage
                        <= args.max_cross_axis_leakage_ratio * principal_magnitude
                    )
                label = f"{sign:+.0f}{axis_name.upper()}"
                _write_csv(output_dir / f"wrench_direction_{label}.csv", records)
                if args.raw_wrench_frame != "auto_child_joint":
                    _make_visualization(
                        output_dir / f"wrench_direction_{label}.png",
                        output_dir / f"wrench_direction_{label}.mp4",
                        records_array,
                        float(applied[axis_index]),
                        axis_index,
                        axis_name,
                        baseline_base,
                        baseline_final,
                        args.skip_video,
                    )
                directions.append(
                    {
                        "label": label,
                        "applied_force_base": applied.tolist(),
                        "baseline_raw_mean": baseline_raw.tolist(),
                        "baseline_base_mean": baseline_base.tolist(),
                        "baseline_final_mean": baseline_final.tolist(),
                        "wrench_raw_mean": stable_raw.mean(axis=0).tolist(),
                        "wrench_base_mean": stable_base.mean(axis=0).tolist(),
                        "wrench_final_mean": stable_final.mean(axis=0).tolist(),
                        "wrench_raw_response": raw_response.tolist(),
                        "wrench_base_response": base_response.tolist(),
                        "wrench_final_response": final_response.tolist(),
                        "target_force_parent": target_parent.tolist(),
                        "base_principal_signed_vs_applied_n": signed_base_principal,
                        "principal_force_magnitude_n": principal_magnitude,
                        "recommended_base_to_external_sign": recommended_sign,
                        "cross_axis_leakage_n": leakage,
                        "cross_axis_leakage_ratio": (
                            None
                            if leakage is None or principal_magnitude is None
                            else leakage / max(principal_magnitude, 1.0e-12)
                        ),
                        "passed": passed,
                    }
                )
                for _ in range(10):
                    step_once(None)

        frame_calibration = None
        if args.raw_wrench_frame == "auto_child_joint":
            positive_directions = [
                item for item in directions if max(item["applied_force_base"]) > 0.0
            ]
            if len(positive_directions) != 3:
                raise RuntimeError(
                    "--raw-wrench-frame auto_child_joint requires --axis all so +XYZ fit "
                    "and -XYZ validation are independent"
                )

            raw_fit = np.asarray(
                [item["wrench_raw_response"][:3] for item in positive_directions], dtype=np.float64
            )
            target_parent_fit = np.asarray(
                [item["target_force_parent"] for item in positive_directions], dtype=np.float64
            )
            candidates = []
            for raw_global_sign in (1.0, -1.0):
                signed_raw_fit = raw_global_sign * raw_fit
                u_matrix, _singular_values, vt_matrix = np.linalg.svd(
                    signed_raw_fit.T @ target_parent_fit
                )
                rotation_row = u_matrix @ vt_matrix
                if np.linalg.det(rotation_row) < 0.0:
                    u_matrix[:, -1] *= -1.0
                    rotation_row = u_matrix @ vt_matrix
                prediction = signed_raw_fit @ rotation_row
                rmse = float(np.sqrt(np.mean(np.square(prediction - target_parent_fit))))
                candidates.append((rmse, raw_global_sign, rotation_row))
            fit_rmse, fitted_raw_sign, child_to_parent_row = min(candidates, key=lambda item: item[0])

            negative_errors = []
            for item in directions:
                raw_force_response = np.asarray(item["wrench_raw_response"][:3], dtype=np.float64)
                corrected_parent = fitted_raw_sign * raw_force_response @ child_to_parent_row
                corrected_base = corrected_parent @ parent_to_base_row
                applied_force = np.asarray(item["applied_force_base"], dtype=np.float64)
                axis_index = int(np.argmax(np.abs(applied_force)))
                applied_sign = float(np.sign(applied_force[axis_index]))
                signed_base_principal = float(applied_sign * corrected_base[axis_index])
                principal_magnitude = abs(signed_base_principal)
                leakage = float(np.max(np.abs(np.delete(corrected_base, axis_index))))
                norm_ratio = float(np.linalg.norm(corrected_base) / args.force)
                item.update(
                    {
                        "wrench_child_joint_corrected_parent_response": corrected_parent.tolist(),
                        "wrench_child_joint_corrected_base_response": corrected_base.tolist(),
                        "base_principal_signed_vs_applied_n": signed_base_principal,
                        "principal_force_magnitude_n": principal_magnitude,
                        "recommended_base_to_external_sign": fitted_raw_sign,
                        "cross_axis_leakage_n": leakage,
                        "cross_axis_leakage_ratio": leakage / max(principal_magnitude, 1.0e-12),
                        "force_norm_ratio": norm_ratio,
                        "passed": bool(
                            principal_magnitude >= 0.5 * args.force
                            and 0.5 <= norm_ratio <= 1.5
                            and leakage
                            <= args.max_cross_axis_leakage_ratio * principal_magnitude
                        ),
                    }
                )
                if applied_sign < 0.0:
                    negative_errors.append(corrected_base - applied_force)

            validation_rmse = float(
                np.sqrt(np.mean(np.square(np.asarray(negative_errors, dtype=np.float64))))
            )
            frame_calibration = {
                "fit_directions": ["+X", "+Y", "+Z"],
                "validation_directions": ["-X", "-Y", "-Z"],
                "raw_global_sign": fitted_raw_sign,
                "child_joint_to_parent_rotation_row_vector": child_to_parent_row.tolist(),
                "rotation_determinant": float(np.linalg.det(child_to_parent_row)),
                "fit_rmse_n": fit_rmse,
                "validation_rmse_n": validation_rmse,
            }
            _make_frame_calibration_plot(output_dir / "frame_calibration.png", directions)

        recommended_signs = {item["recommended_base_to_external_sign"] for item in directions}
        sign_consistent = len(recommended_signs) == 1
        all_passed = bool(directions) and all(item["passed"] for item in directions) and sign_consistent
        result = {
            "task": args.task,
            "force_frame": "robot_base",
            "force_magnitude_n": float(args.force),
            "force_application_body": args.apply_body,
            "force_application_body_index": int(apply_body_idx),
            "incoming_wrench_read_body": args.read_body,
            "incoming_wrench_read_body_index": int(read_body_idx),
            "incoming_wrench_source": str(base_env.wrench_source),
            "raw_wrench_frame_hypothesis": args.raw_wrench_frame,
            "frame_calibration": frame_calibration,
            "robot_pose_fixed": True,
            "baseline_steps": int(args.baseline_steps),
            "settle_steps": int(args.settle_steps),
            "measure_steps": int(args.measure_steps),
            "physics_dt_s": sim_dt,
            "wrench_component_order": ["Fx", "Fy", "Fz", "Tx", "Ty", "Tz"],
            "wrench_units": ["N", "N", "N", "N*m", "N*m", "N*m"],
            "directions": directions,
            "summary": {
                "passed": all_passed,
                "cross_axis_leakage_limit": float(
                    args.max_cross_axis_leakage_ratio
                ),
                "sign_consistent": sign_consistent,
                "recommended_base_to_external_sign": (
                    next(iter(recommended_signs)) if sign_consistent else None
                ),
            },
            "note": "Robot-only physical-source diagnostic; failure blocks calibration and data export.",
        }
        (output_dir / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(json.dumps(result, indent=2))
        if not all_passed:
            raise RuntimeError("Directed wrench validation failed; do not generate calibrated force data")
        print(f"outputs: {output_dir}")
    except BaseException as exc:
        failure = {
            "status": "failed",
            "exception_type": type(exc).__name__,
            "exception": str(exc),
            "traceback": traceback.format_exc(),
        }
        failure_path = output_dir / "failure.json"
        failure_path.write_text(json.dumps(failure, indent=2), encoding="utf-8")
        print("[DirectedWrenchTest] FAILED", file=sys.stderr, flush=True)
        traceback.print_exc()
        print(f"[DirectedWrenchTest] failure_report={failure_path}", file=sys.stderr, flush=True)
        raise
    finally:
        print("[DirectedWrenchTest] closing environment", flush=True)
        if env is not None:
            env.close()
        print("[DirectedWrenchTest] closing simulation app", flush=True)
        simulation_app.close()


if __name__ == "__main__":
    main()
