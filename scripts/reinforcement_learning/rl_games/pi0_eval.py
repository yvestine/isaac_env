"""Run the current image/state-only Pi0 policy inside the RealSim task."""
from __future__ import annotations

import argparse
import atexit
import csv
import json
import os
import tempfile
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm
from isaaclab.app import AppLauncher

PI0_ACTION_HORIZON = 50

parser = argparse.ArgumentParser(description="Pi0 image/state-only RealSim evaluation")
parser.add_argument("--task", type=str, default="TacEx-RealSim-PegInsert-PI0-Direct-v0")
parser.add_argument("--pi0-host", type=str, default="114.214.164.112")
parser.add_argument("--pi0-port", type=int, default=8001)
parser.add_argument(
    "--pi0-timeout-s",
    type=float,
    default=10.0,
    help="Maximum wait for one Pi0 reset/inference response.",
)
parser.add_argument("--steps", type=int, default=200)
parser.add_argument("--episodes", type=int, default=1)
parser.add_argument(
    "--replan-actions",
    type=int,
    default=3,
    help=(
        "Number of actions consumed from each Pi0 chunk before replanning. "
        "The default is 3; use 5 to amortize the current Pi0 inference latency."
    ),
)
parser.add_argument(
    "--action-start-index",
    type=int,
    default=2,
    help="First future action index consumed from each predicted chunk.",
)
parser.add_argument(
    "--direct-reset-h5",
    type=Path,
    default=None,
    help="Optional real_data/*/data.h5 used by the DirectJointReset task.",
)
parser.add_argument(
    "--direct-reset-h5-list",
    type=Path,
    nargs="+",
    default=None,
    help=(
        "Ordered direct-reset H5 files. One Isaac process cycles through this "
        "list on successive episode resets."
    ),
)
parser.add_argument(
    "--control-mode",
    choices=(
        "implicit",
        "implicit_slow",
        "implicit_contact_safe",
        "implicit_fast",
        "explicit_pd",
    ),
    default="implicit",
    help="Joint control path used after Pi0 target shaping.",
)
parser.add_argument(
    "--episode-length-s",
    type=float,
    default=None,
    help="Override the environment timeout; success still terminates immediately.",
)
parser.add_argument(
    "--stop-on-success",
    action=argparse.BooleanOptionalAction,
    default=True,
    help="Stop the episode immediately when the PPO success predicate is true.",
)
parser.add_argument("--output-dir", type=Path, default=Path("outputs/pi0_eval"))
parser.add_argument(
    "--gt-reference-report",
    type=Path,
    default=None,
    help="Optional replay_metadata.json from the configured traj_0 GT validation.",
)
parser.add_argument("--save-video", action=argparse.BooleanOptionalAction, default=True)
parser.add_argument("--video-fps", type=float, default=10.0)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if args.replan_actions <= 0:
    raise ValueError("--replan-actions must be positive")
if args.action_start_index < 0:
    raise ValueError("--action-start-index must be non-negative")
if args.pi0_timeout_s <= 0.0:
    raise ValueError("--pi0-timeout-s must be positive")
# Pi0 always consumes camera observations, including headless video capture.
if hasattr(args, "enable_cameras"):
    args.enable_cameras = True

simulation_app = AppLauncher(args).app

import gymnasium as gym  # noqa: E402
import tacex_tasks  # noqa: E402,F401
from isaaclab_tasks.direct.factory import factory_utils  # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402
from tacex_tasks.real2sim.pi0_env import (  # noqa: E402
    _composite_pi0_cable_grommets,
    _composite_pi0_green_table_front,
    _composite_pi0_reference_cylinder,
    _match_pi0_wrist_appearance,
    _resize_pi0_rgb,
)
from tacex_tasks.real2sim.realsim_env import _write_h264_mp4  # noqa: E402
import cv2  # noqa: E402
from pxr import Usd, UsdGeom  # noqa: E402


def _create_physics_free_background_usd() -> Path:
    """Keep the visual background but remove its embedded PhysicsScene."""
    source = Path(__file__).resolve().parents[3] / "franka_env_background_edit" / "franka_env.usd"
    if not source.is_file():
        raise FileNotFoundError(f"Background USD not found: {source}")
    fd, filename = tempfile.mkstemp(prefix="tacex_pi0_visual_background_", suffix=".usd")
    os.close(fd)
    wrapper_path = Path(filename)
    stage = Usd.Stage.CreateNew(str(wrapper_path))
    root = UsdGeom.Xform.Define(stage, "/World")
    root.GetPrim().GetReferences().AddReference(str(source), "/World")
    stage.SetDefaultPrim(root.GetPrim())
    stage.OverridePrim("/World/PhysicsScene").SetActive(False)
    stage.GetRootLayer().Save()
    return wrapper_path


_PI0_BACKGROUND_USD = _create_physics_free_background_usd()
atexit.register(lambda: _PI0_BACKGROUND_USD.unlink(missing_ok=True))


def _scalar(value):
    if torch.is_tensor(value):
        return float(value.detach().reshape(-1)[0].cpu())
    return float(value)

def _video_frame(frame):
    """Convert an Isaac camera frame to contiguous HWC RGB uint8."""
    frame = np.asarray(frame)
    if frame.ndim == 4 and frame.shape[0] == 1:
        frame = frame[0]
    if frame.ndim != 3:
        raise ValueError(f"Camera frame must be rank-3, got {frame.shape}")
    if frame.shape[-1] not in (1, 3, 4) and frame.shape[0] in (1, 3, 4):
        frame = np.transpose(frame, (1, 2, 0))
    if frame.shape[-1] == 4:
        frame = frame[..., :3]
    elif frame.shape[-1] == 1:
        frame = np.repeat(frame, 3, axis=-1)
    if frame.shape[-1] != 3:
        raise ValueError(f"Camera frame must have 1, 3, or 4 channels, got {frame.shape}")
    if frame.dtype != np.uint8:
        frame = frame.astype(np.float32)
        if np.isfinite(frame).all() and (frame.max() if frame.size else 0.0) <= 1.0:
            frame *= 255.0
        frame = np.clip(frame, 0.0, 255.0).astype(np.uint8)
    return np.ascontiguousarray(frame)


def _capture_video_frames(env):
    """Read the frames already produced by Isaac's camera managers.

    ``_camera_observation`` is the policy path and deliberately performs
    material swaps plus several forced RTX renders.  Calling it again only
    for video recording can deadlock the renderer after a long rollout.  The
    environment has already rendered both cameras during ``env.step``; read
    those buffers directly and apply only the CPU-side image transforms used
    by the policy.
    """
    raw_env = env.unwrapped
    front = _video_frame(
        raw_env.tiled_camera.data.output["rgb"][0].detach().cpu().numpy()
    )
    wrist = _video_frame(
        raw_env.wrist_tiled_camera.data.output["rgb"][0].detach().cpu().numpy()
    )
    front = _resize_pi0_rgb(front, raw_env.cfg)
    front = _composite_pi0_green_table_front(front, raw_env.cfg)
    front = _composite_pi0_reference_cylinder(front, raw_env.cfg, "rgb")
    front = _composite_pi0_cable_grommets(front, raw_env.cfg, "rgb")
    wrist = _resize_pi0_rgb(wrist, raw_env.cfg)
    wrist = _match_pi0_wrist_appearance(wrist, raw_env.cfg, "rgb")
    return front, wrist


def _insertion_metrics(env):
    """Use the same peg-base geometry and success predicate as GT validation."""
    raw_env = env.unwrapped
    held_base_pos, _ = factory_utils.get_held_base_pose(
        raw_env.held_pos,
        raw_env.held_quat,
        raw_env.cfg_task.name,
        raw_env.cfg_task.fixed_asset_cfg,
        raw_env.num_envs,
        raw_env.device,
    )
    target_base_pos, _ = factory_utils.get_target_held_base_pose(
        raw_env.fixed_pos,
        raw_env.fixed_quat,
        raw_env.cfg_task.name,
        raw_env.cfg_task.fixed_asset_cfg,
        raw_env.num_envs,
        raw_env.device,
    )
    delta = target_base_pos[0] - held_base_pos[0]
    success = raw_env._get_curr_successes(
        success_threshold=raw_env.cfg_task.success_threshold,
        check_rot=False,
    )[0]
    return {
        "xy_error_m": float(torch.linalg.vector_norm(delta[:2]).detach().cpu()),
        "z_disp_m": float((held_base_pos[0, 2] - target_base_pos[0, 2]).detach().cpu()),
        "geometric_success": bool(success.detach().cpu()),
    }


def _held_asset_diagnostics(env):
    """Record peg-root motion relative to the fingertip.

    The held peg is a simulated articulation.  Its root can move under
    contact even when the gripper target is unchanged, which is the signal
    needed to distinguish a fast arm command from a slipping peg mount.
    """
    raw_env = env.unwrapped
    held_asset = getattr(raw_env, "_held_asset", None)
    data = getattr(held_asset, "data", None)
    root_state = getattr(data, "root_state_w", None)
    fingertip = getattr(raw_env, "fingertip_midpoint_pos", None)
    if root_state is None or fingertip is None:
        return {}
    root_state = root_state[0].detach().cpu().numpy()
    fingertip = fingertip[0].detach().cpu().numpy()
    root_pos = root_state[:3]
    root_linvel = root_state[7:10]
    root_to_fingertip = fingertip - root_pos
    return {
        "held_root_pos": root_pos.tolist(),
        "held_root_linvel": root_linvel.tolist(),
        "held_root_to_fingertip_vector_m": root_to_fingertip.tolist(),
        "held_root_to_fingertip_distance_m": float(np.linalg.norm(root_to_fingertip)),
        "held_root_to_fingertip_z_m": float(root_to_fingertip[2]),
    }


def _load_gt_reference(path):
    if path is None:
        return None
    path = path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"GT reference report not found: {path}")
    report = json.loads(path.read_text(encoding="utf-8"))
    return {
        "path": str(path),
        "geometric_success": bool(report.get("geometric_success", False)),
        "first_success_frame": report.get("first_success_frame"),
        "best_xy_error_m": report.get("best_xy_error_m"),
        "minimum_z_disp_m": report.get("minimum_z_disp_m"),
        "final_xy_error_m": report.get("final_xy_error_m"),
        "final_z_disp_m": report.get("final_z_disp_m"),
    }


def _reward_terms(extras):
    terms = {}
    for name, value in (extras or {}).items():
        if not name.startswith("logs_rew_"):
            continue
        try:
            terms[name[len("logs_rew_"):]] = _scalar(value)
        except (TypeError, ValueError, RuntimeError):
            continue
    return terms


def _apply_control_mode(env_cfg, mode: str) -> None:
    """Apply a named controller profile before Isaac creates the articulation."""
    cfg = env_cfg.pi0_policy_cfg
    if mode == "implicit":
        # Tracking profile: send the predicted absolute joint target directly
        # through a sufficiently stiff implicit actuator. The old 80/20 gains
        # left the measured state 2-3 control steps behind the target.
        cfg.use_implicit_position_controller = True
        cfg.implicit_arm_stiffness = 200.0
        cfg.implicit_arm_damping = 40.0
        # Safe default for hardware-facing validation. At 10 Hz this limits
        # each commanded joint target change to 0.0015 rad per control step.
        cfg.joint_velocity_limits = [0.015] * 7
    elif mode == "implicit_slow":
        # Contact-isolation profile: keep the same PD gains as the baseline,
        # but reduce the target slew from 1.0 rad/s to 0.2 rad/s.  At 10 Hz
        # this limits each commanded joint target change to 0.02 rad.
        cfg.use_implicit_position_controller = True
        cfg.implicit_arm_stiffness = 200.0
        cfg.implicit_arm_damping = 40.0
        cfg.joint_velocity_limits = [0.2] * 7
    elif mode == "implicit_contact_safe":
        # Conservative contact-isolation profile. At 10 Hz the commanded
        # target can move by at most 0.0015 rad per control step.
        cfg.use_implicit_position_controller = True
        cfg.implicit_arm_stiffness = 200.0
        cfg.implicit_arm_damping = 40.0
        cfg.joint_velocity_limits = [0.015] * 7
    elif mode == "implicit_fast":
        cfg.use_implicit_position_controller = True
        cfg.implicit_arm_stiffness = 400.0
        cfg.implicit_arm_damping = 80.0
        cfg.joint_velocity_limits = [2.0] * 7
    elif mode == "explicit_pd":
        cfg.use_implicit_position_controller = False
        cfg.joint_kp = [300.0] * 4 + [150.0] * 3
        cfg.joint_kd = [40.0] * 4 + [20.0] * 3
        cfg.joint_velocity_limits = [1.0] * 7
    else:
        raise ValueError(f"Unknown controller mode: {mode}")


def _write_rgb_video(path, frames, fps):
    if len(frames) <= 1:
        return
    bgr_frames = [np.ascontiguousarray(frame[..., ::-1]) for frame in frames]
    _write_h264_mp4(path, bgr_frames, fps)


def _write_reward_video(path, frames, rewards, reward_terms, diagnostic_rows, fps):
    if len(frames) <= 1:
        return
    overlay_frames = []
    cumulative = 0.0
    for frame_index, rgb in enumerate(frames):
        frame = np.ascontiguousarray(rgb[..., ::-1].copy())
        if frame_index == 0:
            lines = ["reset frame"]
        else:
            reward_index = frame_index - 1
            reward = rewards[reward_index]
            cumulative += reward
            lines = [
                f"step: {reward_index:04d}",
                f"PPO reward: {reward:+.4f}",
                f"cumulative: {cumulative:+.3f}",
            ]
            for name in ("kp_fine", "curr_engaged", "curr_success", "insertion_progress"):
                if reward_index < len(reward_terms) and name in reward_terms[reward_index]:
                    lines.append(f"{name}: {reward_terms[reward_index][name]:+.3f}")
            if reward_index < len(diagnostic_rows):
                diagnostic = diagnostic_rows[reward_index]
                lines.extend(
                    (
                        f"peg XY error: {diagnostic['xy_error_m'] * 1000.0:.2f} mm",
                        f"peg Z disp: {diagnostic['z_disp_m'] * 1000.0:+.2f} mm",
                        f"geometric success: {diagnostic['geometric_success']}",
                    )
                )
        for line_index, line in enumerate(lines):
            cv2.putText(
                frame,
                line,
                (12, 30 + 25 * line_index),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 0, 0),
                3,
                cv2.LINE_AA,
            )
            cv2.putText(
                frame,
                line,
                (12, 30 + 25 * line_index),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )
        overlay_frames.append(frame)
    _write_h264_mp4(path, overlay_frames, fps)


def main():
    env_cfg = parse_env_cfg(
        args.task,
        device=args.device,
        num_envs=1,
        use_fabric=not getattr(args, "disable_fabric", False),
    )
    env_cfg.pi0_policy_cfg.host_ip = args.pi0_host
    env_cfg.pi0_policy_cfg.host_port = args.pi0_port
    env_cfg.pi0_policy_cfg.inference_timeout_s = float(args.pi0_timeout_s)
    env_cfg.pi0_policy_cfg.action_start_index = int(args.action_start_index)
    env_cfg.pi0_action_start_index = int(args.action_start_index)
    env_cfg.pi0_replan_actions = args.replan_actions
    _apply_control_mode(env_cfg, args.control_mode)
    if args.direct_reset_h5 is not None:
        if not hasattr(env_cfg, "pi0_direct_reset_h5"):
            raise ValueError(
                "--direct-reset-h5 requires the TacEx-RealSim-PegInsert-PI0-DirectJointReset-v0 task"
            )
        env_cfg.pi0_direct_reset_h5 = str(args.direct_reset_h5)
    if args.direct_reset_h5_list is not None:
        if not hasattr(env_cfg, "pi0_direct_reset_h5_list"):
            raise ValueError(
                "--direct-reset-h5-list requires the TacEx-RealSim-PegInsert-PI0-DirectJointReset-v0 task"
            )
        if not args.direct_reset_h5_list:
            raise ValueError("--direct-reset-h5-list cannot be empty")
        env_cfg.pi0_direct_reset_h5_list = tuple(
            str(path) for path in args.direct_reset_h5_list
        )
        env_cfg.pi0_direct_reset_h5 = str(args.direct_reset_h5_list[0])
    if args.episode_length_s is not None:
        env_cfg.episode_length_s = args.episode_length_s
    env_cfg.data_collect_cfg["immediate_stop"] = bool(args.stop_on_success)
    env_cfg.data_collect_cfg["collect_data"] = False
    args.output_dir.mkdir(parents=True, exist_ok=True)
    gt_reference = _load_gt_reference(args.gt_reference_report)

    env = None
    rows = []
    ppo_rewards = []
    reward_terms = []
    episode_returns = []
    episode_return = 0.0
    front_frames = []
    wrist_frames = []
    video_error = ""
    pi0_inference_count_peak = 0
    pi0_failures_peak = 0
    episodes = 0
    executed_steps = 0
    task_success_detected = False
    first_success_step_detected = None
    steps_file = None
    try:
        steps_file = (args.output_dir / "steps.jsonl").open(
            "w", encoding="utf-8", buffering=1
        )
        previous_background_usd = os.environ.get("TACEX_BACKGROUND_USD")
        os.environ["TACEX_BACKGROUND_USD"] = str(_PI0_BACKGROUND_USD)
        try:
            env = gym.make(args.task, cfg=env_cfg, output_dir=str(args.output_dir))
        finally:
            if previous_background_usd is None:
                os.environ.pop("TACEX_BACKGROUND_USD", None)
            else:
                os.environ["TACEX_BACKGROUND_USD"] = previous_background_usd
        env.reset()
        reset_state8 = env.unwrapped._current_pi0_state()[0].detach().cpu().tolist()
        reset_target = (
            env.unwrapped._pi0_target[0].detach().cpu().tolist()
            if env.unwrapped._pi0_target is not None
            else None
        )
        print(
            "[Pi0Eval] post-reset "
            f"state8={reset_state8} pi0_target={reset_target}",
            flush=True,
        )
        if args.save_video:
            front, wrist = _capture_video_frames(env)
            front_frames.append(front)
            wrist_frames.append(wrist)
        progress = tqdm(
            total=args.steps * args.episodes,
            desc="Pi0 rollout",
            unit="step",
            dynamic_ncols=True,
        )
        atexit.register(progress.close)
        for step in range(args.steps * args.episodes):
            action = torch.zeros(
                (1, env.unwrapped.cfg.action_space),
                device=env.unwrapped.device,
            )
            _, reward, terminated, truncated, info = env.step(action)
            executed_steps += 1
            pi0_inference_count_peak = max(pi0_inference_count_peak, int(env.unwrapped.pi0_inference_count))
            pi0_failures_peak = max(pi0_failures_peak, int(env.unwrapped.pi0_failures))
            progress.update(1)
            progress.set_postfix(
                infer=pi0_inference_count_peak,
                latency=f"{env.unwrapped.pi0_last_latency_s:.3f}s",
                failures=pi0_failures_peak,
                frames=len(front_frames) if args.save_video else 0,
            )
            reward_value = _scalar(reward)
            ppo_rewards.append(reward_value)
            episode_return += reward_value
            current_reward_terms = _reward_terms(info)
            reward_terms.append(current_reward_terms)
            terminated_value = bool(_scalar(terminated))
            truncated_value = bool(_scalar(truncated))
            step_success_detected = bool(
                current_reward_terms.get("curr_success", 0.0) > 0.0
                or (terminated_value and not truncated_value)
            )
            if step_success_detected and first_success_step_detected is None:
                first_success_step_detected = step
            task_success_detected = task_success_detected or step_success_detected
            if terminated_value or truncated_value:
                # DirectRLEnv has already reset here. Do not save the reset
                # observation as if it were the final rollout frame/state.
                episode_returns.append(episode_return)
                episode_return = 0.0
                episodes += 1
                if episodes >= args.episodes:
                    break
                continue
            if args.save_video:
                front, wrist = _capture_video_frames(env)
                front_frames.append(front)
                wrist_frames.append(wrist)
            insertion = _insertion_metrics(env)
            target = env.unwrapped._pi0_target[0].detach().cpu().tolist()
            state = env.unwrapped._current_pi0_state()[0].detach().cpu().tolist()
            fingertip_pos = env.unwrapped.fingertip_midpoint_pos[0].detach().cpu().numpy()
            hole_tip_pos = env.unwrapped.fixed_pos_obs_frame[0].detach().cpu().numpy()
            tip_to_hole = hole_tip_pos - fingertip_pos
            rows.append({
                "step": step,
                "ppo_reward": reward_value,
                "reward": reward_value,
                "cumulative_ppo_reward": float(sum(ppo_rewards)),
                "reward_terms": current_reward_terms,
                "pi0_target": target,
                "pi0_raw_chunk": np.asarray(
                    getattr(env.unwrapped, "_pi0_last_chunk", np.empty((0, 8))),
                    dtype=np.float32,
                ).tolist(),
                "pi0_chunk_id": int(getattr(env.unwrapped, "_pi0_last_chunk_id", -1)),
                "pi0_chunk_action_index": int(
                    getattr(env.unwrapped, "_pi0_last_chunk_action_index", -1)
                ),
                "pi0_desired_q": (
                    env.unwrapped._pi0_desired_q[0].detach().cpu().tolist()
                    if getattr(env.unwrapped, "_pi0_desired_q", None) is not None
                    else None
                ),
                "pi0_command_q": (
                    env.unwrapped._pi0_command_q[0].detach().cpu().tolist()
                    if getattr(env.unwrapped, "_pi0_command_q", None) is not None
                    else None
                ),
                "state8": state,
                "fingertip_pos": fingertip_pos.tolist(),
                "hole_tip_pos": hole_tip_pos.tolist(),
                "tip_to_hole_vector": tip_to_hole.tolist(),
                "tip_to_hole_distance_m": float(np.linalg.norm(tip_to_hole)),
                **_held_asset_diagnostics(env),
                **insertion,
                "pi0_inference_count": pi0_inference_count_peak,
                "pi0_failures": pi0_failures_peak,
                "pi0_last_latency_s": env.unwrapped.pi0_last_latency_s,
                "direct_reset_q_real": (
                    getattr(env.unwrapped, "pi0_direct_reset_q_real", None).tolist()
                    if hasattr(env.unwrapped, "pi0_direct_reset_q_real")
                    else None
                ),
                "direct_reset_q_sim": getattr(env.unwrapped, "pi0_direct_reset_q_sim", None),
                "direct_reset_q_error": getattr(env.unwrapped, "pi0_direct_reset_q_error", None),
                "direct_reset_h5": str(
                    getattr(env.unwrapped, "pi0_direct_reset_h5", "")
                ),
            })
            steps_file.write(json.dumps(rows[-1], ensure_ascii=False) + "\n")
            steps_file.flush()
            progress.set_postfix(
                infer=pi0_inference_count_peak,
                latency=f"{env.unwrapped.pi0_last_latency_s:.3f}s",
                failures=pi0_failures_peak,
                frames=len(front_frames) if args.save_video else 0,
            )
        progress.close()
        if args.save_video and len(front_frames) > 1:
            fps = max(1.0, float(args.video_fps))
            try:
                _write_rgb_video(args.output_dir / "front.mp4", front_frames, fps)
                _write_rgb_video(args.output_dir / "wrist.mp4", wrist_frames, fps)
                _write_reward_video(
                    args.output_dir / "front_reward.mp4",
                    front_frames,
                    ppo_rewards,
                    reward_terms,
                    rows,
                    fps,
                )
                _write_reward_video(
                    args.output_dir / "wrist_reward.mp4",
                    wrist_frames,
                    ppo_rewards,
                    reward_terms,
                    rows,
                    fps,
                )
            except Exception as exc:
                video_error = str(exc)
                print(f"Video export failed: {video_error}")
        reward_array = np.asarray(ppo_rewards, dtype=np.float64)
        task_success = task_success_detected or (
            bool(env.unwrapped.ep_succeeded[0].detach().cpu())
            if hasattr(env.unwrapped, "ep_succeeded")
            else False
        )
        metric_rows = [row for row in rows if row.get("xy_error_m") is not None]
        best_xy_row = min(metric_rows, key=lambda row: row["xy_error_m"], default=None)
        minimum_z_row = min(metric_rows, key=lambda row: row["z_disp_m"], default=None)
        successful_rows = [row for row in metric_rows if row["geometric_success"]]
        final_metric_row = metric_rows[-1] if metric_rows else None
        insertion_summary = {
            "geometric_success": bool(successful_rows) or task_success_detected,
            "first_success_step": (
                successful_rows[0]["step"]
                if successful_rows
                else first_success_step_detected
            ),
            "successful_step_count": max(len(successful_rows), int(task_success_detected)),
            "best_xy_error_m": best_xy_row["xy_error_m"] if best_xy_row else None,
            "best_xy_step": best_xy_row["step"] if best_xy_row else None,
            "minimum_z_disp_m": minimum_z_row["z_disp_m"] if minimum_z_row else None,
            "minimum_z_step": minimum_z_row["step"] if minimum_z_row else None,
            "final_xy_error_m": final_metric_row["xy_error_m"] if final_metric_row else None,
            "final_z_disp_m": final_metric_row["z_disp_m"] if final_metric_row else None,
            "success_xy_threshold_m": float(env.unwrapped.cfg_task.success_xy_threshold),
            "success_z_threshold_m": float(
                env.unwrapped.cfg_task.fixed_asset_cfg.height
                * env.unwrapped.cfg_task.success_threshold
            ),
        }
        report = {
            "task": args.task,
            "pi0_host": args.pi0_host,
            "pi0_port": args.pi0_port,
            "pi0_timeout_s": args.pi0_timeout_s,
            "steps": executed_steps,
            "recorded_non_reset_steps": len(rows),
            "episodes": episodes,
            "action_horizon": PI0_ACTION_HORIZON,
            "control_hz": 10.0,
            "replan_actions": args.replan_actions,
            "action_start_index": args.action_start_index,
            "control_mode": args.control_mode,
            "direct_reset_h5_list": (
                [str(path) for path in args.direct_reset_h5_list]
                if args.direct_reset_h5_list is not None
                else None
            ),
            "pi0_inference_count": pi0_inference_count_peak,
            "pi0_failures": pi0_failures_peak,
            "pi0_last_latency_s": env.unwrapped.pi0_last_latency_s,
            "pi0_target_clipped": env.unwrapped.pi0_target_clipped,
            "controller_config": {
                "use_implicit_position_controller": bool(
                    env.unwrapped._pi0_cfg.use_implicit_position_controller
                ),
                "implicit_arm_stiffness": float(
                    env.unwrapped._pi0_cfg.implicit_arm_stiffness
                ),
                "implicit_arm_damping": float(
                    env.unwrapped._pi0_cfg.implicit_arm_damping
                ),
                "joint_velocity_limits": [
                    float(value) for value in env.unwrapped._pi0_cfg.joint_velocity_limits
                ],
                "joint_kp": [float(value) for value in env.unwrapped._pi0_cfg.joint_kp],
                "joint_kd": [float(value) for value in env.unwrapped._pi0_cfg.joint_kd],
            },
            "pi0_last_error": env.unwrapped.pi0_last_error,
            "direct_joint_reset": {
                "enabled": hasattr(env.unwrapped, "pi0_direct_reset_q_real"),
                "h5": str(getattr(env.unwrapped, "pi0_direct_reset_h5", "")),
                "q_real": (
                    env.unwrapped.pi0_direct_reset_q_real.tolist()
                    if hasattr(env.unwrapped, "pi0_direct_reset_q_real")
                    else None
                ),
                "q_sim": getattr(env.unwrapped, "pi0_direct_reset_q_sim", None),
                "q_error": getattr(env.unwrapped, "pi0_direct_reset_q_error", None),
                "q_sim_after_reset": getattr(
                    env.unwrapped, "pi0_direct_reset_q_sim_after_reset", None
                ),
                "q_error_after_reset": getattr(
                    env.unwrapped, "pi0_direct_reset_q_error_after_reset", None
                ),
                "state8_after_reset": reset_state8,
                "pi0_target_after_reset": reset_target,
                "max_abs_error_rad": (
                    float(
                        np.max(
                            np.abs(
                                np.asarray(
                                    getattr(
                                        env.unwrapped,
                                        "pi0_direct_reset_q_error_after_reset",
                                        None,
                                    )
                                    or env.unwrapped.pi0_direct_reset_q_error
                                )
                            )
                        )
                    )
                    if getattr(env.unwrapped, "pi0_direct_reset_q_error", None) is not None
                    else None
                ),
            },
            "task_success_latched": task_success,
            "insertion": insertion_summary,
            "gt_reference": gt_reference,
            "ppo_reward": {
                "steps": int(reward_array.size),
                "return": float(reward_array.sum()) if reward_array.size else 0.0,
                "mean": float(reward_array.mean()) if reward_array.size else 0.0,
                "min": float(reward_array.min()) if reward_array.size else 0.0,
                "max": float(reward_array.max()) if reward_array.size else 0.0,
                "last": float(reward_array[-1]) if reward_array.size else 0.0,
                "completed_episode_returns": episode_returns,
                "open_episode_return": float(episode_return),
            },
            "video": {
                "enabled": bool(args.save_video),
                "fps": float(args.video_fps),
                "frames": len(front_frames),
                "front": str(args.output_dir / "front.mp4") if args.save_video and not video_error else "",
                "wrist": str(args.output_dir / "wrist.mp4") if args.save_video and not video_error else "",
                "front_reward": str(args.output_dir / "front_reward.mp4") if args.save_video and not video_error else "",
                "wrist_reward": str(args.output_dir / "wrist_reward.mp4") if args.save_video and not video_error else "",
                "error": video_error,
            },
        }
        (args.output_dir / "report.json").write_text(
            json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        with (args.output_dir / "ppo_reward.csv").open("w", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerow(("step", "ppo_reward", "cumulative_ppo_reward"))
            cumulative = 0.0
            for index, value in enumerate(ppo_rewards):
                cumulative += value
                writer.writerow((index, value, cumulative))
        with (args.output_dir / "insertion_metrics.csv").open(
            "w", newline="", encoding="utf-8"
        ) as file:
            writer = csv.writer(file)
            writer.writerow(("step", "xy_error_m", "z_disp_m", "geometric_success"))
            for row in rows:
                writer.writerow(
                    (
                        row["step"],
                        row["xy_error_m"],
                        row["z_disp_m"],
                        row["geometric_success"],
                    )
                )
        print(json.dumps(report, indent=2, ensure_ascii=False))
    finally:
        if steps_file is not None:
            steps_file.close()
        if env is not None:
            env.close()
        simulation_app.close()


if __name__ == "__main__":
    main()
