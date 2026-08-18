
"""Replay real EE XYZ with the original PPO Franka articulation and cameras.

This entry point intentionally uses the same RealSimEnv scene as PPO:
``/franka_env/Robot/franka`` is the controlled articulation, the existing
front camera is kept, and the wrist camera is the PPO camera attached to
panda_hand.  The real XYZ is mapped to sim as
``(-real_y, -real_x, real_z)``.  Every target is solved with PPO's DLS IK,
which writes the complete 7-joint articulation state.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import cv2
import h5py
import numpy as np
import torch
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--h5", type=Path, default=Path("real_data/traj_0/data.h5"))
parser.add_argument("--task", type=str, default="TacEx-RealSim-PegInsert-Direct-v0")
parser.add_argument("--output-dir", type=Path, default=Path("outputs/traj0_ppo_cartesian_rollout"))
parser.add_argument("--endpoint-window", type=int, default=3)
parser.add_argument("--fps", type=float, default=10.0)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
# This script always captures the PPO front and wrist cameras.  The old
# working replay enables the Isaac Sim camera extensions before AppLauncher;
# setting only cfg.enable_cameras is too late and can make startup terminate
# after simulation reset without a Python traceback.
if hasattr(args, "enable_cameras"):
    args.enable_cameras = True

simulation_app = AppLauncher(args).app

import isaacsim.core.utils.torch as torch_utils  # noqa: E402
import tacex_tasks  # noqa: E402,F401
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402
from tacex_tasks.real2sim.realsim_env import RealSimEnv, _write_h264_mp4  # noqa: E402

class PPOReplayEnv(RealSimEnv):
    """PPO-compatible bootstrap with one complete Franka articulation."""

    def __init__(self, cfg, **kwargs):
        # DirectRLEnv can invoke _apply_action while the parent is building
        # the scene. Prepare only the whole-articulation position target;
        # no USD link is ever moved directly.
        # FactoryEnv defaults to collection mode during parent construction;
        # set the replay lifecycle before super() so bootstrap cannot exit.
        self.collect_data = False
        self.immediate_stop = False
        self.save_failed_trajectory = False
        self.num_trajectories = 1_000_000
        self.cur_num_traj = 0
        self._replay_gripper_open_width_m = 0.04
        self._gripper_force_n = 5.0
        self._couple_contact_reaction = False
        self._physical_load_cell = False
        self._kinematic_held_asset = False
        self._grasp_constraint = False
        self._replay_target_q = None
        self._replay_target_gripper = 0.0
        super().__init__(cfg, **kwargs)
        self._replay_kp = torch.tensor(
            [80.0, 80.0, 80.0, 80.0, 50.0, 30.0, 20.0],
            dtype=torch.float32,
            device=self.device,
        ).view(1, 7)
        self._replay_kd = torch.tensor(
            [18.0, 18.0, 18.0, 18.0, 10.0, 6.0, 4.0],
            dtype=torch.float32,
            device=self.device,
        ).view(1, 7)
        self._replay_effort_limits = torch.tensor(
            [87.0, 87.0, 87.0, 87.0, 12.0, 12.0, 12.0],
            dtype=torch.float32,
            device=self.device,
        ).view(1, 7)

    def _setup_scene(self):
        # Keep the PPO scene construction path explicit. The background USD
        # and the Articulation use the same /Robot/franka prim path.
        super()._setup_scene()

    def _prepare_robot_usd_for_replay(self) -> None:
        # PPO coordinate replay has no extra rigid tool authoring. This hook
        # exists only to keep scene bootstrap ordering identical to replay's
        # proven articulation path.
        return

    def _apply_action(self) -> None:
        # Match the PPO geometry replay bootstrap: a complete articulation
        # target is applied, never an individual USD link transform.
        required = ("_robot", "joint_pos", "ctrl_target_joint_pos", "_replay_kp")
        if not all(hasattr(self, name) for name in required):
            return
        target_q = self.joint_pos[:, :7] if self._replay_target_q is None else self._replay_target_q
        q_error = (target_q - self.joint_pos[:, :7] + torch.pi) % (2.0 * torch.pi) - torch.pi
        arm_torque = torch.clamp(
            self._replay_kp * q_error - self._replay_kd * self.joint_vel[:, :7],
            -self._replay_effort_limits,
            self._replay_effort_limits,
        )
        self.ctrl_target_joint_pos[:, :7] = target_q
        self._robot.set_joint_position_target(self.ctrl_target_joint_pos)
        full_torque = torch.zeros_like(self.joint_pos)
        full_torque[:, :7] = arm_torque
        self._robot.set_joint_effort_target(full_torque)

def load_real_data(path: Path):
    with h5py.File(path, "r") as h5:
        q = np.asarray(h5["obs/state/joint_pos"][:], dtype=np.float32)
        ee_pose = np.asarray(h5["obs/state/ee_pose"][:], dtype=np.float32)
        timestamps = np.asarray(h5["timestamps"][:], dtype=np.float64)
    if q.ndim != 2 or q.shape[1] != 7:
        raise ValueError(f"joint_pos must be (N,7), got {q.shape}")
    if ee_pose.ndim != 2 or ee_pose.shape[0] != len(q) or ee_pose.shape[1] != 7:
        raise ValueError(f"ee_pose must be (N,7), got {ee_pose.shape}")
    if timestamps.shape != (len(q),):
        raise ValueError("timestamps and trajectory length differ")
    if not all(np.isfinite(x).all() for x in (q, ee_pose, timestamps)):
        raise ValueError("real trajectory contains NaN or Inf")
    return q, ee_pose, timestamps

def transform_real_xyz(ee_pose: np.ndarray) -> np.ndarray:
    real_xyz = np.asarray(ee_pose[:, :3], dtype=np.float32)
    return np.stack((-real_xyz[:, 1], -real_xyz[:, 0], real_xyz[:, 2]), axis=1)

def camera_to_bgr(value: torch.Tensor) -> np.ndarray:
    frame = value.detach().cpu().numpy()
    if frame.dtype != np.uint8:
        if frame.size and float(frame.max()) <= 1.0:
            frame = frame * 255.0
        frame = np.clip(frame, 0.0, 255.0).astype(np.uint8)
    if frame.ndim == 3 and frame.shape[0] in (3, 4) and frame.shape[-1] not in (3, 4):
        frame = np.transpose(frame, (1, 2, 0))
    if frame.ndim != 3 or frame.shape[-1] < 3:
        raise ValueError(f"camera output must be HxWx3/4, got {frame.shape}")
    return cv2.cvtColor(np.ascontiguousarray(frame[:, :, :3]), cv2.COLOR_RGB2BGR)

def write_video(path: Path, frames: list[np.ndarray], fps: float) -> None:
    # Use the same FFmpeg/libx264 writer as the PPO/RealSim pipeline.
    # OpenCV's mp4v output is MPEG-4 Part 2, not H.264.
    _write_h264_mp4(path, frames, fps)

def write_asset_pose(env: RealSimEnv, asset, pos: torch.Tensor, quat: torch.Tensor) -> None:
    state = asset.data.root_state_w.clone()
    state[:, :3] = pos + env.scene.env_origins
    state[:, 3:7] = quat
    state[:, 7:] = 0.0
    asset.write_root_pose_to_sim(state[:, :7])
    asset.write_root_velocity_to_sim(state[:, 7:])
    asset.reset()

def held_peg_pose(env: RealSimEnv) -> tuple[torch.Tensor, torch.Tensor]:
    flip_z = torch.tensor([0.0, 0.0, 1.0, 0.0], device=env.device).view(1, 4)
    zero = torch.zeros((1, 3), device=env.device)
    flipped_quat, flipped_pos = torch_utils.tf_combine(
        env.fingertip_midpoint_quat, env.fingertip_midpoint_pos, flip_z, zero
    )
    relative_pos, relative_quat = env.get_handheld_asset_relative_pose()
    asset_in_hand_quat, asset_in_hand_pos = torch_utils.tf_inverse(relative_quat, relative_pos)
    return torch_utils.tf_combine(
        flipped_quat, flipped_pos, asset_in_hand_quat, asset_in_hand_pos
    )

def ppo_hand_down_quat(env: RealSimEnv) -> torch.Tensor:
    euler = torch.as_tensor(env.cfg.task.hand_init_orn, dtype=torch.float32, device=env.device)
    euler = euler.view(1, 3)
    return torch_utils.quat_from_euler_xyz(euler[:, 0], euler[:, 1], euler[:, 2])

def move_ee_with_ppo_ik(env: RealSimEnv, xyz: np.ndarray, quat: torch.Tensor) -> float:
    env._compute_intermediate_values(dt=env.physics_dt)
    target_pos = torch.as_tensor(xyz, dtype=torch.float32, device=env.device).view(1, 3)
    env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.long)
    pos_error, rot_error = env.set_pos_inverse_kinematics(
        ctrl_target_fingertip_midpoint_pos=target_pos,
        ctrl_target_fingertip_midpoint_quat=quat,
        env_ids=env_ids,
    )
    env._compute_intermediate_values(dt=env.physics_dt)
    return float(torch.linalg.norm(pos_error, dim=1).max().detach().cpu()) + float(
        torch.linalg.norm(rot_error, dim=1).max().detach().cpu()
    )

def set_direct_grasp(env: RealSimEnv) -> None:
    clamp_q = float(env.cfg.task.held_asset_cfg.diameter) / 2.0
    full_q = env.joint_pos.clone()
    full_q[:, 7:9] = clamp_q
    zero_velocity = torch.zeros_like(full_q)
    env._robot.write_joint_state_to_sim(full_q, zero_velocity)
    env.ctrl_target_joint_pos[:] = full_q
    env._robot.set_joint_position_target(full_q)
    env._robot.set_joint_effort_target(torch.zeros_like(full_q))
    env.scene.write_data_to_sim()
    env.sim.forward()
    env.scene.update(dt=env.physics_dt)
    env._compute_intermediate_values(dt=env.physics_dt)

def place_peg(env: RealSimEnv) -> None:
    peg_quat, peg_pos = held_peg_pose(env)
    write_asset_pose(env, env._held_asset, peg_pos, peg_quat)
    env.scene.write_data_to_sim()
    env.sim.forward()
    env.scene.update(dt=env.physics_dt)
    env._compute_intermediate_values(dt=env.physics_dt)

def capture(env: RealSimEnv) -> tuple[np.ndarray, np.ndarray]:
    update_wrist = getattr(env, "_update_wrist_camera_pose", None)
    if update_wrist is not None:
        update_wrist()
    env.sim.render()
    for camera in (env.tiled_camera, env.wrist_tiled_camera):
        camera.update(env.physics_dt, force_recompute=True)
    return (
        camera_to_bgr(env.tiled_camera.data.output["rgb"][0]),
        camera_to_bgr(env.wrist_tiled_camera.data.output["rgb"][0]),
    )

def main() -> None:
    q, ee_pose, timestamps = load_real_data(args.h5)
    sim_xyz = transform_real_xyz(ee_pose)
    if args.endpoint_window < 1:
        raise ValueError("endpoint-window must be positive")
    window = min(args.endpoint_window, len(sim_xyz))

    # Match the PPO/old replay bootstrap.  The scene and camera sensors were
    # authored for the Fabric path; the non-Fabric path can leave the physics
    # scene stepping subscription invalid during startup.
    cfg = parse_env_cfg(args.task, device=args.device, num_envs=1, use_fabric=True)
    cfg.scene.num_envs = 1
    cfg.policy_cfg = None
    cfg.teacher_policy_cfg = None
    cfg.enable_cameras = True
    cfg.data_collect_cfg["collect_data"] = False
    cfg.data_collect_cfg["immediate_stop"] = False
    cfg.data_collect_cfg["save_failed_trajectory"] = False
    cfg.data_collect_cfg["num_trajectories"] = 1_000_000
    cfg.teacher_eval_only = True
    cfg.episode_length_s = max(30.0, float(timestamps[-1] - timestamps[0] + 2.0))
    cfg.ctrl.reset_joints = q[0].tolist()
    cfg.task.fixed_asset_init_pos_noise = [0.0, 0.0, 0.0]
    cfg.task.hand_init_pos_noise = [0.0, 0.0, 0.0]
    cfg.task.hand_init_orn_noise = [0.0, 0.0, 0.0]
    cfg.task.held_asset_pos_noise = [0.0, 0.0, 0.0]
    cfg.task.fixed_asset_init_orn_range_deg = 0.0
    cfg.sim.render_interval = 4

    args.output_dir.mkdir(parents=True, exist_ok=True)
    env = PPOReplayEnv(cfg, render_mode="rgb_array", output_dir=str(args.output_dir))
    try:
        env.reset()
        hand_quat = ppo_hand_down_quat(env)

        # First establish the PPO-consistent real first-frame arm pose.
        init_error = move_ee_with_ppo_ik(env, sim_xyz[0], hand_quat)
        set_direct_grasp(env)
        place_peg(env)

        # Visit the last real frames with the same PPO IK and infer one
        # fixed hole root from the resulting held-peg roots.
        endpoint_roots = []
        for xyz in sim_xyz[-window:]:
            move_ee_with_ppo_ik(env, xyz, hand_quat)
            _, peg_pos = held_peg_pose(env)
            endpoint_roots.append(peg_pos[0].detach().cpu().numpy())
        hole_pos = np.median(np.asarray(endpoint_roots), axis=0)
        identity = torch.tensor([1.0, 0.0, 0.0, 0.0], device=env.device).view(1, 4)
        hole_pos_t = torch.as_tensor(hole_pos, dtype=torch.float32, device=env.device).view(1, 3)
        write_asset_pose(env, env._fixed_asset, hole_pos_t, identity)

        # Return to the real first frame, clamp the peg, and record the
        # complete coordinate replay. No per-link USD transform is used.
        move_ee_with_ppo_ik(env, sim_xyz[0], hand_quat)
        set_direct_grasp(env)
        place_peg(env)
        front_frames, wrist_frames = [], []
        sim_ee = []
        for index, xyz in enumerate(sim_xyz):
            error = move_ee_with_ppo_ik(env, xyz, hand_quat)
            place_peg(env)
            front, wrist = capture(env)
            front_frames.append(front)
            wrist_frames.append(wrist)
            sim_ee.append(
                np.concatenate((env.fingertip_midpoint_pos[0].detach().cpu().numpy(), hand_quat[0].detach().cpu().numpy()))
            )
            if index == 0 or index + 1 == len(sim_xyz) or (index + 1) % 10 == 0:
                print(f"[Replay] frame {index + 1}/{len(sim_xyz)} error={error:.6f}", flush=True)

        write_video(args.output_dir / "front_camera.mp4", front_frames, args.fps)
        write_video(args.output_dir / "wrist_camera.mp4", wrist_frames, args.fps)
        with (args.output_dir / "ee_pose_sim.csv").open("w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["x", "y", "z", "qw", "qx", "qy", "qz"])
            writer.writerows(np.asarray(sim_ee).tolist())
        metadata = {
            "controller": "RealSimEnv PPO DLS IK",
            "robot_articulation_path": "/World/envs/env_.*/franka_env/Robot/franka",
            "base_pose_unchanged": True,
            "real_to_sim_xyz": "sim=(-real_y,-real_x,real_z)",
            "first_frame_ik_error": init_error,
            "endpoint_window": window,
            "hole_position_sim_m": hole_pos.tolist(),
            "peg_diameter_m": float(cfg.task.held_asset_cfg.diameter),
            "hole_diameter_m": float(cfg.task.fixed_asset_cfg.diameter),
            "frame_count": len(sim_xyz),
            "fps": float(args.fps),
        }
        (args.output_dir / "replay_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        print(f"[DONE] front: {args.output_dir / 'front_camera.mp4'}", flush=True)
        print(f"[DONE] wrist: {args.output_dir / 'wrist_camera.mp4'}", flush=True)
    finally:
        env.close()

try:
    main()
finally:
    simulation_app.close()
