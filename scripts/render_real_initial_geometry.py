"""Render a real-data-initialized peg-in-hole scene without PPO initialization.

The final visible scene is constructed from the first and last real joint
states.  The simulator computes the fingertip and held-peg root from its own
geometry, so the fixed hole root is inferred with the peg height and Franka
fingerpad offset included.  No PPO checkpoint or TAVLA server is used.
"""

from __future__ import annotations

import argparse
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
parser.add_argument(
    "--output-dir",
    type=Path,
    default=Path("outputs/real_initial_geometry_scene"),
)
parser.add_argument("--final-index", type=int, default=-1)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

simulation_app = AppLauncher(args).app

import isaacsim.core.utils.torch as torch_utils  # noqa: E402
import tacex_tasks  # noqa: E402,F401
from isaaclab.utils.math import convert_camera_frame_orientation_convention  # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402
from tacex_tasks.real2sim.realsim_env import RealSimEnv  # noqa: E402


def load_real_data(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with h5py.File(path, "r") as h5:
        q = np.asarray(h5["obs/state/joint_pos"][:], dtype=np.float32)
        pose = np.asarray(h5["obs/state/ee_pose"][:], dtype=np.float64)
        timestamps = np.asarray(h5["timestamps"][:], dtype=np.float64)
        if "action/actual/gripper" in h5:
            gripper = np.asarray(h5["action/actual/gripper"][:], dtype=np.float32).reshape(-1)
        else:
            gripper = np.full((len(q),), 0.15, dtype=np.float32)
    if q.ndim != 2 or q.shape[1] != 7:
        raise ValueError(f"Expected joint_pos shape (N,7), got {q.shape}")
    if pose.shape != (len(q), 7) or timestamps.shape != (len(q),):
        raise ValueError("Real q, ee_pose, and timestamps have different lengths")
    if gripper.shape != (len(q),):
        raise ValueError("Real gripper length does not match q")
    if not all(np.isfinite(value).all() for value in (q, pose, timestamps, gripper)):
        raise ValueError("Real trajectory contains NaN or Inf")
    return q, np.clip(gripper, 0.0, 1.0), pose


class GeometryRenderEnv(RealSimEnv):
    """Disable task-space actions; this script only writes measured qpos."""

    def __init__(self, cfg, **kwargs):
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

    def _apply_action(self) -> None:
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

def camera_frame_to_bgr(camera_tensor: torch.Tensor) -> np.ndarray:
    frame = camera_tensor.detach().cpu().numpy()
    if frame.dtype != np.uint8:
        if frame.size and float(frame.max()) <= 1.0:
            frame = frame * 255.0
        frame = np.clip(frame, 0.0, 255.0).astype(np.uint8)
    if frame.ndim == 3 and frame.shape[0] in (3, 4) and frame.shape[-1] not in (3, 4):
        frame = np.transpose(frame, (1, 2, 0))
    if frame.ndim != 3 or frame.shape[-1] < 3:
        raise ValueError(f"Expected camera frame HxWx3/4, got {frame.shape}")
    return cv2.cvtColor(np.ascontiguousarray(frame[:, :, :3]), cv2.COLOR_RGB2BGR)


def set_robot_q(env: RealSimEnv, q: np.ndarray, gripper: float) -> None:
    q_tensor = torch.as_tensor(q, dtype=torch.float32, device=env.device).view(1, 7)
    full_q = env.joint_pos.clone()
    full_q[:, :7] = q_tensor
    full_q[:, 7:9] = float(gripper) * 0.04
    zero_velocity = torch.zeros_like(full_q)
    env._robot.write_joint_state_to_sim(full_q, zero_velocity)
    env.ctrl_target_joint_pos[:] = full_q
    env._robot.set_joint_position_target(full_q)
    env._robot.set_joint_effort_target(torch.zeros_like(full_q))
    env.scene.write_data_to_sim()
    env.sim.forward()
    env.scene.update(dt=env.physics_dt)
    env._compute_intermediate_values(dt=env.physics_dt)


def compute_held_peg_pose(env: RealSimEnv) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the held peg root pose from the current fingertip pose."""
    flip_z_quat = torch.tensor([0.0, 0.0, 1.0, 0.0], device=env.device).view(1, 4)
    zero_pos = torch.zeros((1, 3), device=env.device)
    flipped_quat, flipped_pos = torch_utils.tf_combine(
        env.fingertip_midpoint_quat,
        env.fingertip_midpoint_pos,
        flip_z_quat,
        zero_pos,
    )
    relative_pos, relative_quat = env.get_handheld_asset_relative_pose()
    asset_in_hand_quat, asset_in_hand_pos = torch_utils.tf_inverse(relative_quat, relative_pos)
    held_quat, held_pos = torch_utils.tf_combine(
        flipped_quat,
        flipped_pos,
        asset_in_hand_quat,
        asset_in_hand_pos,
    )
    return held_quat, held_pos


def write_asset_pose(asset, pose_pos: torch.Tensor, pose_quat: torch.Tensor) -> None:
    state = asset.data.root_state_w.clone()
    state[:, :3] = pose_pos
    state[:, 3:7] = pose_quat
    state[:, 7:] = 0.0
    asset.write_root_pose_to_sim(state[:, :7])
    asset.write_root_velocity_to_sim(state[:, 7:])
    asset.reset()


def update_wrist_camera_pose(env: RealSimEnv) -> None:
    offset = env.cfg.wrist_camera.offset
    offset_pos = torch.as_tensor(offset.pos, dtype=torch.float32, device=env.device).view(1, 3)
    offset_rot = torch.as_tensor(offset.rot, dtype=torch.float32, device=env.device).view(1, 4)
    offset_rot = convert_camera_frame_orientation_convention(
        offset_rot, origin=offset.convention, target="opengl"
    )
    camera_quat, camera_pos = torch_utils.tf_combine(
        env.fingertip_midpoint_quat[0:1],
        env.fingertip_midpoint_pos[0:1],
        offset_rot,
        offset_pos,
    )
    env.wrist_tiled_camera.set_world_poses(camera_pos, camera_quat, env_ids=[0], convention="opengl")


def pose_numpy(env: RealSimEnv) -> np.ndarray:
    return torch.cat(
        (env.fingertip_midpoint_pos[0], env.fingertip_midpoint_quat[0]), dim=0
    ).detach().cpu().numpy().astype(np.float64)


def main() -> None:
    q, gripper, real_pose = load_real_data(args.h5)
    final_index = args.final_index if args.final_index >= 0 else len(q) - 1
    if final_index >= len(q):
        raise ValueError(f"final-index {final_index} is outside trajectory with {len(q)} frames")

    env_cfg = parse_env_cfg(args.task, device=args.device, num_envs=1, use_fabric=True)
    env_cfg.scene.num_envs = 1
    env_cfg.policy_cfg = None
    env_cfg.enable_cameras = True
    env_cfg.data_collect_cfg["collect_data"] = False
    env_cfg.data_collect_cfg["immediate_stop"] = False
    env_cfg.data_collect_cfg["save_failed_trajectory"] = False
    env_cfg.data_collect_cfg["num_trajectories"] = 1_000_000
    env_cfg.episode_length_s = 30.0
    env_cfg.ctrl.reset_joints = q[0].tolist()
    env_cfg.task.fixed_asset_init_pos_noise = [0.0, 0.0, 0.0]
    env_cfg.task.hand_init_pos_noise = [0.0, 0.0, 0.0]
    env_cfg.task.hand_init_orn_noise = [0.0, 0.0, 0.0]
    env_cfg.task.held_asset_pos_noise = [0.0, 0.0, 0.0]
    env_cfg.task.fixed_asset_init_orn_range_deg = 0.0

    args.output_dir.mkdir(parents=True, exist_ok=True)
    env = None
    try:
        print("[GeometryRender] creating environment", flush=True)
        env = GeometryRenderEnv(
            env_cfg,
            render_mode="rgb_array",
            output_dir=str(args.output_dir),
        )
        # This reset only creates the USD scene.  The visible final scene is
        # overwritten below from real q[0] and q[final_index].
        env.reset()
        print("[GeometryRender] reset returned", flush=True)

        set_robot_q(env, q[0], gripper[0])
        first_sim_pose = pose_numpy(env)
        first_held_quat, first_held_pos = compute_held_peg_pose(env)

        set_robot_q(env, q[final_index], gripper[final_index])
        final_sim_pose = pose_numpy(env)
        target_held_quat, target_hole_root = compute_held_peg_pose(env)

        # For peg_insert, the effective peg base and the fixed hole root share
        # the same geometric origin.  This includes peg height (0.050 m) and
        # fingerpad offset (0.017608 m) through get_handheld_asset_relative_pose().
        identity_quat = torch.tensor([1.0, 0.0, 0.0, 0.0], device=env.device).view(1, 4)
        write_asset_pose(env._fixed_asset, target_hole_root, identity_quat)

        # Return the robot to the real first-frame state and place the peg in
        # the gripper at that state.  The final hole remains fixed at the pose
        # inferred from the real final frame.
        set_robot_q(env, q[0], gripper[0])
        first_held_quat, first_held_pos = compute_held_peg_pose(env)
        write_asset_pose(env._held_asset, first_held_pos, first_held_quat)
        env.scene.write_data_to_sim()
        env.sim.forward()
        env.scene.update(dt=env.physics_dt)
        env._compute_intermediate_values(dt=env.physics_dt)

        update_wrist_camera_pose(env)
        env.sim.render()
        front = camera_frame_to_bgr(env.tiled_camera.data.output["rgb"][0])
        wrist = camera_frame_to_bgr(env.wrist_tiled_camera.data.output["rgb"][0])
        cv2.imwrite(str(args.output_dir / "sim_front_initial_real_state.png"), front)
        cv2.imwrite(str(args.output_dir / "sim_wrist_initial_real_state.png"), wrist)

        # Include a compact report so the inferred geometry is auditable.
        peg_height = float(env_cfg.task.held_asset_cfg.height)
        hole_height = float(env_cfg.task.fixed_asset_cfg.height)
        fingerpad_length = float(env_cfg.task.robot_cfg.franka_fingerpad_length)
        report = {
            "ppo_reset_used_for_final_visible_scene": False,
            "tavla_server_used": False,
            "robot_base_pose": {
                "position_m": [0.0, 0.0, 0.0],
                "quaternion_wxyz": [1.0, 0.0, 0.0, 0.0],
            },
            "real_source": str(args.h5.resolve()),
            "first_frame": int(0),
            "final_frame": int(final_index),
            "real_first_q": q[0].tolist(),
            "real_final_q": q[final_index].tolist(),
            "real_first_ee_pose_xyz_quat": real_pose[0].tolist(),
            "real_final_ee_pose_xyz_quat": real_pose[final_index].tolist(),
            "sim_first_fingertip_pose_xyz_quat_wxyz": first_sim_pose.tolist(),
            "sim_final_fingertip_pose_xyz_quat_wxyz": final_sim_pose.tolist(),
            "inferred_fixed_hole_root_m": target_hole_root[0].detach().cpu().tolist(),
            "initial_held_peg_root_m": first_held_pos[0].detach().cpu().tolist(),
            "geometry": {
                "peg_height_m": peg_height,
                "hole_height_m": hole_height,
                "franka_fingerpad_length_m": fingerpad_length,
                "peg_relative_z_from_fingertip_m": peg_height - fingerpad_length,
                "hole_root_semantics": "same effective geometric base as held peg for peg_insert",
            },
            "note": "The final hole root is inferred from simulator FK of the real final joint state; no PPO reset pose is used in the saved image.",
        }
        (args.output_dir / "scene_report.json").write_text(
            json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        print(json.dumps(report, indent=2, ensure_ascii=False))
        print(f"[OK] wrote {args.output_dir.resolve()}")
    finally:
        if env is not None:
            env.close()
        simulation_app.close()


if __name__ == "__main__":
    main()

