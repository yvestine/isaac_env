# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
import numpy as np
import torch

import isaacsim.core.utils.torch as torch_utils

from isaaclab.utils.math import axis_angle_from_quat

from isaaclab_tasks.direct.factory import factory_utils
from isaaclab_tasks.direct.factory.factory_env import FactoryEnv

from . import utils
from .realsim_env_cfg import RealSimEnvCfg


class ForgeEnv(FactoryEnv):
    cfg: RealSimEnvCfg

    def __init__(self, cfg: RealSimEnvCfg, render_mode: str | None = None, **kwargs):
        """Initialize additional randomization and logging tensors."""
        super().__init__(cfg, render_mode, **kwargs)

        # Success prediction.
        self.success_pred_scale = 0.0
        self.first_pred_success_tx = {}
        for thresh in [0.5, 0.6, 0.7, 0.8, 0.9]:
            self.first_pred_success_tx[thresh] = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)

        # Flip quaternions.
        self.flip_quats = torch.ones((self.num_envs,), dtype=torch.float32, device=self.device)

        # Force sensor information. Keep every stage of the wrench pipeline
        # available for calibration and comparison. The defaults match the
        # IsaacLab 2.x body_incoming_joint_wrench_b contract; all alternatives
        # remain explicit for backend/version checks.
        self.force_sensor_body_idx = self._robot.body_names.index("force_sensor")
        # The load-cell wrench is read on the fixed-joint child link. Replay
        # overrides this to panda_hand when the rigid tool is below that joint.
        self._incoming_wrench_body_idx = self.force_sensor_body_idx
        if self.cfg.ft_parent_body_name not in self._robot.body_names:
            raise ValueError(
                f"Force sensor parent body {self.cfg.ft_parent_body_name!r} not found. "
                f"Available bodies: {self._robot.body_names}"
            )
        self.force_sensor_parent_body_idx = self._robot.body_names.index(self.cfg.ft_parent_body_name)
        if self.cfg.ft_raw_wrench_frame not in {"parent_body", "sensor_body"}:
            raise ValueError("ft_raw_wrench_frame must be parent_body or sensor_body")
        if self.cfg.ft_raw_torque_reference not in {"parent_origin", "joint_anchor", "sensor_origin"}:
            raise ValueError("ft_raw_torque_reference must be parent_origin, joint_anchor, or sensor_origin")
        if self.cfg.ft_corrected_reference not in {
            "base_origin", "panda_link7_origin", "fingertip_K", "fingertip_midpoint_K"
        }:
            raise ValueError(
                "ft_corrected_reference must be base_origin, panda_link7_origin, "
                "fingertip_K, or fingertip_midpoint_K"
            )
        self.force_sensor_parent = torch.zeros((self.num_envs, 6), device=self.device)
        self.wrench_raw = torch.zeros((self.num_envs, 6), device=self.device)
        self.wrench_source = "uninitialized"
        self.wrench_anchor = torch.zeros((self.num_envs, 6), device=self.device)
        self.wrench_base = torch.zeros((self.num_envs, 6), device=self.device)
        self.wrench_corrected = torch.zeros((self.num_envs, 6), device=self.device)
        self.wrench_final = torch.zeros((self.num_envs, 6), device=self.device)
        self.force_sensor_parent_smooth = torch.zeros((self.num_envs, 6), device=self.device)
        self.wrench_tool = torch.zeros((self.num_envs, 6), device=self.device)
        self.wrench_tool_smooth = torch.zeros((self.num_envs, 6), device=self.device)
        self.wrench_tool_bias = torch.zeros((self.num_envs, 6), device=self.device)
        self.wrench_tool_bias_count = torch.zeros((self.num_envs, 1), device=self.device)
        self.wrench_tool_zeroed = torch.zeros((self.num_envs, 6), device=self.device)
        self.wrench_model_clean = torch.zeros((self.num_envs, 6), device=self.device)
        self.wrench_model = torch.zeros((self.num_envs, 6), device=self.device)

        self.real_wrench_bias = torch.tensor(
            self.cfg.ft_real_wrench_bias, dtype=torch.float32, device=self.device
        ).unsqueeze(0)
        self.wrench_axis_scale = torch.tensor(
            self.cfg.ft_axis_scale, dtype=torch.float32, device=self.device
        ).unsqueeze(0)
        self.wrench_noise_std = torch.tensor(
            self.cfg.obs_rand.ft_wrench_noise_std, dtype=torch.float32, device=self.device
        ).unsqueeze(0)
        self.ft_corrected_force_matrix = torch.as_tensor(
            self.cfg.ft_corrected_force_matrix, dtype=torch.float32, device=self.device
        ).reshape(3, 3)
        self.ft_corrected_torque_matrix = torch.as_tensor(
            self.cfg.ft_corrected_torque_matrix, dtype=torch.float32, device=self.device
        ).reshape(3, 3)
        self.ft_corrected_wrench_sign = torch.as_tensor(
            self.cfg.ft_corrected_wrench_sign, dtype=torch.float32, device=self.device
        ).reshape(1, 6)
        self.ft_corrected_torque_offset_base_m = torch.as_tensor(
            self.cfg.ft_corrected_torque_offset_base_m, dtype=torch.float32, device=self.device
        ).reshape(1, 3)

        # Backward-compatible names used by existing collection code.
        self.force_sensor_smooth = torch.zeros((self.num_envs, 6), device=self.device)
        self.force_sensor_world_smooth = torch.zeros((self.num_envs, 6), device=self.device)

        # Set nominal dynamics parameters for randomization.
        self.default_gains = torch.tensor(self.cfg.ctrl.default_task_prop_gains, dtype=torch.float32, device=self.device).repeat(
            (self.num_envs, 1)
        )
        self.default_pos_threshold = torch.tensor(self.cfg.ctrl.pos_action_threshold, device=self.device).repeat(
            (self.num_envs, 1)
        )
        self.default_rot_threshold = torch.tensor(self.cfg.ctrl.rot_action_threshold, device=self.device).repeat(
            (self.num_envs, 1)
        )
        self.default_dead_zone = torch.tensor(self.cfg.ctrl.default_dead_zone, device=self.device).repeat(
            (self.num_envs, 1)
        )

        self.pos_threshold = self.default_pos_threshold.clone()
        self.rot_threshold = self.default_rot_threshold.clone()

    def _raw_wrench_geometry(self):
        """Return raw wrench frame and candidate torque reference poses."""
        parent_pos_w = self._robot.data.body_pos_w[:, self.force_sensor_parent_body_idx]
        parent_quat_w = self._robot.data.body_quat_w[:, self.force_sensor_parent_body_idx]
        anchor_pos_w = self._robot.data.body_pos_w[:, self.force_sensor_body_idx]
        anchor_quat_w = self._robot.data.body_quat_w[:, self.force_sensor_body_idx]
        raw_quat_w = parent_quat_w if self.cfg.ft_raw_wrench_frame == "parent_body" else anchor_quat_w
        if self.cfg.ft_raw_torque_reference == "parent_origin":
            raw_reference_pos_w = parent_pos_w
        elif self.cfg.ft_raw_torque_reference == "joint_anchor":
            raw_reference_pos_w = anchor_pos_w
        else:
            raw_reference_pos_w = self._robot.data.body_pos_w[:, self.force_sensor_body_idx]
        return parent_pos_w, anchor_pos_w, anchor_quat_w, raw_quat_w, raw_reference_pos_w

    def _transform_raw_wrench_to_base(self, wrench_raw):
        """Express the incoming wrench as ``O_F_ext_hat_K``.

        For a wrench translated from A to B: tau_B = tau_A + (p_A - p_B) x F.
        The real dataset's ``ee_wrench_base`` is Franka's
        ``O_F_ext_hat_K``: force and torque are expressed in the robot-base
        frame, while torque is referenced at the stiffness-frame K origin.
        ``panda_fingertip_centered`` is the simulation counterpart of K.
        """
        _parent_pos_w, _anchor_pos_w, _anchor_quat_w, raw_quat_w, raw_reference_pos_w = (
            self._raw_wrench_geometry()
        )
        k_pos_w = self._robot.data.body_pos_w[:, self.fingertip_body_idx]
        root_quat_w = self._robot.data.root_quat_w
        force_w = torch_utils.quat_apply(raw_quat_w, wrench_raw[:, :3])
        torque_w_at_raw_reference = torch_utils.quat_apply(raw_quat_w, wrench_raw[:, 3:6])
        torque_w_at_k = torque_w_at_raw_reference + torch.cross(
            raw_reference_pos_w - k_pos_w, force_w, dim=-1
        )
        world_to_base_quat = torch_utils.quat_conjugate(root_quat_w)
        force_base = torch_utils.quat_apply(world_to_base_quat, force_w)
        torque_base = torch_utils.quat_apply(world_to_base_quat, torque_w_at_k)
        return torch.cat((force_base, torque_base), dim=-1)

    def _transform_raw_wrench_to_anchor(self, wrench_raw):
        """Express the incoming wrench at the force-sensor joint anchor."""
        _parent_pos_w, anchor_pos_w, anchor_quat_w, raw_quat_w, raw_reference_pos_w = (
            self._raw_wrench_geometry()
        )
        force_w = torch_utils.quat_apply(raw_quat_w, wrench_raw[:, :3])
        torque_w_at_raw_reference = torch_utils.quat_apply(raw_quat_w, wrench_raw[:, 3:6])
        torque_w_at_anchor = torque_w_at_raw_reference + torch.cross(
            raw_reference_pos_w - anchor_pos_w, force_w, dim=-1
        )
        world_to_anchor_quat = torch_utils.quat_conjugate(anchor_quat_w)
        force_anchor = torch_utils.quat_apply(world_to_anchor_quat, force_w)
        torque_anchor = torch_utils.quat_apply(world_to_anchor_quat, torque_w_at_anchor)
        return torch.cat((force_anchor, torque_anchor), dim=-1)

    def _transform_raw_wrench_to_corrected(self, wrench_raw):
        """Build the calibrated wrench used by TAVLA when explicitly enabled."""
        reference = self.cfg.ft_corrected_reference
        if reference == "base_origin":
            wrench_reference = self._transform_raw_wrench_to_base(wrench_raw)
            force_base = wrench_reference[:, :3]
            torque_base = wrench_reference[:, 3:]
        else:
            parent_pos_w, _anchor_pos_w, _anchor_quat_w, raw_quat_w, raw_reference_pos_w = (
                self._raw_wrench_geometry()
            )
            if reference == "panda_link7_origin":
                target_pos_w = parent_pos_w
            else:
                target_pos_w = self._robot.data.body_pos_w[:, self.fingertip_body_idx]
            root_pos_w = self._robot.data.root_pos_w
            root_quat_w = self._robot.data.root_quat_w
            force_w = torch_utils.quat_apply(raw_quat_w, wrench_raw[:, :3])
            torque_w_at_raw_reference = torch_utils.quat_apply(raw_quat_w, wrench_raw[:, 3:6])
            torque_w_at_target = torque_w_at_raw_reference + torch.cross(
                raw_reference_pos_w - target_pos_w, force_w, dim=-1
            )
            world_to_base_quat = torch_utils.quat_conjugate(root_quat_w)
            force_base = torch_utils.quat_apply(world_to_base_quat, force_w)
            torque_base = torch_utils.quat_apply(world_to_base_quat, torque_w_at_target)

        force_corrected = torch.einsum("ij,nj->ni", self.ft_corrected_force_matrix, force_base)
        torque_corrected = torch.einsum("ij,nj->ni", self.ft_corrected_torque_matrix, torque_base)
        torque_corrected = torque_corrected + torch.cross(
            self.ft_corrected_torque_offset_base_m.expand_as(force_corrected),
            force_corrected,
            dim=-1,
        )
        return torch.cat((force_corrected, torque_corrected), dim=-1) * self.ft_corrected_wrench_sign

    def _transform_parent_wrench_to_tool(self, wrench_parent):
        """Express a parent-frame joint wrench at the fingertip stiffness frame."""
        parent_pos_w = self._robot.data.body_pos_w[:, self.force_sensor_parent_body_idx]
        parent_quat_w = self._robot.data.body_quat_w[:, self.force_sensor_parent_body_idx]
        tool_pos_w = self._robot.data.body_pos_w[:, self.fingertip_body_idx]
        tool_quat_w = self._robot.data.body_quat_w[:, self.fingertip_body_idx]

        force_w = torch_utils.quat_apply(parent_quat_w, wrench_parent[:, :3])
        torque_w_at_parent = torch_utils.quat_apply(parent_quat_w, wrench_parent[:, 3:6])
        parent_to_tool_w = parent_pos_w - tool_pos_w
        torque_w_at_tool = torque_w_at_parent + torch.cross(parent_to_tool_w, force_w, dim=-1)

        world_to_tool_quat = torch_utils.quat_conjugate(tool_quat_w)
        force_tool = torch_utils.quat_apply(world_to_tool_quat, force_w)
        torque_tool = torch_utils.quat_apply(world_to_tool_quat, torque_w_at_tool)
        return torch.cat((force_tool, torque_tool), dim=-1)

    def _read_force_sensor_incoming_joint_wrench(self):
        body_idx = int(getattr(self, "_incoming_wrench_body_idx", self.force_sensor_body_idx))
        view = getattr(self._robot, "root_physx_view", None)
        getter = getattr(view, "get_link_incoming_joint_force", None)
        if callable(getter):
            incoming = getter()
            if incoming.ndim != 3 or incoming.shape[1] <= body_idx:
                raise RuntimeError("incoming joint-force tensor has an invalid body dimension")
            self.wrench_source = "root_physx_view.get_link_incoming_joint_force"
            return incoming[:, body_idx]
        legacy = getattr(self._robot.data, "body_incoming_joint_wrench_b", None)
        if legacy is None:
            raise RuntimeError("Isaac Sim does not expose link incoming joint force")
        self.wrench_source = "robot.data.body_incoming_joint_wrench_b_legacy"
        return legacy[:, body_idx]

    def _update_wrench(self):
        """Build the real-data-compatible 6D wrench used by RL and TA-VLA."""
        self.wrench_raw = self._read_force_sensor_incoming_joint_wrench()
        # Backward-compatible alias; frame/reference are still calibration hypotheses.
        self.force_sensor_parent = self.wrench_raw
        self.wrench_anchor = self._transform_raw_wrench_to_anchor(self.wrench_raw)
        self.wrench_base = self._transform_raw_wrench_to_base(self.wrench_raw)
        self.wrench_corrected = self._transform_raw_wrench_to_corrected(self.wrench_raw)
        # PhysX incoming wrench has the opposite sign from the Franka
        # O_F_ext_hat_K convention confirmed by the +/-X diagnostic.
        self.wrench_final = -self.wrench_base
        if not torch.isfinite(self.wrench_final).all():
            raise FloatingPointError("wrench_final contains NaN or Inf")
        alpha = float(self.cfg.ft_smoothing_factor)
        self.force_sensor_parent_smooth = (
            alpha * self.force_sensor_parent + (1.0 - alpha) * self.force_sensor_parent_smooth
        )

        self.wrench_tool = self._transform_parent_wrench_to_tool(self.force_sensor_parent)
        self.wrench_tool_smooth = alpha * self.wrench_tool + (1.0 - alpha) * self.wrench_tool_smooth

        calibration_mask = self.wrench_tool_bias_count[:, 0] < int(self.cfg.ft_bias_calibration_steps)
        if torch.any(calibration_mask):
            old_count = self.wrench_tool_bias_count[calibration_mask]
            new_count = old_count + 1.0
            self.wrench_tool_bias[calibration_mask] = (
                self.wrench_tool_bias[calibration_mask] * old_count
                + self.wrench_tool_smooth[calibration_mask]
            ) / new_count
            self.wrench_tool_bias_count[calibration_mask] = new_count

        self.wrench_tool_zeroed = self.wrench_tool_smooth - self.wrench_tool_bias
        self.wrench_model_clean = self.wrench_tool_zeroed * self.wrench_axis_scale + self.real_wrench_bias
        self.wrench_model = (
            self.wrench_model_clean + torch.randn_like(self.wrench_model_clean) * self.wrench_noise_std
        )

        # Keep old attributes valid while giving them explicit semantics.
        self.force_sensor_world_smooth = self.force_sensor_parent_smooth
        self.force_sensor_smooth = self.wrench_model_clean
        self.noisy_force = self.wrench_model

    def _compute_intermediate_values(self, dt):
        """Add noise to observations for force sensing."""
        super()._compute_intermediate_values(dt)

        # Add noise to fingertip position.
        pos_noise_level, rot_noise_level_deg = self.cfg.obs_rand.fingertip_pos, self.cfg.obs_rand.fingertip_rot_deg
        fingertip_pos_noise = torch.randn((self.num_envs, 3), dtype=torch.float32, device=self.device)
        fingertip_pos_noise = fingertip_pos_noise @ torch.diag(
            torch.tensor([pos_noise_level, pos_noise_level, pos_noise_level], dtype=torch.float32, device=self.device)
        )
        self.noisy_fingertip_pos = self.fingertip_midpoint_pos + fingertip_pos_noise

        rot_noise_axis = torch.randn((self.num_envs, 3), dtype=torch.float32, device=self.device)
        rot_noise_axis /= torch.linalg.norm(rot_noise_axis, dim=1, keepdim=True).clamp_min(1.0e-6)
        rot_noise_angle = torch.randn((self.num_envs,), dtype=torch.float32, device=self.device) * np.deg2rad(
            rot_noise_level_deg
        )
        self.noisy_fingertip_quat = torch_utils.quat_mul(
            self.fingertip_midpoint_quat, torch_utils.quat_from_angle_axis(rot_noise_angle, rot_noise_axis)
        )
        self.noisy_fingertip_quat[:, [0, 3]] = 0.0
        self.noisy_fingertip_quat = self.noisy_fingertip_quat * self.flip_quats.unsqueeze(-1)

        # Repeat finite differencing with noisy fingertip positions.
        self.ee_linvel_fd = (self.noisy_fingertip_pos - self.prev_fingertip_pos) / dt
        self.prev_fingertip_pos = self.noisy_fingertip_pos.clone()

        # Add state differences if velocity isn't being added.
        rot_diff_quat = torch_utils.quat_mul(
            self.noisy_fingertip_quat, torch_utils.quat_conjugate(self.prev_fingertip_quat)
        )
        rot_diff_quat *= torch.sign(rot_diff_quat[:, 0]).unsqueeze(-1)
        rot_diff_aa = axis_angle_from_quat(rot_diff_quat)
        self.ee_angvel_fd = rot_diff_aa / dt
        self.ee_angvel_fd[:, 0:2] = 0.0
        self.prev_fingertip_quat = self.noisy_fingertip_quat.clone()

        self._update_wrench()

    def _get_observations(self):
        """Add additional FORGE observations."""
        obs_dict, state_dict = self._get_factory_obs_state_dict()

        noisy_fixed_pos = self.fixed_pos_obs_frame + self.init_fixed_pos_obs_noise
        prev_actions = self.actions.clone()
        prev_actions[:, 3:5] = 0.0

        obs_dict.update({
            "fingertip_pos": self.noisy_fingertip_pos,
            "fingertip_pos_rel_fixed": self.noisy_fingertip_pos - noisy_fixed_pos,
            "fingertip_quat": self.noisy_fingertip_quat,
            "held_pos_rel_fixed": self.held_pos - noisy_fixed_pos,
            "held_quat": self.held_quat,
            "force_threshold": self.contact_penalty_thresholds[:, None],
            "ft_force": self.noisy_force,
            "prev_actions": prev_actions,
        })

        state_dict.update({
            "ema_factor": self.ema_factor,
            "ft_force": self.wrench_model_clean,
            "force_threshold": self.contact_penalty_thresholds[:, None],
            "prev_actions": prev_actions,
        })

        obs_tensors = factory_utils.collapse_obs_dict(obs_dict, self.cfg.obs_order + ["prev_actions"])
        state_tensors = factory_utils.collapse_obs_dict(state_dict, self.cfg.state_order + ["prev_actions"])
        return {"policy": obs_tensors, "critic": state_tensors}

    def _apply_action(self):
        """FORGE actions are defined as targets relative to the fixed asset."""
        if self.last_update_timestamp < self._robot._data._sim_timestamp:
            self._compute_intermediate_values(dt=self.physics_dt)

        # Step (0): Scale actions to allowed range.
        pos_actions = self.actions[:, 0:3]
        pos_actions = pos_actions @ torch.diag(torch.tensor(self.cfg.ctrl.pos_action_bounds, dtype=torch.float32, device=self.device))

        rot_actions = self.actions[:, 3:6]
        rot_actions = rot_actions @ torch.diag(torch.tensor(self.cfg.ctrl.rot_action_bounds, dtype=torch.float32, device=self.device))

        # Step (1): Compute desired pose targets in EE frame.
        # (1.a) Position. Action frame is assumed to be the top of the bolt (noisy estimate).
        fixed_pos_action_frame = self.fixed_pos_obs_frame + self.init_fixed_pos_obs_noise
        ctrl_target_fingertip_preclipped_pos = fixed_pos_action_frame + pos_actions
        # (1.b) Enforce rotation action constraints.
        rot_actions[:, 0:2] = 0.0

        # Assumes joint limit is in (+x, -y)-quadrant of world frame.
        rot_actions[:, 2] = np.deg2rad(-180.0) + np.deg2rad(270.0) * (rot_actions[:, 2] + 1.0) / 2.0  # Joint limit.
        # (1.c) Get desired orientation target.
        bolt_frame_quat = torch_utils.quat_from_euler_xyz(
            roll=rot_actions[:, 0], pitch=rot_actions[:, 1], yaw=rot_actions[:, 2]
        )

        rot_180_euler = torch.tensor([np.pi, 0.0, 0.0], dtype=torch.float32, device=self.device).repeat(self.num_envs, 1)
        quat_bolt_to_ee = torch_utils.quat_from_euler_xyz(
            roll=rot_180_euler[:, 0], pitch=rot_180_euler[:, 1], yaw=rot_180_euler[:, 2]
        )

        ctrl_target_fingertip_preclipped_quat = torch_utils.quat_mul(quat_bolt_to_ee, bolt_frame_quat)

        # Step (2): Clip targets if they are too far from current EE pose.
        # (2.a): Clip position targets.
        self.delta_pos = ctrl_target_fingertip_preclipped_pos - self.fingertip_midpoint_pos  # Used for action_penalty.
        pos_error_clipped = torch.clip(self.delta_pos, -self.pos_threshold, self.pos_threshold)
        ctrl_target_fingertip_midpoint_pos = self.fingertip_midpoint_pos + pos_error_clipped

        # (2.b) Clip orientation targets. Use Euler angles. We assume we are near upright, so
        # clipping yaw will effectively cause slow motions. When we clip, we also need to make
        # sure we avoid the joint limit.

        # (2.b.i) Get current and desired Euler angles.
        curr_roll, curr_pitch, curr_yaw = torch_utils.get_euler_xyz(self.fingertip_midpoint_quat)
        desired_roll, desired_pitch, desired_yaw = torch_utils.get_euler_xyz(ctrl_target_fingertip_preclipped_quat)
        desired_xyz = torch.stack([desired_roll, desired_pitch, desired_yaw], dim=1)

        # (2.b.ii) Correct the direction of motion to avoid joint limit.
        # Map yaws between [-125, 235] degrees (so that angles appear on a continuous span uninterrupted by the joint limit).
        curr_yaw = factory_utils.wrap_yaw(curr_yaw)
        desired_yaw = factory_utils.wrap_yaw(desired_yaw)

        # (2.b.iii) Clip motion in the correct direction.
        self.delta_yaw = desired_yaw - curr_yaw  # Used later for action_penalty.
        clipped_yaw = torch.clip(self.delta_yaw, -self.rot_threshold[:, 2], self.rot_threshold[:, 2])
        desired_xyz[:, 2] = curr_yaw + clipped_yaw

        # (2.b.iv) Clip roll and pitch.
        desired_roll = torch.where(desired_roll < 0.0, desired_roll + 2 * torch.pi, desired_roll)
        desired_pitch = torch.where(desired_pitch < 0.0, desired_pitch + 2 * torch.pi, desired_pitch)

        delta_roll = desired_roll - curr_roll
        clipped_roll = torch.clip(delta_roll, -self.rot_threshold[:, 0], self.rot_threshold[:, 0])
        desired_xyz[:, 0] = curr_roll + clipped_roll

        curr_pitch = torch.where(curr_pitch > torch.pi, curr_pitch - 2 * torch.pi, curr_pitch)
        desired_pitch = torch.where(desired_pitch > torch.pi, desired_pitch - 2 * torch.pi, desired_pitch)

        delta_pitch = desired_pitch - curr_pitch
        clipped_pitch = torch.clip(delta_pitch, -self.rot_threshold[:, 1], self.rot_threshold[:, 1])
        desired_xyz[:, 1] = curr_pitch + clipped_pitch

        ctrl_target_fingertip_midpoint_quat = torch_utils.quat_from_euler_xyz(
            roll=desired_xyz[:, 0], pitch=desired_xyz[:, 1], yaw=desired_xyz[:, 2]
        )

        self.generate_ctrl_signals(
            ctrl_target_fingertip_midpoint_pos=ctrl_target_fingertip_midpoint_pos,
            ctrl_target_fingertip_midpoint_quat=ctrl_target_fingertip_midpoint_quat,
            ctrl_target_gripper_dof_pos=0.0,
        )

    def _get_rewards(self):
        """FORGE reward includes a contact penalty and success prediction error."""
        # Use same base rewards as Factory.
        rew_buf = super()._get_rewards()

        rew_dict, rew_scales = {}, {}
        # Calculate action penalty for the asset-relative action space.
        pos_error = torch.norm(self.delta_pos, p=2, dim=-1) / self.cfg.ctrl.pos_action_threshold[0]
        rot_error = torch.abs(self.delta_yaw) / self.cfg.ctrl.rot_action_threshold[0]
        # Contact penalty.
        contact_force = torch.norm(self.wrench_tool_zeroed[:, 0:3], p=2, dim=-1, keepdim=False)
        contact_penalty = torch.nn.functional.relu(contact_force - self.contact_penalty_thresholds)
        # Add success prediction rewards.
        check_rot = self.cfg_task.name == "nut_thread"
        true_successes = self._get_curr_successes(
            success_threshold=self.cfg_task.success_threshold, check_rot=check_rot
        )
        has_success_pred_action = self.actions.shape[1] > 6 and getattr(
            self.cfg, "use_success_prediction", True
        )
        if has_success_pred_action:
            policy_success_pred = (self.actions[:, 6] + 1) / 2  # rescale from [-1, 1] to [0, 1]
            success_pred_error = (true_successes.float() - policy_success_pred).abs()
            # Delay success prediction penalty until some successes have occurred.
            if true_successes.float().mean() >= self.cfg_task.delay_until_ratio:
                self.success_pred_scale = 1.0
        else:
            policy_success_pred = true_successes.float()
            success_pred_error = torch.zeros_like(policy_success_pred)

        # Add new FORGE reward terms.
        rew_dict = {
            "action_penalty_asset": pos_error + rot_error,
            "contact_penalty": contact_penalty,
            "success_pred_error": success_pred_error,
        }
        rew_scales = {
            "action_penalty_asset": -self.cfg_task.action_penalty_asset_scale,
            "contact_penalty": -self.cfg_task.contact_penalty_scale,
            "success_pred_error": -self.success_pred_scale if has_success_pred_action else 0.0,
        }
        for rew_name, rew in rew_dict.items():
            rew_buf += rew_dict[rew_name] * rew_scales[rew_name]

        self._log_forge_metrics(rew_dict, policy_success_pred)
        return rew_buf

    def _reset_idx(self, env_ids):
        """Perform additional randomizations."""
        super()._reset_idx(env_ids)

        # Compute initial action for correct EMA computation.
        fixed_pos_action_frame = self.fixed_pos_obs_frame + self.init_fixed_pos_obs_noise
        pos_actions = self.fingertip_midpoint_pos - fixed_pos_action_frame
        pos_action_bounds = torch.tensor(self.cfg.ctrl.pos_action_bounds, dtype=torch.float32, device=self.device)
        pos_actions = pos_actions @ torch.diag(1.0 / pos_action_bounds)
        self.actions[:, 0:3] = self.prev_actions[:, 0:3] = pos_actions

        # Relative yaw to bolt.
        unrot_180_euler = torch.tensor([-np.pi, 0.0, 0.0], dtype=torch.float32, device=self.device).repeat(self.num_envs, 1)
        unrot_quat = torch_utils.quat_from_euler_xyz(
            roll=unrot_180_euler[:, 0], pitch=unrot_180_euler[:, 1], yaw=unrot_180_euler[:, 2]
        )

        fingertip_quat_rel_bolt = torch_utils.quat_mul(unrot_quat, self.fingertip_midpoint_quat)
        fingertip_yaw_bolt = torch_utils.get_euler_xyz(fingertip_quat_rel_bolt)[-1]
        fingertip_yaw_bolt = torch.where(
            fingertip_yaw_bolt > torch.pi / 2, fingertip_yaw_bolt - 2 * torch.pi, fingertip_yaw_bolt
        )
        fingertip_yaw_bolt = torch.where(
            fingertip_yaw_bolt < -torch.pi, fingertip_yaw_bolt + 2 * torch.pi, fingertip_yaw_bolt
        )

        yaw_action = (fingertip_yaw_bolt + np.deg2rad(180.0)) / np.deg2rad(270.0) * 2.0 - 1.0
        self.actions[:, 5] = self.prev_actions[:, 5] = yaw_action
        if self.actions.shape[1] > 6 and getattr(self.cfg, "use_success_prediction", True):
            self.actions[:, 6] = self.prev_actions[:, 6] = -1.0

        # EMA randomization.
        ema_rand = torch.rand((self.num_envs, 1), dtype=torch.float32, device=self.device)
        ema_lower, ema_upper = self.cfg.ctrl.ema_factor_range
        self.ema_factor = ema_lower + ema_rand * (ema_upper - ema_lower)

        # Set initial gains for the episode.
        prop_gains = self.default_gains.clone()
        self.pos_threshold = self.default_pos_threshold.clone()
        self.rot_threshold = self.default_rot_threshold.clone()
        prop_gains = utils.get_random_prop_gains(
            prop_gains, self.cfg.ctrl.task_prop_gains_noise_level, self.num_envs, self.device
        )
        self.pos_threshold = utils.get_random_prop_gains(
            self.pos_threshold, self.cfg.ctrl.pos_threshold_noise_level, self.num_envs, self.device
        )
        self.rot_threshold = utils.get_random_prop_gains(
            self.rot_threshold, self.cfg.ctrl.rot_threshold_noise_level, self.num_envs, self.device
        )
        self.task_prop_gains = prop_gains
        self.task_deriv_gains = factory_utils.get_deriv_gains(prop_gains)

        contact_rand = torch.rand((self.num_envs,), dtype=torch.float32, device=self.device)
        contact_lower, contact_upper = self.cfg.task.contact_penalty_threshold_range
        self.contact_penalty_thresholds = contact_lower + contact_rand * (contact_upper - contact_lower)

        self.dead_zone_thresholds = (
            torch.rand((self.num_envs, 6), dtype=torch.float32, device=self.device) * self.default_dead_zone
        )

        for wrench_buffer in (
            self.wrench_raw,
            self.wrench_anchor,
            self.wrench_base,
            self.wrench_corrected,
            self.wrench_final,
            self.force_sensor_parent_smooth,
            self.wrench_tool,
            self.wrench_tool_smooth,
            self.wrench_tool_bias,
            self.wrench_tool_zeroed,
            self.wrench_model_clean,
            self.wrench_model,
            self.force_sensor_smooth,
            self.force_sensor_world_smooth,
        ):
            wrench_buffer[env_ids] = 0.0
        self.wrench_tool_bias_count[env_ids] = 0.0

        self.flip_quats = torch.ones((self.num_envs,), dtype=torch.float32, device=self.device)
        rand_flips = torch.rand(self.num_envs) > 0.5
        self.flip_quats[rand_flips] = -1.0

    def _reset_buffers(self, env_ids):
        """Reset additional logging metrics."""
        super()._reset_buffers(env_ids)
        # Reset success pred metrics.
        for thresh in [0.5, 0.6, 0.7, 0.8, 0.9]:
            self.first_pred_success_tx[thresh][env_ids] = 0
        self.wrench_tool_bias[env_ids] = 0.0
        self.wrench_tool_bias_count[env_ids] = 0.0

    def _log_forge_metrics(self, rew_dict, policy_success_pred):
        """Log metrics to evaluate success prediction performance."""
        for rew_name, rew in rew_dict.items():
            self.extras[f"logs_rew_{rew_name}"] = rew.mean()

        for thresh, first_success_tx in self.first_pred_success_tx.items():
            curr_predicted_success = policy_success_pred > thresh
            first_success_idxs = torch.logical_and(curr_predicted_success, first_success_tx == 0)

            first_success_tx[:] = torch.where(first_success_idxs, self.episode_length_buf, first_success_tx)

            # Only log at the end.
            if torch.any(self.reset_buf):
                # Log prediction delay.
                delay_ids = torch.logical_and(self.ep_success_times != 0, first_success_tx != 0)
                num_delay_ids = delay_ids.sum()
                if num_delay_ids.item() > 0:
                    delay_times = (first_success_tx[delay_ids] - self.ep_success_times[delay_ids]).sum() / num_delay_ids
                    self.extras[f"early_term_delay_all/{thresh}"] = delay_times

                correct_delay_ids = torch.logical_and(delay_ids, first_success_tx > self.ep_success_times)
                num_correct_delay_ids = correct_delay_ids.sum()
                if num_correct_delay_ids.item() > 0:
                    correct_delay_times = (
                        first_success_tx[correct_delay_ids] - self.ep_success_times[correct_delay_ids]
                    ).sum() / num_correct_delay_ids
                    self.extras[f"early_term_delay_correct/{thresh}"] = correct_delay_times.item()

                # Log early-term success rate (for all episodes we have "stopped", did we succeed?).
                pred_success_idxs = first_success_tx != 0  # Episodes which we have predicted success.

                true_success_preds = torch.logical_and(
                    self.ep_success_times[pred_success_idxs] > 0,  # Success has actually occurred.
                    self.ep_success_times[pred_success_idxs]
                    < first_success_tx[pred_success_idxs],  # Success occurred before we predicted it.
                )

                num_pred_success = pred_success_idxs.sum().item()
                if num_pred_success > 0:
                    et_prec = true_success_preds.sum() / num_pred_success
                    self.extras[f"early_term_precision/{thresh}"] = et_prec

                true_success_idxs = self.ep_success_times > 0
                num_true_success = true_success_idxs.sum().item()
                if num_true_success > 0:
                    et_recall = true_success_preds.sum() / num_true_success
                    self.extras[f"early_term_recall/{thresh}"] = et_recall
