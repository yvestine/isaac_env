from __future__ import annotations

import json
import os
import cv2
import time

import gymnasium as gym
import numpy as np
import torch
import isaacsim.core.utils.torch as torch_utils
from isaaclab_tasks.direct.factory import factory_control, factory_utils

from .policy.modeling_pi0remote import PI0RemotePolicyTAVLA
from .realsim_env import RealSimEnv


class TavlaResidualEnv(RealSimEnv):
    """Peg-insertion environment with a frozen TAVLA joint-target teacher."""

    def __init__(self, cfg, render_mode=None, **kwargs):
        self._tavla_ready = False
        self._teacher_control_mode = getattr(cfg, "teacher_control_mode", "aligned_joint")
        if self._teacher_control_mode not in {"kinematic_taskspace", "aligned_joint", "ppo_cartesian"}:
            raise ValueError(
                f"Unsupported TAVLA control mode: {self._teacher_control_mode!r}; "
                "expected kinematic_taskspace, aligned_joint, or ppo_cartesian"
            )
        self._use_implicit_position_servo = bool(
            getattr(cfg, "use_implicit_position_servo", False)
            and self._teacher_control_mode == "aligned_joint"
        )
        if self._use_implicit_position_servo:
            # Match the saved replay validation: the aligned Kp/Kd are the
            # articulation implicit position-servo gains, not an additional
            # explicit torque controller.
            cfg.robot.actuators["panda_arm1"].stiffness = float(cfg.joint_target_kp[0])
            cfg.robot.actuators["panda_arm1"].damping = float(cfg.joint_target_kd[0])
            cfg.robot.actuators["panda_arm2"].stiffness = float(cfg.joint_target_kp[4])
            cfg.robot.actuators["panda_arm2"].damping = float(cfg.joint_target_kd[4])
        super().__init__(cfg, render_mode, **kwargs)

        if cfg.teacher_policy_cfg is None:
            raise ValueError("TavlaResidualEnv requires cfg.teacher_policy_cfg")
        if self.num_envs != 1:
            raise ValueError(
                "The initial remote TAVLA adapter supports one environment; "
                "use num_envs=1 until the server supports batched inference."
            )

        policy_class = getattr(self, "teacher_policy_class", PI0RemotePolicyTAVLA)
        self.teacher_policy = policy_class(cfg.teacher_policy_cfg)
        self.teacher_hold_steps = max(1, int(cfg.teacher_hold_steps))
        self.teacher_replan_actions = max(1, int(getattr(cfg, "teacher_replan_actions", 5)))
        self._teacher_visual_profile = getattr(cfg, "teacher_visual_profile", "raw")
        self._teacher_camera_calibration = getattr(cfg, "teacher_camera_calibration", "")
        if self._teacher_visual_profile not in {"raw", "real_aligned"}:
            raise ValueError("teacher_visual_profile must be raw or real_aligned")
        self._teacher_visual_calibration = self._load_teacher_visual_calibration()
        self._teacher_taskspace_velocity_limits = torch.as_tensor(
            getattr(cfg, "teacher_taskspace_velocity_limits", [0.12, 0.12, 0.30, 1.20, 1.20, 1.20]),
            dtype=torch.float32, device=self.device,
        ).view(1, 6)
        if torch.any(self._teacher_taskspace_velocity_limits <= 0.0):
            raise ValueError("teacher task-space velocity limits must be positive")
        self._teacher_force_norm_p99 = float(getattr(cfg, "teacher_force_norm_p99", 100.0))
        if self._teacher_force_norm_p99 <= 0.0:
            raise ValueError("teacher_force_norm_p99 must be positive")
        self.gripper_open_width_m = float(cfg.gripper_open_width_m)
        self._teacher_wrench_scale = torch.as_tensor(
            getattr(cfg, "teacher_wrench_scale", [1.0] * 6), dtype=torch.float32, device=self.device
        ).view(1, 6)
        self._teacher_wrench_bias = torch.as_tensor(
            getattr(cfg, "teacher_wrench_bias", [0.0] * 6), dtype=torch.float32, device=self.device
        ).view(1, 6)
        self._joint_kp = torch.as_tensor(cfg.joint_target_kp, dtype=torch.float32, device=self.device).view(1, 7)
        self._joint_kd = torch.as_tensor(cfg.joint_target_kd, dtype=torch.float32, device=self.device).view(1, 7)
        self._joint_effort_limits = torch.as_tensor(
            cfg.joint_target_effort_limits, dtype=torch.float32, device=self.device
        ).view(1, 7)
        self._teacher_execution_position_servo = bool(
            getattr(cfg, "teacher_execution_position_servo", False)
        )
        self._teacher_position_servo_activated = False
        self._joint_residual_scale = torch.as_tensor(
            cfg.joint_residual_scale, dtype=torch.float32, device=self.device
        ).view(1, 7)
        self._gripper_residual_scale = float(cfg.gripper_residual_scale)
        self._teacher_action_interpolation = bool(getattr(cfg, "teacher_action_interpolation", True))
        self._teacher_speed_scale = float(getattr(cfg, "teacher_speed_scale", 1.0))
        if self._teacher_speed_scale <= 0.0:
            raise ValueError("teacher_speed_scale must be positive")
        self._teacher_joint_velocity_limits = torch.as_tensor(
            getattr(cfg, "teacher_joint_velocity_limits", [float("inf")] * 7),
            dtype=torch.float32, device=self.device,
        ).view(1, 7)
        self._teacher_joint_acceleration_limits = torch.as_tensor(
            getattr(cfg, "teacher_joint_acceleration_limits", [float("inf")] * 7),
            dtype=torch.float32, device=self.device,
        ).view(1, 7)
        if torch.any(self._teacher_joint_velocity_limits <= 0.0) or torch.any(self._teacher_joint_acceleration_limits <= 0.0):
            raise ValueError("teacher joint velocity and acceleration limits must be positive")
        self._teacher_gripper_velocity_limit = float(getattr(cfg, "teacher_gripper_velocity_limit", 2.0))
        if self._teacher_gripper_velocity_limit <= 0.0:
            raise ValueError("teacher_gripper_velocity_limit must be positive")
        self._privileged_xy_guidance = bool(getattr(cfg, "privileged_xy_guidance", False))
        self._privileged_xy_guidance_weight = float(getattr(cfg, "privileged_xy_guidance_weight", 1.0))
        self._privileged_xy_guidance_gain = float(getattr(cfg, "privileged_xy_guidance_gain", 1.0))
        self._privileged_xy_guidance_max_joint_step = float(
            getattr(cfg, "privileged_xy_guidance_max_joint_step", 0.02)
        )
        self._privileged_xy_guidance_gate_m = float(getattr(cfg, "privileged_xy_guidance_gate_m", 0.006))
        self._privileged_xyz_guidance = bool(getattr(cfg, "privileged_xyz_guidance", False))
        self._privileged_xyz_guidance_weight = float(getattr(cfg, "privileged_xyz_guidance_weight", 1.0))
        self._privileged_xyz_guidance_gain = float(getattr(cfg, "privileged_xyz_guidance_gain", 1.0))
        self._privileged_xyz_guidance_max_joint_step = float(
            getattr(cfg, "privileged_xyz_guidance_max_joint_step", 0.20)
        )
        self._privileged_insert_quat = torch.as_tensor(
            getattr(cfg, "privileged_insert_quat", [0.0, 1.0, 0.0, 0.0]),
            dtype=torch.float32, device=self.device,
        ).view(1, 4).repeat(self.num_envs, 1)
        self._teacher_state_alignment = bool(getattr(cfg, "teacher_state_alignment", False))
        self._teacher_action_state_alignment = bool(getattr(cfg, "teacher_action_state_alignment", False))
        self._teacher_policy_reference_state = torch.as_tensor(
            getattr(cfg, "teacher_policy_reference_state", cfg.ctrl.reset_joints[:7]),
            dtype=torch.float32, device=self.device,
        ).view(1, 7)
        self._teacher_sim_reference_state = torch.as_tensor(
            cfg.ctrl.reset_joints[:7], dtype=torch.float32, device=self.device
        ).view(1, 7)
        if self._teacher_control_mode == "kinematic_taskspace" and not hasattr(self, "_tavla_twin_robot"):
            raise RuntimeError("kinematic_taskspace requires a RealSim Franka twin")
        self._initialize_teacher_runtime()
        self._extend_observation_space(16)
        self._tavla_ready = True

    def _load_teacher_visual_calibration(self):
        if self._teacher_visual_profile == "raw":
            return {}
        path = self._teacher_camera_calibration
        if not path:
            raise ValueError("real_aligned visual profile requires --tavla-camera-calibration")
        if not os.path.isfile(path):
            raise FileNotFoundError(f"TAVLA camera calibration not found: {path}")
        with open(path, "r", encoding="utf-8") as file:
            calibration = json.load(file)
        if not isinstance(calibration, dict):
            raise ValueError("TAVLA camera calibration must be a JSON object")
        return calibration

    def _activate_teacher_position_servo(self):
        if not self._teacher_execution_position_servo or self._teacher_position_servo_activated:
            return
        stiffness = self._robot.data.joint_stiffness.clone()
        damping = self._robot.data.joint_damping.clone()
        stiffness[:, :7] = self._joint_kp
        damping[:, :7] = self._joint_kd
        self._robot.write_joint_stiffness_to_sim(stiffness)
        self._robot.write_joint_damping_to_sim(damping)
        self._use_implicit_position_servo = True
        self._teacher_position_servo_activated = True

    def _apply_teacher_visual_profile(self, image, camera_name):
        if self._teacher_visual_profile == "raw":
            return image
        array = image[0].detach().cpu().numpy()
        if array.dtype != np.uint8:
            if float(array.max(initial=0.0)) <= 1.0:
                array = array * 255.0
            array = np.clip(array, 0.0, 255.0).astype(np.uint8)
        camera_cfg = self._teacher_visual_calibration.get(
            camera_name, self._teacher_visual_calibration.get("default", {})
        )
        crop = camera_cfg.get("crop")
        if crop is not None:
            if len(crop) != 4:
                raise ValueError(f"{camera_name} crop must be [x, y, width, height]")
            x, y, width, height = [int(value) for value in crop]
            array = array[max(0, y):max(0, y) + max(1, height), max(0, x):max(0, x) + max(1, width)]
        height, width = int(image.shape[-3]), int(image.shape[-2])
        array = cv2.resize(array, (width, height), interpolation=cv2.INTER_LINEAR)
        matrix = camera_cfg.get("color_matrix")
        bias = camera_cfg.get("color_bias", [0.0, 0.0, 0.0])
        if matrix is not None:
            matrix = np.asarray(matrix, dtype=np.float32).reshape(3, 3)
            array = np.einsum("hwc,dc->hwd", array.astype(np.float32), matrix)
        array = array + np.asarray(bias, dtype=np.float32).reshape(1, 1, 3)
        gamma = float(camera_cfg.get("gamma", 1.0))
        if gamma <= 0.0:
            raise ValueError("camera calibration gamma must be positive")
        if gamma != 1.0:
            array = 255.0 * np.power(np.clip(array, 0.0, 255.0) / 255.0, gamma)
        blur = int(camera_cfg.get("blur_kernel", 0))
        if blur > 1:
            if blur % 2 == 0:
                blur += 1
            array = cv2.GaussianBlur(array, (blur, blur), 0)
        array = np.ascontiguousarray(np.clip(array, 0.0, 255.0).astype(np.uint8))
        return torch.from_numpy(array).unsqueeze(0)

    def _twin_jacobian(self):
        jacobians = self._tavla_twin_robot.root_physx_view.get_jacobians()
        return jacobians[:, self._tavla_twin_jacobian_body_idx, 0:6, 0:7]

    def _sync_twin_to_actual_pose(self):
        if self._teacher_control_mode != "kinematic_taskspace":
            return True
        actual_pos = self._robot.data.body_pos_w[:, self.fingertip_body_idx]
        actual_quat = self._robot.data.body_quat_w[:, self.fingertip_body_idx]
        target_pos = actual_pos + self._tavla_twin_robot.data.root_pos_w - self._robot.data.root_pos_w
        target_quat = actual_quat
        if not getattr(self, "_tavla_twin_initialized", False):
            q = self._teacher_policy_reference_state.clone()
        else:
            q = self._tavla_twin_q.clone()
        lower, upper = self._joint_limits()
        q_reference = self._teacher_policy_reference_state
        full_state = self._tavla_twin_robot.data.joint_pos.clone()
        if full_state.shape[1] > 7 and self.joint_pos.shape[1] > 7:
            finger_count = min(full_state.shape[1] - 7, self.joint_pos.shape[1] - 7)
            full_state[:, 7:7 + finger_count] = self.joint_pos[:, 7:7 + finger_count]
        converged = False
        position_error = torch.full((self.num_envs,), float("inf"), device=self.device)
        rotation_error = torch.full((self.num_envs,), float("inf"), device=self.device)
        iteration_count = 0
        for iteration in range(100):
            full_state[:, :7] = q
            self._tavla_twin_robot.write_joint_state_to_sim(full_state, torch.zeros_like(full_state))
            self.sim.forward()
            twin_pos = self._tavla_twin_robot.data.body_pos_w[:, self._tavla_twin_body_idx]
            twin_quat = self._tavla_twin_robot.data.body_quat_w[:, self._tavla_twin_body_idx]
            position_delta, rotation_delta = factory_control.get_pose_error(
                fingertip_midpoint_pos=twin_pos,
                fingertip_midpoint_quat=twin_quat,
                ctrl_target_fingertip_midpoint_pos=target_pos,
                ctrl_target_fingertip_midpoint_quat=target_quat,
                jacobian_type="geometric",
                rot_error_type="axis_angle",
            )
            position_error = torch.linalg.vector_norm(position_delta, dim=-1)
            rotation_error = torch.linalg.vector_norm(rotation_delta, dim=-1)
            iteration_count = iteration + 1
            if bool(torch.all((position_error < 0.002) & (rotation_error < np.deg2rad(2.0)))):
                converged = True
                break
            jacobian = self._twin_jacobian()
            delta_pose = torch.cat((position_delta, rotation_delta), dim=-1)
            # A smaller DLS damping is needed for the reset-to-reference
            # pose gap; the previous 0.1 damping stalled at ~7 mm.
            lambda_matrix = (0.02 ** 2) * torch.eye(6, device=self.device).unsqueeze(0)
            jacobian_t = jacobian.transpose(1, 2)
            dls_inverse = torch.inverse(jacobian @ jacobian_t + lambda_matrix)
            delta_task = (jacobian_t @ dls_inverse @ delta_pose.unsqueeze(-1)).squeeze(-1)
            jacobian_pinv = jacobian_t @ dls_inverse
            null_projector = torch.eye(7, device=self.device).unsqueeze(0) - jacobian_pinv @ jacobian
            delta_null = (null_projector @ (q_reference - q).unsqueeze(-1)).squeeze(-1)
            # Prioritize pose convergence; nullspace is only a gentle
            # secondary pull toward the real-data reference configuration.
            q = q + delta_task + 0.02 * delta_null
            q = torch.clamp(q, lower, upper)
        self._tavla_twin_q = q.detach().clone()
        self._teacher_taskspace_prev_q = self._tavla_twin_q.clone()
        self._tavla_twin_initialized = True
        self.tavla_twin_ik_position_error = position_error
        self.tavla_twin_ik_rotation_error = rotation_error
        self.tavla_twin_ik_iterations = torch.full(
            (self.num_envs,), iteration_count, dtype=torch.long, device=self.device
        )
        self.tavla_twin_ik_converged = torch.full(
            (self.num_envs,), converged, dtype=torch.bool, device=self.device
        )
        if not converged:
            self.tavla_mapping_failures += 1
            self._tavla_mapping_failed[:] = True
            print(
                "[TAVLA] twin IK mapping failed: "
                f"position_error={float(position_error.max().detach().cpu()):.6f} m, "
                f"rotation_error={float(rotation_error.max().detach().cpu()):.6f} rad, "
                f"iterations={iteration_count}"
            )
        return converged

    def _prepare_taskspace_action(self, target):
        # The twin is synchronized immediately before every remote inference.
        # Re-running the 100-iteration IK here would add a second expensive
        # solve for every action and could change the policy state/action frame
        # between the request and its response.
        if bool(torch.any(self._tavla_mapping_failed)) or not self._tavla_twin_initialized:
            self._teacher_taskspace_delta_pose.zero_()
            self._teacher_taskspace_target_pos = self.fingertip_midpoint_pos.clone()
            self._teacher_taskspace_target_quat = self.fingertip_midpoint_quat.clone()
            return
        q_previous = getattr(self, "_teacher_taskspace_prev_q", self._tavla_twin_q)
        q_error = (target[:, :7] - q_previous + torch.pi) % (2.0 * torch.pi) - torch.pi
        q_next = q_previous + q_error
        self._teacher_taskspace_q_delta = q_error.detach().clone()

        # Each TAVLA chunk contains absolute joint targets.  Read the Jacobian
        # at the preceding target so the chunk becomes q1-q0, q2-q1, ... .
        # The twin is synchronized to the real robot again before the next
        # remote inference.
        full_state = self._tavla_twin_robot.data.joint_pos.clone()
        full_state[:, :7] = q_previous
        if full_state.shape[1] > 7 and self.joint_pos.shape[1] > 7:
            finger_count = min(full_state.shape[1] - 7, self.joint_pos.shape[1] - 7)
            full_state[:, 7:7 + finger_count] = self.joint_pos[:, 7:7 + finger_count]
        self._tavla_twin_robot.write_joint_state_to_sim(full_state, torch.zeros_like(full_state))
        self.sim.forward()

        delta_pose = torch.bmm(self._twin_jacobian(), q_error.unsqueeze(-1)).squeeze(-1)
        action_dt = max(float(self.step_dt) * self.teacher_hold_steps, 1.0e-6)
        max_delta = self._teacher_taskspace_velocity_limits * (action_dt * self._teacher_speed_scale)
        scale = torch.min(max_delta / torch.abs(delta_pose).clamp_min(1.0e-6), dim=-1).values
        scale = torch.minimum(scale, torch.ones_like(scale)).unsqueeze(-1)
        delta_pose = delta_pose * scale
        force_norm = torch.linalg.vector_norm(self.wrench_base, dim=-1)
        abnormal_force = force_norm > self._teacher_force_norm_p99
        self.tavla_force_abort[:] = abnormal_force
        if bool(torch.any(abnormal_force)):
            delta_pose[abnormal_force, :3] = 0.0
            target = target.clone()
            target[abnormal_force, 7] = self._current_tavla_state()[abnormal_force, 7]
            self._teacher_chunk = torch.empty((0, 8), device=self.device)
            self._teacher_chunk_index = 0
            self._teacher_chunk_end = 0
            self._teacher_hold_count = self.teacher_hold_steps
            self._teacher_target_updated = False
            self.tavla_force_abort_count += int(abnormal_force.sum().item())
            q_next = torch.where(abnormal_force.unsqueeze(-1), q_previous, q_next)
        self._teacher_taskspace_prev_q = q_next.detach().clone()
        self._tavla_twin_q = q_next.detach().clone()
        clipped_pose = delta_pose.clone()
        clipped_pose[:, :3] = torch.clamp(clipped_pose[:, :3], -self.pos_threshold, self.pos_threshold)
        clipped_pose[:, 3:6] = torch.clamp(clipped_pose[:, 3:6], -self.rot_threshold, self.rot_threshold)
        self._update_reward_action_metrics(clipped_pose)
        self._teacher_taskspace_delta_pose = clipped_pose.detach().clone()
        self._teacher_taskspace_target_pos, self._teacher_taskspace_target_quat = self._pose_from_delta(
            clipped_pose,
            base_pos=self._teacher_taskspace_target_pos,
            base_quat=self._teacher_taskspace_target_quat,
        )

    def _extend_observation_space(self, extra_dim):
        self.cfg.observation_space += extra_dim
        self.cfg.state_space += extra_dim
        if hasattr(self, "observation_space"):
            self.observation_space = gym.spaces.Box(
                low=-np.inf, high=np.inf, shape=(self.cfg.observation_space,), dtype=np.float32
            )
        if hasattr(self, "state_space") and hasattr(self.state_space, "shape"):
            self.state_space = gym.spaces.Box(
                low=-np.inf, high=np.inf, shape=(self.cfg.state_space,), dtype=np.float32
            )

    def _current_tavla_state(self):
        if self.joint_pos.shape[1] < 9:
            raise RuntimeError("TAVLA residual control requires 7 arm joints and 2 finger joints")
        finger_pos = self.joint_pos[:, 7:9]
        if finger_pos.shape[1] == 0:
            gripper = torch.zeros((self.num_envs,), device=self.device)
        else:
            gripper = finger_pos.mean(dim=1) / self.gripper_open_width_m
        gripper = torch.clamp(gripper, 0.0, 1.0).unsqueeze(-1)
        return torch.cat((self.joint_pos[:, :7], gripper), dim=-1)

    def _initialize_teacher_runtime(self):
        if self._teacher_control_mode == "kinematic_taskspace":
            self._tavla_twin_body_idx = self._tavla_twin_robot.body_names.index("panda_fingertip_centered")
            self._tavla_twin_jacobian_body_idx = self._tavla_twin_body_idx - (1 if self._tavla_twin_robot.is_fixed_base else 0)
        else:
            self._tavla_twin_body_idx = -1
            self._tavla_twin_jacobian_body_idx = -1
        if hasattr(self, "joint_pos"):
            current = self._current_tavla_state()
        else:
            # Factory/RealSim allocates joint_pos on the first reset. Keep
            # construction safe with the same arm reset used by PPO; the
            # actual state replaces this placeholder in _reset_idx().
            current = torch.zeros((self.num_envs, 8), device=self.device)
            current[:, :7] = torch.as_tensor(
                self.cfg.ctrl.reset_joints, dtype=torch.float32, device=self.device
            )
        self.teacher_target = current.clone()
        self.combined_joint_target = current.clone()
        self.teacher_joint_error = torch.zeros_like(current)
        self.residual_action = torch.zeros_like(current)
        self.next_action = current.clone()
        self._teacher_command_target = current[:, :7].clone()
        self._teacher_command_velocity = torch.zeros((self.num_envs, 7), device=self.device)
        self._teacher_command_gripper = current[:, 7].clone()
        self.teacher_action_index = torch.full(
            (self.num_envs,), -1, dtype=torch.long, device=self.device
        )
        self._teacher_chunk = torch.empty((0, 8), device=self.device)
        self._teacher_chunk_index = 0
        self._teacher_chunk_end = 0
        self._teacher_hold_count = self.teacher_hold_steps
        self._teacher_target_updated = False
        self._teacher_started = False
        self.teacher_timeouts = 0
        self.teacher_failures = 0
        self.teacher_target_out_of_limits_count = 0
        self.teacher_inference_latency_s = 0.0
        self.teacher_inference_count = 0
        self.last_teacher_inference_event = False
        self.last_teacher_inference_timeout = False
        self.last_teacher_action_nonfinite = False
        self.last_teacher_target_out_of_limits = False
        self._teacher_error_reported = False
        self.last_tavla_actual_state = torch.zeros((self.num_envs, 8), device=self.device)
        self.last_tavla_policy_state = torch.zeros((self.num_envs, 8), device=self.device)
        self.last_tavla_effort = torch.zeros((self.num_envs, 6), device=self.device)
        self.last_tavla_wrench_base = torch.zeros((self.num_envs, 6), device=self.device)
        self.last_tavla_wrench_final = torch.zeros((self.num_envs, 6), device=self.device)
        self.last_tavla_adapted_wrench = torch.zeros((self.num_envs, 6), device=self.device)
        self.last_tavla_server_effort = torch.zeros((self.num_envs, 6), device=self.device)
        self.last_tavla_wrench_matches_neg_base = torch.zeros(
            (self.num_envs,), dtype=torch.bool, device=self.device
        )
        self.last_tavla_server_effort_matches_final = torch.zeros(
            (self.num_envs,), dtype=torch.bool, device=self.device
        )
        self.last_tavla_server_effort_is_finite = torch.zeros(
            (self.num_envs,), dtype=torch.bool, device=self.device
        )
        self.last_tavla_payload_matches_sent = torch.zeros(
            (self.num_envs,), dtype=torch.bool, device=self.device
        )
        self.tavla_delta_pose = torch.zeros((self.num_envs, 6), device=self.device)
        self.delta_pos = torch.zeros((self.num_envs, 3), device=self.device)
        self.delta_yaw = torch.zeros((self.num_envs,), device=self.device)
        self.privileged_xy_error = torch.zeros((self.num_envs, 2), device=self.device)
        self.privileged_xy_guidance_delta = torch.zeros((self.num_envs, 7), device=self.device)
        self.privileged_xy_guidance_active = torch.zeros((self.num_envs,), dtype=torch.bool, device=self.device)
        self._tavla_twin_q = self._teacher_policy_reference_state.clone()
        self._tavla_twin_initialized = False
        self._tavla_mapping_failed = torch.zeros((self.num_envs,), dtype=torch.bool, device=self.device)
        self.tavla_mapping_failures = 0
        self.tavla_twin_ik_position_error = torch.full((self.num_envs,), float("inf"), device=self.device)
        self.tavla_twin_ik_rotation_error = torch.full((self.num_envs,), float("inf"), device=self.device)
        self.tavla_twin_ik_iterations = torch.zeros((self.num_envs,), dtype=torch.long, device=self.device)
        self.tavla_twin_ik_converged = torch.zeros((self.num_envs,), dtype=torch.bool, device=self.device)
        self._teacher_target_updated = False
        self._teacher_camera_warmup_done = False
        self._teacher_taskspace_delta_pose = torch.zeros((self.num_envs, 6), device=self.device)
        self._teacher_taskspace_q_delta = torch.zeros((self.num_envs, 7), device=self.device)
        self._teacher_taskspace_prev_q = self._tavla_twin_q.clone()
        self._teacher_taskspace_target_pos = torch.zeros((self.num_envs, 3), device=self.device)
        self._teacher_taskspace_target_quat = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device).repeat(self.num_envs, 1)
        if hasattr(self, "fingertip_midpoint_pos"):
            self._teacher_taskspace_target_pos = self.fingertip_midpoint_pos.clone()
            self._teacher_taskspace_target_quat = self.fingertip_midpoint_quat.clone()
        self.tavla_force_abort = torch.zeros((self.num_envs,), dtype=torch.bool, device=self.device)
        self.tavla_force_abort_count = 0


    def _teacher_wrench(self):
        """Return the sign-corrected wrench sent to TAVLA."""
        # Server-side norm_stats handle normalization. Do not apply the RL
        # observation bias/scale or simulated noise to the TAVLA payload.
        return self.wrench_final

    def _teacher_batch(self):
        if not hasattr(self, "tiled_camera") or self.tiled_camera is None:
            raise RuntimeError("TAVLA teacher requires the front camera")
        if not hasattr(self, "wrist_tiled_camera") or self.wrist_tiled_camera is None:
            raise RuntimeError("TAVLA teacher requires the wrist camera")
        if not hasattr(self, "wrench_base") or not hasattr(self, "wrench_final"):
            raise RuntimeError("TAVLA teacher requires base-frame and final wrench data")

        # TAVLA infers before the first physics substep. Refresh RTX cameras
        # here so the payload is the current reset/command frame, not the
        # previous render cached by the sensor during reset.
        self.sim.render()
        self.tiled_camera.update(self.physics_dt, force_recompute=True)
        self.wrist_tiled_camera.update(self.physics_dt, force_recompute=True)
        front_raw = self.tiled_camera.data.output["rgb"][0].detach().cpu().unsqueeze(0)
        wrist_raw = self.wrist_tiled_camera.data.output["rgb"][0].detach().cpu().unsqueeze(0)
        self.last_tavla_front = front_raw[0].clone()
        self.last_tavla_wrist = wrist_raw[0].clone()
        front = self._apply_teacher_visual_profile(front_raw, "front")
        wrist = self._apply_teacher_visual_profile(wrist_raw, "wrist")
        self.last_tavla_transformed_front = front[0].clone()
        self.last_tavla_transformed_wrist = wrist[0].clone()
        self._tavla_visual_frame_ready = True
        actual_state = self._current_tavla_state()
        if self._teacher_control_mode == "kinematic_taskspace":
            self._sync_twin_to_actual_pose()
            # Start a new absolute task-space command path at the actual pose.
            # Subsequent actions in this chunk accumulate from the previous
            # command target, so controller drift is not silently re-based.
            self._teacher_taskspace_target_pos = self.fingertip_midpoint_pos.clone()
            self._teacher_taskspace_target_quat = self.fingertip_midpoint_quat.clone()
            policy_state = torch.cat((self._tavla_twin_q, actual_state[:, 7:8]), dim=-1)
        else:
            if self._teacher_state_alignment and not self._teacher_started:
                self._teacher_sim_reference_state = actual_state[:, :7].detach().clone()
            policy_state = actual_state.clone()
            if self._teacher_state_alignment:
                policy_state[:, :7] += self._teacher_policy_reference_state - self._teacher_sim_reference_state
        self.last_tavla_actual_state = actual_state.detach().clone()
        self.last_tavla_policy_state = policy_state.detach().clone()
        state = policy_state[0].detach().cpu().unsqueeze(0)
        effort = self._teacher_wrench()
        if not torch.isfinite(self.wrench_base).all() or not torch.isfinite(effort).all():
            raise FloatingPointError("TAVLA wrench_base/wrench_final contains NaN or Inf")
        expected_final = -self.wrench_base
        matches_neg_base = torch.equal(effort, expected_final)
        if not matches_neg_base:
            raise RuntimeError("wrench_final must equal -wrench_base before TAVLA inference")
        self.last_tavla_effort = effort.detach().clone()
        self.last_tavla_wrench_base = self.wrench_base.detach().clone()
        self.last_tavla_wrench_final = effort.detach().clone()
        self.last_tavla_adapted_wrench = effort.detach().clone()
        self.last_tavla_wrench_matches_neg_base[:] = matches_neg_base
        effort = effort[0].detach().cpu().unsqueeze(0)
        return {
            "observation.images.front": front,
            "observation.images.left_wrist": wrist,
            "observation.state": state,
            "observation.effort": effort,
            "task": getattr(self.cfg, "teacher_prompt", "peg-in-hole"),
        }

    def _fetch_teacher_chunk(self):
        self.last_teacher_inference_event = True
        self.last_teacher_inference_timeout = False
        self.last_teacher_action_nonfinite = False
        self.last_teacher_target_out_of_limits = False
        try:
            start = time.perf_counter()
            chunk = self.teacher_policy.predict_action_chunk(self._teacher_batch())
            payload_effort = getattr(self.teacher_policy, "last_server_payload_effort", None)
            if payload_effort is None:
                raise RuntimeError("TAVLA client did not record the effort payload sent to Server")
            payload_effort = np.asarray(payload_effort, dtype=np.float32).reshape(-1)
            expected_effort = self.last_tavla_effort[0].detach().cpu().numpy().astype(
                np.float32, copy=False
            )
            payload_finite = bool(np.isfinite(payload_effort).all())
            payload_matches_final = bool(np.array_equal(payload_effort, expected_effort))
            if not payload_finite:
                raise FloatingPointError("Server effort payload contains NaN or Inf")
            if not payload_matches_final:
                raise RuntimeError("Server effort payload does not equal the adapted TAVLA effort")
            payload_matches_sent = bool(
                getattr(self.teacher_policy, "last_server_payload_matches_sent", False)
            )
            if not payload_matches_sent:
                raise RuntimeError("WebSocket effort bytes were not verified against the TAVLA payload")
            self.last_tavla_server_effort[0] = torch.as_tensor(
                payload_effort, dtype=torch.float32, device=self.device
            )
            self.last_tavla_server_effort_is_finite[:] = payload_finite
            self.last_tavla_server_effort_matches_final[:] = payload_matches_final
            self.last_tavla_payload_matches_sent[:] = payload_matches_sent
            elapsed = time.perf_counter() - start
            if torch.is_tensor(chunk):
                chunk = chunk.detach().cpu()
            else:
                chunk = torch.as_tensor(chunk, dtype=torch.float32)
            if chunk.ndim == 3:
                chunk = chunk[0]
            if chunk.ndim != 2 or chunk.shape[1] != 8 or chunk.shape[0] < 2:
                raise ValueError(f"TAVLA chunk must be (chunk, 8), got {tuple(chunk.shape)}")
            if not torch.isfinite(chunk).all():
                self.last_teacher_action_nonfinite = True
                raise ValueError("TAVLA chunk contains non-finite values")
            # The checkpoint uses normalized gripper semantics:
            # 0=closed and 1=open. Keep the teacher target in that domain
            # before converting it to the simulated finger joint position.
            chunk[:, 7] = torch.clamp(chunk[:, 7], 0.0, 1.0)

            self._teacher_chunk = chunk.to(self.device)
            start_index = max(1, int(getattr(self.cfg, "teacher_action_start_index", 1)))
            if start_index >= self._teacher_chunk.shape[0]:
                raise ValueError(f"teacher_action_start_index={start_index} is outside chunk length {self._teacher_chunk.shape[0]}")
            lower, upper = self._joint_limits()
            selected_target = self._teacher_chunk[start_index]
            out_of_limits = torch.any(
                (selected_target[:7] < lower[0]) | (selected_target[:7] > upper[0])
            )
            self.last_teacher_target_out_of_limits = bool(out_of_limits.detach().cpu())
            if self.last_teacher_target_out_of_limits:
                self.teacher_target_out_of_limits_count += 1
            self._teacher_chunk_index = start_index
            self._teacher_chunk_end = min(
                self._teacher_chunk.shape[0], start_index + self.teacher_replan_actions
            )
            self._teacher_started = True
            self._teacher_hold_count = 0
            self._teacher_target_updated = False
            self.teacher_inference_latency_s = float(elapsed)
            self.teacher_inference_count += 1
            self._teacher_error_reported = False
        except Exception as exc:
            self.teacher_failures += 1
            if isinstance(exc, TimeoutError):
                self.last_teacher_inference_timeout = True
                self.teacher_timeouts += 1
            self._teacher_chunk = torch.empty((0, 8), device=self.device)
            self._teacher_chunk_index = 0
            self._teacher_chunk_end = 0
            self._teacher_hold_count = self.teacher_hold_steps
            self._teacher_target_updated = False
            if not self._teacher_error_reported or self.teacher_failures % 100 == 0:
                print(f"[TAVLA] teacher inference failed; holding last safe target: {exc}")
                self._teacher_error_reported = True

    def _advance_teacher_target(self):
        if self._teacher_hold_count >= self.teacher_hold_steps:
            self._teacher_hold_count = 0

        if self._teacher_hold_count == 0:
            if self._teacher_chunk_index >= self._teacher_chunk_end:
                self._fetch_teacher_chunk()
            if self._teacher_chunk.shape[0] > 0 and self._teacher_chunk_index < self._teacher_chunk_end:
                self.teacher_action_index[:] = self._teacher_chunk_index
                self.teacher_target = self._teacher_chunk[self._teacher_chunk_index].unsqueeze(0)
                self._teacher_chunk_index += 1
                self._teacher_target_updated = True
                self._teacher_hold_count = 1
            return

        self._teacher_hold_count += 1

    def _joint_limits(self):
        limits = getattr(self._robot.data, "soft_joint_pos_limits", None)
        if limits is None:
            limits = getattr(self._robot.data, "joint_pos_limits", None)
        if limits is None:
            raise RuntimeError("Isaac Lab articulation does not expose joint position limits")
        if limits.ndim == 2:
            limits = limits.unsqueeze(0)
        return limits[:, :7, 0], limits[:, :7, 1]

    def _smooth_teacher_target(self, target):
        """Interpolate the 10 Hz teacher target through the 30 Hz sim command path."""
        if not self._teacher_action_interpolation:
            self._teacher_command_target = target[:, :7].detach().clone()
            self._teacher_command_velocity.zero_()
            self._teacher_command_gripper = target[:, 7].detach().clone()
            return target

        dt = max(float(self.step_dt), 1.0e-6)
        speed_scale = self._teacher_speed_scale
        velocity_limit = self._teacher_joint_velocity_limits * speed_scale
        acceleration_limit = self._teacher_joint_acceleration_limits * speed_scale
        q_delta = target[:, :7] - self._teacher_command_target
        desired_velocity = torch.clamp(q_delta / dt, -velocity_limit, velocity_limit)
        max_velocity_change = acceleration_limit * dt
        velocity_change = torch.clamp(
            desired_velocity - self._teacher_command_velocity,
            -max_velocity_change,
            max_velocity_change,
        )
        next_velocity = self._teacher_command_velocity + velocity_change
        next_target = self._teacher_command_target + next_velocity * dt
        q_step = next_target - self._teacher_command_target
        overshoot = (torch.abs(q_step) >= torch.abs(q_delta)) | (q_delta * q_step <= 0.0)
        next_target = torch.where(overshoot, target[:, :7], next_target)
        next_velocity = torch.where(overshoot, torch.zeros_like(next_velocity), next_velocity)

        gripper_step_limit = self._teacher_gripper_velocity_limit * speed_scale * dt
        gripper_delta = target[:, 7] - self._teacher_command_gripper
        gripper_step = torch.clamp(gripper_delta, -gripper_step_limit, gripper_step_limit)
        next_gripper = self._teacher_command_gripper + gripper_step
        gripper_done = torch.abs(gripper_step) >= torch.abs(gripper_delta)
        next_gripper = torch.where(gripper_done, target[:, 7], next_gripper)

        self._teacher_command_target = next_target
        self._teacher_command_velocity = next_velocity
        self._teacher_command_gripper = next_gripper
        smoothed = target.clone()
        smoothed[:, :7] = next_target
        smoothed[:, 7] = next_gripper
        return smoothed

    def _combine_teacher_and_residual(self):
        current = self._current_tavla_state()
        self.teacher_joint_error = self.teacher_target - current
        if bool(getattr(self.cfg, "teacher_eval_only", False)):
            self.residual_action.zero_()
        else:
            self.residual_action = self.actions.clone()

        target = self.teacher_target.clone()
        if self._teacher_state_alignment and self._teacher_action_state_alignment and self._teacher_control_mode != "kinematic_taskspace":
            target[:, :7] -= self._teacher_policy_reference_state - self._teacher_sim_reference_state
        self.privileged_xy_guidance_delta.zero_()
        self.privileged_xy_guidance_active.zero_()
        if self._privileged_xy_guidance:
            held_base_pos, _ = factory_utils.get_held_base_pose(
                self.held_pos,
                self.held_quat,
                self.cfg_task.name,
                self.cfg_task.fixed_asset_cfg,
                self.num_envs,
                self.device,
            )
            target_held_base_pos, _ = factory_utils.get_target_held_base_pose(
                self.fixed_pos,
                self.fixed_quat,
                self.cfg_task.name,
                self.cfg_task.fixed_asset_cfg,
                self.num_envs,
                self.device,
            )
            xy_error = target_held_base_pos[:, :2] - held_base_pos[:, :2]
            xy_dist = torch.linalg.vector_norm(xy_error, dim=-1)
            self.privileged_xy_error[:] = xy_error
            active = xy_dist > self._privileged_xy_guidance_gate_m
            self.privileged_xy_guidance_active[:] = active
            delta_pose = torch.zeros((self.num_envs, 2), device=self.device)
            delta_pose[:, :2] = self._privileged_xy_guidance_gain * xy_error
            delta_q = factory_control.get_delta_dof_pos(
                delta_pose,
                "dls",
                self.fingertip_midpoint_jacobian[:, :2, :7],
                self.device,
            )
            delta_q_norm = torch.linalg.vector_norm(delta_q, dim=-1, keepdim=True)
            max_step = max(self._privileged_xy_guidance_max_joint_step, 1e-6)
            delta_q = delta_q * torch.clamp(max_step / delta_q_norm.clamp_min(1e-6), max=1.0)
            delta_q = torch.where(active.unsqueeze(-1), delta_q, torch.zeros_like(delta_q))
            self.privileged_xy_guidance_delta[:] = delta_q
            weight = max(0.0, min(1.0, self._privileged_xy_guidance_weight))
            target[:, :7] += weight * delta_q
        if self._privileged_xyz_guidance:
            held_base_pos, held_base_quat = factory_utils.get_held_base_pose(
                self.held_pos,
                self.held_quat,
                self.cfg_task.name,
                self.cfg_task.fixed_asset_cfg,
                self.num_envs,
                self.device,
            )
            target_held_base_pos, target_held_base_quat = factory_utils.get_target_held_base_pose(
                self.fixed_pos,
                self.fixed_quat,
                self.cfg_task.name,
                self.cfg_task.fixed_asset_cfg,
                self.num_envs,
                self.device,
            )
            # Preserve the grasp translation measured in the running simulation.
            # The fixed-asset target is expressed in the held-peg base frame,
            # while this controller acts on the fingertip midpoint.
            desired_pos = self.fingertip_midpoint_pos + (target_held_base_pos - held_base_pos)
            # The restored Franka body frame has a fixed 90-degree-style
            # fingertip convention difference from the task-frame pose. Use
            # the measured successful original-asset pose as the stable target.
            desired_quat = self._privileged_insert_quat
            position_error = self._privileged_xyz_guidance_gain * (desired_pos - self.fingertip_midpoint_pos)
            _, rotation_error = factory_control.get_pose_error(
                fingertip_midpoint_pos=self.fingertip_midpoint_pos,
                fingertip_midpoint_quat=self.fingertip_midpoint_quat,
                ctrl_target_fingertip_midpoint_pos=self.fingertip_midpoint_pos,
                ctrl_target_fingertip_midpoint_quat=desired_quat,
                jacobian_type="geometric",
                rot_error_type="axis_angle",
            )
            position_pose = torch.cat((position_error, torch.zeros_like(position_error)), dim=-1)
            rotation_pose = torch.cat((torch.zeros_like(position_error), rotation_error), dim=-1)
            z_disp = held_base_pos[:, 2] - target_held_base_pos[:, 2]
            rotate_first = torch.logical_and(
                z_disp > float(getattr(self.cfg, "privileged_rotation_align_height", 0.05)),
                torch.linalg.vector_norm(rotation_error, dim=-1) > 0.01,
            )
            delta_pose = torch.where(rotate_first.unsqueeze(-1), rotation_pose, position_pose)
            delta_q = factory_control.get_delta_dof_pos(
                delta_pose,
                "dls",
                self.fingertip_midpoint_jacobian[:, :6, :7],
                self.device,
            )
            delta_q_norm = torch.linalg.vector_norm(delta_q, dim=-1, keepdim=True)
            max_step = max(self._privileged_xyz_guidance_max_joint_step, 1e-6)
            delta_q = delta_q * torch.clamp(max_step / delta_q_norm.clamp_min(1e-6), max=1.0)
            xyz_target = current[:, :7] + delta_q
            xyz_weight = max(0.0, min(1.0, self._privileged_xyz_guidance_weight))
            target[:, :7] = torch.lerp(target[:, :7], xyz_target, xyz_weight)
        if self._privileged_xyz_guidance and self._privileged_xy_guidance:
            xy_error = target_held_base_pos[:, :2] - held_base_pos[:, :2]
            xy_dist = torch.linalg.vector_norm(xy_error, dim=-1)
            active = xy_dist > self._privileged_xy_guidance_gate_m
            delta_pose_xy = torch.zeros((self.num_envs, 2), device=self.device)
            delta_pose_xy[:, :2] = self._privileged_xy_guidance_gain * xy_error
            delta_q_xy = factory_control.get_delta_dof_pos(
                delta_pose_xy,
                "dls",
                self.fingertip_midpoint_jacobian[:, :2, :7],
                self.device,
            )
            delta_q_norm = torch.linalg.vector_norm(delta_q_xy, dim=-1, keepdim=True)
            max_step = max(self._privileged_xy_guidance_max_joint_step, 1e-6)
            delta_q_xy = delta_q_xy * torch.clamp(max_step / delta_q_norm.clamp_min(1e-6), max=1.0)
            delta_q_xy = torch.where(active.unsqueeze(-1), delta_q_xy, torch.zeros_like(delta_q_xy))
            target[:, :7] += self._privileged_xy_guidance_weight * delta_q_xy
        target[:, 7] += self._gripper_residual_scale * self.residual_action[:, 7]
        target[:, 7] = torch.clamp(target[:, 7], 0.0, 1.0)
        lower, upper = self._joint_limits()
        target[:, :7] = torch.minimum(torch.maximum(target[:, :7], lower), upper)
        if self._teacher_control_mode == "kinematic_taskspace":
            if self._teacher_target_updated:
                self._prepare_taskspace_action(target)
                self._teacher_target_updated = False
        else:
            target = self._smooth_teacher_target(target)
        self.combined_joint_target = target
        self.teacher_joint_error = target - current
        self.next_action = target.clone()

    def _pre_physics_step(self, action):
        action = action.to(self.device)
        self.last_teacher_inference_event = False
        self.last_teacher_inference_timeout = False
        self.last_teacher_action_nonfinite = False
        self.last_teacher_target_out_of_limits = False
        if action.ndim != 2 or action.shape[1] != 8:
            raise ValueError(f"TAVLA residual action must have shape (num_envs, 8), got {tuple(action.shape)}")
        super()._pre_physics_step(action)
        if not self._teacher_started and not self._teacher_camera_warmup_done:
            # The RTX camera output immediately after reset can still be one
            # frame behind. Let one normal 30 Hz environment step complete
            # before the first policy query; keep the reset target for it.
            self._teacher_camera_warmup_done = True
            self._activate_teacher_position_servo()
            # IK/grasp reset can leave a residual joint velocity in PhysX.
            # Clear it only for this reset warmup step so waiting for the RTX
            # frame does not move the robot away from its reset qpos.
            reset_qpos = self._robot.data.joint_pos.clone()
            self._teacher_warmup_reset_qpos = reset_qpos.detach().clone()
            self._robot.write_joint_state_to_sim(reset_qpos, torch.zeros_like(reset_qpos))
            self.teacher_target = self._current_tavla_state().clone()
            self._teacher_target_updated = False
        else:
            if not self._teacher_started and hasattr(self, "_teacher_warmup_reset_qpos"):
                # Restore the exact reset state before the first policy query.
                # The warmup step exists only to advance RTX camera buffers;
                # it must not become part of the physical episode trajectory.
                reset_qpos = self._teacher_warmup_reset_qpos
                self._robot.write_joint_state_to_sim(reset_qpos, torch.zeros_like(reset_qpos))
                self.scene.write_data_to_sim()
                self.sim.forward()
                self.scene.update(dt=0.0)
                self._compute_intermediate_values(dt=self.physics_dt)
            self._advance_teacher_target()
        self._combine_teacher_and_residual()

    def _joint_target_delta_pose(self):
        current_q = self.joint_pos[:, :7]
        q_error = (self.combined_joint_target[:, :7] - current_q + torch.pi) % (2.0 * torch.pi) - torch.pi
        jacobian = self.fingertip_midpoint_jacobian[:, :6, :7]
        return torch.bmm(jacobian, q_error.unsqueeze(-1)).squeeze(-1)

    def _pose_from_delta(self, delta_pose, base_pos=None, base_quat=None):
        if base_pos is None:
            base_pos = self.fingertip_midpoint_pos
        if base_quat is None:
            base_quat = self.fingertip_midpoint_quat
        angle = torch.linalg.vector_norm(delta_pose[:, 3:6], dim=-1)
        axis = delta_pose[:, 3:6] / angle.unsqueeze(-1).clamp_min(1.0e-6)
        axis = torch.where(
            (angle > 1.0e-6).unsqueeze(-1), axis, torch.zeros_like(axis)
        )
        delta_quat = torch_utils.quat_from_angle_axis(angle, axis)
        target_quat = torch_utils.quat_mul(delta_quat, base_quat)
        if getattr(self.cfg.ctrl, "lock_fingertip_downward", False):
            _, _, target_yaw = torch_utils.get_euler_xyz(target_quat)
            target_quat = torch_utils.quat_from_euler_xyz(
                roll=torch.full_like(target_yaw, torch.pi),
                pitch=torch.zeros_like(target_yaw),
                yaw=target_yaw,
            )
        return base_pos + delta_pose[:, :3], target_quat

    def _update_reward_action_metrics(self, delta_pose):
        self.tavla_delta_pose = delta_pose
        self.delta_pos = delta_pose[:, :3]
        _, target_quat = self._pose_from_delta(delta_pose)
        _, _, target_yaw = torch_utils.get_euler_xyz(target_quat)
        _, _, current_yaw = torch_utils.get_euler_xyz(self.fingertip_midpoint_quat)
        self.delta_yaw = factory_utils.wrap_yaw(target_yaw) - factory_utils.wrap_yaw(current_yaw)

    def _apply_ppo_cartesian_action(self):
        delta_pose = self._joint_target_delta_pose()
        self._update_reward_action_metrics(delta_pose)
        clipped_pose = delta_pose.clone()
        clipped_pose[:, :3] = torch.clamp(
            clipped_pose[:, :3], -self.pos_threshold, self.pos_threshold
        )
        clipped_pose[:, 3:6] = torch.clamp(
            clipped_pose[:, 3:6], -self.rot_threshold, self.rot_threshold
        )
        target_pos, target_quat = self._pose_from_delta(clipped_pose)
        gripper_target = self.combined_joint_target[:, 7] * self.gripper_open_width_m
        self.generate_ctrl_signals(
            ctrl_target_fingertip_midpoint_pos=target_pos,
            ctrl_target_fingertip_midpoint_quat=target_quat,
            ctrl_target_gripper_dof_pos=gripper_target,
        )

    def _apply_action(self):
        if self.last_update_timestamp < self._robot._data._sim_timestamp:
            self._compute_intermediate_values(dt=self.physics_dt)

        if self._teacher_control_mode == "ppo_cartesian":
            self._apply_ppo_cartesian_action()
            return
        if self._teacher_control_mode == "kinematic_taskspace":
            gripper_target = self.combined_joint_target[:, 7] * self.gripper_open_width_m
            self.generate_ctrl_signals(
                ctrl_target_fingertip_midpoint_pos=self._teacher_taskspace_target_pos,
                ctrl_target_fingertip_midpoint_quat=self._teacher_taskspace_target_quat,
                ctrl_target_gripper_dof_pos=gripper_target,
            )
            return

        delta_pose = self._joint_target_delta_pose()
        self._update_reward_action_metrics(delta_pose)
        current_q = self.joint_pos[:, :7]
        self.ctrl_target_joint_pos[:, :7] = self.combined_joint_target[:, :7]
        gripper_target = self.combined_joint_target[:, 7:8] * self.gripper_open_width_m
        self.ctrl_target_joint_pos[:, 7:9] = gripper_target

        self._robot.set_joint_position_target(self.ctrl_target_joint_pos)
        if self._use_implicit_position_servo:
            self._robot.set_joint_effort_target(torch.zeros_like(self.joint_pos))
        else:
            q_error = (self.combined_joint_target[:, :7] - current_q + torch.pi) % (2.0 * torch.pi) - torch.pi
            joint_torque = self._joint_kp * q_error - self._joint_kd * self.joint_vel[:, :7]
            joint_torque = torch.clamp(joint_torque, -self._joint_effort_limits, self._joint_effort_limits)
            full_torque = torch.zeros_like(self.joint_pos)
            full_torque[:, :7] = joint_torque
            self._robot.set_joint_effort_target(full_torque)

    def _get_dones(self):
        terminated, time_out = super()._get_dones()
        if self._teacher_control_mode == "kinematic_taskspace":
            terminated = terminated | self._tavla_mapping_failed
        return terminated, time_out

    def _get_observations(self):
        observations = super()._get_observations()
        if hasattr(self, "teacher_target"):
            current = self._current_tavla_state()
            teacher_error = self.teacher_target - current
        else:
            current = torch.zeros((self.num_envs, 8), device=self.device)
            teacher_error = torch.zeros_like(current)
        observations["policy"] = torch.cat((observations["policy"], current, teacher_error), dim=-1)
        observations["critic"] = torch.cat((observations["critic"], current, teacher_error), dim=-1)
        return observations

    def _get_rewards(self):
        # Teacher-only evaluation uses exactly the existing PPO/Forge reward.
        # Residual penalty remains available only for a future residual PPO run.
        reward = super()._get_rewards()
        residual = getattr(self, "residual_action", torch.zeros_like(self.actions))
        if not bool(getattr(self.cfg, "teacher_eval_only", False)):
            penalty = float(self.cfg.residual_penalty_scale) * torch.sum(torch.square(residual), dim=-1)
            reward = reward - penalty
        if hasattr(self, "extras"):
            self.extras["tavla/residual_norm"] = torch.linalg.vector_norm(residual, dim=-1).mean()
            self.extras["tavla/teacher_target_norm"] = torch.linalg.vector_norm(
                self.teacher_target, dim=-1
            ).mean()
            self.extras["tavla/combined_target_norm"] = torch.linalg.vector_norm(
                self.combined_joint_target, dim=-1
            ).mean()
            self.extras["tavla/teacher_joint_error"] = torch.linalg.vector_norm(
                self.teacher_joint_error[:, :7], dim=-1
            ).mean()
            self.extras["tavla/gripper_target"] = self.teacher_target[:, 7].mean()
            self.extras["tavla/gripper_actual"] = self._current_tavla_state()[:, 7].mean()
            self.extras["tavla/wrench_base_mean"] = self.wrench_base.mean()
            self.extras["tavla/wrench_base_std"] = self.wrench_base.std(unbiased=False)
            self.extras["tavla/privileged_xy_error_m"] = torch.linalg.vector_norm(
                self.privileged_xy_error, dim=-1
            ).mean()
            self.extras["tavla/privileged_xy_guidance_active"] = self.privileged_xy_guidance_active.float().mean()
            self.extras["tavla/teacher_latency_s"] = self.teacher_inference_latency_s
            self.extras["tavla/teacher_failures"] = self.teacher_failures
            self.extras["tavla/teacher_timeouts"] = self.teacher_timeouts
        return reward

    def close(self):
        if hasattr(self, "teacher_policy"):
            self.teacher_policy.close()
        super().close()

    def _reset_idx(self, env_ids):
        super()._reset_idx(env_ids)
        # DirectRLEnv may reset while the RealSim base class is still being
        # constructed. The teacher queue is initialized immediately after
        # that construction completes.
        if not hasattr(self, "teacher_target"):
            return
        self.actions[env_ids] = 0.0
        self.prev_actions[env_ids] = 0.0
        if hasattr(self, "teacher_policy"):
            try:
                self.teacher_policy.reset()
            except Exception as exc:
                self.teacher_failures += 1
                print(f"[TAVLA] teacher reset failed; holding safe target: {exc}")
        self.teacher_timeouts = 0
        self.teacher_failures = 0
        self.teacher_target_out_of_limits_count = 0
        self.teacher_inference_latency_s = 0.0
        self.teacher_inference_count = 0
        current = self._current_tavla_state()
        self._teacher_sim_reference_state[env_ids] = current[env_ids, :7]
        self.teacher_target[env_ids] = current[env_ids]
        self.combined_joint_target[env_ids] = current[env_ids]
        self._teacher_command_target[env_ids] = current[env_ids, :7]
        self._teacher_command_velocity[env_ids] = 0.0
        self._teacher_command_gripper[env_ids] = current[env_ids, 7]
        self.teacher_action_index[env_ids] = -1
        self.teacher_joint_error[env_ids] = 0.0
        self.residual_action[env_ids] = 0.0
        self._teacher_chunk = torch.empty((0, 8), device=self.device)
        self._teacher_chunk_index = 0
        self._teacher_chunk_end = 0
        self._teacher_hold_count = self.teacher_hold_steps
        self._teacher_target_updated = False
        self._teacher_camera_warmup_done = False
        if getattr(self, "_privileged_xyz_guidance", False) and self.cfg_task.name == "peg_insert":
            # The restored original Franka asset uses a different fingertip
            # body-frame convention. Re-orient the held peg root to the hole
            # frame only for the privileged oracle reset; pure TAVLA is unchanged.
            _, target_held_quat = factory_utils.get_target_held_base_pose(
                self.fixed_pos,
                self.fixed_quat,
                self.cfg_task.name,
                self.cfg_task.fixed_asset_cfg,
                self.num_envs,
                self.device,
            )
            held_state = self._held_asset.data.root_state_w.clone()
            held_state[env_ids, 3:7] = target_held_quat[env_ids]
            held_state[env_ids, 7:] = 0.0
            self._held_asset.write_root_pose_to_sim(held_state[env_ids, :7], env_ids=env_ids)
            self._held_asset.write_root_velocity_to_sim(held_state[env_ids, 7:], env_ids=env_ids)
            self._held_asset.reset()
            self.scene.write_data_to_sim()
            self.sim.forward()
            self.scene.update(dt=self.physics_dt)
            self._compute_intermediate_values(dt=self.physics_dt)
        self._tavla_twin_q[env_ids] = self._teacher_policy_reference_state[env_ids]
        self._tavla_twin_initialized = False
        self._tavla_mapping_failed[env_ids] = False
        self.tavla_force_abort[env_ids] = False
        if self._teacher_control_mode == "kinematic_taskspace":
            self._sync_twin_to_actual_pose()
        self._teacher_taskspace_target_pos[env_ids] = self.fingertip_midpoint_pos[env_ids]
        self._teacher_taskspace_target_quat[env_ids] = self.fingertip_midpoint_quat[env_ids]
        self._teacher_taskspace_delta_pose[env_ids] = 0.0
        self._teacher_taskspace_q_delta[env_ids] = 0.0
        self._teacher_taskspace_prev_q[env_ids] = self._tavla_twin_q[env_ids]
        self.delta_pos[env_ids] = 0.0
        self.delta_yaw[env_ids] = 0.0
        self._teacher_started = False
        self._tavla_visual_frame_ready = False
        self.next_action[env_ids] = current[env_ids]
        self.privileged_xy_error[env_ids] = 0.0
        self.privileged_xy_guidance_delta[env_ids] = 0.0
        self.privileged_xy_guidance_active[env_ids] = False
