"""Sim-data-aligned force-trend evaluation on the validated TAVLA controller."""

from __future__ import annotations

import importlib
import json
import sys
import time
from dataclasses import replace
from pathlib import Path

import h5py
import numpy as np
import torch
from isaaclab.utils import configclass

from .policy.modeling_sim_data_force_trend import SimDataForceTrendPolicy
from .sim_data_visual import (
    SIM_DATA_GRIPPER_STATE,
    apply_visual_domain_randomization,
    capture_sim_data_rgb,
    configure_sim_data_visual_scene,
    load_sim_data_profile,
    make_image_rng,
)
from .tavla_residual_env import TavlaResidualEnv
from .tavla_residual_env_cfg import RealSimTavlaTeacherPegInsertCfg


def _repo_path(path: str | Path) -> Path:
    resolved = Path(path).expanduser()
    if not resolved.is_absolute():
        resolved = Path(__file__).resolve().parents[4] / resolved
    return resolved.resolve()


def _load_aligned_profile(cfg):
    """Load visuals from sim-data and the matching state from sim-data-aligned."""

    profile = load_sim_data_profile(cfg.sim_data_dir, int(cfg.sim_data_profile_id))
    aligned_directory = _repo_path(cfg.sim_data_aligned_dir) / f"traj_{int(cfg.sim_data_profile_id)}"
    aligned_h5 = aligned_directory / "data.h5"
    if not aligned_h5.is_file():
        raise FileNotFoundError(f"sim-data-aligned H5 not found: {aligned_h5}")
    with h5py.File(aligned_h5, "r") as h5:
        q0 = np.asarray(h5["obs/state/joint_pos"][0], dtype=np.float32)
        frame_count = int(h5["obs/state/joint_pos"].shape[0])
        gripper = float(np.asarray(h5["obs/state/gripper_pos"][0]).reshape(-1)[0])
    if q0.shape != (7,) or not np.isfinite(q0).all():
        raise ValueError(f"Aligned initial joint state must be finite (7,), got {q0}")
    if frame_count != profile.frame_count:
        raise ValueError(f"Aligned/visual frame count mismatch: {frame_count} != {profile.frame_count}")
    if not np.allclose(q0, profile.initial_joint_pos, rtol=0.0, atol=1.0e-7):
        raise RuntimeError("sim-data and sim-data-aligned initial joints differ")
    if abs(gripper - SIM_DATA_GRIPPER_STATE) > 1.0e-6:
        raise RuntimeError(f"Aligned gripper must be {SIM_DATA_GRIPPER_STATE}, got {gripper}")
    return replace(profile, directory=aligned_directory, h5_path=aligned_h5, initial_joint_pos=q0)


@configclass
class RealSimSimDataAlignedPegInsertCfg(RealSimTavlaTeacherPegInsertCfg):
    """The old TAVLA test configuration plus aligned visuals and force features."""

    sim_data_dir: str = "sim-data"
    sim_data_aligned_dir: str = "sim-data-aligned"
    sim_data_profile_id: int = 0
    sim_data_prompt: str = "peg-in-hole"
    openpi_root: str = "sim_side_test_bundle"
    wrench_adapter_path: str = "sim_side_test_bundle/assets/wrench_adapters/sim_aligned_to_real_affine.pt"
    force_trend_config_path: str = "sim_side_test_bundle/configs/tavla_sim_force_trend_affine.json"
    sim_data_reset_joint_tolerance_rad: float = 0.03


class SimDataAlignedEnv(TavlaResidualEnv):
    """Run the force-trend model through the validated 80-run TAVLA path."""

    teacher_policy_class = SimDataForceTrendPolicy

    def __init__(self, cfg, render_mode=None, **kwargs):
        self._capture_force_wrench = False
        self._force_wrench_samples: list[np.ndarray] = []
        self._force_trend_buffer = None
        self._force_trend_buffer_class = None
        self._wrench_adapter = None
        self.force_trend_config = None
        self.last_sim_aligned_wrench = np.zeros(6, dtype=np.float32)
        self.last_real_aligned_wrench = np.zeros(6, dtype=np.float32)
        self.last_force_trend_effort = np.zeros((10, 10), dtype=np.float32)
        self._sim_data_reset_grasp_active = False
        self._sim_data_reset_joint_servo_ready = False

        profile = _load_aligned_profile(cfg)
        self.sim_data_profile = profile
        self.sim_data_reset_validation = None
        self._sim_data_image_rng = make_image_rng(profile.domain_randomization)

        teacher_cfg = cfg.teacher_policy_cfg
        action_start_index = int(cfg.teacher_action_start_index)
        replan_actions = int(cfg.teacher_replan_actions)
        configure_sim_data_visual_scene(cfg)
        # The replay-only left visual shell overlaps the live articulation.
        # Direct joint replay can tolerate that duplicate because it rewrites
        # q every frame; online PhysX control cannot. Render the live robot.
        cfg.replay_left_background_visual_shell = False
        cfg.teacher_policy_cfg = teacher_cfg
        cfg.teacher_action_start_index = action_start_index
        cfg.teacher_replan_actions = replan_actions

        # Preserve the controller/reset contract used by the successful
        # 80-episode TAVLA evaluator. Force processing remains observational.
        cfg.teacher_control_mode = "aligned_joint"
        cfg.teacher_action_interpolation = True
        cfg.teacher_execution_position_servo = True
        cfg.use_implicit_position_servo = False
        cfg.task.skip_reset_ik = True
        cfg.task.use_pi0_reset_pose = True
        cfg.ctrl.reset_joints = profile.initial_joint_pos.tolist()
        cfg.pi0_hole_init_pos = tuple(profile.hole_position_m)
        cfg.task.fixed_asset.init_state.pos = tuple(profile.hole_position_m)
        cfg.task.fixed_asset.init_state.rot = tuple(cfg.pi0_hole_init_rot)
        cfg.task.fixed_asset_init_pos_noise = [0.0, 0.0, 0.0]
        cfg.task.hand_init_pos_noise = [0.0, 0.0, 0.0]
        cfg.pi0_peg_mount_depth_adjust_m = 0.0
        cfg.pi0_policy_cfg.gripper_state_override = SIM_DATA_GRIPPER_STATE
        cfg.teacher_policy_cfg.n_action_steps = 50
        cfg.ft_parent_body_name = "panda_link7"
        cfg.ft_raw_wrench_frame = "parent_body"
        cfg.ft_raw_torque_reference = "parent_origin"
        cfg.ft_apply_child_joint_frame_calibration = True
        apply_visual_domain_randomization(cfg, profile.domain_randomization)

        self._initialize_force_trend_runtime(cfg)
        super().__init__(cfg, render_mode, **kwargs)
        self._capture_force_wrench = True

    def _initialize_force_trend_runtime(self, cfg) -> None:
        openpi_root = _repo_path(cfg.openpi_root)
        for candidate in (openpi_root / "src", openpi_root):
            if candidate.is_dir() and str(candidate) not in sys.path:
                sys.path.insert(0, str(candidate))
        adapter_path = _repo_path(cfg.wrench_adapter_path)
        config_path = _repo_path(cfg.force_trend_config_path)
        if not adapter_path.is_file():
            raise FileNotFoundError(f"wrench adapter not found: {adapter_path}")
        if not config_path.is_file():
            raise FileNotFoundError(f"force-trend config not found: {config_path}")
        with config_path.open(encoding="utf-8") as handle:
            self.force_trend_config = json.load(handle)
        representation = self.force_trend_config["representation"]
        force_features = importlib.import_module("openpi.force_features")
        wrench_adapter = importlib.import_module("openpi.shared.wrench_adapter")
        self._force_trend_buffer_class = force_features.RealForceTrendBuffer
        self._force_trend_buffer_args = {
            "baseline_frames": 20,
            "smoothing_frames": int(representation.get("causal_smoothing_frames", 3)),
            "trend_lag_frames": int(representation.get("trend_lag_frames", 5)),
            "epsilon": float(representation.get("epsilon", 1.0e-6)),
            "active_noise_mad_multiplier": float(representation.get("active_noise_mad_multiplier", 6.0)),
            "active_noise_floor": float(representation.get("active_noise_floor_newton", 1.0e-6)),
        }
        self._wrench_adapter = wrench_adapter.load_adapter(str(adapter_path))
        self._reset_force_trend_runtime()

    def _reset_force_trend_runtime(self) -> None:
        self._force_trend_buffer = self._force_trend_buffer_class(**self._force_trend_buffer_args)
        self._force_wrench_samples.clear()
        self.last_sim_aligned_wrench.fill(0.0)
        self.last_real_aligned_wrench.fill(0.0)
        self.last_force_trend_effort.fill(0.0)

    def _update_wrench(self):
        super()._update_wrench()
        if not self._capture_force_wrench:
            return
        wrench = self.wrench_final[0].detach().cpu().numpy().astype(np.float32)
        if wrench.shape != (6,) or not np.isfinite(wrench).all():
            raise FloatingPointError(f"panda_link7 wrench must be finite (6,), got {wrench}")
        self._force_wrench_samples.append(wrench.copy())

    def _consume_force_trend_sample(self) -> None:
        if self._force_wrench_samples:
            raw = np.mean(np.asarray(self._force_wrench_samples, dtype=np.float32), axis=0)
            self._force_wrench_samples.clear()
        else:
            raw = self.wrench_final[0].detach().cpu().numpy().astype(np.float32)
        aligned = np.asarray(self._wrench_adapter.transform_numpy(raw[None]), dtype=np.float32)
        if aligned.shape != (1, 6) or not np.isfinite(aligned).all():
            raise ValueError(f"adapted wrench must be finite (1, 6), got {aligned.shape}")
        self._force_trend_buffer.update(aligned[0, :3])
        effort = np.asarray(self._force_trend_buffer.history(10), dtype=np.float32)
        if effort.shape != (10, 10) or not np.isfinite(effort).all():
            raise ValueError(f"force-trend history must be finite (10, 10), got {effort.shape}")
        self.last_sim_aligned_wrench = raw.copy()
        self.last_real_aligned_wrench = aligned[0].copy()
        self.last_force_trend_effort = np.ascontiguousarray(effort)

    def _record_tavla_effort(self):
        """Replace only the old 6-D history input; control stays unchanged."""

        self._consume_force_trend_sample()

    def _sync_sim_data_left_visual(self) -> None:
        if (
            not getattr(self.cfg, "replay_left_background_visual_shell", False)
            or not hasattr(self, "joint_pos")
        ):
            return
        self._pose_background_robot_visual(
            self.sim.stage,
            "/World/envs/env_0/franka_env/LeftRobotVisual/Robot/franka",
            self.joint_pos[0, :7].detach().cpu().numpy(),
            float(self._current_tavla_state()[0, 7].detach().cpu()) * 0.04,
        )

    def _sim_data_camera_observation(self):
        self._sync_sim_data_left_visual()
        return capture_sim_data_rgb(
            self,
            self.sim_data_profile.domain_randomization,
            self._sim_data_image_rng,
        )

    def _teacher_batch(self):
        if not np.any(self.last_force_trend_effort) and not self._force_wrench_samples:
            self._consume_force_trend_sample()
        front, wrist = self._sim_data_camera_observation()
        self.last_tavla_front = torch.from_numpy(front.copy())
        self.last_tavla_wrist = torch.from_numpy(wrist.copy())
        self.last_model_input_front = self.last_tavla_front.clone()
        self.last_model_input_wrist = self.last_tavla_wrist.clone()
        self._model_visual_frame_ready = True
        actual_state = self._current_tavla_state()
        self.last_tavla_actual_state = actual_state.detach().clone()
        self.last_tavla_policy_state = self.last_tavla_actual_state.clone()
        return {
            "observation.images.front": torch.from_numpy(front.copy()).unsqueeze(0),
            "observation.images.left_wrist": torch.from_numpy(wrist.copy()).unsqueeze(0),
            "observation.state": actual_state.detach().cpu(),
            "observation.effort": torch.from_numpy(
                self.last_force_trend_effort.copy()
            ).unsqueeze(0),
            "task": str(self.cfg.sim_data_prompt),
        }

    def _fetch_teacher_chunk(self):
        """Use the old TAVLA queue semantics with a [10,10] effort contract."""

        self.last_teacher_inference_event = True
        self.last_teacher_inference_timeout = False
        self.last_teacher_action_nonfinite = False
        self.last_teacher_target_out_of_limits = False
        try:
            started = time.perf_counter()
            chunk = self.teacher_policy.predict_action_chunk(self._teacher_batch())
            effort = np.asarray(self.teacher_policy.last_server_payload_effort, dtype=np.float32)
            if effort.shape != (10, 10) or not np.isfinite(effort).all():
                raise ValueError(f"TAVLA force-trend effort must be finite (10,10), got {effort.shape}")
            if not np.array_equal(effort, self.last_force_trend_effort):
                raise RuntimeError("WebSocket force-trend payload differs from prepared effort")
            chunk = torch.as_tensor(chunk, dtype=torch.float32)
            if chunk.ndim == 3:
                chunk = chunk[0]
            if chunk.shape != (50, 8) or not torch.isfinite(chunk).all():
                raise ValueError(f"TAVLA chunk must be finite (50,8), got {tuple(chunk.shape)}")
            pi0_cfg = self.cfg.pi0_policy_cfg
            chunk[:, 7] = torch.clamp(
                chunk[:, 7],
                float(pi0_cfg.gripper_policy_min),
                float(pi0_cfg.gripper_policy_max),
            )
            self._teacher_chunk = chunk.to(self.device)
            start_index = max(1, int(self.cfg.teacher_action_start_index))
            if start_index >= len(self._teacher_chunk):
                raise ValueError(f"teacher_action_start_index={start_index} is outside chunk")
            lower, upper = self._joint_limits()
            selected = self._teacher_chunk[start_index, :7]
            out_of_limits = torch.any((selected < lower[0]) | (selected > upper[0]))
            self.last_teacher_target_out_of_limits = bool(out_of_limits.detach().cpu())
            self.teacher_target_out_of_limits_count += int(self.last_teacher_target_out_of_limits)
            self._teacher_chunk_index = start_index
            self._teacher_chunk_end = min(len(self._teacher_chunk), start_index + self.teacher_replan_actions)
            self._teacher_started = True
            self._teacher_hold_count = 0
            self._teacher_target_updated = False
            self.teacher_inference_latency_s = time.perf_counter() - started
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
                print(f"[SimDataAlignedTAVLA] inference failed; holding safe target: {exc}", flush=True)
                self._teacher_error_reported = True

    def _refresh_model_input_video_frame(self):
        front, wrist = self._sim_data_camera_observation()
        self.last_model_input_front = torch.from_numpy(front.copy())
        self.last_model_input_wrist = torch.from_numpy(wrist.copy())
        self._model_visual_frame_ready = True
        return True

    def randomize_initial_state(self, env_ids):
        """Clear the default peg before installing the profile arm pose."""

        held_state = self._held_asset.data.default_root_state[env_ids].clone()
        held_state[:, 0:3] = self.scene.env_origins[env_ids]
        held_state[:, 2] += 2.0
        held_state[:, 7:] = 0.0
        self._held_asset.write_root_pose_to_sim(held_state[:, :7], env_ids=env_ids)
        self._held_asset.write_root_velocity_to_sim(held_state[:, 7:], env_ids=env_ids)
        self._held_asset.reset()
        return super().randomize_initial_state(env_ids)

    def close_gripper_in_place(self):
        """Close fingers while the runtime joint servo holds the profile pose."""

        if not self._sim_data_reset_grasp_active:
            return super().close_gripper_in_place()
        if not self._sim_data_reset_joint_servo_ready:
            stiffness = self._robot.data.joint_stiffness.clone()
            damping = self._robot.data.joint_damping.clone()
            stiffness[:, :7] = torch.as_tensor(
                self.cfg.joint_target_kp, dtype=torch.float32, device=self.device
            ).view(1, 7)
            damping[:, :7] = torch.as_tensor(
                self.cfg.joint_target_kd, dtype=torch.float32, device=self.device
            ).view(1, 7)
            self._robot.write_joint_stiffness_to_sim(stiffness)
            self._robot.write_joint_damping_to_sim(damping)
            self._sim_data_reset_joint_servo_ready = True
        self.ctrl_target_joint_pos[:, :7] = torch.as_tensor(
            self.sim_data_profile.initial_joint_pos,
            dtype=torch.float32,
            device=self.device,
        ).view(1, 7)
        self.ctrl_target_joint_pos[:, 7:9] = 0.0
        self._robot.set_joint_position_target(self.ctrl_target_joint_pos)
        self._robot.set_joint_effort_target(torch.zeros_like(self.joint_pos))

    def _reset_idx(self, env_ids):
        self._sim_data_reset_joint_servo_ready = False
        self._sim_data_reset_grasp_active = True
        try:
            super()._reset_idx(env_ids)
        finally:
            self._sim_data_reset_grasp_active = False
        self._reset_force_trend_runtime()
        self._sim_data_image_rng = make_image_rng(self.sim_data_profile.domain_randomization)
        schedule = getattr(self, "tavla_direct_reset_joints_schedule", None)
        if schedule:
            q = self.joint_pos[env_ids, :7].detach().cpu().numpy().reshape(7)
            q_error = q - self.sim_data_profile.initial_joint_pos
            hole = self._fixed_asset.data.root_pos_w[env_ids].detach().cpu().numpy().reshape(3)
            expected_hole = np.asarray(self.sim_data_profile.hole_position_m, dtype=np.float32)
            self.sim_data_reset_validation = {
                "q_expected": self.sim_data_profile.initial_joint_pos.tolist(),
                "q_sim": q.tolist(),
                "q_max_abs_error_rad": float(np.max(np.abs(q_error))),
                "hole_expected_m": expected_hole.tolist(),
                "hole_sim_m": hole.tolist(),
                "hole_max_abs_error_m": float(np.max(np.abs(hole - expected_hole))),
            }
            print(
                "[SimDataAlignedTAVLA] reset_validation="
                f"{self.sim_data_reset_validation}",
                flush=True,
            )
            if self.sim_data_reset_validation["q_max_abs_error_rad"] > float(
                self.cfg.sim_data_reset_joint_tolerance_rad
            ):
                raise RuntimeError(f"Aligned reset drifted: {self.sim_data_reset_validation}")
