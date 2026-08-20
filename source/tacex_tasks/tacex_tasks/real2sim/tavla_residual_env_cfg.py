from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass

from .policy.configuration_pi0remote import PI0RemoteTAVLAConfig
from .realsim_env_cfg import RealSimTaskPegInsertCfg


@configclass
class RealSimTavlaResidualPegInsertCfg(RealSimTaskPegInsertCfg):
    """Single-env residual PPO config around the frozen TAVLA teacher."""

    action_space: int = 8
    scene = InteractiveSceneCfg(
        num_envs=1,
        env_spacing=2.0,
        replicate_physics=True,
        filter_collisions=True,
        clone_in_fabric=False,
    )
    policy_cfg = None
    teacher_policy_cfg = PI0RemoteTAVLAConfig(
        n_action_steps=50,
        num_history_steps=1,
        history_step_interval=1,
    )
    teacher_prompt = "peg-in-hole"
    # Affine sim-to-real wrench adapter used only by the affine deployment
    # task. The legacy Teacher task does not load this adapter.
    baseline_adapter_path: str = "checkpoints/unpaired_sim_to_real_affine.pt"
    # The PPO reset stays unchanged. Translate policy observations and absolute
    # teacher targets through the same per-episode real/simulation joint offset.
    teacher_state_alignment: bool = False
    teacher_action_state_alignment: bool = False
    teacher_policy_reference_state: list = [-0.05648576654493809, 0.06290022935718298, 0.2503179907798767, -1.990307331085205, -0.035843100398778915, 2.102778196334839, 1.0133224725723267]
    # Skip the action corresponding to the current observation on every
    # network round-trip, including replans after the first chunk.
    teacher_action_start_index: int = 1
    teacher_hold_steps: int = 3
    # First closed-loop pass: use actions[1] and replan after one predicted action.
    teacher_replan_actions: int = 5
    teacher_action_interpolation: bool = True
    teacher_speed_scale: float = 1.0
    # Robust P95 limits fitted from the 40 real H5 trajectories. Units are
    # rad/s and rad/s^2, and are applied in the 30 Hz simulation command path.
    teacher_joint_velocity_limits: list = [0.07953, 0.13770, 0.00148, 0.15825, 0.05380, 0.12145, 0.09179]
    teacher_joint_acceleration_limits: list = [0.29079, 1.53436, 0.01800, 0.63842, 0.72472, 1.45085, 0.73457]
    teacher_gripper_velocity_limit: float = 2.0
    teacher_eval_only: bool = False
    teacher_control_mode: str = "aligned_joint"
    teacher_visual_profile: str = "raw"
    teacher_camera_calibration: str = ""
    # p99 from all 40 real_data/traj_*/data.h5 files. Translation is
    # [m/s] and rotation is [rad/s]; one scale preserves the full 6D direction.
    teacher_taskspace_velocity_limits: list = [0.13797, 0.15633, 0.14112, 0.00260, 0.00845, 0.00215]
    # p99 of the complete base-frame [F, T] wrench norm from the same 40 files.
    teacher_force_norm_p99: float = 13.49642
    # TAVLA was trained with the real robot's raw base-frame wrench. These
    # compatibility fields are not applied in the baseline; the server owns
    # normalization through the checkpoint's norm_stats.
    teacher_wrench_scale: list = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    teacher_wrench_bias: list = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    # Optional sim-only oracle alignment. It uses privileged held-peg and hole poses plus the Franka Jacobian to correct XY drift while leaving TAVLA responsible for insertion and gripper behavior.
    # Keep the default evaluation pure TAVLA. Enable this explicitly for a
    # privileged XY-ablation/diagnostic run; it is not part of the teacher.
    privileged_xy_guidance: bool = False
    privileged_xy_guidance_weight: float = 1.0
    privileged_xy_guidance_gain: float = 3.0
    privileged_xy_guidance_max_joint_step: float = 0.20
    privileged_xy_guidance_gate_m: float = 0.003
    # Diagnostic sim-privileged Cartesian execution path. Disabled by default;
    # when enabled, it is an oracle upper-bound check for the reward pipeline.
    privileged_xyz_guidance: bool = False
    privileged_xyz_guidance_weight: float = 1.0
    privileged_insert_quat: list = [0.0243090261, 0.9994032977, -0.0245799627, 0.0001990777]
    privileged_rotation_align_height: float = 0.05
    privileged_xyz_guidance_gain: float = 3.0
    privileged_xyz_guidance_max_joint_step: float = 0.20
    use_success_prediction: bool = False
    gripper_open_width_m: float = 0.04
    joint_residual_scale: list = [0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01]
    gripper_residual_scale: float = 0.05
    residual_penalty_scale: float = 0.01

    # Dynamics-aligned joint-space PD from the saved hierarchical replay
    # validation. The first four joints use the arm1 fit and the last three
    # use the arm2 fit; force/torque alignment remains intentionally separate.
    joint_target_kp: list = [303.3830260094937] * 4 + [148.56203519189626] * 3
    joint_target_kd: list = [36.850833024805276] * 4 + [17.460777084541743] * 3
    use_implicit_position_servo: bool = True
    joint_target_effort_limits: list = [87.0, 87.0, 87.0, 87.0, 12.0, 12.0, 12.0]


@configclass
class RealSimTavlaTeacherPegInsertCfg(RealSimTavlaResidualPegInsertCfg):
    """Teacher-only mode; the RL action is ignored and residual is zero."""

    teacher_eval_only: bool = True
    # Activate the existing joint position servo after reset, so absolute
    # TAVLA joint targets are tracked without changing the robot dynamics.
    teacher_execution_position_servo: bool = True
    # Match the reset-time control path used by the Direct environment that
    # produced sim_wrench_final_50. The implicit servo changes the dynamics
    # during the DLS reset IK iterations and can select a different IK branch.
    # This override is limited to teacher-only evaluation; residual PPO keeps
    # its own configured controller.
    use_implicit_position_servo: bool = False
