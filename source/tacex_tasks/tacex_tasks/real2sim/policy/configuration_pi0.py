from dataclasses import dataclass, field

from .configuration_pi0remote import PI0RemoteConfig


@dataclass
class PI0RemoteJointConfig(PI0RemoteConfig):
    """Configuration for the current image/state-only Pi0 server contract."""

    type: str = "pi0remote_joint"
    host_ip: str = "114.214.164.36"
    host_port: int = 8000
    n_action_steps: int = 50
    action_dim: int = 8

    # The real-data policy runs at 10 Hz. The simulation environment uses a
    # 100 ms environment step for this policy, so one action is consumed per
    # environment step and a fresh chunk is requested after a few actions.
    control_hz: float = 10.0
    action_start_index: int = 2
    replan_actions: int = 3

    # Pi0's gripper field is a raw training-domain value, not a [0, 1]
    # fraction. The real trajectory is effectively constant around this value.
    # Keep the policy observation in that domain while holding the simulator's
    # physical finger target fixed for the whole episode.
    gripper_policy_min: float = 0.0
    gripper_policy_max: float = 0.094847
    gripper_state_override: float = 0.08652404
    hold_gripper: bool = True
    gripper_state_scale: float = 1.0
    gripper_action_to_finger_scale: float = 0.5
    gripper_policy_velocity_limit: float = 0.0

    # Hardware-facing safe default. With the 10 Hz control period, the target
    # slew limiter allows at most 0.0015 rad per control step (0.015 rad/s).
    joint_velocity_limits: list[float] = field(default_factory=lambda: [0.015] * 7)
    joint_kp: list[float] = field(
        default_factory=lambda: [100.0] * 4 + [50.0] * 3
    )
    joint_kd: list[float] = field(
        default_factory=lambda: [20.0] * 4 + [10.0] * 3
    )
    joint_effort_limits: list[float] = field(
        default_factory=lambda: [30.0, 30.0, 30.0, 30.0, 8.0, 8.0, 8.0]
    )
    # Preserve Pi0's absolute joint targets as the rollout replay does. The
    # joint-space path must not replace them with an orientation-projected IK
    # solution; orientation constraints apply only to optional task-space mode.
    use_taskspace_controller: bool = False
    hold_wrist_joints: bool = False
    enforce_fingertip_orientation: bool = False
    use_implicit_position_controller: bool = True
    implicit_arm_stiffness: float = 200.0
    implicit_arm_damping: float = 40.0
