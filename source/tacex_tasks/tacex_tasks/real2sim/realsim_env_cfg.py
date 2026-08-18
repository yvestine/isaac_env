# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from pathlib import Path

import isaaclab.envs.mdp as mdp
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass
from isaaclab.actuators.actuator_cfg import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg

from isaaclab_tasks.direct.factory.factory_env_cfg import OBS_DIM_CFG, STATE_DIM_CFG, CtrlCfg, FactoryEnvCfg, ObsRandCfg

from .events import randomize_dead_zone
from .realsim_tasks_cfg import RealSimTask, RealSimPegInsert, RealSimGearMesh, RealSimNutThread
from isaaclab.sensors import TiledCameraCfg
import isaaclab.sim as sim_utils
from .policy.configuration_pi0remote import PI0RemoteConfig, PI0RemoteTAVLAConfig

# Use local Factory assets so training does not depend on Nucleus/S3 access.
ASSET_DIR = str((Path(__file__).resolve().parents[4] / "assets" / "Factory").resolve())


OBS_DIM_CFG.update({"force_threshold": 1, "ft_force": 6, "held_pos_rel_fixed": 3, "held_quat": 4})

STATE_DIM_CFG.update({"force_threshold": 1, "ft_force": 6})


@configclass
class RealSimCtrlCfg(CtrlCfg):
    ema_factor_range = [0.025, 0.1]
    # SpaceMouse-like Cartesian command limits. RealSim runs at 30 Hz
    # (physics dt 1/120 s with decimation 4), so these are applied once per
    # environment action, not once per PhysX substep.
    cartesian_target_speed_limit_enabled: bool = True
    cartesian_target_speed_limit_mps: float = 0.03
    cartesian_target_speed_cap_mps: float = 0.144
    cartesian_target_acceleration_limit_mps2: float = 0.4
    cartesian_target_angular_speed_limit_radps: float = 0.40
    cartesian_target_angular_acceleration_limit_radps2: float = 1.0
    joint_target_dynamics_limit_enabled: bool = True
    joint_target_velocity_limit_radps: float = 0.5
    joint_target_acceleration_limit_radps2: float = 0.7
    default_task_prop_gains = [565.0, 565.0, 565.0, 28.0, 28.0, 28.0]
    task_prop_gains_noise_level = [0.41, 0.41, 0.41, 0.41, 0.41, 0.41]
    pos_threshold_noise_level = [0.25, 0.25, 0.25]
    rot_threshold_noise_level = [0.29, 0.29, 0.29]
    default_dead_zone = [5.0, 5.0, 5.0, 1.0, 1.0, 1.0]
    kp_null = 10.0
    kd_null = 6.3246
    lock_fingertip_downward = True
    lock_fingertip_yaw: bool = True
    locked_fingertip_yaw: float = -1.5708
    allow_fingertip_tilt: bool = True
    fingertip_tilt_limit_rad: float = 0.0873
    locked_fingertip_quat = [0.0, 1.0, 0.0, 0.0]

@configclass
class RealSimObsRandCfg(ObsRandCfg):
    fingertip_pos = 0.00025
    fingertip_rot_deg = 0.1
    # Measured from the first, static second of the real 10 Hz trajectory.
    ft_wrench_noise_std = [0.045, 0.022, 0.036, 0.005, 0.007, 0.004]


@configclass
class EventCfg:
    object_scale_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("held_asset"),
            "mass_distribution_params": (-0.005, 0.005),
            "operation": "add",
            "distribution": "uniform",
        },
    )

    held_physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("held_asset"),
            "static_friction_range": (0.75, 0.75),
            "dynamic_friction_range": (0.75, 0.75),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 1,
        },
    )

    fixed_physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("fixed_asset"),
            "static_friction_range": (0.25, 1.25),
            "dynamic_friction_range": (0.25, 0.25),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 128,
        },
    )

    robot_physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.75, 0.75),
            "dynamic_friction_range": (0.75, 0.75),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 1,
        },
    )

    # Keep this disabled for early RL training. Randomizing controller dead zones
    # mid-episode makes exploration much harder; re-enable for final robustness.
    # dead_zone_thresholds = EventTerm(
    #     func=randomize_dead_zone, mode="interval", interval_range_s=(2.0, 2.0)
    # )


@configclass
class RealSimEnvCfg(FactoryEnvCfg):
    decimation = 4
    seed = 0
    action_space: int = 6
    # Explicitly document the fixed base pose used by the existing PPO scene.
    # The defaults intentionally preserve the current zero translation and
    # identity quaternion behavior.
    robot_base_pos: tuple[float, float, float] = (0.0, 0.0, 0.0)
    robot_base_rot: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=128,
        env_spacing=2.0,
        replicate_physics=True,
        filter_collisions=True,
        clone_in_fabric=False,
    )
    obs_rand: RealSimObsRandCfg = RealSimObsRandCfg()
    ctrl: RealSimCtrlCfg = RealSimCtrlCfg()
    task: RealSimTask = RealSimTask()
    events: EventCfg = EventCfg()

    ft_smoothing_factor: float = 0.25
    ft_parent_body_name: str = "panda_link7"
    # IsaacLab 2.x exposes body_incoming_joint_wrench_b in the parent-body
    # frame, with its torque referenced at the parent-body origin. Keep these
    # explicit so a different sensor/backend convention cannot be hidden.
    ft_raw_wrench_frame: str = "parent_body"
    ft_raw_torque_reference: str = "parent_origin"  # parent_origin or joint_anchor
    # The real dataset's base/stiffness pair is consistent with a base-frame
    # wrench whose torque is referenced at the robot-base origin. Keep the
    # final sign/calibration gated until a simulator contact check is done.
    ft_corrected_ready: bool = False
    ft_use_corrected_wrench: bool = False
    ft_corrected_force_matrix: list = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
    ft_corrected_torque_matrix: list = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
    ft_corrected_wrench_sign: list = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    ft_corrected_torque_offset_base_m: list = [0.0, 0.0, 0.0]
    ft_corrected_reference: str = "base_origin"
    ft_bias_calibration_steps: int = 30
    # Median of the first, static second in the real trajectory.
    ft_real_wrench_bias: list = [-2.684174, 0.624736, -0.766258, 0.066580, 0.201163, -0.006069]
    # Per-axis sign/gain calibration; update after directed push measurements.
    ft_axis_scale: list = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    ft_sample_hz: int = 10
    override_held_asset_color: bool = True
    held_asset_visual_color: tuple[float, float, float] = (1.0, 0.4157, 0.0627)
    override_fixed_asset_color: bool = True
    fixed_asset_visual_color: tuple[float, float, float] = (0.6471, 0.6627, 0.6667)

    obs_order: list = [
        "fingertip_pos_rel_fixed",
        "fingertip_quat",
        "held_pos_rel_fixed",
        "held_quat",
        "ee_linvel",
        "ee_angvel",
        "ft_force",
        "force_threshold",
    ]
    state_order: list = [
        "fingertip_pos",
        "fingertip_quat",
        "ee_linvel",
        "ee_angvel",
        "joint_pos",
        "held_pos",
        "held_pos_rel_fixed",
        "held_quat",
        "fixed_pos",
        "fixed_quat",
        "task_prop_gains",
        "ema_factor",
        "ft_force",
        "pos_threshold",
        "rot_threshold",
        "force_threshold",
    ]

    robot = ArticulationCfg(
        prim_path="/World/envs/env_.*/franka_env/Robot/franka",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{ASSET_DIR}/franka_mimic.usd",
            activate_contact_sensors=True,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=True,
                max_depenetration_velocity=5.0,
                linear_damping=0.0,
                angular_damping=0.0,
                max_linear_velocity=1000.0,
                max_angular_velocity=3666.0,
                enable_gyroscopic_forces=True,
                solver_position_iteration_count=192,
                solver_velocity_iteration_count=1,
                max_contact_impulse=1e32,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False,
                fix_root_link=True,
                solver_position_iteration_count=192,
                solver_velocity_iteration_count=1,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.005, rest_offset=0.0),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            joint_pos={
                "panda_joint1": 0.0,
                "panda_joint2": -0.7853981634,
                "panda_joint3": 0.0,
                "panda_joint4": -2.3561944902,
                "panda_joint5": 0.0,
                "panda_joint6": 1.5707963268,
                "panda_joint7": 0.7853981634,
                "panda_finger_joint2": 0.04,
            },
            pos=(0.0, 0.0, 0.0),
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
        actuators={
            "panda_arm1": ImplicitActuatorCfg(
                joint_names_expr=["panda_joint[1-4]"],
                stiffness=0.0,
                damping=0.0,
                friction=0.0,
                armature=0.0,
                effort_limit_sim=87,
                velocity_limit_sim=124.6,
            ),
            "panda_arm2": ImplicitActuatorCfg(
                joint_names_expr=["panda_joint[5-7]"],
                stiffness=0.0,
                damping=0.0,
                friction=0.0,
                armature=0.0,
                effort_limit_sim=12,
                velocity_limit_sim=149.5,
            ),
            "panda_hand": ImplicitActuatorCfg(
                joint_names_expr=["panda_finger_joint[1-2]"],
                effort_limit_sim=40.0,
                velocity_limit_sim=0.04,
                stiffness=7500.0,
                damping=173.0,
                friction=0.1,
                armature=0.0,
            ),
        },
    )
    
    wrist_camera = TiledCameraCfg(
        prim_path="/World/envs/env_.*/franka_env/Robot/franka/panda_link7/panda_link8/panda_hand/wrist_camera",
        update_period=0,
        height=480,
        width=640,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0,
            focus_distance=400.0,
            horizontal_aperture=20.955,
            clipping_range=(0.1, 1.0e5),
        ),
        offset=TiledCameraCfg.OffsetCfg(
            pos=(0.07813, -0.00845, -0.0073),
            rot=(0.12057, 0.71266, 0.68644, 0.07985),
            convention="opengl",
        ),
    )

    tiled_camera = TiledCameraCfg(
        prim_path="/World/envs/env_.*/franka_env/front_camera",
        update_period=0,
        height=480,
        width=640,
        data_types=["rgb"],
        spawn=None,
    )

    disable_xy_rot = True
    # Prevent collection-only auto-exit logic from terminating replay.
    teacher_eval_only: bool = False
    data_collect_cfg = {
        "collect_data": False,
        "num_trajectories": 150,
        "save_failed_trajectory": False,
        "minimal_output": False,
        "save_tavla_hdf5": True,
        "tavla_hdf5_dir": "tavla_raw",
        "immediate_stop": False,
        "auto_release_on_alignment": True,
        "auto_release_xy_threshold": 0.004,
        "auto_release_z_threshold": 0.08,
        "auto_release_lower_distance": 0.02,
        "success_on_auto_release": True,
    }
    policy_cfg = None
    # policy_cfg = PI0RemoteConfig(host_ip="127.0.0.1", host_port=8990, n_action_steps=10)
    # At 30 Hz control, interval=3 matches the real sensor 10 Hz samples.
    # policy_cfg = PI0RemoteTAVLAConfig(num_history_steps=10, history_step_interval=3, n_action_steps=32)


@configclass
class RealSimTaskPegInsertCfg(RealSimEnvCfg):
    task_name = "peg_insert"
    task_prompt = "place a peg in a hole"
    task = RealSimPegInsert()
    disable_xy_rot = True
    episode_length_s = 20.0


@configclass
class RealSimTaskGearMeshCfg(RealSimEnvCfg):
    task_name = "gear_mesh"
    task_prompt = "Install the gear between the two gears."
    task = RealSimGearMesh()
    episode_length_s = 20.0


@configclass
class RealSimTaskNutThreadCfg(RealSimEnvCfg):
    task_name = "nut_thread"
    task_prompt = "Thread the nut onto the bolt until it is fully tightened."
    task = RealSimNutThread()
    episode_length_s = 30.0
