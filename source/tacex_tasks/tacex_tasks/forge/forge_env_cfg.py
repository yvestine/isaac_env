# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import isaaclab.envs.mdp as mdp
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass
from isaaclab.sim import PhysxCfg, SimulationCfg
from isaaclab_tasks.direct.factory.factory_env_cfg import OBS_DIM_CFG, STATE_DIM_CFG, CtrlCfg, FactoryEnvCfg, ObsRandCfg
from isaaclab.sim.spawners.materials.physics_materials_cfg import RigidBodyMaterialCfg

from .forge_events import randomize_dead_zone
from .forge_tasks_cfg import (
    ForgeGearMesh,
    ForgeNutThread,
    ForgePegInsert,
    ForgeTask,
)
from isaaclab.sensors import TiledCamera, TiledCameraCfg
import isaaclab.sim as sim_utils
from .policy.configuration_pi0remote import PI0RemoteConfig,PI0RemoteTAVLAConfig


OBS_DIM_CFG.update({"force_threshold": 1, "ft_force": 3})

STATE_DIM_CFG.update({"force_threshold": 1, "ft_force": 3})


@configclass
class ForgeCtrlCfg(CtrlCfg):
    ema_factor_range = [0.025, 0.1]
    default_task_prop_gains = [565.0, 565.0, 565.0, 28.0, 28.0, 28.0]
    task_prop_gains_noise_level = [0.41, 0.41, 0.41, 0.41, 0.41, 0.41]
    pos_threshold_noise_level = [0.25, 0.25, 0.25]
    rot_threshold_noise_level = [0.29, 0.29, 0.29]
    default_dead_zone = [5.0, 5.0, 5.0, 1.0, 1.0, 1.0]
    
    # pos_action_threshold = [0.002, 0.002, 0.002]


@configclass
class ForgeObsRandCfg(ObsRandCfg):
    fingertip_pos = 0.00025
    fingertip_rot_deg = 0.1
    ft_force = 1.0


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
            "static_friction_range": (0.25, 1.25),  # TODO: Set these values based on asset type.
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

    dead_zone_thresholds = EventTerm(
        func=randomize_dead_zone, mode="interval", interval_range_s=(2.0, 2.0)  # (0.25, 0.25)
    )


@configclass
class ForgeEnvCfg(FactoryEnvCfg):
    decimation = 4
    seed = 0
    action_space: int = 7
    obs_rand: ForgeObsRandCfg = ForgeObsRandCfg()
    ctrl: ForgeCtrlCfg = ForgeCtrlCfg()
    task: ForgeTask = ForgeTask()
    events: EventCfg = EventCfg()

    ft_smoothing_factor: float = 0.25

    obs_order: list = [
        "fingertip_pos_rel_fixed",
        "fingertip_quat",
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
    
    rl_training = False
    if not rl_training:
        wrist_camera = TiledCameraCfg(
            prim_path="/World/envs/env_.*/Robot/panda_hand/wrist_camera",
            update_period=0,
            height=480,
            width=640,
            data_types=["rgb"],
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=24.0, 
                focus_distance=400.0, 
                horizontal_aperture=20.955, 
                clipping_range=(0.1, 1.0e5)
            ),
            # 使用你提供的特定 offset
            offset=TiledCameraCfg.OffsetCfg(
                pos=(0.07813, -0.00845, -0.0073), 
                rot=(0.12057, 0.71266, 0.68644, 0.07985), 
                convention="opengl"
            ),
        )

        # 2. 固定位相机配置 (Static/Fixed Camera)
        tiled_camera = TiledCameraCfg(
            prim_path="/World/envs/env_.*/Camera",
            update_period=0,
            height=480,
            width=640,
            data_types=["rgb"],
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=24.0, 
                focus_distance=400.0, 
                horizontal_aperture=20.955, 
                clipping_range=(0.1, 1.0e5)
            ),
            # 使用你提供的特定 offset
            offset=TiledCameraCfg.OffsetCfg(
                pos=(1.29, -0.09, 0.4), 
                rot=(0.61, 0.4278, 0.347, 0.569), 
                convention="opengl"
            )
        )
    
    disable_xy_rot = True
    # Maximum random rotation magnitude (degrees) for peg_insert held asset initialization.
    peg_insert_rot_noise_deg: float = 0.0
    
    collect_data = False if rl_training else True
    immediate_stop = False if rl_training else True
    data_collect_cfg = {
        "collect_data": collect_data,          # Enable data collection during execution
        "num_trajectories": 150,            # Number of trajectories to collect
        "save_failed_trajectory": False,  # Save trajectories even if the task fails
        "immediate_stop": immediate_stop        # Reset the environment immediately once the task succeeds
    }
    
    policy_cfg = None
    # policy_cfg = PI0RemoteConfig(n_action_steps = 32)
    # policy_cfg = PI0RemoteTAVLAConfig(num_history_steps=32, history_step_interval=1, n_action_steps=32)
    # policy_cfg = PI0RemoteTAVLAConfig(num_history_steps=10, history_step_interval=4, n_action_steps=8)
    


@configclass
class ForgeTaskPegInsertCfg(ForgeEnvCfg):
    task_name = "peg_insert"
    task_prompt = "place a peg in a hole"
    task = ForgePegInsert(
        # peg_shape="round",
        # peg_diameter_mm=8,
        use_industreal_obj_assets=False,
        success_threshold=0.12,
    )
    disable_xy_rot = False
    peg_insert_rot_noise_deg = 30
    episode_length_s = 20.0  # 测试时用 20，RL agent 训练时用 10

    def __post_init__(self):
        super().__post_init__()
        self.task.success_threshold = 0.04 if self.rl_training else 0.2
        # self.episode_length_s = 10.0 if self.rl_training else 20.0
        self.episode_length_s = 20.0 if self.rl_training else 20.0

    

@configclass
class ForgeTaskGearMeshCfg(ForgeEnvCfg):
    task_name = "gear_mesh"
    task_prompt = "Install the gear between the two gears."
    task = ForgeGearMesh()
    episode_length_s = 20.0


@configclass
class ForgeTaskNutThreadCfg(ForgeEnvCfg):
    task_name = "nut_thread"
    task_prompt = "Thread the nut onto the bolt until it is fully tightened."
    # task = ForgeNutThread(success_threshold = 0.1)
    task = ForgeNutThread()
    episode_length_s = 30.0
    
