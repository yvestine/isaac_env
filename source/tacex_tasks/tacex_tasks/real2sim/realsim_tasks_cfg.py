# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from pathlib import Path
from isaaclab.utils import configclass
from isaaclab_tasks.direct.factory.factory_tasks_cfg import (FactoryTask,FixedAssetCfg,GearMesh,HeldAssetCfg,NutThread,PegInsert,)
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg

LOCAL_FACTORY_ASSET_DIR = Path(__file__).resolve().parents[4] / "assets" / "Factory"
def _local_factory_asset(filename: str) -> str:
    return str((LOCAL_FACTORY_ASSET_DIR / filename).resolve())


@configclass
class Peg8mm(HeldAssetCfg):
    usd_path = _local_factory_asset("factory_peg_8mm.usd")
    diameter = 0.007986
    height = 0.050
    mass = 0.019


@configclass
class Hole8mm(FixedAssetCfg):
    usd_path = _local_factory_asset("factory_hole_8mm.usd")
    diameter = 0.0081
    height = 0.025
    base_height = 0.0

@configclass
class RealSimTask(FactoryTask):
    action_penalty_ee_scale: float = 0.0
    action_penalty_asset_scale: float = 0.001
    action_grad_penalty_scale: float = 0.1
    contact_penalty_scale: float = 0.05
    delay_until_ratio: float = 0.25
    contact_penalty_threshold_range = [5.0, 10.0]
    success_xy_threshold: float = 0.006
    # Keep the newer reward terms available for later experiments, but disable
    # them for the baseline RL training configuration.
    alignment_reward_scale: float = 0.0
    insertion_reward_scale: float = 0.0
    success_bonus_scale: float = 0.0
    down_action_reward_scale: float = 0.0
    misaligned_down_penalty_scale: float = 0.0
    alignment_reward_tolerance: float = 0.006
    insertion_gate_tolerance: float = 0.02
    insertion_reward_depth: float = 0.025
    pre_insert_reward_height: float = 0.06
    pre_insert_reward_scale: float = 3.0
    align_only: bool = False
    align_above_hole_height: float = 0.05
    align_above_hole_z_tolerance: float = 0.03
    align_success_min_height: float = 0.03
    align_success_max_height: float = 0.08


@configclass
class RealSimPegInsert(PegInsert, RealSimTask):
    contact_penalty_scale: float = 0.2
    align_only: bool = False
    alignment_reward_scale: float = 0.0
    insertion_reward_scale: float = 0.0
    pre_insert_reward_scale: float = 0.0
    down_action_reward_scale: float = 0.0
    misaligned_down_penalty_scale: float = 0.0
    alignment_reward_tolerance: float = 0.004
    insertion_gate_tolerance: float = 0.006
    success_xy_threshold: float = 0.003

    name = "peg_insert"
    fixed_asset_cfg = Hole8mm()
    held_asset_cfg = Peg8mm()
    asset_size = 8.0
    duration_s = 10.0

    hand_init_pos: list = [0.0, 0.0, 0.15]
    hand_init_pos_noise: list = [0.04, 0.04, 0.0]
    hand_init_orn: list = [3.1416, 0.0, -1.5708]
    hand_init_orn_noise: list = [0.0175, 0.0175, 0.0]

    fixed_asset_init_pos_noise: list = [0.05, 0.05, 0.0]
    fixed_asset_init_orn_deg: float = 0.0
    fixed_asset_init_orn_range_deg: float = 5.0

    held_asset_pos_noise: list = [0.0, 0.0, 0.0]
    held_asset_rot_init: float = 0.0
    held_asset_rot_noise_deg: float = 0.0
    grasp_close_time_s: float = 0.6
    grasp_settle_time_s: float = 0.2
    snap_held_asset_after_grasp: bool = True
    reset_ik_debug: bool = True

    keypoint_coef_baseline: list = [5, 4]
    keypoint_coef_coarse: list = [50, 2]
    keypoint_coef_fine: list = [100, 0]
    engage_threshold: float = 0.9

    fixed_asset: ArticulationCfg = ArticulationCfg(
        prim_path="/World/envs/env_.*/FixedAsset",
        spawn=sim_utils.UsdFileCfg(
            usd_path=fixed_asset_cfg.usd_path,
            activate_contact_sensors=True,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
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
            mass_props=sim_utils.MassPropertiesCfg(mass=fixed_asset_cfg.mass),
            collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.0005, rest_offset=0.0),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(-0.07, -0.42, 0.12), rot=(1.0, 0.0, 0.0, 0.0), joint_pos={}, joint_vel={}
        ),
        actuators={},
    )
    held_asset: ArticulationCfg = ArticulationCfg(
        prim_path="/World/envs/env_.*/HeldAsset",
        spawn=sim_utils.UsdFileCfg(
            usd_path=held_asset_cfg.usd_path,
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
            mass_props=sim_utils.MassPropertiesCfg(mass=held_asset_cfg.mass),
            collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.0005, rest_offset=0.0),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(-0.07, -0.42, 0.52), rot=(1.0, 0.0, 0.0, 0.0), joint_pos={}, joint_vel={}
        ),
        actuators={},
    )


@configclass
class RealSimGearMesh(GearMesh, RealSimTask):
    contact_penalty_scale: float = 0.05


@configclass
class RealSimNutThread(NutThread, RealSimTask):
    contact_penalty_scale: float = 0.05
