# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR

from isaaclab_tasks.direct.factory.factory_tasks_cfg import FactoryTask, GearMesh, NutThread

ASSET_DIR = f"{ISAACLAB_NUCLEUS_DIR}/Factory"
LOCAL_PEG_ASSET_DIR = Path(__file__).resolve().parents[4] / "assets" / "Factory"

SUPPORTED_SHAPES = ("round", "rectangular")
SUPPORTED_DIAMETERS_MM = (4, 8, 12, 16)


def _local_asset_path(filename: str) -> str:
    return str((LOCAL_PEG_ASSET_DIR / filename).resolve())


def _make_fixed_asset_articulation(fixed_cfg: "FixedAssetCfg") -> ArticulationCfg:
    return ArticulationCfg(
        prim_path="/World/envs/env_.*/FixedAsset",
        spawn=sim_utils.UsdFileCfg(
            usd_path=fixed_cfg.usd_path,
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
            mass_props=sim_utils.MassPropertiesCfg(mass=fixed_cfg.mass),
            collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.005, rest_offset=0.0),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.6, 0.0, 0.05), rot=(1.0, 0.0, 0.0, 0.0), joint_pos={}, joint_vel={}
        ),
        actuators={},
    )


def _make_held_asset_articulation(held_cfg: "HeldAssetCfg") -> ArticulationCfg:
    return ArticulationCfg(
        prim_path="/World/envs/env_.*/HeldAsset",
        spawn=sim_utils.UsdFileCfg(
            usd_path=held_cfg.usd_path,
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
            mass_props=sim_utils.MassPropertiesCfg(mass=held_cfg.mass),
            collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.005, rest_offset=0.0),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.4, 0.1), rot=(1.0, 0.0, 0.0, 0.0), joint_pos={}, joint_vel={}
        ),
        actuators={},
    )


def _build_held_cfg(path: str, diameter_mm: int) -> "HeldAssetCfg":
    cfg = HeldAssetCfg()
    cfg.usd_path = path
    cfg.diameter = diameter_mm / 1000.0
    cfg.height = 0.050
    cfg.mass = 0.019
    cfg.friction = 0.75
    return cfg


def _build_fixed_cfg(path: str, diameter_mm: int) -> "FixedAssetCfg":
    cfg = FixedAssetCfg()
    cfg.usd_path = path
    cfg.diameter = diameter_mm / 1000.0
    cfg.height = 0.025
    cfg.base_height = 0.0
    cfg.mass = 0.05
    cfg.friction = 0.75
    return cfg


def _resolve_industreal_obj_paths(shape: str, diameter_mm: int) -> tuple[str, str]:
    peg_name = f"peg_{shape}_{diameter_mm}mm.usd"
    hole_name = f"hole_{shape}_{diameter_mm}mm.usd"
    return _local_asset_path(peg_name), _local_asset_path(hole_name)


@configclass
class FixedAssetCfg:
    usd_path: str = ""
    diameter: float = 0.0
    height: float = 0.0
    base_height: float = 0.0  # Used to compute held asset CoM.
    friction: float = 0.75
    mass: float = 0.05


@configclass
class HeldAssetCfg:
    usd_path: str = ""
    diameter: float = 0.0  # Used for gripper width.
    height: float = 0.0
    friction: float = 0.75
    mass: float = 0.05


@configclass
class Peg8mm(HeldAssetCfg):
    usd_path = f"{ASSET_DIR}/factory_peg_8mm.usd"
    diameter = 0.007986
    height = 0.050
    mass = 0.019


@configclass
class Hole8mm(FixedAssetCfg):
    usd_path = f"{ASSET_DIR}/factory_hole_8mm.usd"
    diameter = 0.0081
    height = 0.025
    base_height = 0.0


@configclass
class PegInsert(FactoryTask):
    name = "peg_insert"
    fixed_asset_cfg = Hole8mm()
    held_asset_cfg = Peg8mm()
    asset_size = 8.0
    duration_s = 10.0

    # Robot
    # hand_init_pos: list = [0.0, 0.0, 0.047]  # Relative to fixed asset tip.
    hand_init_pos: list = [0.0, 0.0, 0.08]  # Relative to fixed asset tip.
    # hand_init_pos_noise: list = [0.02, 0.02, 0.01]
    hand_init_pos_noise: list = [0.04, 0.04, 0.02]
    hand_init_orn: list = [3.1416, 0.0, 0.0]
    hand_init_orn_noise: list = [0.0, 0.0, 0.785]

    # Fixed Asset (applies to all tasks)
    fixed_asset_init_pos_noise: list = [0.05, 0.05, 0.05]
    fixed_asset_init_orn_deg: float = 0.0
    fixed_asset_init_orn_range_deg: float = 360.0

    # Held Asset (applies to all tasks)
    held_asset_pos_noise: list = [0.003, 0.0, 0.003]  # noise level of the held asset in gripper
    held_asset_rot_init: float = 0.0

    # Rewards
    keypoint_coef_baseline: list = [5, 4]
    keypoint_coef_coarse: list = [50, 2]
    keypoint_coef_fine: list = [100, 0]
    # Fraction of socket height.
    success_threshold: float = 0.12
    engage_threshold: float = 0.9

    fixed_asset: ArticulationCfg = _make_fixed_asset_articulation(fixed_asset_cfg)
    held_asset: ArticulationCfg = _make_held_asset_articulation(held_asset_cfg)


@configclass
class ForgeTask(FactoryTask):
    action_penalty_ee_scale: float = 0.0
    action_penalty_asset_scale: float = 0.001
    action_grad_penalty_scale: float = 0.1
    contact_penalty_scale: float = 0.05
    delay_until_ratio: float = 0.25
    contact_penalty_threshold_range = [5.0, 10.0]


@configclass
class ForgePegInsert(PegInsert, ForgeTask):
    contact_penalty_scale: float = 0.2
    success_threshold: float = 0.12  # 强化学习专家训练时设置为 0.04

    # User-facing selection parameters.
    peg_shape: str = "round"  # round | rectangular
    peg_diameter_mm: int = 8  # 4 | 8 | 12 | 16
    use_industreal_obj_assets: bool = False

    def __post_init__(self):
        if self.peg_shape not in SUPPORTED_SHAPES:
            raise ValueError(f"Unsupported peg_shape={self.peg_shape}, choose from {SUPPORTED_SHAPES}.")
        if self.peg_diameter_mm not in SUPPORTED_DIAMETERS_MM:
            raise ValueError(
                f"Unsupported peg_diameter_mm={self.peg_diameter_mm}, choose from {SUPPORTED_DIAMETERS_MM}."
            )

        if self.use_industreal_obj_assets:
            peg_path, hole_path = _resolve_industreal_obj_paths(self.peg_shape, self.peg_diameter_mm)
            self.held_asset_cfg = _build_held_cfg(peg_path, self.peg_diameter_mm)
            self.fixed_asset_cfg = _build_fixed_cfg(hole_path, self.peg_diameter_mm)
            self.asset_size = float(self.peg_diameter_mm)
        else:
            # Backward-compatible default: IsaacLab factory USD supports only 8mm round.
            self.held_asset_cfg = Peg8mm()
            self.fixed_asset_cfg = Hole8mm()
            self.asset_size = 8.0
            self.peg_shape = "round"
            self.peg_diameter_mm = 8

        # peg_path = "assets/Factory/peg_rectangle_8mm.usd"
        # peg_path = "assets/Factory/USB.usd"
        # self.held_asset_cfg = _build_held_cfg(peg_path, self.peg_diameter_mm)
        # hole_path = "assets/Factory/hole_rectangle_16mm.usd"
        # hole_path = "assets/Factory/socket.usd"
        # self.fixed_asset_cfg = _build_fixed_cfg(hole_path, self.peg_diameter_mm)
        # self.fixed_asset_cfg = Hole8mm()
        self.fixed_asset = _make_fixed_asset_articulation(self.fixed_asset_cfg)
        self.held_asset = _make_held_asset_articulation(self.held_asset_cfg)


@configclass
class ForgeGearMesh(GearMesh, ForgeTask):
    contact_penalty_scale: float = 0.05


@configclass
class ForgeNutThread(NutThread, ForgeTask):
    contact_penalty_scale: float = 0.05
