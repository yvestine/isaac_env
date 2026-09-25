# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import gymnasium as gym

from . import agents
from .realsim_env import RealSimEnv
from .realsim_env_cfg import RealSimTaskPegInsertCfg, RealSimTaskGearMeshCfg, RealSimTaskNutThreadCfg
from .tavla_residual_env import TavlaResidualEnv
from .tavla_residual_env_cfg import RealSimTavlaResidualPegInsertCfg, RealSimTavlaTeacherPegInsertCfg
from .pi0_env import Pi0RealSimEnv
from .pi0_env_cfg import RealSimPi0PegInsertCfg
from .pi0_direct_joint_reset_env import (
    Pi0DirectJointResetEnv,
    RealSimPi0DirectJointResetPegInsertCfg,
)


def __getattr__(name):
    """Load the optional TAVLA baseline only when it is requested."""
    if name == "TavlaAffineResidualEnv":
        from .tavla_baseline.isaac_env import TavlaAffineResidualEnv
        return TavlaAffineResidualEnv
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

##
# Register Gym environments.
##

gym.register(
    id="TacEx-RealSim-PegInsert-Direct-v0",
    entry_point=f"{__name__}.realsim_env:RealSimEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": RealSimTaskPegInsertCfg,
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
    },
)

gym.register(
    id="TacEx-RealSim-GearMesh-Direct-v0",
    entry_point=f"{__name__}.realsim_env:RealSimEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": RealSimTaskGearMeshCfg,
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
    },
)

gym.register(
    id="TacEx-RealSim-NutThread-Direct-v0",
    entry_point=f"{__name__}.realsim_env:RealSimEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": RealSimTaskNutThreadCfg,
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg_nut_thread.yaml",
    },
)


gym.register(
    id="TacEx-RealSim-PegInsert-TAVLA-Teacher-v0",
    entry_point=f"{__name__}.tavla_residual_env:TavlaResidualEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": RealSimTavlaTeacherPegInsertCfg,
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_tavla_residual_cfg.yaml",
    },
)


gym.register(
    id="TacEx-RealSim-PegInsert-TAVLA-Affine-Teacher-v0",
    entry_point=f"{__name__}.tavla_baseline.isaac_env:TavlaAffineResidualEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": RealSimTavlaTeacherPegInsertCfg,
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_tavla_residual_cfg.yaml",
    },
)

gym.register(
    id="TacEx-RealSim-PegInsert-TAVLAResidual-Direct-v0",
    entry_point=f"{__name__}.tavla_residual_env:TavlaResidualEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": RealSimTavlaResidualPegInsertCfg,
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_tavla_residual_cfg.yaml",
    },
)

gym.register(
    id="TacEx-RealSim-PegInsert-PI0-Direct-v0",
    entry_point=f"{__name__}.pi0_env:Pi0RealSimEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": RealSimPi0PegInsertCfg,
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
    },
)

gym.register(
    id="TacEx-RealSim-PegInsert-PI0-DirectJointReset-v0",
    entry_point=f"{__name__}.pi0_direct_joint_reset_env:Pi0DirectJointResetEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": RealSimPi0DirectJointResetPegInsertCfg,
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
    },
)
