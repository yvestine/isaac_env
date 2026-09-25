"""Isaac Sim 6.0 standalone RealSim peg-in-hole PPO smoke trainer.

This file intentionally does not modify or import the legacy RealSimEnv step
and reset implementations.  It reuses the existing RealSim Franka/peg/hole
configuration, but follows the Isaac Lab 3.x FactoryEnv scene, reset, IK, and
rendering path.  The background USD is deliberately omitted: it contains a
per-environment PhysicsScene and a duplicate Franka articulation, both of
which are invalid for Isaac Sim 6 environment replication.

The default run is deliberately small:
    num_envs=4, horizon_length=64, minibatch_size=256, max_iterations=1

Override --max_iterations only after the smoke run is clean.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--num_envs", type=int, default=4)
parser.add_argument("--max_iterations", type=int, default=1)
parser.add_argument("--seed", type=int, default=0)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0], *hydra_args]

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


# Isaac Sim 6 keeps the legacy Semantics Python extension in an extension
# cache.  TacEx's sensor package still imports the legacy module while its
# task package is scanned.  Add the cache before importing tacex_tasks.
import isaacsim  # noqa: E402

_isaacsim_root = Path(isaacsim.__file__).resolve().parent
_isaacsim_exts = _isaacsim_root / "exts"
_isaacsim_extscache = _isaacsim_root / "extscache"
for _cache_dir in _isaacsim_extscache.glob("omni.usd.schema.semantics-*"):
    if _cache_dir.is_dir():
        sys.path.append(str(_cache_dir))
for _extension_name in ("isaacsim.core.prims", "isaacsim.core.utils", "isaacsim.core.api"):
    _extension_pkg = _isaacsim_exts / _extension_name / "isaacsim"
    if _extension_pkg.is_dir() and str(_extension_pkg) not in isaacsim.__path__:
        isaacsim.__path__.append(str(_extension_pkg))


import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
import yaml  # noqa: E402
from isaaclab.assets import Articulation  # noqa: E402
from isaaclab.envs import DirectMARLEnv  # noqa: E402
from isaaclab_rl.rl_games import RlGamesGpuEnv, RlGamesVecEnvWrapper  # noqa: E402
from isaaclab.utils.assets import retrieve_file_path  # noqa: E402
from rl_games.common import env_configurations, vecenv  # noqa: E402
from rl_games.common.algo_observer import IsaacAlgoObserver  # noqa: E402
from rl_games.torch_runner import Runner  # noqa: E402


from isaaclab_tasks.direct.factory.factory_env import FactoryEnv  # noqa: E402
from isaaclab_tasks.direct.factory.factory_env_cfg import FactoryTaskPegInsertCfg  # noqa: E402
from isaaclab.envs import multi_agent_to_single_agent  # noqa: E402
from pxr import Usd  # noqa: E402
_LOCAL_ASSET_DIR = Path(__file__).resolve().parents[3] / "assets" / "Factory"
_BACKGROUND_SOURCE = Path(__file__).resolve().parents[3] / "franka_env_background_edit" / "franka_env.usd"

# The calibrated peg root offset from the fingertip center is 50 - 17.608 =
# 32.392 mm. Keep the same geometry used by the original replay script.
_PEG_FRANKA_FINGERPAD_LENGTH_M = 0.017608
_PEG_ROOT_OFFSET_M = 0.050 - _PEG_FRANKA_FINGERPAD_LENGTH_M


def _make_rollout_franka_usd():
    """Expose rollout's complete black-gripper Franka as one active asset."""
    generated = Path("/tmp/tacex_isaac6_rollout_franka_full_black_v1.usd")
    if not generated.is_file():
        if not _BACKGROUND_SOURCE.is_file():
            raise FileNotFoundError(_BACKGROUND_SOURCE)
        stage = Usd.Stage.CreateNew(str(generated))
        root = stage.DefinePrim("/franka", "Xform")
        root.GetReferences().AddReference(
            str(_BACKGROUND_SOURCE), "/World/Robot/franka"
        )
        stage.SetDefaultPrim(root)
        stage.GetRootLayer().Save()
    return generated


class Isaac6PegFactoryEnv(FactoryEnv):
    """Official FactoryEnv mechanics with the existing RealSim peg assets."""

    def __init__(self, cfg, render_mode=None, **kwargs):
        # The legacy train.py passes collect_data=False to gym.make().  The
        # official FactoryEnv does not need that argument.
        kwargs.pop("collect_data", None)
        kwargs.pop("output_dir", None)
        self.last_ik_pos_error = None
        self.last_ik_rot_error = None
        self.success_times = 0
        self.total_times = 0
        super().__init__(cfg, render_mode, **kwargs)

    def get_handheld_asset_relative_pose(self):
        """Use the same 32.392 mm calibrated peg root offset as play."""
        relative_pos, relative_quat = super().get_handheld_asset_relative_pose()
        if self.cfg_task.name == "peg_insert":
            relative_pos[:, 2] = _PEG_ROOT_OFFSET_M
        return relative_pos, relative_quat

    def _setup_scene(self):
        """Create only the official three-articulation Factory scene.

        No visual background, camera, or nested PhysicsScene is loaded.
        Isaac Lab creates the single global PhysicsScene and clones only the
        environment assets below.
        """
        self._robot = Articulation(self.cfg.robot)
        self._fixed_asset = Articulation(self.cfg_task.fixed_asset)
        self._held_asset = Articulation(self.cfg_task.held_asset)

        self.scene.clone_environments(copy_from_source=False)
        if self.device == "cpu":
            self.scene.filter_collisions()

        self.scene.articulations["robot"] = self._robot
        self.scene.articulations["fixed_asset"] = self._fixed_asset
        self.scene.articulations["held_asset"] = self._held_asset

    def set_pos_inverse_kinematics(self, *args, **kwargs):
        pos_error, rot_error = super().set_pos_inverse_kinematics(*args, **kwargs)
        self.last_ik_pos_error = torch.linalg.norm(pos_error.detach(), dim=-1)
        self.last_ik_rot_error = torch.linalg.norm(rot_error.detach(), dim=-1)
        return pos_error, rot_error

    def step(self, action):
        result = super().step(action)
        reset_mask = result[2] | result[3]
        if bool(torch.any(reset_mask)):
            success_rate = result[4].get("log", {}).get("Metrics/success_rate")
            if success_rate is not None:
                current_total = int(reset_mask.sum().item())
                current_successes = int(round(float(success_rate) * self.num_envs))
                current_successes = min(current_successes, current_total)
                current_rate = (current_successes / current_total) * 100 if current_total else 0.0
                self.success_times += current_successes
                self.total_times += current_total
                cumulative_rate = (self.success_times / self.total_times) * 100 if self.total_times else 0.0
                print(f"Current Success rate: {current_successes} / {current_total} = {current_rate:.2f}%", flush=True)
                print(f"Cumulative Success rate: {self.success_times} / {self.total_times} = {cumulative_rate:.2f}%", flush=True)
        return result


def _make_cfg():
    cfg = FactoryTaskPegInsertCfg()
    cfg.robot.spawn.usd_path = str(_make_rollout_franka_usd())
    cfg.task.fixed_asset_cfg.usd_path = str(_LOCAL_ASSET_DIR / "factory_hole_8mm.usd")
    cfg.task.fixed_asset.spawn.usd_path = cfg.task.fixed_asset_cfg.usd_path
    cfg.task.held_asset_cfg.usd_path = str(_LOCAL_ASSET_DIR / "factory_peg_8mm.usd")
    cfg.task.held_asset.spawn.usd_path = cfg.task.held_asset_cfg.usd_path
    cfg.task.robot_cfg.franka_fingerpad_length = _PEG_FRANKA_FINGERPAD_LENGTH_M
    cfg.scene.num_envs = int(args_cli.num_envs)
    cfg.scene.env_spacing = 2.0
    cfg.scene.replicate_physics = True
    cfg.scene.clone_in_fabric = False
    cfg.seed = int(args_cli.seed)
    cfg.sim.device = args_cli.device or cfg.sim.device

    # No camera/data collection in this test.  Keep the simulation buffers
    # bounded for four environments while retaining the Factory contact
    # solver settings needed by the peg/hole geometry.
    # No camera sensors are created by this standalone training scene.
    cfg.sim.render_interval = cfg.decimation
    cfg.sim.physics.gpu_max_rigid_contact_count = 2**20
    cfg.sim.physics.gpu_max_rigid_patch_count = 2**20
    cfg.sim.physics.gpu_collision_stack_size = 2**26
    cfg.sim.physics.gpu_max_num_partitions = 1
    return cfg


def _load_agent_cfg():
    cfg_path = Path(__file__).resolve().parents[3] / "source/tacex_tasks/tacex_tasks/real2sim/agents/rl_games_ppo_cfg.yaml"
    with cfg_path.open("r", encoding="utf-8") as stream:
        agent_cfg = yaml.safe_load(stream)

    params = agent_cfg["params"]
    params["seed"] = int(args_cli.seed)
    network = params["network"]
    network["mlp"]["units"] = [128, 64]
    network["rnn"]["units"] = 128
    network["rnn"]["layers"] = 1
    params["config"].update(
        {
            "name": "RealSimIsaac6Peg4",
            "full_experiment_name": "smoke",
            "device": args_cli.device or "cuda:0",
            "num_actors": int(args_cli.num_envs),
            "horizon_length": 64,
            "minibatch_size": 256,
            "seq_length": 32,
            "mini_epochs": 2,
            "max_epochs": int(args_cli.max_iterations),
            "save_frequency": 1000,
            "save_best_after": 1000,
        }
    )
    central = params["config"].get("central_value_config", {})
    if central:
        central["minibatch_size"] = 256
        central["mini_epochs"] = 2
        central.setdefault("rnn", {})["units"] = 128
        central.setdefault("rnn", {})["layers"] = 1
        central.setdefault("mlp", {})["units"] = [128, 64]
    return agent_cfg


def _print_memory(tag):
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**2
        reserved = torch.cuda.memory_reserved() / 1024**2
        peak = torch.cuda.max_memory_allocated() / 1024**2
        print(f"[Isaac6][memory] {tag}: allocated={allocated:.1f}MiB reserved={reserved:.1f}MiB peak={peak:.1f}MiB", flush=True)


def main():
    cfg = _make_cfg()
    agent_cfg = _load_agent_cfg()
    print(f"[Isaac6] task=TacEx-RealSim-PegInsert-Direct-v0 num_envs={cfg.scene.num_envs}", flush=True)
    print("[Isaac6] background USD disabled; cameras disabled; official Factory reset/IK enabled", flush=True)

    env = Isaac6PegFactoryEnv(cfg, render_mode=None)
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    _print_memory("after_env_create")
    env.reset(seed=int(args_cli.seed))
    unwrapped = env.unwrapped
    if unwrapped.last_ik_pos_error is None or unwrapped.last_ik_rot_error is None:
        raise RuntimeError("IK did not execute during reset")
    max_pos = float(unwrapped.last_ik_pos_error.max().item())
    max_rot = float(unwrapped.last_ik_rot_error.max().item())
    ik_success = max_pos <= 2.0e-3 and max_rot <= 2.0e-3
    print(f"[Isaac6][IK] success={ik_success} max_position_error={max_pos:.6e} max_rotation_error={max_rot:.6e}", flush=True)
    if max_pos > 2.0e-3 or max_rot > 2.0e-3:
        raise RuntimeError(f"Isaac6 peg reset IK did not converge: pos={max_pos} rot={max_rot}")

    rl_device = agent_cfg["params"]["config"]["device"]
    env = RlGamesVecEnvWrapper(
        env,
        rl_device,
        agent_cfg["params"]["env"].get("clip_observations", float("inf")),
        agent_cfg["params"]["env"].get("clip_actions", float("inf")),
    )
    vecenv.register(
        "Isaac6Peg4VecEnv",
        lambda config_name, num_actors, **kwargs: RlGamesGpuEnv(config_name, num_actors, **kwargs),
    )
    env_configurations.register(
        "rlgpu_isaac6_peg4",
        {"vecenv_type": "Isaac6Peg4VecEnv", "env_creator": lambda **kwargs: env},
    )
    agent_cfg["params"]["config"]["env_name"] = "rlgpu_isaac6_peg4"
    agent_cfg["params"]["config"]["num_actors"] = int(env.unwrapped.num_envs)

    runner = Runner(IsaacAlgoObserver())
    runner.load(agent_cfg)
    runner.reset()
    _print_memory("before_ppo")
    runner.run({"train": True, "play": False})
    _print_memory("after_ppo")
    env.close()


gym.register(
    id="TacEx-RealSim-PegInsert-Isaac6-Peg4-v0",
    entry_point=f"{__name__}:Isaac6PegFactoryEnv",
    disable_env_checker=True,
)


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
