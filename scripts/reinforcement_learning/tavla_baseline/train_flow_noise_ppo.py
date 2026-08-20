"""Flow-Noise PPO entry point for a local TA-VLA bridge.

The remote WebSocket Server intentionally cannot be used here: this trainer
needs the local action-expert velocity field and gradients.  A bridge module
is supplied by the TA-VLA machine with ``--bridge module:function`` and must
implement the small interface documented in the baseline README.
"""

from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path

import torch
from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="Flow-Noise PPO for a local TA-VLA bridge")
parser.add_argument("--bridge", required=True, help="Python factory path module:function")
parser.add_argument("--config", type=Path, default=Path("scripts/reinforcement_learning/tavla_baseline/config/tavla_baseline.json"))
parser.add_argument("--updates", type=int, default=100)
parser.add_argument("--output-dir", type=Path, default=Path("logs/tavla_baseline/flow_noise_ppo"))
parser.add_argument("--device", type=str, default=None)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
simulation_app = AppLauncher(args).app

import tacex_tasks  # noqa: E402,F401
from tacex_tasks.real2sim.tavla_baseline.checkpointing import JsonlLogger, save_checkpoint  # noqa: E402
from tacex_tasks.real2sim.tavla_baseline.config import BaselineConfig  # noqa: E402
from tacex_tasks.real2sim.tavla_baseline.observations import ActorCriticObservationSplitter  # noqa: E402
from tacex_tasks.real2sim.tavla_baseline.rollout import _reset_env  # noqa: E402
from tacex_tasks.real2sim.tavla_residual_env_cfg import RealSimTavlaResidualPegInsertCfg  # noqa: E402
from tacex_tasks.real2sim.tavla_baseline.isaac_env import TavlaAffineResidualEnv  # noqa: E402


def _resolve(path: str):
    module_name, function_name = path.split(":", 1)
    return getattr(importlib.import_module(module_name), function_name)


def main() -> None:
    config = BaselineConfig.load_json(args.config)
    factory = _resolve(args.bridge)
    bridge = factory(config)
    required = ("sampler", "critic", "actor_parameters", "critic_parameters", "sample", "evaluate")
    missing = [name for name in required if not hasattr(bridge, name)]
    if missing:
        raise TypeError(f"Flow bridge is missing required fields: {missing}")
    device = torch.device(args.device or "cuda:0")
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = JsonlLogger(output_dir / "metrics.jsonl")

    env_cfg = RealSimTavlaResidualPegInsertCfg()
    env_cfg.scene.num_envs = 1
    env_cfg.teacher_eval_only = True
    env_cfg.teacher_policy_cfg.host_ip = config.tavla_host
    env_cfg.teacher_policy_cfg.host_port = config.tavla_port
    env = TavlaAffineResidualEnv(env_cfg, baseline_adapter_path=config.adapter_checkpoint)
    try:
        observation, _ = _reset_env(env)
        splitter = ActorCriticObservationSplitter()
        split = splitter.split(observation)
        obs = torch.as_tensor(split.actor, device=device).view(1, -1)
        critic_obs = torch.as_tensor(split.critic, device=device).view(1, -1)
        for update in range(args.updates):
            action, old_log_prob, trace = bridge.sample(obs)
            next_observation, reward, terminated, truncated, info = env.step(action)
            next_split = splitter.split(next_observation)
            next_critic_obs = torch.as_tensor(next_split.critic, device=device).view(1, -1)
            with torch.no_grad():
                next_value = bridge.critic(next_critic_obs).reshape(-1)
            value = bridge.critic(critic_obs).reshape(-1)
            done = torch.as_tensor(terminated, device=device).bool().reshape(-1)
            reward = torch.as_tensor(reward, device=device).float().reshape(-1)
            returns = reward + config.ppo.gamma * next_value * (~done).float()
            advantages = returns - value.detach()
            metrics = bridge.update(obs, critic_obs, trace, old_log_prob.detach(), returns.detach(), advantages.detach())
            metrics.update({"update": update + 1, "reward": float(reward[0]), "terminated": float(done[0])})
            logger.write(metrics)
            save_checkpoint(output_dir / "latest.pt", {
                "format": "tavla_flow_noise_ppo_v1",
                "update": update + 1,
                "bridge": bridge.state_dict(),
                "config": config.__dict__,
            })
            obs, critic_obs = torch.as_tensor(next_split.actor, device=device).view(1, -1), next_critic_obs
            if bool(done[0]) or bool(torch.as_tensor(truncated).reshape(-1)[0]):
                observation, _ = _reset_env(env)
                split = splitter.split(observation)
                obs = torch.as_tensor(split.actor, device=device).view(1, -1)
                critic_obs = torch.as_tensor(split.critic, device=device).view(1, -1)
            if (update + 1) % 10 == 0:
                print(json.dumps(metrics, default=float))
    finally:
        env.close()


if __name__ == "__main__":
    main()

