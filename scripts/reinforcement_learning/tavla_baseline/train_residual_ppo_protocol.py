"""Residual PPO baseline using the explicit 224x224/reset-message protocol."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="Protocol-accurate TA-VLA residual PPO")
parser.add_argument("--config", type=Path, default=Path("scripts/reinforcement_learning/tavla_baseline/config/tavla_baseline.json"))
parser.add_argument("--updates", type=int, default=100)
parser.add_argument("--rollout-steps", type=int, default=None)
parser.add_argument("--output-dir", type=Path, default=Path("logs/tavla_baseline/residual_ppo_protocol"))
parser.add_argument("--tavla-host", type=str, default=None)
parser.add_argument("--tavla-port", type=int, default=None)
parser.add_argument("--seed", type=int, default=None)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
simulation_app = AppLauncher(args).app

import tacex_tasks  # noqa: E402,F401
from tacex_tasks.real2sim.tavla_baseline.checkpointing import JsonlLogger, save_checkpoint  # noqa: E402
from tacex_tasks.real2sim.tavla_baseline.config import BaselineConfig  # noqa: E402
from tacex_tasks.real2sim.tavla_baseline.isaac_env_protocol import TavlaAffineProtocolResidualEnv  # noqa: E402
from tacex_tasks.real2sim.tavla_baseline.observations import ActorCriticObservationSplitter  # noqa: E402
from tacex_tasks.real2sim.tavla_baseline.ppo import AsymmetricPPO, GaussianResidualActorCritic  # noqa: E402
from tacex_tasks.real2sim.tavla_baseline.rollout import TransitionCollector, _reset_env  # noqa: E402
from tacex_tasks.real2sim.tavla_residual_env_cfg import RealSimTavlaResidualPegInsertCfg  # noqa: E402


def main() -> None:
    config = BaselineConfig.load_json(args.config)
    if args.tavla_host:
        config.tavla_host = args.tavla_host
    if args.tavla_port:
        config.tavla_port = args.tavla_port
    if args.seed is not None:
        config.ppo.seed = args.seed
    if args.rollout_steps:
        config.ppo.rollout_steps = args.rollout_steps
    random.seed(config.ppo.seed)
    np.random.seed(config.ppo.seed)
    torch.manual_seed(config.ppo.seed)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = JsonlLogger(output_dir / "metrics.jsonl")
    env_cfg = RealSimTavlaResidualPegInsertCfg()
    env_cfg.scene.num_envs = 1
    env_cfg.teacher_eval_only = False
    env_cfg.teacher_prompt = config.prompt
    env_cfg.teacher_action_interpolation = False
    env_cfg.teacher_replan_actions = 1
    env_cfg.teacher_policy_cfg.host_ip = config.tavla_host
    env_cfg.teacher_policy_cfg.host_port = config.tavla_port
    env_cfg.data_collect_cfg["collect_data"] = False
    env = TavlaAffineProtocolResidualEnv(env_cfg, output_dir=str(output_dir), baseline_adapter_path=config.adapter_checkpoint)
    try:
        observation, _ = _reset_env(env)
        splitter = ActorCriticObservationSplitter()
        split = splitter.split(observation)
        model = GaussianResidualActorCritic(split.actor.size, split.critic.size, config.action_dim, config.ppo.actor_hidden_dims, config.ppo.critic_hidden_dims)
        model.config = config.ppo
        trainer = AsymmetricPPO(model, config.ppo, device=args.device or "cuda:0")
        collector = TransitionCollector(splitter, trainer.device)
        for update in range(args.updates):
            batch, _, raw = collector.collect(env, trainer.model, config.ppo.rollout_steps)
            metrics = trainer.update(batch)
            metrics.update({"update": update + 1, "rollout_reward_mean": float(raw["rewards"].mean()), "done_fraction": float(raw["dones"].float().mean())})
            logger.write(metrics)
            save_checkpoint(output_dir / "latest.pt", {"format": "tavla_residual_ppo_protocol_v1", "update": update + 1, "model": trainer.model.state_dict(), "actor_optimizer": trainer.actor_optimizer.state_dict(), "critic_optimizer": trainer.critic_optimizer.state_dict()})
            if (update + 1) % 10 == 0:
                print(json.dumps(metrics, default=float))
    finally:
        env.close()


if __name__ == "__main__":
    main()

