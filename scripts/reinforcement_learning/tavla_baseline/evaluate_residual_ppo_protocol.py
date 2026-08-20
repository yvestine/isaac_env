"""Deterministic evaluation using the explicit TA-VLA wire protocol."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="Evaluate protocol-accurate TA-VLA residual PPO")
parser.add_argument("--checkpoint", type=Path, required=True)
parser.add_argument("--config", type=Path, default=Path("scripts/reinforcement_learning/tavla_baseline/config/tavla_baseline.json"))
parser.add_argument("--episodes", type=int, default=100)
parser.add_argument("--output", type=Path, default=Path("logs/tavla_baseline/residual_eval_protocol.json"))
parser.add_argument("--tavla-host", type=str, default=None)
parser.add_argument("--tavla-port", type=int, default=None)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
simulation_app = AppLauncher(args).app

import tacex_tasks  # noqa: E402,F401
from tacex_tasks.real2sim.tavla_baseline.checkpointing import load_checkpoint  # noqa: E402
from tacex_tasks.real2sim.tavla_baseline.config import BaselineConfig  # noqa: E402
from tacex_tasks.real2sim.tavla_baseline.isaac_env_protocol import TavlaAffineProtocolResidualEnv  # noqa: E402
from tacex_tasks.real2sim.tavla_baseline.observations import ActorCriticObservationSplitter  # noqa: E402
from tacex_tasks.real2sim.tavla_baseline.ppo import GaussianResidualActorCritic  # noqa: E402
from tacex_tasks.real2sim.tavla_baseline.rollout import _reset_env  # noqa: E402
from tacex_tasks.real2sim.tavla_residual_env_cfg import RealSimTavlaResidualPegInsertCfg  # noqa: E402


def main() -> None:
    config = BaselineConfig.load_json(args.config)
    if args.tavla_host:
        config.tavla_host = args.tavla_host
    if args.tavla_port:
        config.tavla_port = args.tavla_port
    device = args.device or "cuda:0"
    cfg = RealSimTavlaResidualPegInsertCfg()
    cfg.scene.num_envs = 1
    cfg.teacher_eval_only = False
    cfg.teacher_prompt = config.prompt
    cfg.teacher_action_interpolation = False
    cfg.teacher_replan_actions = 1
    cfg.teacher_policy_cfg.host_ip = config.tavla_host
    cfg.teacher_policy_cfg.host_port = config.tavla_port
    cfg.data_collect_cfg["collect_data"] = False
    env = TavlaAffineProtocolResidualEnv(cfg, baseline_adapter_path=config.adapter_checkpoint)
    try:
        observation, _ = _reset_env(env)
        splitter = ActorCriticObservationSplitter()
        split = splitter.split(observation)
        model = GaussianResidualActorCritic(
            split.actor.size,
            split.critic.size,
            config.action_dim,
            config.ppo.actor_hidden_dims,
            config.ppo.critic_hidden_dims,
        ).to(device)
        model.load_state_dict(load_checkpoint(args.checkpoint, device)["model"])
        rows = []
        for episode in range(args.episodes):
            observation, _ = _reset_env(env)
            total_reward = 0.0
            steps = 0
            done = False
            while steps < env.max_episode_length:
                split = splitter.split(observation)
                actor = torch.as_tensor(split.actor, device=device).view(1, -1)
                critic = torch.as_tensor(split.critic, device=device).view(1, -1)
                with torch.no_grad():
                    action, _, _, _ = model.sample(actor, critic, deterministic=True)
                observation, reward, terminated, truncated, _ = env.step(action)
                total_reward += float(torch.as_tensor(reward).reshape(-1)[0])
                steps += 1
                done = bool(torch.as_tensor(terminated).reshape(-1)[0])
                if done or bool(torch.as_tensor(truncated).reshape(-1)[0]):
                    break
            rows.append({"episode": episode, "success": int(done), "return": total_reward, "steps": steps})
        summary = {
            "episodes": args.episodes,
            "success_rate": float(np.mean([row["success"] for row in rows])),
            "return_mean": float(np.mean([row["return"] for row in rows])),
            "steps_mean": float(np.mean([row["steps"] for row in rows])),
            "episodes_detail": rows,
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(json.dumps({key: value for key, value in summary.items() if key != "episodes_detail"}, indent=2))
    finally:
        env.close()


if __name__ == "__main__":
    main()

