"""Environment-agnostic rollout utilities used by the additive trainers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
import torch

from .observations import ActorCriticObservationSplitter
from .ppo import RolloutBatch, compute_gae


@dataclass
class RawTransition:
    actor_obs: torch.Tensor
    critic_obs: torch.Tensor
    action: torch.Tensor
    log_prob: torch.Tensor
    value: torch.Tensor
    reward: torch.Tensor
    done: torch.Tensor


class TransitionCollector:
    """Collect fixed-length rollouts while preserving actor/critic separation."""

    def __init__(self, splitter: ActorCriticObservationSplitter, device: str | torch.device = "cpu"):
        self.splitter = splitter
        self.device = torch.device(device)

    def collect(self, env: Any, policy: Any, steps: int, action_transform: Callable[[torch.Tensor], torch.Tensor] | None = None) -> tuple[RolloutBatch, Any, dict[str, Any]]:
        observation, _ = _reset_env(env)
        transitions: list[RawTransition] = []
        info_rows: list[Any] = []
        for _ in range(steps):
            split = self.splitter.split(observation)
            actor_obs = torch.as_tensor(split.actor, device=self.device).view(1, -1)
            critic_obs = torch.as_tensor(split.critic, device=self.device).view(1, -1)
            action, log_prob, value, _entropy = policy.sample(actor_obs, critic_obs)
            env_action = action if action_transform is None else action_transform(action)
            next_observation, reward, terminated, truncated, info = env.step(env_action)
            done = _as_tensor_bool(terminated) | _as_tensor_bool(truncated)
            transitions.append(RawTransition(actor_obs.squeeze(0), critic_obs.squeeze(0), action.squeeze(0), log_prob.squeeze(0), value.squeeze(0), _as_tensor(reward), done))
            info_rows.append(info)
            observation = next_observation
            if bool(done.reshape(-1)[0]):
                observation, _ = _reset_env(env)
        split = self.splitter.split(observation)
        next_value = policy.value(torch.as_tensor(split.critic, device=self.device).view(1, -1)).squeeze(0).detach()
        actor_obs = torch.stack([item.actor_obs for item in transitions])
        critic_obs = torch.stack([item.critic_obs for item in transitions])
        actions = torch.stack([item.action for item in transitions])
        log_probs = torch.stack([item.log_prob for item in transitions])
        rewards = torch.stack([item.reward for item in transitions]).reshape(-1)
        dones = torch.stack([item.done for item in transitions]).reshape(-1)
        values = torch.stack([item.value for item in transitions]).reshape(-1)
        returns, advantages = compute_gae(rewards, values, dones, next_value, policy.config.gamma, policy.config.gae_lambda)
        return RolloutBatch(actor_obs, critic_obs, actions, log_probs, returns, advantages), observation, {"infos": info_rows, "rewards": rewards, "dones": dones}


def _reset_env(env: Any) -> tuple[Any, Any]:
    result = env.reset()
    if isinstance(result, tuple) and len(result) == 2:
        return result
    return result, {}


def _as_tensor(value: Any) -> torch.Tensor:
    if torch.is_tensor(value):
        return value.detach().float().reshape(-1)[0]
    return torch.as_tensor(np.asarray(value), dtype=torch.float32).reshape(-1)[0]


def _as_tensor_bool(value: Any) -> torch.Tensor:
    if torch.is_tensor(value):
        return value.detach().bool()
    return torch.as_tensor(np.asarray(value), dtype=torch.bool)

