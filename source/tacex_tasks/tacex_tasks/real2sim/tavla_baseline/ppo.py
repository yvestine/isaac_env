"""Small, explicit PPO implementation for asymmetric actor-critic baselines."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.distributions import Normal


def _mlp(input_dim: int, hidden_dims: tuple[int, ...], output_dim: int) -> nn.Sequential:
    layers: list[nn.Module] = []
    current = input_dim
    for hidden in hidden_dims:
        layers.extend((nn.Linear(current, hidden), nn.LayerNorm(hidden), nn.Tanh()))
        current = hidden
    layers.append(nn.Linear(current, output_dim))
    return nn.Sequential(*layers)


class GaussianResidualActorCritic(nn.Module):
    """Gaussian policy over normalized residual actions and privileged critic."""

    def __init__(self, actor_dim: int, critic_dim: int, action_dim: int, actor_hidden_dims=(256, 256), critic_hidden_dims=(256, 256, 128)):
        super().__init__()
        self.actor = _mlp(actor_dim, tuple(actor_hidden_dims), action_dim)
        self.critic = _mlp(critic_dim, tuple(critic_hidden_dims), 1)
        self.log_std = nn.Parameter(torch.full((action_dim,), -2.0))

    def distribution(self, actor_obs: torch.Tensor) -> Normal:
        mean = self.actor(actor_obs)
        std = self.log_std.clamp(-6.0, 1.0).exp().expand_as(mean)
        return Normal(mean, std)

    def value(self, critic_obs: torch.Tensor) -> torch.Tensor:
        return self.critic(critic_obs).squeeze(-1)

    def sample(self, actor_obs: torch.Tensor, critic_obs: torch.Tensor, deterministic: bool = False) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        distribution = self.distribution(actor_obs)
        latent = distribution.mean if deterministic else distribution.rsample()
        action = torch.tanh(latent)
        log_prob = distribution.log_prob(latent).sum(-1) - torch.log((1.0 - action.square()).clamp_min(1.0e-6)).sum(-1)
        entropy = distribution.entropy().sum(-1)
        return action, log_prob, self.value(critic_obs), entropy

    def evaluate(self, actor_obs: torch.Tensor, critic_obs: torch.Tensor, action: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        action = action.clamp(-0.999999, 0.999999)
        latent = torch.atanh(action)
        distribution = self.distribution(actor_obs)
        log_prob = distribution.log_prob(latent).sum(-1) - torch.log((1.0 - action.square()).clamp_min(1.0e-6)).sum(-1)
        return log_prob, self.value(critic_obs), distribution.entropy().sum(-1)


@dataclass
class RolloutBatch:
    actor_obs: torch.Tensor
    critic_obs: torch.Tensor
    actions: torch.Tensor
    old_log_probs: torch.Tensor
    returns: torch.Tensor
    advantages: torch.Tensor

    def __len__(self) -> int:
        return int(self.actions.shape[0])


def compute_gae(rewards: torch.Tensor, values: torch.Tensor, dones: torch.Tensor, next_value: torch.Tensor, gamma: float, gae_lambda: float) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute GAE with terminal masking and return normalized advantages."""
    rewards = rewards.float()
    values = values.float()
    dones = dones.float()
    advantages = torch.zeros_like(rewards)
    gae = torch.zeros_like(next_value)
    for step in range(rewards.shape[0] - 1, -1, -1):
        next_values = next_value if step == rewards.shape[0] - 1 else values[step + 1]
        nonterminal = 1.0 - dones[step]
        delta = rewards[step] + gamma * next_values * nonterminal - values[step]
        gae = delta + gamma * gae_lambda * nonterminal * gae
        advantages[step] = gae
    returns = advantages + values
    normalized = (advantages - advantages.mean()) / advantages.std(unbiased=False).clamp_min(1.0e-8)
    return returns, normalized


def explained_variance(values: torch.Tensor, returns: torch.Tensor) -> torch.Tensor:
    variance = torch.var(returns, unbiased=False)
    if float(variance) < 1.0e-8:
        return torch.zeros((), device=returns.device)
    return 1.0 - torch.var(returns - values, unbiased=False) / variance


class AsymmetricPPO:
    def __init__(self, model: GaussianResidualActorCritic, config: Any, device: str | torch.device = "cpu"):
        self.model = model.to(device)
        self.config = config
        self.device = torch.device(device)
        self.actor_optimizer = torch.optim.Adam(self.model.actor.parameters(), lr=float(config.actor_lr))
        self.actor_optimizer.add_param_group({"params": [self.model.log_std], "lr": float(config.actor_lr)})
        self.critic_optimizer = torch.optim.Adam(self.model.critic.parameters(), lr=float(config.critic_lr))
        self.update_count = 0

    def update(self, batch: RolloutBatch) -> dict[str, float]:
        actor_obs = batch.actor_obs.to(self.device)
        critic_obs = batch.critic_obs.to(self.device)
        actions = batch.actions.to(self.device)
        old_log_probs = batch.old_log_probs.to(self.device)
        returns = batch.returns.to(self.device)
        advantages = batch.advantages.to(self.device)
        size = len(batch)
        minibatch = min(int(self.config.minibatch_size), size)
        metrics: dict[str, list[float]] = {name: [] for name in ("policy_loss", "value_loss", "entropy", "approx_kl", "clip_fraction")}
        actor_enabled = self.update_count >= int(getattr(self.config, "critic_warmup_updates", 0))
        indices = torch.arange(size, device=self.device)
        for _ in range(int(self.config.update_epochs)):
            permutation = indices[torch.randperm(size, device=self.device)]
            for start in range(0, size, minibatch):
                idx = permutation[start : start + minibatch]
                new_log_probs, values, entropy = self.model.evaluate(actor_obs[idx], critic_obs[idx], actions[idx])
                ratio = torch.exp((new_log_probs - old_log_probs[idx]).clamp(-20.0, 20.0))
                unclipped = ratio * advantages[idx]
                clipped = ratio.clamp(1.0 - self.config.clip_ratio, 1.0 + self.config.clip_ratio) * advantages[idx]
                policy_loss = -torch.minimum(unclipped, clipped).mean()
                value_loss = 0.5 * torch.square(values - returns[idx]).mean()
                if actor_enabled:
                    self.actor_optimizer.zero_grad(set_to_none=True)
                    (policy_loss - self.config.entropy_weight * entropy.mean()).backward()
                    nn.utils.clip_grad_norm_(list(self.model.actor.parameters()) + [self.model.log_std], self.config.gradient_clip)
                    self.actor_optimizer.step()
                self.critic_optimizer.zero_grad(set_to_none=True)
                (self.config.value_weight * value_loss).backward()
                nn.utils.clip_grad_norm_(self.model.critic.parameters(), self.config.gradient_clip)
                self.critic_optimizer.step()
                metrics["policy_loss"].append(float(policy_loss.detach()))
                metrics["value_loss"].append(float(value_loss.detach()))
                metrics["entropy"].append(float(entropy.mean().detach()))
                metrics["approx_kl"].append(float((old_log_probs[idx] - new_log_probs).mean().detach()))
                metrics["clip_fraction"].append(float((torch.abs(ratio - 1.0) > self.config.clip_ratio).float().mean().detach()))
        self.update_count += 1
        with torch.no_grad():
            _, values, _ = self.model.evaluate(actor_obs, critic_obs, actions)
            result = {name: float(np.mean(values_)) if values_ else 0.0 for name, values_ in metrics.items()}
            result["explained_variance"] = float(explained_variance(values, returns))
            result["actor_enabled"] = float(actor_enabled)
            result["update"] = float(self.update_count)
        return result

