"""PPO update for the Flow-Noise denoising-transition likelihood."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import torch
from torch import nn

from .flow_noise import FlowNoiseSampler, FlowTrace


@dataclass
class FlowNoiseBatch:
    observation: torch.Tensor
    critic_observation: torch.Tensor
    traces: FlowTrace
    old_log_probs: torch.Tensor
    returns: torch.Tensor
    advantages: torch.Tensor


class FlowNoisePPO:
    """Optimize the complete flow transition sequence with PPO clipping."""

    def __init__(self, sampler: FlowNoiseSampler, critic: nn.Module, actor_parameters: Iterable[nn.Parameter], critic_parameters: Iterable[nn.Parameter], config: Any, device: str | torch.device = "cpu"):
        self.sampler = sampler
        self.critic = critic.to(device)
        self.config = config
        self.device = torch.device(device)
        self.actor_optimizer = torch.optim.Adam(list(actor_parameters), lr=float(config.actor_lr))
        self.critic_optimizer = torch.optim.Adam(list(critic_parameters), lr=float(config.critic_lr))
        self.update_count = 0

    def update(self, batch: FlowNoiseBatch) -> dict[str, float]:
        observation = batch.observation.to(self.device)
        critic_observation = batch.critic_observation.to(self.device)
        old_log_probs = batch.old_log_probs.to(self.device)
        returns = batch.returns.to(self.device)
        advantages = batch.advantages.to(self.device)
        traces = batch.traces
        new_log_probs = self.sampler.log_prob(observation, traces)
        ratio = torch.exp((new_log_probs - old_log_probs).clamp(-20.0, 20.0))
        unclipped = ratio * advantages
        clipped = ratio.clamp(1.0 - self.config.clip_ratio, 1.0 + self.config.clip_ratio) * advantages
        policy_loss = -torch.minimum(unclipped, clipped).mean()
        values = self.critic(critic_observation).reshape(-1)
        value_loss = 0.5 * torch.square(values - returns).mean()
        if self.update_count >= int(getattr(self.config, "critic_warmup_updates", 0)):
            self.actor_optimizer.zero_grad(set_to_none=True)
            policy_loss.backward(retain_graph=True)
            nn.utils.clip_grad_norm_(list(self.actor_optimizer.param_groups[0]["params"]), self.config.gradient_clip)
            self.actor_optimizer.step()
        self.critic_optimizer.zero_grad(set_to_none=True)
        (self.config.value_weight * value_loss).backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), self.config.gradient_clip)
        self.critic_optimizer.step()
        self.update_count += 1
        return {
            "policy_loss": float(policy_loss.detach()),
            "value_loss": float(value_loss.detach()),
            "approx_kl": float((old_log_probs - new_log_probs).mean().detach()),
            "clip_fraction": float((torch.abs(ratio - 1.0) > self.config.clip_ratio).float().mean().detach()),
            "explained_variance": float(1.0 - torch.var(returns - values.detach()) / torch.var(returns).clamp_min(1.0e-8)),
            "update": float(self.update_count),
        }

