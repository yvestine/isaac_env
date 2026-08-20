"""Flow-Noise sampler and PPO likelihood support for Conditional Flow Matching."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Protocol

import torch
from torch.distributions import Normal


class VelocityModel(Protocol):
    def __call__(self, observation: torch.Tensor, action_state: torch.Tensor, tau: torch.Tensor) -> torch.Tensor: ...


class NoiseModel(Protocol):
    def __call__(self, observation: torch.Tensor, action_state: torch.Tensor, tau: torch.Tensor) -> torch.Tensor: ...


@dataclass
class FlowNoiseConfig:
    integration_steps: int = 10
    initial_std: float = 1.0
    min_std: float = 1.0e-4
    max_std: float = 1.0
    action_horizon: int = 1
    action_dim: int = 8


@dataclass
class FlowTrace:
    states: torch.Tensor
    next_states: torch.Tensor
    means: torch.Tensor
    stds: torch.Tensor
    taus: torch.Tensor
    initial_noise: torch.Tensor

    def detach(self) -> "FlowTrace":
        return FlowTrace(*(value.detach() for value in (self.states, self.next_states, self.means, self.stds, self.taus, self.initial_noise)))


class FlowNoiseSampler:
    """Discretize the flow path as Gaussian transitions with exact log-prob."""

    def __init__(self, velocity: VelocityModel, noise: NoiseModel | None, config: FlowNoiseConfig):
        if config.integration_steps <= 0:
            raise ValueError("integration_steps must be positive")
        self.velocity = velocity
        self.noise = noise
        self.config = config

    def sample(self, observation: torch.Tensor, deterministic: bool = False) -> tuple[torch.Tensor, torch.Tensor, FlowTrace]:
        batch = observation.shape[0]
        shape = (batch, self.config.action_horizon, self.config.action_dim)
        delta = 1.0 / self.config.integration_steps
        initial_noise = torch.randn(shape, device=observation.device)
        state = initial_noise * self.config.initial_std
        initial_log_prob = Normal(torch.zeros_like(state), torch.full_like(state, self.config.initial_std)).log_prob(state).sum((-1, -2))
        states, next_states, means, stds, taus = [], [], [], [], []
        log_prob = initial_log_prob
        for index in range(self.config.integration_steps):
            tau = torch.full((batch,), index / self.config.integration_steps, device=observation.device, dtype=observation.dtype)
            velocity = self.velocity(observation, state, tau)
            mean = state + velocity * delta
            if self.noise is None:
                std = torch.full_like(state, self.config.min_std)
            else:
                log_std = self.noise(observation, state, tau).clamp(torch.log(torch.tensor(self.config.min_std, device=state.device)), torch.log(torch.tensor(self.config.max_std, device=state.device)))
                std = log_std.exp()
            if deterministic:
                next_state = mean
            else:
                next_state = Normal(mean, std).rsample()
            log_prob = log_prob + Normal(mean, std).log_prob(next_state).sum((-1, -2))
            states.append(state)
            next_states.append(next_state)
            means.append(mean)
            stds.append(std)
            taus.append(tau)
            state = next_state
        trace = FlowTrace(
            states=torch.stack(states, dim=1),
            next_states=torch.stack(next_states, dim=1),
            means=torch.stack(means, dim=1),
            stds=torch.stack(stds, dim=1),
            taus=torch.stack(taus, dim=1),
            initial_noise=initial_noise,
        )
        return state, log_prob, trace

    def log_prob(self, observation: torch.Tensor, trace: FlowTrace) -> torch.Tensor:
        initial_std = torch.full_like(trace.states[:, 0], self.config.initial_std)
        log_prob = Normal(torch.zeros_like(trace.initial_noise), initial_std).log_prob(trace.states[:, 0]).sum((-1, -2))
        for index in range(self.config.integration_steps):
            state = trace.states[:, index]
            tau = trace.taus[:, index]
            velocity = self.velocity(observation, state, tau)
            delta = 1.0 / self.config.integration_steps
            mean = state + velocity * delta
            if self.noise is None:
                std = torch.full_like(state, self.config.min_std)
            else:
                log_std = self.noise(observation, state, tau).clamp(torch.log(torch.tensor(self.config.min_std, device=state.device)), torch.log(torch.tensor(self.config.max_std, device=state.device)))
                std = log_std.exp()
            log_prob = log_prob + Normal(mean, std).log_prob(trace.next_states[:, index]).sum((-1, -2))
        return log_prob

