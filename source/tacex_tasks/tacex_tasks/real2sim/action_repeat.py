"""Shared action-repeat wrapper for PPO train and play."""

from __future__ import annotations

import numpy as np
import torch
import gymnasium as gym


def _any_done(value) -> bool:
    if torch.is_tensor(value):
        return bool(torch.any(value).item())
    return bool(np.asarray(value).any())


class PPOActionRepeatWrapper(gym.Wrapper):
    """Expose one PPO decision after several underlying environment steps."""

    def __init__(self, env, repeat_steps: int, gamma: float):
        super().__init__(env)
        if repeat_steps < 1:
            raise ValueError("repeat_steps must be positive")
        if not 0.0 < gamma <= 1.0:
            raise ValueError("gamma must be in (0, 1]")
        self.repeat_steps = int(repeat_steps)
        self.inner_gamma = float(gamma)

    def step(self, action):
        total_reward = None
        observation = terminated = truncated = info = None
        for inner_step in range(self.repeat_steps):
            observation, reward, terminated, truncated, info = self.env.step(action)
            discounted_reward = reward * (self.inner_gamma ** inner_step)
            total_reward = discounted_reward if total_reward is None else total_reward + discounted_reward
            if _any_done(terminated) or _any_done(truncated):
                break
        return observation, total_reward, terminated, truncated, info
