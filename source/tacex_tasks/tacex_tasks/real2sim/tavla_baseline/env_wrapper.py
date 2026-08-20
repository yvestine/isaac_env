"""Optional external wrapper for baseline reward terms and diagnostics."""

from __future__ import annotations

from typing import Any, Callable

import numpy as np

from .rewards import RewardEngine, TerminationChecker


class BaselineRewardWrapper:
    """Keep existing environment behavior while exposing optional baseline reward.

    ``state_provider`` is intentionally injected because the existing task has
    several geometry conventions.  The wrapper never guesses peg/hole frames.
    """

    def __init__(self, env: Any, state_provider: Callable[[Any, Any, Any], dict[str, Any]], reward_engine: RewardEngine, termination_checker: TerminationChecker, replace_reward: bool = False):
        self.env = env
        self.state_provider = state_provider
        self.reward_engine = reward_engine
        self.termination_checker = termination_checker
        self.replace_reward = bool(replace_reward)
        self.previous_state: dict[str, Any] | None = None

    def reset(self, *args, **kwargs):
        result = self.env.reset(*args, **kwargs)
        self.previous_state = None
        return result

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        state = self.state_provider(self.env, action, info)
        if self.previous_state is not None:
            state["previous_insertion_depth"] = self.previous_state["insertion_depth"]
            state["previous_alignment_error"] = self.previous_state["alignment_error"]
            state["previous_action"] = self.previous_state.get("action", action)
        output = self.reward_engine.compute(state)
        baseline_terminated, reasons = self.termination_checker.check(state)
        info = dict(info or {})
        info["baseline_reward"] = output.total
        info["baseline_reward_terms"] = output.terms
        info["baseline_termination_reasons"] = reasons
        if self.replace_reward:
            reward = output.total
            terminated = baseline_terminated
        self.previous_state = dict(state)
        return observation, reward, terminated, truncated, info

    def __getattr__(self, name: str):
        return getattr(self.env, name)

