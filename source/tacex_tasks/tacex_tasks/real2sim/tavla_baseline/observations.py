"""Actor/critic observation boundaries for asymmetric PPO."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np


PRIVILEGED_NAMES = frozenset(
    {
        "peg_hole_relative_pose",
        "peg_hole_relative_position",
        "insertion_depth",
        "contact_points",
        "contact_normals",
        "contact_count",
        "non_target_collision",
        "domain_parameters",
        "raw_wrench",
        "wrench_base",
        "wrench_final",
        "adapted_wrench",
    }
)


def _flatten(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=np.float32).reshape(-1)


@dataclass(frozen=True)
class ActorCriticObservation:
    actor: np.ndarray
    critic: np.ndarray


class ActorCriticObservationSplitter:
    """Normalize common IsaacLab observation layouts without leaking critic state."""

    def __init__(self, actor_key: str = "policy", critic_key: str = "critic"):
        self.actor_key = actor_key
        self.critic_key = critic_key

    def split(self, observation: Any) -> ActorCriticObservation:
        if isinstance(observation, Mapping):
            actor = self._read_actor(observation)
            critic = self._read_critic(observation, actor)
        else:
            actor = _flatten(observation)
            critic = actor.copy()
        if actor.size == 0 or critic.size == 0:
            raise ValueError("actor and critic observations must be non-empty")
        if not np.isfinite(actor).all() or not np.isfinite(critic).all():
            raise FloatingPointError("observation contains NaN or Inf")
        return ActorCriticObservation(actor=actor, critic=critic)

    def _read_actor(self, observation: Mapping[str, Any]) -> np.ndarray:
        if self.actor_key in observation:
            return _flatten(observation[self.actor_key])
        if "actor" in observation:
            return _flatten(observation["actor"])
        names = observation.get("actor_keys")
        if names is None:
            names = [key for key in sorted(observation) if key not in PRIVILEGED_NAMES and key not in {self.critic_key, "critic"}]
        self._assert_no_privileged(names, "actor")
        return np.concatenate([_flatten(observation[name]) for name in names])

    def _read_critic(self, observation: Mapping[str, Any], actor: np.ndarray) -> np.ndarray:
        if self.critic_key in observation:
            return _flatten(observation[self.critic_key])
        if "critic" in observation:
            return _flatten(observation["critic"])
        names = observation.get("critic_keys")
        if names is None:
            return actor.copy()
        return np.concatenate([_flatten(observation[name]) for name in names])

    @staticmethod
    def _assert_no_privileged(names: Any, side: str) -> None:
        leaked = sorted(set(names).intersection(PRIVILEGED_NAMES))
        if leaked:
            raise ValueError(f"privileged fields leaked into {side}: {leaked}")

    @staticmethod
    def validate_actor_keys(actor_keys: list[str] | tuple[str, ...]) -> None:
        ActorCriticObservationSplitter._assert_no_privileged(actor_keys, "actor")

