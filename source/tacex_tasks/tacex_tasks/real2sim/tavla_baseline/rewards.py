"""Reward terms and safety termination for contact-rich PegInsert PPO."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


def _vector(value: Any, count: int | None = None) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32)
    if count is not None and array.ndim == 0:
        array = np.full(count, float(array), dtype=np.float32)
    return array


@dataclass
class RewardOutput:
    total: np.ndarray
    terms: dict[str, np.ndarray]


class SuccessTracker:
    def __init__(self, count: int, hold_steps: int):
        self.count = int(count)
        self.hold_steps = max(1, int(hold_steps))
        self._streak = np.zeros(self.count, dtype=np.int64)
        self.latched = np.zeros(self.count, dtype=bool)

    def reset(self, indices: Any | None = None) -> None:
        if indices is None:
            self._streak.fill(0)
            self.latched.fill(False)
            return
        indices = np.asarray(indices, dtype=np.int64)
        self._streak[indices] = 0
        self.latched[indices] = False

    def update(self, eligible: Any) -> np.ndarray:
        eligible = np.asarray(eligible, dtype=bool).reshape(self.count)
        self._streak = np.where(eligible, self._streak + 1, 0)
        newly_successful = eligible & (self._streak >= self.hold_steps) & ~self.latched
        self.latched |= newly_successful
        return newly_successful


class RewardEngine:
    def __init__(self, config: Any):
        self.config = config

    def compute(self, state: dict[str, Any]) -> RewardOutput:
        depth = _vector(state["insertion_depth"])
        previous_depth = _vector(state.get("previous_insertion_depth", depth), depth.size)
        alignment = _vector(state["alignment_error"])
        previous_alignment = _vector(state.get("previous_alignment_error", alignment), depth.size)
        force = _vector(state.get("force", np.zeros((depth.size, 3), dtype=np.float32)))
        torque = _vector(state.get("torque", np.zeros((depth.size, 3), dtype=np.float32)))
        if force.ndim == 1:
            force = force.reshape(-1, 1)
        if torque.ndim == 1:
            torque = torque.reshape(-1, 1)
        action = _vector(state.get("action", np.zeros((depth.size, 8), dtype=np.float32)))
        previous_action = _vector(state.get("previous_action", action), depth.size * action.size // max(depth.size, 1))
        action = action.reshape(depth.size, -1)
        previous_action = previous_action.reshape(depth.size, -1)

        force_norm = np.linalg.norm(force, axis=-1)
        torque_norm = np.linalg.norm(torque, axis=-1)
        terms = {
            "success_reward": _vector(state.get("success", np.zeros(depth.size, dtype=bool))).astype(np.float32),
            "depth_progress": depth - previous_depth,
            "alignment_reward": previous_alignment - alignment,
            "force_exceed_penalty": np.maximum(force_norm - self.config.force_soft_threshold, 0.0),
            "torque_exceed_penalty": np.maximum(torque_norm - self.config.torque_soft_threshold, 0.0),
            "action_delta_penalty": np.linalg.norm(action - previous_action, axis=-1),
            "collision_penalty": _vector(state.get("collision", np.zeros(depth.size, dtype=np.float32))),
            "timeout_penalty": _vector(state.get("timeout", np.zeros(depth.size, dtype=np.float32))),
        }
        total = (
            self.config.success_reward * terms["success_reward"]
            + self.config.depth_progress_weight * terms["depth_progress"]
            + self.config.alignment_weight * terms["alignment_reward"]
            - self.config.force_exceed_weight * terms["force_exceed_penalty"]
            - self.config.torque_exceed_weight * terms["torque_exceed_penalty"]
            - self.config.action_delta_weight * terms["action_delta_penalty"]
            - self.config.collision_weight * terms["collision_penalty"]
            - self.config.timeout_penalty * terms["timeout_penalty"]
        )
        if not np.isfinite(total).all():
            raise FloatingPointError("reward contains NaN or Inf")
        return RewardOutput(total=total.astype(np.float32), terms={name: value.astype(np.float32) for name, value in terms.items()})


class TerminationChecker:
    """Evaluate the complete safety and task termination contract."""

    def __init__(self, config: Any):
        self.config = config

    def check(self, state: dict[str, Any]) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        depth = _vector(state["insertion_depth"])
        count = depth.size
        alignment = _vector(state.get("alignment_error", np.full(count, np.inf)))
        orientation = _vector(state.get("orientation_error", np.full(count, np.inf)))
        force = _vector(state.get("force", np.zeros((count, 3), dtype=np.float32))).reshape(count, -1)
        torque = _vector(state.get("torque", np.zeros((count, 3), dtype=np.float32))).reshape(count, -1)
        force_norm = np.linalg.norm(force, axis=-1)
        torque_norm = np.linalg.norm(torque, axis=-1)
        success = (
            depth >= self.config.success_depth_threshold
        ) & (alignment <= self.config.success_alignment_threshold) & (
            orientation <= self.config.success_orientation_threshold
        ) & (force_norm < self.config.force_hard_threshold) & (torque_norm < self.config.torque_hard_threshold)
        reasons = {
            "success": success,
            "timeout": _vector(state.get("timeout", np.zeros(count, dtype=bool))).astype(bool),
            "force_hard": force_norm >= self.config.force_hard_threshold,
            "torque_hard": torque_norm >= self.config.torque_hard_threshold,
            "collision": _vector(state.get("severe_collision", np.zeros(count, dtype=bool))).astype(bool),
            "workspace": _vector(state.get("out_of_workspace", np.zeros(count, dtype=bool))).astype(bool),
            "joint_limit": _vector(state.get("joint_limit", np.zeros(count, dtype=bool))).astype(bool),
            "invalid": ~np.isfinite(np.concatenate((depth.reshape(count, -1), alignment.reshape(count, -1), orientation.reshape(count, -1)), axis=-1)).all(axis=-1),
        }
        terminated = np.zeros(count, dtype=bool)
        for value in reasons.values():
            terminated |= value
        return terminated, reasons

