"""Deterministic curriculum domain randomization for PegInsert."""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import Any

import numpy as np


class CurriculumStage(IntEnum):
    NOMINAL = 0
    POSE = 1
    CONTACT = 2
    SENSOR_DELAY = 3


@dataclass(frozen=True)
class ScalarRange:
    low: float
    high: float

    def sample(self, rng: np.random.Generator, shape: tuple[int, ...]) -> np.ndarray:
        return rng.uniform(self.low, self.high, size=shape).astype(np.float32)


class DomainRandomizer:
    """Sample only the randomization groups enabled by the curriculum stage."""

    def __init__(self, config: Any, seed: int = 0):
        self.config = config
        self.rng = np.random.default_rng(seed)
        self.stage = CurriculumStage.NOMINAL

    def set_stage(self, stage: CurriculumStage | int) -> None:
        self.stage = CurriculumStage(int(stage))

    def sample(self, count: int = 1) -> dict[str, np.ndarray]:
        if count <= 0:
            raise ValueError("count must be positive")
        c = self.config
        result: dict[str, np.ndarray] = {
            "position_error_m": np.zeros(count, dtype=np.float32),
            "orientation_error_rad": np.zeros(count, dtype=np.float32),
            "insertion_depth_m": np.zeros(count, dtype=np.float32),
            "friction": np.ones(count, dtype=np.float32),
            "contact_stiffness": np.ones(count, dtype=np.float32),
            "contact_damping": np.ones(count, dtype=np.float32),
            "controller_gain": np.ones(count, dtype=np.float32),
            "action_delay_steps": np.zeros(count, dtype=np.int64),
            "qpos_noise_rad": np.zeros(count, dtype=np.float32),
            "wrench_bias": np.zeros(count, dtype=np.float32),
            "wrench_scale": np.ones(count, dtype=np.float32),
            "wrench_noise_std": np.zeros(count, dtype=np.float32),
            "wrench_delay_steps": np.zeros(count, dtype=np.int64),
        }
        if self.stage >= CurriculumStage.POSE:
            for name in ("position_error_m", "orientation_error_rad", "insertion_depth_m"):
                result[name] = self._uniform(getattr(c, name), count)
            result["qpos_noise_rad"] = self._uniform(c.qpos_noise_rad, count)
        if self.stage >= CurriculumStage.CONTACT:
            for name in ("friction", "contact_stiffness", "contact_damping", "controller_gain"):
                result[name] = self._uniform(getattr(c, name), count)
        if self.stage >= CurriculumStage.SENSOR_DELAY:
            result["action_delay_steps"] = self._integer(c.action_delay_steps, count)
            result["wrench_bias"] = self._uniform(c.wrench_bias, count)
            result["wrench_scale"] = self._uniform(c.wrench_scale, count)
            result["wrench_noise_std"] = self._uniform(c.wrench_noise_std, count)
            result["wrench_delay_steps"] = self._integer(c.wrench_delay_steps, count)
        return result

    def _uniform(self, bounds: tuple[float, float], count: int) -> np.ndarray:
        low, high = map(float, bounds)
        if high < low:
            raise ValueError(f"invalid randomization range {bounds}")
        return self.rng.uniform(low, high, size=count).astype(np.float32)

    def _integer(self, bounds: tuple[int, int], count: int) -> np.ndarray:
        low, high = map(int, bounds)
        if high < low:
            raise ValueError(f"invalid integer randomization range {bounds}")
        return self.rng.integers(low, high + 1, size=count, dtype=np.int64)


def curriculum_stage_for_update(update: int, pose_start: int = 10, contact_start: int = 30, sensor_start: int = 60) -> CurriculumStage:
    if update >= sensor_start:
        return CurriculumStage.SENSOR_DELAY
    if update >= contact_start:
        return CurriculumStage.CONTACT
    if update >= pose_start:
        return CurriculumStage.POSE
    return CurriculumStage.NOMINAL

