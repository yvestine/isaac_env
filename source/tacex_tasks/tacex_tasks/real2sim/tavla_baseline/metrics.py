"""Rollout diagnostics required for contact-rich baseline comparison."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np


class EpisodeMetrics:
    def __init__(self):
        self._episodes: list[dict[str, float]] = []

    def add(self, **values: Any) -> None:
        converted: dict[str, float] = {}
        for name, value in values.items():
            array = np.asarray(value, dtype=np.float32)
            if array.size:
                converted[name] = float(array.reshape(-1)[0])
        self._episodes.append(converted)

    def summary(self) -> dict[str, float]:
        keys = sorted({key for row in self._episodes for key in row})
        result: dict[str, float] = {"episodes": float(len(self._episodes))}
        for key in keys:
            values = np.asarray([row[key] for row in self._episodes if key in row], dtype=np.float32)
            if values.size:
                result[f"{key}/mean"] = float(values.mean())
                result[f"{key}/p95"] = float(np.percentile(values, 95))
                result[f"{key}/max"] = float(values.max())
        return result

    @property
    def episodes(self) -> list[dict[str, float]]:
        return list(self._episodes)


class StepMetrics:
    def __init__(self):
        self._values: dict[str, list[float]] = defaultdict(list)

    def add(self, **values: Any) -> None:
        for name, value in values.items():
            array = np.asarray(value, dtype=np.float32).reshape(-1)
            self._values[name].extend(float(item) for item in array if np.isfinite(item))

    def summary(self) -> dict[str, float]:
        result: dict[str, float] = {}
        for name, values in self._values.items():
            array = np.asarray(values, dtype=np.float32)
            if array.size:
                result[f"{name}/mean"] = float(array.mean())
                result[f"{name}/p95"] = float(np.percentile(array, 95))
                result[f"{name}/max"] = float(array.max())
        return result

