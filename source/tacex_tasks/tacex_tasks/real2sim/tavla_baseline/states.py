"""Portable pre-insertion state database without pickle deserialization."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


@dataclass
class PreInsertState:
    qpos: list[float]
    qvel: list[float] = field(default_factory=list)
    peg_pose: list[float] = field(default_factory=list)
    hole_pose: list[float] = field(default_factory=list)
    insertion_depth: float = 0.0
    domain_parameters: dict[str, float] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_arrays(cls, qpos: Any, qvel: Any = (), peg_pose: Any = (), hole_pose: Any = (), **kwargs: Any) -> "PreInsertState":
        def values(item: Any) -> list[float]:
            return np.asarray(item, dtype=np.float32).reshape(-1).tolist()

        return cls(qpos=values(qpos), qvel=values(qvel), peg_pose=values(peg_pose), hole_pose=values(hole_pose), **kwargs)

    def validate(self, qpos_dim: int = 8) -> None:
        if len(self.qpos) != qpos_dim:
            raise ValueError(f"pre-insert qpos must contain {qpos_dim} values")
        for name in ("qpos", "qvel", "peg_pose", "hole_pose"):
            values = np.asarray(getattr(self, name), dtype=np.float32)
            if values.size and not np.isfinite(values).all():
                raise FloatingPointError(f"state field {name} contains NaN or Inf")


class PreInsertStateDatabase:
    def __init__(self, records: list[PreInsertState] | None = None):
        self.records = records or []

    def add(self, state: PreInsertState) -> None:
        state.validate()
        self.records.append(state)

    def __len__(self) -> int:
        return len(self.records)

    def sample(self, rng: np.random.Generator, count: int = 1, replace: bool = True) -> list[PreInsertState]:
        if not self.records:
            raise RuntimeError("pre-insert state database is empty")
        indices = rng.choice(len(self.records), size=count, replace=replace or count > len(self.records))
        return [self.records[int(index)] for index in np.asarray(indices).reshape(-1)]

    def save(self, path: str | Path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = {"format": "tavla_preinsert_states_v1", "records": [asdict(record) for record in self.records]}
        with target.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=False)

    @classmethod
    def load(cls, path: str | Path) -> "PreInsertStateDatabase":
        with Path(path).open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if payload.get("format") != "tavla_preinsert_states_v1":
            raise ValueError("unsupported pre-insert state database format")
        records = [PreInsertState(**item) for item in payload.get("records", [])]
        database = cls(records)
        for record in records:
            record.validate()
        return database

