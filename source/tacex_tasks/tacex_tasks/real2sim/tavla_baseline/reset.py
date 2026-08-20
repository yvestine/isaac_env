"""Pre-insertion reset sampling and a small environment integration contract."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np

from .randomization import DomainRandomizer
from .states import PreInsertState, PreInsertStateDatabase


class PreInsertStateApplier(Protocol):
    def apply_preinsert_state(self, state: PreInsertState, domain_parameters: dict[str, float]) -> None: ...


@dataclass
class ResetSample:
    state: PreInsertState
    domain_parameters: dict[str, float]


class PreInsertResetSampler:
    """Sample a saved pre-insertion state and a curriculum domain together."""

    def __init__(self, database: PreInsertStateDatabase, randomizer: DomainRandomizer):
        self.database = database
        self.randomizer = randomizer

    def sample(self) -> ResetSample:
        state = self.database.sample(self.randomizer.rng, count=1)[0]
        params = {name: float(value[0]) for name, value in self.randomizer.sample(1).items()}
        state.domain_parameters = params
        return ResetSample(state=state, domain_parameters=params)

    def apply(self, env: Any) -> ResetSample:
        sample = self.sample()
        applier = getattr(env, "apply_preinsert_state", None)
        if applier is None:
            raise TypeError(
                "environment must implement apply_preinsert_state(state, domain_parameters) "
                "before pre-insertion database reset is enabled"
            )
        applier(sample.state, sample.domain_parameters)
        return sample

