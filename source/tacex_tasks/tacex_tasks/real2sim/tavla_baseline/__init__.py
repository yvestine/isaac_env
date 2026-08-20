"""Additive TA-VLA simulation-RL baseline components.

This package is deliberately not imported by the existing task registration
path.  It can therefore be used beside the current deployment and training
jobs without changing their behavior.
"""

from .config import BaselineConfig, PPOConfig, RandomizationConfig, RewardConfig
from .observations import ActorCriticObservationSplitter
from .randomization import CurriculumStage, DomainRandomizer
from .rewards import RewardEngine, SuccessTracker, TerminationChecker
from .states import PreInsertStateDatabase, PreInsertState
from .wrench import FixedAffineWrenchAdapter, WrenchPipeline

__all__ = [
    "ActorCriticObservationSplitter",
    "BaselineConfig",
    "CurriculumStage",
    "DomainRandomizer",
    "FixedAffineWrenchAdapter",
    "PPOConfig",
    "PreInsertState",
    "PreInsertStateDatabase",
    "RandomizationConfig",
    "RewardConfig",
    "RewardEngine",
    "SuccessTracker",
    "TerminationChecker",
    "WrenchPipeline",
]
