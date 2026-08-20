"""Run baseline components that do not require Isaac Sim or the TA-VLA Server."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from tacex_tasks.real2sim.tavla_baseline.config import RandomizationConfig, RewardConfig
from tacex_tasks.real2sim.tavla_baseline.observations import ActorCriticObservationSplitter
from tacex_tasks.real2sim.tavla_baseline.randomization import CurriculumStage, DomainRandomizer
from tacex_tasks.real2sim.tavla_baseline.rewards import RewardEngine, SuccessTracker, TerminationChecker
from tacex_tasks.real2sim.tavla_baseline.wrench import FixedAffineWrenchAdapter, WrenchPipeline


parser = argparse.ArgumentParser(description="Pure TA-VLA baseline smoke test")
parser.add_argument("--adapter", type=Path, default=Path("checkpoints/unpaired_sim_to_real_affine.pt"))
args = parser.parse_args()


def main() -> None:
    adapter = FixedAffineWrenchAdapter(args.adapter)
    pipeline = WrenchPipeline(adapter)
    wrench = pipeline.convert(np.zeros((2, 6), dtype=np.float32))
    assert np.array_equal(wrench["wrench_final"], -wrench["wrench_base"])
    assert np.isfinite(wrench["adapted_wrench"]).all()

    splitter = ActorCriticObservationSplitter()
    split = splitter.split({"policy": np.zeros(16), "critic": np.zeros(32)})
    assert split.actor.shape == (16,) and split.critic.shape == (32,)

    randomizer = DomainRandomizer(RandomizationConfig(), seed=0)
    randomizer.set_stage(CurriculumStage.SENSOR_DELAY)
    domain = randomizer.sample(4)
    assert domain["wrench_noise_std"].shape == (4,)

    reward_config = RewardConfig()
    reward = RewardEngine(reward_config).compute({
        "insertion_depth": np.asarray([0.01, 0.02]),
        "previous_insertion_depth": np.asarray([0.0, 0.01]),
        "alignment_error": np.asarray([0.001, 0.003]),
        "previous_alignment_error": np.asarray([0.002, 0.004]),
        "force": np.zeros((2, 3)),
        "torque": np.zeros((2, 3)),
        "action": np.zeros((2, 8)),
        "previous_action": np.zeros((2, 8)),
        "success": np.asarray([False, True]),
    })
    terminated, reasons = TerminationChecker(reward_config).check({
        "insertion_depth": np.asarray([0.0, 0.03]),
        "alignment_error": np.asarray([0.01, 0.001]),
        "orientation_error": np.asarray([0.2, 0.01]),
        "force": np.zeros((2, 3)),
        "torque": np.zeros((2, 3)),
    })
    tracker = SuccessTracker(2, 3)
    tracker.update([False, True])
    summary = {
        "adapter": "ok",
        "wrench_shape": list(wrench["adapted_wrench"].shape),
        "reward_mean": float(reward.total.mean()),
        "terminated_count": int(terminated.sum()),
        "success_reason_count": int(reasons["success"].sum()),
        "curriculum": "ok",
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

