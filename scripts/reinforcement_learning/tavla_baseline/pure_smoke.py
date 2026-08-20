"""Pure baseline smoke test; intentionally does not import IsaacLab."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

BASELINE_ROOT = Path(__file__).resolve().parents[3] / "source/tacex_tasks/tacex_tasks/real2sim"
sys.path.insert(0, str(BASELINE_ROOT))

from tavla_baseline.config import RandomizationConfig, RewardConfig
from tavla_baseline.observations import ActorCriticObservationSplitter
from tavla_baseline.randomization import CurriculumStage, DomainRandomizer
from tavla_baseline.rewards import RewardEngine, TerminationChecker
from tavla_baseline.wrench import FixedAffineWrenchAdapter, WrenchPipeline


parser = argparse.ArgumentParser(description="Pure TA-VLA baseline smoke test")
parser.add_argument("--adapter", type=Path, default=Path("checkpoints/unpaired_sim_to_real_affine.pt"))
args = parser.parse_args()


def main() -> None:
    adapter = FixedAffineWrenchAdapter(args.adapter)
    pipeline = WrenchPipeline(adapter)
    wrench = pipeline.convert(np.zeros((2, 6), dtype=np.float32))
    assert np.array_equal(wrench["wrench_final"], -wrench["wrench_base"])
    assert np.isfinite(wrench["adapted_wrench"]).all()
    split = ActorCriticObservationSplitter().split({"policy": np.zeros(16), "critic": np.zeros(32)})
    assert split.actor.shape == (16,) and split.critic.shape == (32,)
    randomizer = DomainRandomizer(RandomizationConfig(), seed=0)
    randomizer.set_stage(CurriculumStage.SENSOR_DELAY)
    assert randomizer.sample(4)["wrench_noise_std"].shape == (4,)
    reward = RewardEngine(RewardConfig()).compute({
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
    terminated, reasons = TerminationChecker(RewardConfig()).check({
        "insertion_depth": np.asarray([0.0, 0.03]),
        "alignment_error": np.asarray([0.01, 0.001]),
        "orientation_error": np.asarray([0.2, 0.01]),
        "force": np.zeros((2, 3)),
        "torque": np.zeros((2, 3)),
    })
    summary = {
        "adapter": "ok",
        "adapted_shape": list(wrench["adapted_wrench"].shape),
        "reward_mean": float(reward.total.mean()),
        "terminated_count": int(terminated.sum()),
        "success_reason_count": int(reasons["success"].sum()),
        "curriculum": "ok",
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

