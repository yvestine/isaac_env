"""Pure-Python regression tests for the additive TA-VLA baseline package."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest


PACKAGE_ROOT = Path(__file__).parents[1] / "source/tacex_tasks/tacex_tasks/real2sim"
sys.path.insert(0, str(PACKAGE_ROOT))

from tavla_baseline.config import RandomizationConfig, RewardConfig  # noqa: E402
from tavla_baseline.observations import ActorCriticObservationSplitter  # noqa: E402
from tavla_baseline.randomization import CurriculumStage, DomainRandomizer  # noqa: E402
from tavla_baseline.rewards import RewardEngine, SuccessTracker, TerminationChecker  # noqa: E402
from tavla_baseline.states import PreInsertState, PreInsertStateDatabase  # noqa: E402
from tavla_baseline.wrench import FixedAffineWrenchAdapter, WrenchPipeline  # noqa: E402


ROOT = Path(__file__).parents[1]


def test_affine_adapter_matches_declared_formula():
    adapter = FixedAffineWrenchAdapter(ROOT / "checkpoints/unpaired_sim_to_real_affine.pt")
    sample = np.asarray([[1.0, -2.0, 3.0, 0.1, -0.2, 0.3]], dtype=np.float32)
    expected = (sample - adapter.source_center) / adapter.source_scale * adapter.target_scale + adapter.target_center
    np.testing.assert_allclose(adapter.apply(sample), expected, rtol=1e-6, atol=1e-6)


def test_wrench_pipeline_preserves_sign_and_order():
    adapter = FixedAffineWrenchAdapter(ROOT / "checkpoints/unpaired_sim_to_real_affine.pt")
    base = np.asarray([[1, 2, 3, 4, 5, 6]], dtype=np.float32)
    result = WrenchPipeline(adapter).convert(base)
    np.testing.assert_array_equal(result["wrench_final"], -base)
    np.testing.assert_allclose(result["adapted_wrench"], adapter.apply(-base))


def test_actor_observation_does_not_accept_privileged_keys():
    splitter = ActorCriticObservationSplitter()
    with pytest.raises(ValueError, match="privileged"):
        splitter.validate_actor_keys(["qpos", "wrench_base"])
    split = splitter.split({"policy": np.ones(4), "critic": np.ones(8)})
    assert split.actor.shape == (4,)
    assert split.critic.shape == (8,)


def test_curriculum_reveals_randomization_in_order():
    randomizer = DomainRandomizer(RandomizationConfig(), seed=7)
    nominal = randomizer.sample(32)
    assert np.all(nominal["friction"] == 1.0)
    randomizer.set_stage(CurriculumStage.POSE)
    pose = randomizer.sample(32)
    assert np.any(pose["position_error_m"] != 0.0)
    randomizer.set_stage(CurriculumStage.SENSOR_DELAY)
    delayed = randomizer.sample(32)
    assert delayed["wrench_noise_std"].shape == (32,)


def test_reward_and_safety_termination_have_named_terms():
    config = RewardConfig(force_soft_threshold=2.0, force_hard_threshold=5.0, torque_soft_threshold=1.0, torque_hard_threshold=3.0)
    reward = RewardEngine(config).compute({
        "insertion_depth": [0.01],
        "previous_insertion_depth": [0.0],
        "alignment_error": [0.001],
        "previous_alignment_error": [0.002],
        "force": [[3.0, 0.0, 0.0]],
        "torque": [[0.0, 2.0, 0.0]],
        "action": [[0.0] * 8],
        "previous_action": [[0.1] * 8],
        "success": [True],
    })
    assert {"success_reward", "depth_progress", "force_exceed_penalty", "action_delta_penalty"} <= reward.terms.keys()
    terminated, reasons = TerminationChecker(config).check({
        "insertion_depth": [0.03],
        "alignment_error": [0.001],
        "orientation_error": [0.01],
        "force": [[1.0, 0.0, 0.0]],
        "torque": [[0.5, 0.0, 0.0]],
    })
    assert bool(terminated[0])
    assert bool(reasons["success"][0])


def test_success_tracker_requires_consecutive_steps():
    tracker = SuccessTracker(1, hold_steps=3)
    assert not tracker.update([True])[0]
    assert not tracker.update([True])[0]
    assert tracker.update([True])[0]
    assert not tracker.update([True])[0]


def test_preinsert_state_database_roundtrip(tmp_path):
    database = PreInsertStateDatabase()
    database.add(PreInsertState.from_arrays(np.zeros(8), qvel=np.zeros(8), metadata={"seed": 3}))
    path = tmp_path / "states.json"
    database.save(path)
    restored = PreInsertStateDatabase.load(path)
    assert len(restored) == 1
    assert restored.records[0].metadata["seed"] == 3


def test_flow_noise_log_probability_is_finite_when_torch_available():
    torch = pytest.importorskip("torch")
    from tavla_baseline.flow_noise import FlowNoiseConfig, FlowNoiseSampler

    class Velocity:
        def __call__(self, observation, action_state, tau):
            return torch.zeros_like(action_state)

    class Noise:
        def __call__(self, observation, action_state, tau):
            return torch.full_like(action_state, -2.0)

    sampler = FlowNoiseSampler(Velocity(), Noise(), FlowNoiseConfig(integration_steps=4, action_horizon=1, action_dim=2))
    observation = torch.zeros((3, 5))
    action, log_prob, trace = sampler.sample(observation)
    assert action.shape == (3, 1, 2)
    assert torch.isfinite(log_prob).all()
    assert torch.isfinite(sampler.log_prob(observation, trace)).all()
