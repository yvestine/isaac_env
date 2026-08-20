"""Configuration objects for the additive TA-VLA RL baselines."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class RewardConfig:
    """Configurable contact-rich reward.

    Force and torque thresholds are deliberately configuration values.  They
    must be replaced with thresholds measured from the successful real-data
    distribution before a real experiment is reported.
    """

    success_reward: float = 10.0
    depth_progress_weight: float = 2.0
    alignment_weight: float = 1.0
    force_exceed_weight: float = 0.25
    torque_exceed_weight: float = 0.25
    action_delta_weight: float = 0.02
    collision_weight: float = 2.0
    timeout_penalty: float = 1.0
    force_soft_threshold: float = 13.49642
    force_hard_threshold: float = 20.24463
    torque_soft_threshold: float = 3.0
    torque_hard_threshold: float = 6.0
    success_depth_threshold: float = 0.02
    success_alignment_threshold: float = 0.002
    success_orientation_threshold: float = 0.05
    success_hold_steps: int = 3


@dataclass
class RandomizationConfig:
    """Curriculum ranges for the pre-insertion reset distribution."""

    position_error_m: tuple[float, float] = (0.0, 0.003)
    orientation_error_rad: tuple[float, float] = (0.0, 0.08)
    insertion_depth_m: tuple[float, float] = (0.0, 0.002)
    friction: tuple[float, float] = (0.7, 0.9)
    contact_stiffness: tuple[float, float] = (0.8, 1.2)
    contact_damping: tuple[float, float] = (0.8, 1.2)
    controller_gain: tuple[float, float] = (0.9, 1.1)
    action_delay_steps: tuple[int, int] = (0, 0)
    qpos_noise_rad: tuple[float, float] = (0.0, 0.002)
    wrench_bias: tuple[float, float] = (0.0, 0.05)
    wrench_scale: tuple[float, float] = (1.0, 1.0)
    wrench_noise_std: tuple[float, float] = (0.0, 0.02)
    wrench_delay_steps: tuple[int, int] = (0, 0)


@dataclass
class PPOConfig:
    """Conservative PPO defaults for a pretrained contact policy."""

    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_ratio: float = 0.2
    actor_lr: float = 3.0e-6
    critic_lr: float = 1.0e-4
    update_epochs: int = 3
    minibatch_size: int = 64
    gradient_clip: float = 0.5
    entropy_weight: float = 0.0
    value_weight: float = 0.5
    target_kl: float = 0.03
    critic_warmup_updates: int = 1
    actor_hidden_dims: tuple[int, ...] = (256, 256)
    critic_hidden_dims: tuple[int, ...] = (256, 256, 128)
    residual_action_scale: tuple[float, ...] = (0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.05)
    rollout_steps: int = 128
    seed: int = 0


@dataclass
class BaselineConfig:
    """Top-level baseline configuration shared by all entry points."""

    task: str = "TacEx-RealSim-PegInsert-TAVLAResidual-Direct-v0"
    prompt: str = "peg-in-hole"
    adapter_checkpoint: str = "checkpoints/unpaired_sim_to_real_affine.pt"
    adapter_metadata: str = "checkpoints/unpaired_sim_to_real_affine.json"
    tavla_host: str = "10.0.40.113"
    tavla_port: int = 8000
    decision_hz: int = 10
    simulation_hz: int = 30
    exec_horizon: int = 1
    action_chunk_length: int = 50
    action_dim: int = 8
    reward: RewardConfig = field(default_factory=RewardConfig)
    randomization: RandomizationConfig = field(default_factory=RandomizationConfig)
    ppo: PPOConfig = field(default_factory=PPOConfig)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "BaselineConfig":
        data = dict(data)
        data["reward"] = RewardConfig(**data.get("reward", {}))
        data["randomization"] = RandomizationConfig(**data.get("randomization", {}))
        ppo_data = data.get("ppo", {})
        if "actor_hidden_dims" in ppo_data:
            ppo_data["actor_hidden_dims"] = tuple(ppo_data["actor_hidden_dims"])
        if "critic_hidden_dims" in ppo_data:
            ppo_data["critic_hidden_dims"] = tuple(ppo_data["critic_hidden_dims"])
        if "residual_action_scale" in ppo_data:
            ppo_data["residual_action_scale"] = tuple(ppo_data["residual_action_scale"])
        data["ppo"] = PPOConfig(**ppo_data)
        return cls(**data)

    @classmethod
    def load_json(cls, path: str | Path) -> "BaselineConfig":
        with Path(path).open("r", encoding="utf-8") as handle:
            return cls.from_dict(json.load(handle))

    def save_json(self, path: str | Path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("w", encoding="utf-8") as handle:
            json.dump(asdict(self), handle, indent=2, ensure_ascii=False)

