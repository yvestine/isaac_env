"""Optional Co-SFT reference regularization for Flow-Noise PPO."""

from __future__ import annotations

from typing import Any

import torch


class FlowVelocityReference:
    """Keep the trainable flow velocity close to a frozen Co-SFT reference."""

    def __init__(self, reference_model: Any, weight: float = 0.0):
        self.reference_model = reference_model
        self.weight = float(weight)
        for parameter in getattr(reference_model, "parameters", lambda: ())():
            parameter.requires_grad_(False)
        self.reference_model.eval()

    def loss(self, observation: torch.Tensor, action_state: torch.Tensor, tau: torch.Tensor, trainable_velocity: torch.Tensor) -> torch.Tensor:
        if self.weight <= 0.0:
            return trainable_velocity.new_zeros(())
        with torch.no_grad():
            reference_velocity = self.reference_model(observation, action_state, tau)
        return self.weight * torch.square(trainable_velocity - reference_velocity).mean()

