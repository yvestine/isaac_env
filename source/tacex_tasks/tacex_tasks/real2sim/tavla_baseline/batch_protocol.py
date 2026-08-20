"""Batch WebSocket payload contract for future parallel TA-VLA rollouts."""

from __future__ import annotations

from typing import Any

import numpy as np


class BatchInferencePayload:
    """Validate the proposed ``infer_batch`` MessagePack contract."""

    def __init__(self, episode_ids: Any, cam_high: Any, cam_left_wrist: Any, state: Any, effort: Any, prompts: Any):
        self.episode_ids = np.asarray(episode_ids, dtype=np.int64).reshape(-1)
        self.cam_high = np.asarray(cam_high, dtype=np.uint8)
        self.cam_left_wrist = np.asarray(cam_left_wrist, dtype=np.uint8)
        self.state = np.asarray(state, dtype=np.float32)
        self.effort = np.asarray(effort, dtype=np.float32)
        self.prompts = list(prompts)
        batch = self.episode_ids.size
        if self.cam_high.shape != (batch, 224, 224, 3):
            raise ValueError(f"cam_high must have shape (B,224,224,3), got {self.cam_high.shape}")
        if self.cam_left_wrist.shape != (batch, 224, 224, 3):
            raise ValueError(f"cam_left_wrist must have shape (B,224,224,3), got {self.cam_left_wrist.shape}")
        if self.state.shape != (batch, 8):
            raise ValueError(f"state must have shape (B,8), got {self.state.shape}")
        if self.effort.shape != (batch, 1, 6):
            raise ValueError(f"effort must have shape (B,1,6), got {self.effort.shape}")
        if len(self.prompts) != batch:
            raise ValueError("prompts length must match batch size")
        if not np.isfinite(self.state).all() or not np.isfinite(self.effort).all():
            raise FloatingPointError("batch state/effort contains NaN or Inf")

    def to_message(self) -> dict[str, Any]:
        return {
            "type": "infer_batch",
            "episode_ids": self.episode_ids,
            "images": {"cam_high": self.cam_high, "cam_left_wrist": self.cam_left_wrist},
            "state": self.state,
            "effort": self.effort,
            "prompt": self.prompts,
        }


def validate_batch_actions(actions: Any, batch_size: int, chunk_length: int = 50, action_dim: int = 8) -> np.ndarray:
    values = np.asarray(actions, dtype=np.float32)
    expected = (batch_size, chunk_length, action_dim)
    if values.shape != expected:
        raise ValueError(f"batch actions must have shape {expected}, got {values.shape}")
    if not np.isfinite(values).all():
        raise FloatingPointError("batch actions contain NaN or Inf")
    return values

