"""TAVLA websocket client with a 10-frame, 10-D force-trend input."""

from __future__ import annotations

import os
from typing import Any

import numpy as np

from .modeling_pi0remote import PI0RemotePolicyTAVLA


class SimDataForceTrendPolicy(PI0RemotePolicyTAVLA):
    """Keep the original TAVLA client and change only the effort shape."""

    name = "sim_data_force_trend"

    def __init__(self, config, **kwargs):
        host = str(config.host_ip).strip()
        for variable in ("NO_PROXY", "no_proxy"):
            entries = [
                value.strip()
                for value in os.environ.get(variable, "").split(",")
                if value.strip()
            ]
            if host and host not in entries:
                entries.append(host)
                os.environ[variable] = ",".join(entries)
        self.last_payload_manifest: dict[str, Any] | None = None
        self.last_actions: np.ndarray | None = None
        self.total_inference_count = 0
        super().__init__(config, **kwargs)

    def process_obs(self, batch):
        state = self._to_single_array(batch["observation.state"]).astype(
            np.float32, copy=False
        ).reshape(-1)
        effort = self._to_single_array(batch["observation.effort"]).astype(
            np.float32, copy=False
        )
        if state.shape != (8,) or not np.isfinite(state).all():
            raise ValueError(f"TAVLA state must be finite (8,), got {state.shape}")
        if effort.shape != (10, 10) or not np.isfinite(effort).all():
            raise ValueError(
                f"TAVLA force-trend effort must be finite (10,10), got {effort.shape}"
            )
        effort = np.ascontiguousarray(effort, dtype=np.float32)
        task = batch.get("task", "peg-in-hole")
        if isinstance(task, (list, tuple, np.ndarray)):
            task = task[0]
        front = self._to_rgb_image(batch["observation.images.front"])
        wrist = self._to_rgb_image(batch["observation.images.left_wrist"])
        payload = {
            "images": {
                "cam_high": front,
                "cam_left_wrist": wrist,
                "cam_right_wrist": wrist.copy(),
            },
            "state": np.ascontiguousarray(state),
            "effort": effort,
            "prompt": str(task),
        }
        self.last_payload_manifest = {
            "state_shape": list(state.shape),
            "effort_shape": list(effort.shape),
            "front_shape": list(front.shape),
            "wrist_shape": list(wrist.shape),
            "prompt": str(task),
        }
        return payload

    def predict_action_chunk(self, observations):
        actions = super().predict_action_chunk(observations)
        array = actions.detach().cpu().numpy() if hasattr(actions, "detach") else np.asarray(actions)
        self.last_actions = np.asarray(array, dtype=np.float32).copy()
        self.total_inference_count += 1
        return actions
