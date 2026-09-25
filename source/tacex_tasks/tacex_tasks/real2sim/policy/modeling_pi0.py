from __future__ import annotations

from typing import Any, Mapping

import numpy as np
from openpi_client import msgpack_numpy
from openpi_client import websocket_client_policy

from .configuration_pi0 import PI0RemoteJointConfig


class PI0RemoteJointPolicy:
    """WebSocket client for the current image/state/effort Pi0 server."""

    config_class = PI0RemoteJointConfig
    name = "pi0remote_joint"

    def __init__(self, config: PI0RemoteJointConfig):
        self.config = config
        self._client = websocket_client_policy.WebsocketClientPolicy(
            host=config.host_ip,
            port=config.host_port,
        )

    @staticmethod
    def _to_rgb_uint8(value: Any) -> np.ndarray:
        image = value.detach().cpu().numpy() if hasattr(value, "detach") else np.asarray(value)
        while image.ndim > 3 and image.shape[0] == 1:
            image = image[0]
        if image.ndim == 3 and image.shape[0] in (3, 4) and image.shape[-1] not in (3, 4):
            image = np.transpose(image, (1, 2, 0))
        if image.ndim != 3 or image.shape[-1] not in (3, 4):
            raise ValueError(f"Pi0 image must be HxWx3 or HxWx4, got {image.shape}")
        if image.shape[-1] == 4:
            image = image[..., :3]
        if image.dtype != np.uint8:
            if image.size and float(np.nanmax(image)) <= 1.0:
                image = image * 255.0
            image = np.clip(image, 0.0, 255.0).astype(np.uint8)
        return np.ascontiguousarray(image)

    @staticmethod
    def _first(mapping: Mapping[str, Any], *keys: str) -> Any:
        for key in keys:
            if key in mapping:
                return mapping[key]
        raise KeyError(f"Missing Pi0 observation field; expected one of {keys}")

    def process_obs(self, observation: Mapping[str, Any]) -> dict[str, Any]:
        """Build exactly the image/state-only payload accepted by Pi0."""
        images = observation.get("images", {})
        front = self._first(images, "cam_high") if images else self._first(observation, "front_rgb", "front")
        wrist = self._first(images, "cam_left_wrist") if images else self._first(observation, "wrist_rgb", "wrist")
        state = np.asarray(self._first(observation, "state", "state8"), dtype=np.float32).reshape(-1)
        if state.shape != (8,):
            raise ValueError(f"Pi0 state must have shape (8,), got {state.shape}")
        effort = np.asarray(self._first(observation, "effort"), dtype=np.float32).reshape(-1)
        if effort.shape != (6,):
            raise ValueError(f"Pi0 effort must have shape (6,), got {effort.shape}")
        if not np.isfinite(effort).all():
            raise ValueError("Pi0 effort contains NaN or Inf")
        prompt = str(observation.get("prompt", "peg-in-hole"))
        front_rgb = self._to_rgb_uint8(front)
        wrist_rgb = self._to_rgb_uint8(wrist)
        return {
            "images": {
                "cam_high": front_rgb,
                "cam_left_wrist": wrist_rgb,
                "cam_right_wrist": wrist_rgb.copy(),
            },
            "state": state,
            # The 8001 checkpoint was trained with one current 6-D wrench
            # frame. Keep the history dimension explicit as (1, 6); the
            # server adds only the batch dimension.
            "effort": effort.reshape(1, 6),
            "prompt": prompt,
        }

    def reset(self) -> None:
        """Reset the server-side policy state on the current connection."""
        websocket = getattr(self._client, "_ws", None)
        packer = getattr(self._client, "_packer", None)
        if websocket is None or packer is None:
            raise RuntimeError("openpi-client does not expose its WebSocket connection")
        websocket.send(packer.pack({"reset": True}))
        response = websocket.recv(timeout=float(self.config.inference_timeout_s))
        if isinstance(response, str):
            raise RuntimeError(f"Pi0 reset failed: {response}")
        reset_result = msgpack_numpy.unpackb(response)
        if reset_result != {"reset": True}:
            raise RuntimeError(f"Unexpected Pi0 reset response: {reset_result!r}")

    def predict_action_chunk(self, observation: Mapping[str, Any]) -> np.ndarray:
        payload = self.process_obs(observation)
        # WebsocketClientPolicy.infer() calls recv() without a timeout. Use
        # the already-open client connection directly so a stalled inference
        # server cannot block Isaac Sim forever at a chunk boundary.
        websocket = getattr(self._client, "_ws", None)
        packer = getattr(self._client, "_packer", None)
        if websocket is None or packer is None:
            raise RuntimeError("openpi-client does not expose its WebSocket connection")
        websocket.send(packer.pack(payload))
        try:
            response = websocket.recv(timeout=float(self.config.inference_timeout_s))
        except TimeoutError as exc:
            raise TimeoutError(
                f"Pi0 inference timed out after {self.config.inference_timeout_s:.1f}s"
            ) from exc
        if isinstance(response, str):
            raise RuntimeError(f"Error in inference server:\n{response}")
        result = msgpack_numpy.unpackb(response)
        if "actions" not in result:
            raise KeyError("Pi0 server response does not contain 'actions'")
        actions = np.asarray(result["actions"], dtype=np.float32)
        if actions.shape == (1, self.config.n_action_steps, self.config.action_dim):
            actions = actions[0]
        expected = (self.config.n_action_steps, self.config.action_dim)
        if actions.shape != expected:
            raise ValueError(f"Pi0 actions must have shape {expected}, got {actions.shape}")
        if not np.isfinite(actions).all():
            raise ValueError("Pi0 actions contain NaN or Inf")
        return actions

    def close(self) -> None:
        close = getattr(self._client, "close", None)
        if callable(close):
            close()
        else:
            websocket = getattr(self._client, "_ws", None)
            if websocket is not None:
                websocket.close()
