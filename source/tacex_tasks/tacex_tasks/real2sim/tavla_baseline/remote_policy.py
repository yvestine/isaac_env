"""TA-VLA Server client for the explicitly documented baseline wire protocol."""

from __future__ import annotations

import logging

import numpy as np
import websockets.sync.client
from openpi_client import image_tools, msgpack_numpy

from ..policy.modeling_pi0remote import PI0RemotePolicyTAVLA


class BaselineWebsocketClient:
    """WebSocket + MessagePack/NumPy client with an explicit reset message."""

    def __init__(self, host: str, port: int, connect_timeout_s: float = 15.0, inference_timeout_s: float = 30.0):
        self.uri = host if str(host).startswith("ws") else f"ws://{host}:{port}"
        if str(host).startswith("ws") and port is not None and f":{port}" not in self.uri:
            self.uri = f"{self.uri}:{port}"
        self.connect_timeout_s = float(connect_timeout_s)
        self.inference_timeout_s = float(inference_timeout_s)
        self.packer = msgpack_numpy.Packer()
        self.last_sent_effort = None
        self._ws = websockets.sync.client.connect(self.uri, compression=None, max_size=None, open_timeout=self.connect_timeout_s)
        metadata = self._ws.recv(timeout=self.connect_timeout_s)
        if isinstance(metadata, str):
            raise RuntimeError(f"TAVLA Server sent text metadata: {metadata}")
        self.server_metadata = msgpack_numpy.unpackb(metadata)
        logging.info("Connected to TA-VLA Server at %s", self.uri)

    def infer(self, observation: dict):
        if "effort" in observation:
            self.last_sent_effort = np.asarray(observation["effort"], dtype=np.float32).copy()
        self._ws.send(self.packer.pack(observation))
        response = self._ws.recv(timeout=self.inference_timeout_s)
        if isinstance(response, str):
            raise RuntimeError(f"TAVLA Server inference error: {response}")
        return msgpack_numpy.unpackb(response)

    def reset(self):
        self._ws.send(self.packer.pack({"reset": True}))
        response = self._ws.recv(timeout=self.inference_timeout_s)
        if isinstance(response, str):
            raise RuntimeError(f"TAVLA Server reset error: {response}")
        result = msgpack_numpy.unpackb(response)
        if result != {"reset": True}:
            raise RuntimeError(f"unexpected TAVLA Server reset response: {result}")

    def close(self):
        if self._ws is not None:
            self._ws.close()
            self._ws = None


class BaselineRemoteTavlaPolicy(PI0RemotePolicyTAVLA):
    """Use 224x224 RGB inputs and the reset-message protocol."""

    def _make_client(self):
        return BaselineWebsocketClient(
            self.config.host_ip,
            self.config.host_port,
            getattr(self.config, "connection_timeout_s", 15.0),
            getattr(self.config, "inference_timeout_s", 30.0),
        )

    @classmethod
    def _to_rgb_image(cls, value):
        image = cls._to_single_array(value)
        if image.ndim == 3 and image.shape[0] == 3 and image.shape[-1] != 3:
            image = np.transpose(image, (1, 2, 0))
        if image.ndim != 3 or image.shape[-1] != 3:
            raise ValueError(f"TAVLA image must be HxWx3, got {image.shape}")
        if image.dtype != np.uint8:
            if image.size and float(image.max()) <= 1.0:
                image = image * 255.0
            image = np.clip(image, 0.0, 255.0).astype(np.uint8)
        if image.shape[:2] != (224, 224):
            image = np.asarray(image_tools.resize_with_pad(image[None], width=224, height=224))[0]
        return np.ascontiguousarray(image.astype(np.uint8, copy=False))

    def process_obs(self, batch):
        state = self._to_single_array(batch["observation.state"]).astype(np.float32, copy=False).reshape(-1)
        effort = self._to_single_array(batch["observation.effort"]).astype(np.float32, copy=False).reshape(-1)
        if state.shape != (8,):
            raise ValueError(f"TAVLA state must be (8,), got {state.shape}")
        if effort.shape != (6,):
            raise ValueError(f"TAVLA effort must be (6,), got {effort.shape}")
        task = batch.get("task", "peg-in-hole")
        if isinstance(task, (list, tuple, np.ndarray)):
            task = task[0]
        return {
            "images": {
                "cam_high": self._to_rgb_image(batch["observation.images.front"]),
                "cam_left_wrist": self._to_rgb_image(batch["observation.images.left_wrist"]),
            },
            "state": state,
            "effort": effort.reshape(1, 6),
            "prompt": str(task),
        }


class AffineRemoteTavlaPolicy(PI0RemotePolicyTAVLA):
    """Affine deployment client with the exact 224x224 RGB wire shape.

    It keeps the normal reconnect-based reset protocol used by the active
    Server and only specializes image preprocessing for the affine task.
    """

    @classmethod
    def _to_rgb_image(cls, value):
        image = cls._to_single_array(value)
        if image.ndim == 3 and image.shape[0] == 3 and image.shape[-1] != 3:
            image = np.transpose(image, (1, 2, 0))
        if image.ndim != 3 or image.shape[-1] != 3:
            raise ValueError(f"TAVLA image must be HxWx3, got {image.shape}")
        if image.dtype != np.uint8:
            if image.size and float(image.max()) <= 1.0:
                image = image * 255.0
            image = np.clip(image, 0.0, 255.0).astype(np.uint8)
        if image.shape[:2] != (224, 224):
            image = np.asarray(image_tools.resize_with_pad(image[None], width=224, height=224))[0]
        return np.ascontiguousarray(image.astype(np.uint8, copy=False))

