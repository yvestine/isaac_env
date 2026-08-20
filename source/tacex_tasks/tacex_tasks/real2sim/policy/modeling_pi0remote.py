from collections import deque
from typing import Any, Dict, Optional

import numpy as np
import torch
from openpi_client import image_tools, websocket_client_policy
from .configuration_pi0remote import PI0RemoteConfig,PI0RemoteTAVLAConfig


class PI0RemotePolicy():
    """
    Policy that infers actions from a remote OpenPI server via websocket.
    """

    config_class = PI0RemoteConfig
    name = "pi0remote"  # Add the policy name

    def __init__(self, config: PI0RemoteConfig, **kwargs):
        self.config = config

        if config.temporal_ensemble_coeff is not None:
            self.temporal_ensembler = ACTTemporalEnsembler(config.temporal_ensemble_coeff, config.n_action_steps) # We assume that the chunk size equals n_action_steps

        self.reset()  # Call reset during initialization

    def _make_client(self):
        return websocket_client_policy.WebsocketClientPolicy(host=self.config.host_ip, port=self.config.host_port)

    def reset(self):
        """This should be called whenever the environment is reset."""
        # For a remote policy, reset might involve sending a reset command to the server
        # or just resetting internal state if any.
        old_client = getattr(self, "_client", None)
        if old_client is None:
            self._client = self._make_client()
        else:
            reset = getattr(self._client, "reset", None)
            if reset is not None:
                reset()

        self._action_queue = deque([], maxlen=self.config.n_action_steps)

        print("PI0RemotePolicy reset called.")

        if self.config.temporal_ensemble_coeff is not None:
            self.temporal_ensembler.reset()

    def _new_client(self):
        return self._make_client()

    def _close_client(self):
        client = getattr(self, "_client", None)
        if client is not None:
            close = getattr(client, "close", None)
            if close is not None:
                close()
        self._client = None

    def _infer_with_retry(self, observation):
        try:
            return self._client.infer(observation)
        except Exception:
            # A timed-out websocket is not reusable. Reconnect once so the
            # environment can hold the last safe target and retry next chunk.
            self._close_client()
            self._client = self._new_client()
            return self._client.infer(observation)

    def close(self):
        self._close_client()

    def get_optim_params(self) -> dict:
        """
        Returns the policy-specific parameters dict to be passed on to the optimizer.
        For a remote policy, this is not applicable, so return an empty dict.
        """
        return {}

    @torch.no_grad()  # Inference should be done without gradients
    def select_action(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        if self.config.temporal_ensemble_coeff is not None:
            actions = self.predict_action_chunk(batch)[:, : self.config.n_action_steps]
            # print(">>> [select_action] actions chunk (temporal_ensemble branch)")
            # print("    chunk shape:", tuple(actions.shape))   # 比如 (1, 32, 7)
            action = self.temporal_ensembler.update(actions)
            return action[0] # batch size = 1 is assumed
         
        if len(self._action_queue) == 0:
            assert batch["observation.state"].shape[0] == 1, "Batch size should be 1 for remote inference."
            actions = self.predict_action_chunk(batch)[:, : self.config.n_action_steps]
            # print(">>> [select_action] actions chunk (temporal_ensemble branch)")
            # print("    chunk shape:", tuple(actions.shape))   # 比如 (1, 32, 7)
            # `self.model.forward` returns a (batch_size, n_action_steps, action_dim) tensor, but the queue
            # effectively has shape (n_action_steps, batch_size, *), hence the transpose.
            self._action_queue.extend(actions.transpose(0, 1))


        # 此时 len(self._action_queue) 是“还没拿的步数”
        remaining = len(self._action_queue)  # popleft 前
        used = self.config.n_action_steps - remaining  # 当前这一步是第 used 步
        # print(f"[select_action] step_in_chunk={used}/{self.config.n_action_steps}")
        next_action = self._action_queue.popleft()
        return next_action[0]

    def process_obs(self,batch):
        observation = {}
        key_map = {
            "observation.images.front": "observation/image",
            "observation.images.left_wrist": "observation/wrist_image",
            "observation.state": "observation/state",
        }
        for name, tensor in batch.items():
            if name in key_map:
                restored_tensor = tensor.cpu()  # 假设原始数据在 CPU 上处理

                if "image" in name:
                    restored_tensor = image_tools.convert_to_uint8(
                        image_tools.resize_with_pad(restored_tensor.numpy(), width=224, height=224)
                    )
                    restored_tensor = torch.from_numpy(restored_tensor)

                observation[key_map[name]] = restored_tensor.numpy()

        observation["prompt"] = batch["task"]
        missing_keys = {"observation/image", "observation/wrist_image", "observation/state"} - observation.keys()
        if missing_keys:
            raise KeyError(f"Missing OpenPI observation keys after remapping: {sorted(missing_keys)}")
        return observation
         
    def forward(self, batch):
        """
            expect keys in batch:
            {
                'observation.images.head_camera', 
                'observation.images.wrist_left_camera', 
                'observation.images.gelsight_left', 
                'action', 
                'observation.state', 
                'timestamp', 
                'frame_index', 
                'episode_index', 
                'index', 
                'task_index', 
                'action_is_pad', 
                'task'
            }
        """
        batch = self.process_obs(batch) 
        batch_size = batch["observation/state"].shape[0]
        remote_result = []
        for i in range(batch_size):
            # 发送数据给服务器
            obs = {key: batch[key][i] for key in batch if key != "prompt"}
            obs["prompt"] = batch["prompt"]
            res = self._client.infer(obs)
            remote_result.append(res["actions"])

        # 处理结果
        actions = np.stack(remote_result)
        return actions

    def predict_action_chunk(self, observations: Dict[str, Any]) -> np.ndarray:
        """
        Internal method to call the remote inference via the websocket client.
        Handles the client interaction and extracts actions.
        """
        try:
            # Call the remote inference using the client
            # The original snippet was action_chunk = self.client.infer(observation)["actions"]
            # Assuming client.infer takes a dict and returns a dict with "actions" key.
            action_chunk = self.forward(observations)
            if action_chunk is None:
                raise ValueError("Remote inference did not return 'actions' key.")

            if not action_chunk.flags["WRITEABLE"]:
                action_chunk = action_chunk.copy()

            return torch.from_numpy(action_chunk)
            # The action_chunk is expected to be a numpy array

        except Exception as e:
            print(f"Error during remote inference: {e}")
            raise
            
class PI0RemotePolicyTAVLA(PI0RemotePolicy):
    """Remote TAVLA adapter matching the single-arm EE-wrench report."""

    config_class = PI0RemoteTAVLAConfig
    name = "pi0remote_tavla"
    effort_history_list = deque(maxlen=10000)

    def _make_client(self):
        return _TimedWebsocketClient(
            host=self.config.host_ip,
            port=self.config.host_port,
            connect_timeout_s=getattr(self.config, "connection_timeout_s", 15.0),
            inference_timeout_s=getattr(self.config, "inference_timeout_s", 30.0),
        )

    def reset(self):
        self.effort_history_list = deque(maxlen=10000)
        self.last_server_payload_effort = None
        self.last_server_payload_is_finite = False
        self.last_server_sent_effort = None
        self.last_server_payload_matches_sent = False
        super().reset()

    @staticmethod
    def _to_single_array(value):
        if torch.is_tensor(value):
            value = value.detach().cpu().numpy()
        array = np.asarray(value)
        while array.ndim > 1 and array.shape[0] == 1:
            array = array[0]
        return array

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
        if image.shape[:2] != (480, 640):
            image = image_tools.resize_with_pad(image[None], width=640, height=480)
            image = np.asarray(image)[0]
            image = image.astype(np.uint8, copy=False)
        return np.ascontiguousarray(image)

    def process_obs(self, batch):
        """Build the exact server payload used by the fine-tuned checkpoint."""
        state = self._to_single_array(batch["observation.state"]).astype(np.float32, copy=False).reshape(-1)
        effort = self._to_single_array(batch["observation.effort"]).astype(np.float32, copy=False).reshape(-1)
        if state.shape != (8,):
            raise ValueError(f"TAVLA state must be (8,), got {state.shape}")
        if effort.shape != (6,):
            raise ValueError(f"TAVLA effort must be (6,), got {effort.shape}")
        # The fine-tuned server signature is (batch, effort_history, 6).
        # This checkpoint uses one current wrench frame, not a flat vector.
        # The server adds the batch dimension; send one history frame.
        effort = effort.reshape(1, 6)

        task = batch.get("task", "peg-in-hole")
        if isinstance(task, (list, tuple, np.ndarray)):
            task = task[0]

        wrist = self._to_rgb_image(batch["observation.images.left_wrist"])
        return {
            "images": {
                "cam_high": self._to_rgb_image(batch["observation.images.front"]),
                "cam_left_wrist": wrist,
                "cam_right_wrist": wrist.copy(),
            },
            "state": state,
            "effort": effort,
            "prompt": str(task),
        }

    def forward(self, batch):
        payload = self.process_obs(batch)
        payload_effort = np.asarray(payload["effort"], dtype=np.float32).copy()
        self.last_server_payload_effort = payload_effort
        self.last_server_payload_is_finite = bool(np.isfinite(payload_effort).all())
        if not self.last_server_payload_is_finite:
            raise ValueError("TAVLA payload effort contains NaN or Inf")
        result = self._infer_with_retry(payload)
        sent_effort = getattr(self._client, "last_sent_effort", None)
        if sent_effort is not None:
            self.last_server_sent_effort = np.asarray(sent_effort, dtype=np.float32).copy()
            self.last_server_payload_matches_sent = bool(
                np.array_equal(self.last_server_payload_effort, self.last_server_sent_effort)
            )
        if "actions" not in result:
            raise KeyError("TAVLA server response does not contain 'actions'")
        actions = np.asarray(result["actions"], dtype=np.float32)
        if actions.ndim == 2:
            actions = actions[None]
        if actions.ndim != 3 or actions.shape[0] != 1 or actions.shape[1] != self.config.n_action_steps or actions.shape[-1] != 8:
            raise ValueError(f"TAVLA actions must have shape (1, chunk, 8), got {actions.shape}")
        if not np.isfinite(actions).all():
            raise ValueError("TAVLA server returned non-finite actions")
        return actions

    def select_action(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Use current effort only for the checkpoint trained with effort_history=(0,)."""
        effort = batch["observation.effort"]
        if getattr(self.config, "num_history_steps", 1) <= 1:
            batch["observation.effort"] = effort
        else:
            self.effort_history_list.append(effort)
            history_efforts = []
            for idx in self.config.history_idx:
                if len(self.effort_history_list) + idx >= 0:
                    history_efforts.append(self.effort_history_list[idx])
                else:
                    history_efforts.append(self.effort_history_list[0])
            batch["observation.effort"] = torch.stack(history_efforts, dim=0).permute(1, 0, 2)
        return super().select_action(batch)
import logging

import websockets.sync.client
from openpi_client import msgpack_numpy

class _TimedWebsocketClient:
    """OpenPI-compatible client with bounded handshake and inference waits."""

    def __init__(self, host, port, connect_timeout_s, inference_timeout_s):
        self._uri = host if str(host).startswith("ws") else f"ws://{host}"
        if port is not None:
            self._uri += f":{port}"
        self._connect_timeout_s = float(connect_timeout_s)
        self._inference_timeout_s = float(inference_timeout_s)
        self._packer = msgpack_numpy.Packer()
        self._ws = None
        self.last_sent_effort = None
        try:
            logging.info("Connecting to TAVLA server at %s", self._uri)
            self._ws = websockets.sync.client.connect(
                self._uri,
                compression=None,
                max_size=None,
                open_timeout=self._connect_timeout_s,
            )
            metadata = self._ws.recv(timeout=self._connect_timeout_s)
            if isinstance(metadata, str):
                raise RuntimeError(f"TAVLA server sent text metadata: {metadata}")
            self._server_metadata = msgpack_numpy.unpackb(metadata)
        except Exception as exc:
            if self._ws is not None:
                self._ws.close()
            raise TimeoutError(
                f"TAVLA WebSocket handshake failed at {self._uri} within "
                f"{self._connect_timeout_s:.1f}s: {exc}"
            ) from exc

    def infer(self, obs):
        if "effort" in obs:
            self.last_sent_effort = np.asarray(obs["effort"], dtype=np.float32).copy()
        self._ws.send(self._packer.pack(obs))
        response = self._ws.recv(timeout=self._inference_timeout_s)
        if isinstance(response, str):
            raise RuntimeError(f"Error in TAVLA inference server:\n{response}")
        return msgpack_numpy.unpackb(response)

    def close(self):
        if self._ws is not None:
            self._ws.close()

    def reset(self):
        """Match openpi_client's episode-reset API.

        The current websocket protocol has no reset message, so reconnecting
        starts a fresh server-side connection state for each episode.
        """
        self.close()
        self.__init__(self._uri, None, self._connect_timeout_s, self._inference_timeout_s)
