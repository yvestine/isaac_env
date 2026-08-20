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

    def reset(self):
        """This should be called whenever the environment is reset."""
        # For a remote policy, reset might involve sending a reset command to the server
        # or just resetting internal state if any.
        self._client = websocket_client_policy.WebsocketClientPolicy(host=self.config.host_ip, port=self.config.host_port)

        self._action_queue = deque([], maxlen=self.config.n_action_steps)

        print("PI0RemotePolicy reset called.")

        if self.config.temporal_ensemble_coeff is not None:
            self.temporal_ensembler.reset()

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
        for name, tensor in batch.items():
            if name.startswith("observation"):
                restored_tensor = tensor.cpu()  # 假设原始数据在 CPU 上处理

                if "image" in name:
                    restored_tensor = image_tools.convert_to_uint8(
                        image_tools.resize_with_pad(restored_tensor.numpy(), width=224, height=224)
                    )
                    restored_tensor = torch.from_numpy(restored_tensor)

                observation[name] = restored_tensor.numpy()

        observation["prompt"] = batch["task"]
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
        batch_size = batch["observation.state"].shape[0]
        remote_result = []
        for i in range(batch_size):
            # 发送数据给服务器
            obs = {key: batch[key][i] for key in batch}
            obs["task"] = batch["prompt"] # fix a bug
            # print( batch["prompt"])
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
            
class PI0RemotePolicyTAVLA(PI0RemotePolicy):
    config_class = PI0RemoteTAVLAConfig
    name = "pi0remote_tavla"  # Add the policy name
    effort_history_list = deque(maxlen=10000)

    def reset(self):
        self.effort_history_list = deque(maxlen=10000)
        super().reset()


    def select_action(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        effort = batch["observation.effort"]  # (batch, 6)
        self.effort_history_list.append(effort)  # 存当前帧

        # 构建历史effort序列
        history_efforts = []

        for idx in self.config.history_idx:
            if len(self.effort_history_list) + idx >= 0:
                e = self.effort_history_list[idx]
            else:
                e = self.effort_history_list[0] # 不足时用最后一个填充
            history_efforts.append(e)

        # 拼接成张量: (10, batch, 6)
        batch["observation.effort"] = torch.stack(history_efforts, dim=0).permute(1, 0, 2)

        # 调用父类的select_action
        return super().select_action(batch)
    
     
