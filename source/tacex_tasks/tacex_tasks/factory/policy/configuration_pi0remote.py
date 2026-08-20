from dataclasses import dataclass, field
from typing import List

def _make_history_idx(num_steps: int = 10, step_interval: int = 4) -> List[int]:
    """
    生成历史帧索引列表。
    
    Args:
        num_steps: 要取的历史帧数量（包括最近的一帧）
        step_interval: 帧之间的间隔（单位：帧数）
    
    Returns:
        List[int]: 负索引列表，如 [-37, -33, ..., -1]
    """
    if num_steps <= 0 or step_interval <= 0:
        return []
    start = -(step_interval * (num_steps - 1) + 1)
    return list(range(start, 0, step_interval))


@dataclass
class PI0RemoteConfig():
    """
    Configuration class for the PI0Remote policy.
    """

    type: str = "pi0remote"
    host_ip: str = "127.0.0.1"
    # host_ip: str = "172.16.110.61"
    host_port: int = 8990
    n_action_steps: int = 10
    temporal_ensemble_coeff: float | None = None # If not None, use temporPal ensembling with this coefficient
    # temporal_ensemble_coeff: float = 0.01
    # Add any other specific configuration parameters for pi0_remote here
    # For example, observation keys, action space details, etc.
    # observation_keys: list[str] = field(default_factory=lambda: ["image", "joint_states"])
    # action_dimension: int = 7 # Example action dimension

    @property
    def observation_delta_indices(self) -> list | None:
        return None

    @property
    def action_delta_indices(self) -> list | None:
        return None

    @property
    def reward_delta_indices(self) -> list | None:
        return None

    def get_optimizer_preset(self):
        # Configuration class doesn't define optimizer presets
        return None

    def get_scheduler_preset(self):
        # Configuration class doesn't define scheduler presets
        return None

    def validate_features(self):
        # No feature validation needed in the configuration class
        pass

@dataclass
class PI0RemoteTAVLAConfig(PI0RemoteConfig):
    type: str = "pi0remote_tavla"
    
    num_history_steps: int = 20      # 使用多少个历史时间步
    history_step_interval: int = 2   # 每隔多少帧取一帧
    history_idx: List[int] = field(init=False)

    def __post_init__(self):
        if hasattr(super(), '__post_init__'):
            super().__post_init__()
        self.history_idx = _make_history_idx(
            self.num_history_steps,
            self.history_step_interval
        )
