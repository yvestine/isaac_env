"""Protocol-accurate variant of the additive affine residual environment."""

from __future__ import annotations

from .. import tavla_residual_env as residual_module
from .isaac_env import TavlaAffineResidualEnv
from .remote_policy import BaselineRemoteTavlaPolicy


class TavlaAffineProtocolResidualEnv(TavlaAffineResidualEnv):
    """Instantiate the parent environment with the additive Server client.

    The temporary module substitution is limited to the parent constructor;
    it avoids editing the existing task registration while ensuring its policy
    factory creates the new 224x224/reset-message client from the first connect.
    """

    def __init__(self, cfg, render_mode=None, baseline_adapter_path=None, **kwargs):
        original_policy = residual_module.PI0RemotePolicyTAVLA
        residual_module.PI0RemotePolicyTAVLA = BaselineRemoteTavlaPolicy
        try:
            super().__init__(cfg, render_mode=render_mode, baseline_adapter_path=baseline_adapter_path, **kwargs)
        finally:
            residual_module.PI0RemotePolicyTAVLA = original_policy

