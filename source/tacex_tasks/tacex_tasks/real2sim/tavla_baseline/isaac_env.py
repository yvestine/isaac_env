"""IsaacLab-only compatibility subclass for the affine TA-VLA baseline.

The existing deployment environment remains untouched.  This subclass adapts
the wrench after the parent has produced ``wrench_final`` and adds arm-joint
residuals to the teacher target, making the additive residual PPO baseline
actually train all eight action dimensions.
"""

from __future__ import annotations

import torch

from ..tavla_residual_env import TavlaResidualEnv
from .remote_policy import AffineRemoteTavlaPolicy
from .wrench import FixedAffineWrenchAdapter


class TavlaAffineResidualEnv(TavlaResidualEnv):
    teacher_policy_class = AffineRemoteTavlaPolicy
    def __init__(self, cfg, render_mode=None, baseline_adapter_path: str | None = None, **kwargs):
        self._baseline_adapter_path = baseline_adapter_path or getattr(
            cfg, "baseline_adapter_path", "checkpoints/unpaired_sim_to_real_affine.pt"
        )
        self._baseline_wrench_adapter = FixedAffineWrenchAdapter(self._baseline_adapter_path)
        super().__init__(cfg, render_mode=render_mode, **kwargs)
        self.last_tavla_raw_wrench_final = torch.zeros_like(self.last_tavla_wrench_final)
        self.last_tavla_adapted_wrench = torch.zeros_like(self.last_tavla_wrench_final)
        self._baseline_joint_residual_scale = torch.as_tensor(
            getattr(cfg, "baseline_joint_residual_scale", cfg.joint_residual_scale),
            dtype=torch.float32,
            device=self.device,
        ).view(1, 7)
        self._baseline_max_joint_residual_step = float(getattr(cfg, "baseline_max_joint_residual_step", 0.04))

    def _teacher_batch(self):
        payload = super()._teacher_batch()
        raw_final = self.last_tavla_wrench_final.detach().clone()
        adapted = self._baseline_wrench_adapter.apply_torch(raw_final)
        self.last_tavla_raw_wrench_final = raw_final
        self.last_tavla_adapted_wrench = adapted.detach().clone()
        self.last_tavla_effort = adapted.detach().clone()
        payload["observation.effort"] = adapted[0].detach().cpu().unsqueeze(0)
        return payload

    def _combine_teacher_and_residual(self):
        super()._combine_teacher_and_residual()
        if bool(getattr(self.cfg, "teacher_eval_only", False)):
            return
        residual = self.residual_action[:, :7]
        target = self.combined_joint_target.clone()
        delta = residual * self._baseline_joint_residual_scale
        if self._baseline_max_joint_residual_step > 0.0:
            delta = delta.clamp(-self._baseline_max_joint_residual_step, self._baseline_max_joint_residual_step)
        target[:, :7] += delta
        lower, upper = self._joint_limits()
        target[:, :7] = torch.minimum(torch.maximum(target[:, :7], lower), upper)
        self.combined_joint_target = target
        self.next_action = target.clone()
        self.teacher_joint_error = target - self._current_tavla_state()

