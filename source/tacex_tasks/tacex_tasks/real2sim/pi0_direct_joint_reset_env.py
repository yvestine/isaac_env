"""Pi0 diagnostic environment with a real-trajectory joint reset.

This module deliberately keeps the normal Pi0 environment and its IK reset
unchanged. The registered diagnostic environment only replaces the reset-time
arm pose with the first 7-D joint position from a real-data H5 trajectory.
"""
from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import torch
from isaaclab.utils import configclass

from .pi0_env import Pi0RealSimEnv
from .pi0_env_cfg import RealSimPi0PegInsertCfg


def _resolve_data_path(value: str | Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    # Resolve relative paths from the repository root, independent of the
    # directory from which IsaacLab's launcher was invoked.
    repo_root = Path(__file__).resolve().parents[4]
    return repo_root / path


def load_first_real_joint_state(path: str | Path) -> np.ndarray:
    """Load and validate the first real trajectory arm joint state."""
    resolved = _resolve_data_path(path)
    if not resolved.is_file():
        raise FileNotFoundError(f"Real trajectory H5 not found: {resolved}")
    with h5py.File(resolved, "r") as h5:
        if "obs/state/joint_pos" not in h5:
            raise KeyError(f"Missing obs/state/joint_pos in {resolved}")
        q = np.asarray(h5["obs/state/joint_pos"][0], dtype=np.float32)
    if q.shape != (7,):
        raise ValueError(f"First real joint state must have shape (7,), got {q.shape}")
    if not np.isfinite(q).all():
        raise ValueError(f"First real joint state contains NaN/Inf: {q}")
    return q


@configclass
class RealSimPi0DirectJointResetPegInsertCfg(RealSimPi0PegInsertCfg):
    """Pi0 config for the first-frame direct-joint-reset diagnostic."""

    pi0_direct_reset_h5: str = "real_data/traj_0/data.h5"
    # Optional ordered reset sources for a single Isaac process. The first
    # reset uses pi0_direct_reset_h5; subsequent episode resets advance through
    # this list without recreating the simulator.
    pi0_direct_reset_h5_list: tuple[str, ...] = ()


class Pi0DirectJointResetEnv(Pi0RealSimEnv):
    """Pi0 environment whose reset arm pose is taken directly from H5."""

    def __init__(self, cfg, render_mode=None, **kwargs):
        configured_paths = tuple(getattr(cfg, "pi0_direct_reset_h5_list", ()) or ())
        if configured_paths:
            self._pi0_direct_reset_h5_paths = tuple(
                _resolve_data_path(path) for path in configured_paths
            )
        else:
            self._pi0_direct_reset_h5_paths = (
                _resolve_data_path(cfg.pi0_direct_reset_h5),
            )
        self._pi0_direct_reset_h5_index = 0
        self.pi0_direct_reset_h5 = self._pi0_direct_reset_h5_paths[0]
        self.pi0_direct_reset_q_real = load_first_real_joint_state(self.pi0_direct_reset_h5)
        self.pi0_direct_reset_q_sim = None
        self.pi0_direct_reset_q_error = None
        self.pi0_direct_reset_q_sim_after_reset = None
        self.pi0_direct_reset_q_error_after_reset = None
        self._preserve_direct_reset_arm = False
        super().__init__(cfg, render_mode, **kwargs)

    def close_gripper_in_place(self):
        """Close only the fingers so the real reset arm q is not overwritten."""
        if not self._preserve_direct_reset_arm:
            return super().close_gripper_in_place()
        self.ctrl_target_joint_pos[:, 7:9] = 0.0
        self._robot.set_joint_position_target(self.ctrl_target_joint_pos)
        self._robot.set_joint_effort_target(torch.zeros_like(self.joint_pos))

    def _reset_idx(self, env_ids):
        # RealSimEnv calls this method automatically at episode termination.
        # Select the next H5 before the shared reset writes the arm state, so
        # all requested initial positions run inside one Isaac process.
        if self._pi0_direct_reset_h5_index > 0:
            next_index = min(
                self._pi0_direct_reset_h5_index,
                len(self._pi0_direct_reset_h5_paths) - 1,
            )
            self.pi0_direct_reset_h5 = self._pi0_direct_reset_h5_paths[next_index]
            self.pi0_direct_reset_q_real = load_first_real_joint_state(
                self.pi0_direct_reset_h5
            )
        super()._reset_idx(env_ids)
        q_sim = self.joint_pos[env_ids, :7].detach().cpu().numpy().reshape(7)
        q_error = q_sim - self.pi0_direct_reset_q_real
        self.pi0_direct_reset_q_sim_after_reset = q_sim.tolist()
        self.pi0_direct_reset_q_error_after_reset = q_error.tolist()
        print(
            "[Pi0DirectJointReset] post-reset "
            f"q_sim={np.array2string(q_sim, precision=7)} "
            f"max_abs_error={np.max(np.abs(q_error)):.8f} rad",
            flush=True,
        )
        self._pi0_direct_reset_h5_index += 1

    def _reset_pi0_joint_state_direct(self, env_ids) -> bool:
        """Write q_real directly, clear qdot, then refresh IsaacLab state."""
        if self.num_envs != 1 or env_ids.numel() != 1:
            raise ValueError("Direct Pi0 joint reset currently supports one environment")

        self._preserve_direct_reset_arm = True
        q_real = torch.as_tensor(
            self.pi0_direct_reset_q_real, dtype=torch.float32, device=self.device
        ).view(1, 7)
        full_q = self.joint_pos.detach().clone()
        full_q[env_ids, :7] = q_real
        zero_velocity = torch.zeros_like(full_q)

        self._robot.write_joint_state_to_sim(full_q, zero_velocity)
        self.ctrl_target_joint_pos[:] = full_q
        self._robot.set_joint_position_target(full_q)
        self._robot.set_joint_effort_target(torch.zeros_like(full_q))

        # Push the write through the current scene and recompute the cached
        # articulation/body values before the shared peg/grasp reset continues.
        self.scene.write_data_to_sim()
        self.sim.forward()
        self.scene.update(dt=self.physics_dt)
        self._compute_intermediate_values(dt=self.physics_dt)

        q_sim = self.joint_pos[env_ids, :7].detach().cpu().numpy().reshape(7)
        q_error = q_sim - self.pi0_direct_reset_q_real
        self.pi0_direct_reset_q_sim = q_sim.tolist()
        self.pi0_direct_reset_q_error = q_error.tolist()
        print(
            "[Pi0DirectJointReset] "
            f"h5={self.pi0_direct_reset_h5} "
            f"q_real={np.array2string(self.pi0_direct_reset_q_real, precision=7)} "
            f"q_sim={np.array2string(q_sim, precision=7)} "
            f"max_abs_error={np.max(np.abs(q_error)):.8f} rad",
            flush=True,
        )
        return True
