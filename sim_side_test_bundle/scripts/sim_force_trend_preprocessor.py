"""Runtime sim-wrench to Route A 10-D force-trend history wrapper."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from openpi.force_features import RealForceTrendBuffer
from openpi.shared.wrench_adapter import load_adapter


class SimForceTrendPreprocessor:
    """Convert a base-frame [Fx,Fy,Fz,Tx,Ty,Tz] sample into [10,10] history."""

    def __init__(self, adapter_path: str | Path, force_config_path: str | Path):
        self.adapter = load_adapter(adapter_path, device="cpu")
        with Path(force_config_path).open(encoding="utf-8") as f:
            config = json.load(f)
        rep = config["representation"]
        expected_axis = np.array([0.0, 0.0, -1.0], dtype=np.float32)
        axis = np.asarray(rep.get("insertion_axis", expected_axis), dtype=np.float32)
        if axis.shape != (3,) or np.linalg.norm(axis) == 0 or not np.allclose(axis / np.linalg.norm(axis), expected_axis):
            raise ValueError("This runtime wrapper supports the configured -Z insertion axis only")
        self.history_frames = int(rep.get("history_frames", 10))
        self.buffer_args = dict(
            baseline_frames=20,
            smoothing_frames=int(rep.get("causal_smoothing_frames", 3)),
            trend_lag_frames=int(rep.get("trend_lag_frames", 5)),
            epsilon=float(rep.get("epsilon", 1e-6)),
            active_noise_mad_multiplier=float(rep.get("active_noise_mad_multiplier", 6.0)),
            active_noise_floor=float(rep.get("active_noise_floor_newton", 1e-6)),
        )
        self.reset()

    def reset(self) -> None:
        """Reset calibration/history at the start of a new episode."""
        self.buffer = RealForceTrendBuffer(**self.buffer_args)

    def update(self, raw_wrench: np.ndarray) -> np.ndarray:
        """Consume one raw wrench sample and return its 10-D feature."""
        raw = np.asarray(raw_wrench, dtype=np.float32).reshape(-1)
        if raw.shape != (6,) or not np.isfinite(raw).all():
            raise ValueError(f"Expected one finite wrench [Fx,Fy,Fz,Tx,Ty,Tz], got {raw.shape}")
        aligned = self.adapter.transform_numpy(raw[None, :])[0]
        return self.buffer.update(aligned[:3])

    def history(self) -> np.ndarray:
        """Return the newest causal effort window with shape [10,10]."""
        return self.buffer.history(self.history_frames)
