"""The single immutable wrench path used by the TA-VLA baseline."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


WRENCH_NAMES = ("Fx", "Fy", "Fz", "Tx", "Ty", "Tz")


class FixedAffineWrenchAdapter:
    """Load and apply the frozen sim-to-real affine adapter.

    The implementation intentionally uses the JSON metadata as the numerical
    source of truth and never executes an arbitrary pickle from the ``.pt``
    file.  The checkpoint is still required to exist and is retained in the
    metadata for reproducibility.
    """

    def __init__(self, checkpoint_path: str | Path, metadata_path: str | Path | None = None):
        self.checkpoint_path = Path(checkpoint_path).expanduser()
        if not self.checkpoint_path.is_file():
            raise FileNotFoundError(f"wrench adapter checkpoint not found: {self.checkpoint_path}")
        self.metadata_path = Path(metadata_path).expanduser() if metadata_path else self.checkpoint_path.with_suffix(".json")
        if not self.metadata_path.is_file():
            raise FileNotFoundError(f"wrench adapter metadata not found: {self.metadata_path}")
        with self.metadata_path.open("r", encoding="utf-8") as handle:
            metadata = json.load(handle)
        self._validate_metadata(metadata)
        self.metadata = metadata
        self.source_center = np.asarray(metadata["source_center"], dtype=np.float32)
        self.source_scale = np.asarray(metadata["source_scale"], dtype=np.float32)
        self.target_scale = np.asarray(metadata["target_scale"], dtype=np.float32)
        self.target_center = np.asarray(metadata["target_center"], dtype=np.float32)

    @staticmethod
    def _validate_metadata(metadata: dict[str, Any]) -> None:
        if metadata.get("format") != "tavla_wrench_adapter":
            raise ValueError("unsupported wrench adapter format")
        if tuple(metadata.get("wrench_names", ())) != WRENCH_NAMES:
            raise ValueError("wrench order must be [Fx,Fy,Fz,Tx,Ty,Tz]")
        if metadata.get("config", {}).get("direction") != "sim_to_real":
            raise ValueError("wrench adapter direction must be sim_to_real")
        for name in ("source_center", "source_scale", "target_center", "target_scale"):
            values = metadata.get(name)
            if not isinstance(values, list) or len(values) != 6:
                raise ValueError(f"adapter field {name!r} must contain six values")
        if any(float(value) == 0.0 for value in metadata["source_scale"]):
            raise ValueError("source_scale contains zero")

    def apply(self, wrench_final: np.ndarray) -> np.ndarray:
        values = np.asarray(wrench_final, dtype=np.float32)
        if values.shape[-1] != 6:
            raise ValueError(f"wrench must end in six values, got {values.shape}")
        if not np.isfinite(values).all():
            raise FloatingPointError("wrench_final contains NaN or Inf")
        # Map the source robust-standardized coordinates into the target
        # physical wrench domain. The Server applies its own norm_stats after
        # this step, so this method must return real-domain values rather than
        # already-normalized values.
        result = (values - self.source_center) / self.source_scale
        result = result * self.target_scale + self.target_center
        if not np.isfinite(result).all():
            raise FloatingPointError("adapted wrench contains NaN or Inf")
        return result.astype(np.float32, copy=False)

    def apply_torch(self, wrench_final: Any) -> Any:
        """Torch equivalent used by the Isaac Sim-side subclass."""
        import torch

        if not torch.is_tensor(wrench_final):
            raise TypeError("wrench_final must be a torch.Tensor")
        if wrench_final.shape[-1] != 6:
            raise ValueError(f"wrench must end in six values, got {tuple(wrench_final.shape)}")
        source_center = torch.as_tensor(self.source_center, device=wrench_final.device, dtype=wrench_final.dtype)
        source_scale = torch.as_tensor(self.source_scale, device=wrench_final.device, dtype=wrench_final.dtype)
        target_scale = torch.as_tensor(self.target_scale, device=wrench_final.device, dtype=wrench_final.dtype)
        target_center = torch.as_tensor(self.target_center, device=wrench_final.device, dtype=wrench_final.dtype)
        if not torch.isfinite(wrench_final).all():
            raise FloatingPointError("wrench_final contains NaN or Inf")
        result = (wrench_final - source_center) / source_scale
        result = result * target_scale + target_center
        if not torch.isfinite(result).all():
            raise FloatingPointError("adapted wrench contains NaN or Inf")
        return result


class WrenchPipeline:
    """Make every sign and coordinate transition explicit and auditable."""

    def __init__(self, adapter: FixedAffineWrenchAdapter):
        self.adapter = adapter

    def convert(self, wrench_base: np.ndarray) -> dict[str, np.ndarray]:
        base = np.asarray(wrench_base, dtype=np.float32)
        if base.shape[-1] != 6:
            raise ValueError(f"wrench_base must end in six values, got {base.shape}")
        final = -base
        adapted = self.adapter.apply(final)
        return {"wrench_base": base, "wrench_final": final, "adapted_wrench": adapted}
