"""Causal three-axis force alignment shared by replay and online evaluation.

The module deliberately contains no Isaac Sim dependencies.  It consumes a
force that has already been expressed in the robot-base frame and applies the
same deployable processing on simulation and real-robot streams:

    motion baseline -> initial residual tare -> causal median -> causal LPF
    -> optional AR(1) sensor residual

Torque is outside the v1 contract.  Compatibility helpers append three zero
torque channels when a six-axis HDF5 field is required.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any

import numpy as np


FORCE_ALIGNMENT_VERSION = "causal_force_alignment_v1"
MOTION_FEATURE_NAMES = ("1", *[f"q{i}" for i in range(7)], *[f"qd{i}" for i in range(7)])


def _finite_vector(value: Any, size: int, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64).reshape(-1)
    if array.shape != (size,) or not np.isfinite(array).all():
        raise ValueError(f"{name} must be a finite ({size},) vector, got {array.shape}")
    return array


def _finite_matrix(value: Any, shape: tuple[int, int], name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != shape or not np.isfinite(array).all():
        raise ValueError(f"{name} must be a finite {shape} matrix, got {array.shape}")
    return array


def causal_joint_velocity(joint_pos: np.ndarray, timestamps: np.ndarray) -> np.ndarray:
    """Backward-difference joint velocity without using future samples."""

    q = np.asarray(joint_pos, dtype=np.float64)
    time = np.asarray(timestamps, dtype=np.float64).reshape(-1)
    if q.ndim != 2 or q.shape[1] != 7 or len(q) != len(time):
        raise ValueError("joint_pos and timestamps must have shapes (T,7) and (T,)")
    if not np.isfinite(q).all() or not np.isfinite(time).all():
        raise ValueError("joint_pos/timestamps contain NaN or Inf")
    if len(time) > 1 and np.any(np.diff(time) <= 0.0):
        raise ValueError("timestamps must be strictly increasing")
    velocity = np.zeros_like(q)
    if len(q) > 1:
        velocity[1:] = np.diff(q, axis=0) / np.diff(time)[:, None]
        velocity[0] = velocity[1]
    return velocity


def motion_features(joint_pos: np.ndarray, joint_vel: np.ndarray) -> np.ndarray:
    q = np.asarray(joint_pos, dtype=np.float64)
    qd = np.asarray(joint_vel, dtype=np.float64)
    if q.shape != qd.shape or q.ndim != 2 or q.shape[1] != 7:
        raise ValueError("joint_pos and joint_vel must both have shape (T,7)")
    if not np.isfinite(q).all() or not np.isfinite(qd).all():
        raise ValueError("joint state contains NaN or Inf")
    return np.column_stack((np.ones(len(q), dtype=np.float64), q, qd))


@dataclass(frozen=True)
class MotionBaselineModel:
    """Linear no-contact model ``force = [1,q,qd] @ weights``."""

    weights: np.ndarray = field(
        default_factory=lambda: np.zeros((len(MOTION_FEATURE_NAMES), 3), dtype=np.float64)
    )

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "weights",
            _finite_matrix(
                self.weights,
                (len(MOTION_FEATURE_NAMES), 3),
                "motion baseline weights",
            ),
        )

    def predict_one(self, joint_pos: np.ndarray, joint_vel: np.ndarray) -> np.ndarray:
        q = _finite_vector(joint_pos, 7, "joint_pos")
        qd = _finite_vector(joint_vel, 7, "joint_vel")
        return np.concatenate(([1.0], q, qd)) @ self.weights

    def predict(self, joint_pos: np.ndarray, joint_vel: np.ndarray) -> np.ndarray:
        return motion_features(joint_pos, joint_vel) @ self.weights

    def to_dict(self) -> dict[str, Any]:
        return {
            "features": list(MOTION_FEATURE_NAMES),
            "weights": self.weights.tolist(),
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any] | None) -> "MotionBaselineModel":
        if not value:
            return cls()
        features = tuple(value.get("features", ()))
        if features and features != MOTION_FEATURE_NAMES:
            raise ValueError(f"unsupported motion baseline features: {features}")
        return cls(weights=np.asarray(value.get("weights"), dtype=np.float64))


@dataclass(frozen=True)
class AR1NoiseModel:
    """Post-filter real-sensor residual model used only for model inputs."""

    phi: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float64))
    innovation_covariance: np.ndarray = field(
        default_factory=lambda: np.zeros((3, 3), dtype=np.float64)
    )

    def __post_init__(self) -> None:
        phi = _finite_vector(self.phi, 3, "AR(1) phi")
        if np.any(np.abs(phi) >= 1.0):
            raise ValueError("AR(1) phi must be strictly inside (-1, 1)")
        covariance = _finite_matrix(
            self.innovation_covariance, (3, 3), "AR(1) innovation covariance"
        )
        covariance = 0.5 * (covariance + covariance.T)
        eigenvalues = np.linalg.eigvalsh(covariance)
        if float(eigenvalues.min()) < -1.0e-10:
            raise ValueError("AR(1) innovation covariance must be positive semidefinite")
        object.__setattr__(self, "phi", phi)
        object.__setattr__(self, "innovation_covariance", covariance)

    @property
    def enabled(self) -> bool:
        return bool(np.any(self.innovation_covariance > 0.0))

    def factor(self) -> np.ndarray:
        if not self.enabled:
            return np.zeros((3, 3), dtype=np.float64)
        values, vectors = np.linalg.eigh(self.innovation_covariance)
        return vectors @ np.diag(np.sqrt(np.clip(values, 0.0, None)))

    def to_dict(self) -> dict[str, Any]:
        return {
            "phi": self.phi.tolist(),
            "innovation_covariance": self.innovation_covariance.tolist(),
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any] | None) -> "AR1NoiseModel":
        if not value:
            return cls()
        return cls(
            phi=np.asarray(value.get("phi", [0.0, 0.0, 0.0]), dtype=np.float64),
            innovation_covariance=np.asarray(
                value.get("innovation_covariance", np.zeros((3, 3))), dtype=np.float64
            ),
        )


@dataclass(frozen=True)
class ForceAlignmentConfig:
    cutoff_hz: float = 0.35
    median_window: int = 3
    tare_seconds: float = 3.0
    output_hz: float = 10.0
    sim_baseline: MotionBaselineModel = field(default_factory=MotionBaselineModel)
    real_baseline: MotionBaselineModel = field(default_factory=MotionBaselineModel)
    sim_noise: AR1NoiseModel = field(default_factory=AR1NoiseModel)
    noise_seed: int = 0

    def __post_init__(self) -> None:
        if self.cutoff_hz <= 0.0 or self.tare_seconds <= 0.0 or self.output_hz <= 0.0:
            raise ValueError("cutoff_hz, tare_seconds and output_hz must be positive")
        if self.median_window <= 0 or self.median_window % 2 == 0:
            raise ValueError("median_window must be a positive odd integer")

    def baseline_for(self, domain: str) -> MotionBaselineModel:
        if domain == "sim":
            return self.sim_baseline
        if domain == "real":
            return self.real_baseline
        raise ValueError("force domain must be 'sim' or 'real'")

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": FORCE_ALIGNMENT_VERSION,
            "contract": {
                "input": "three-axis force expressed in robot_base_O",
                "output": "force_clean and force_model XYZ; torque is zero",
                "causal": True,
                "fixed_delay_frames": 0,
                "amplitude_mapping": "identity",
            },
            "filter": {
                "cutoff_hz": float(self.cutoff_hz),
                "median_window": int(self.median_window),
                "tare_seconds": float(self.tare_seconds),
                "output_hz": float(self.output_hz),
            },
            "baseline": {
                "sim": self.sim_baseline.to_dict(),
                "real": self.real_baseline.to_dict(),
            },
            "sim_noise": self.sim_noise.to_dict(),
            "noise_seed": int(self.noise_seed),
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "ForceAlignmentConfig":
        version = value.get("version")
        if version != FORCE_ALIGNMENT_VERSION:
            raise ValueError(f"unsupported force alignment version: {version!r}")
        contract = value.get("contract", {})
        if contract.get("fixed_delay_frames", 0) != 0:
            raise ValueError("v1 force alignment does not allow an added fixed delay")
        if contract.get("amplitude_mapping", "identity") != "identity":
            raise ValueError("v1 force alignment requires identity amplitude mapping")
        filter_cfg = value.get("filter", {})
        baseline_cfg = value.get("baseline", {})
        return cls(
            cutoff_hz=float(filter_cfg.get("cutoff_hz", 0.35)),
            median_window=int(filter_cfg.get("median_window", 3)),
            tare_seconds=float(filter_cfg.get("tare_seconds", 3.0)),
            output_hz=float(filter_cfg.get("output_hz", 10.0)),
            sim_baseline=MotionBaselineModel.from_dict(baseline_cfg.get("sim")),
            real_baseline=MotionBaselineModel.from_dict(baseline_cfg.get("real")),
            sim_noise=AR1NoiseModel.from_dict(value.get("sim_noise")),
            noise_seed=int(value.get("noise_seed", 0)),
        )

    @classmethod
    def load(cls, path: Path | str) -> "ForceAlignmentConfig":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


@dataclass(frozen=True)
class ForceAlignmentSample:
    raw_base: np.ndarray
    predicted_baseline: np.ndarray
    residual_tare: np.ndarray
    baseline_corrected: np.ndarray
    median_filtered: np.ndarray
    force_clean: np.ndarray
    force_model: np.ndarray
    tare_ready: bool


class CausalForceAligner:
    """Stateful, timestamp-driven online implementation of the v1 contract."""

    def __init__(
        self,
        config: ForceAlignmentConfig,
        *,
        domain: str,
        add_model_noise: bool = False,
        seed: int | None = None,
    ) -> None:
        self.config = config
        self.domain = domain
        self.baseline_model = config.baseline_for(domain)
        self.add_model_noise = bool(add_model_noise and domain == "sim" and config.sim_noise.enabled)
        self.rng = np.random.default_rng(config.noise_seed if seed is None else seed)
        self._noise_factor = config.sim_noise.factor()
        self.reset()

    def reset(self) -> None:
        self._start_time: float | None = None
        self._previous_time: float | None = None
        self._tare_samples: list[np.ndarray] = []
        self._tare = np.zeros(3, dtype=np.float64)
        self._median_history: deque[np.ndarray] = deque(maxlen=self.config.median_window)
        self._lowpass = np.zeros(3, dtype=np.float64)
        self._lowpass_initialized = False
        self._noise_state = np.zeros(3, dtype=np.float64)

    def step(
        self,
        timestamp: float,
        raw_force_base: np.ndarray,
        joint_pos: np.ndarray,
        joint_vel: np.ndarray,
    ) -> ForceAlignmentSample:
        time = float(timestamp)
        if not np.isfinite(time):
            raise ValueError("force timestamp must be finite")
        if self._previous_time is not None and time <= self._previous_time:
            raise ValueError("force timestamps must be strictly increasing")
        if self._start_time is None:
            self._start_time = time
        raw = _finite_vector(raw_force_base, 3, "raw_force_base")
        predicted = self.baseline_model.predict_one(joint_pos, joint_vel)
        residual = raw - predicted
        elapsed = time - self._start_time
        if elapsed <= self.config.tare_seconds + 1.0e-12:
            self._tare_samples.append(residual.copy())
            self._tare = np.median(np.asarray(self._tare_samples), axis=0)
        corrected = residual - self._tare
        self._median_history.append(corrected)
        median_filtered = np.median(np.asarray(self._median_history), axis=0)

        if self._previous_time is None:
            dt = 1.0 / self.config.output_hz
        else:
            dt = time - self._previous_time
        alpha = 1.0 - np.exp(-2.0 * np.pi * self.config.cutoff_hz * dt)
        if not self._lowpass_initialized:
            self._lowpass = median_filtered.copy()
            self._lowpass_initialized = True
        else:
            self._lowpass = alpha * median_filtered + (1.0 - alpha) * self._lowpass
        clean = self._lowpass.copy()

        if self.add_model_noise:
            innovation = self.rng.standard_normal(3) @ self._noise_factor.T
            self._noise_state = self.config.sim_noise.phi * self._noise_state + innovation
            model = clean + self._noise_state
        else:
            model = clean.copy()
        self._previous_time = time
        return ForceAlignmentSample(
            raw_base=raw.copy(),
            predicted_baseline=predicted.copy(),
            residual_tare=self._tare.copy(),
            baseline_corrected=corrected.copy(),
            median_filtered=median_filtered.copy(),
            force_clean=clean,
            force_model=model,
            tare_ready=elapsed >= self.config.tare_seconds,
        )


def process_force_series(
    timestamps: np.ndarray,
    raw_force_base: np.ndarray,
    joint_pos: np.ndarray,
    *,
    config: ForceAlignmentConfig,
    domain: str,
    joint_vel: np.ndarray | None = None,
    add_model_noise: bool = False,
    seed: int | None = None,
) -> dict[str, np.ndarray]:
    """Process one complete series by repeatedly invoking the online state machine."""

    time = np.asarray(timestamps, dtype=np.float64).reshape(-1)
    raw = np.asarray(raw_force_base, dtype=np.float64)
    q = np.asarray(joint_pos, dtype=np.float64)
    if raw.shape != (len(time), 3) or q.shape != (len(time), 7):
        raise ValueError("expected timestamps (T,), raw_force_base (T,3), joint_pos (T,7)")
    qd = causal_joint_velocity(q, time) if joint_vel is None else np.asarray(joint_vel, dtype=np.float64)
    if qd.shape != q.shape:
        raise ValueError("joint_vel must have shape (T,7)")
    aligner = CausalForceAligner(
        config,
        domain=domain,
        add_model_noise=add_model_noise,
        seed=seed,
    )
    rows = [aligner.step(t, f, qp, qv) for t, f, qp, qv in zip(time, raw, q, qd)]
    names = (
        "raw_base",
        "predicted_baseline",
        "residual_tare",
        "baseline_corrected",
        "median_filtered",
        "force_clean",
        "force_model",
    )
    result = {
        name: np.asarray([getattr(row, name) for row in rows], dtype=np.float64)
        for name in names
    }
    result["tare_ready"] = np.asarray([row.tare_ready for row in rows], dtype=np.bool_)
    return result


def interval_mean_to_frames(
    sample_timestamps: np.ndarray,
    values: np.ndarray,
    frame_timestamps: np.ndarray,
) -> np.ndarray:
    """Causally average high-rate samples over each output-frame interval."""

    sample_time = np.asarray(sample_timestamps, dtype=np.float64).reshape(-1)
    frame_time = np.asarray(frame_timestamps, dtype=np.float64).reshape(-1)
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 2 or len(array) != len(sample_time):
        raise ValueError("high-rate timestamps and values are not aligned")
    if not len(frame_time):
        raise ValueError("frame_timestamps must not be empty")
    if np.any(np.diff(sample_time) <= 0.0) or np.any(np.diff(frame_time) <= 0.0):
        raise ValueError("sample and frame timestamps must be strictly increasing")
    tolerance = 2.0 * float(np.median(np.diff(sample_time))) if len(sample_time) > 1 else 1.0e-6
    if frame_time[0] < sample_time[0] - tolerance or frame_time[-1] > sample_time[-1] + tolerance:
        raise ValueError("high-rate stream does not cover all output-frame timestamps")
    rows = []
    for index, end in enumerate(frame_time):
        if index == 0:
            nearest = int(np.argmin(np.abs(sample_time - end)))
            rows.append(array[nearest])
            continue
        start = frame_time[index - 1]
        mask = (sample_time > start + 1.0e-12) & (sample_time <= end + 1.0e-12)
        if np.any(mask):
            rows.append(np.mean(array[mask], axis=0))
        else:
            nearest = int(np.argmin(np.abs(sample_time - end)))
            rows.append(array[nearest])
    return np.asarray(rows, dtype=np.float64)


def force_to_zero_torque_wrench(force_xyz: np.ndarray) -> np.ndarray:
    force = np.asarray(force_xyz)
    if force.ndim != 2 or force.shape[1] != 3 or not np.isfinite(force).all():
        raise ValueError("force_xyz must be finite with shape (T,3)")
    return np.column_stack((force, np.zeros_like(force)))


def fit_ridge_motion_baseline(
    joint_pos: np.ndarray,
    joint_vel: np.ndarray,
    force_xyz: np.ndarray,
    *,
    ridge: float = 1.0e-3,
) -> MotionBaselineModel:
    if ridge < 0.0:
        raise ValueError("ridge must be non-negative")
    features = motion_features(joint_pos, joint_vel)
    force = np.asarray(force_xyz, dtype=np.float64)
    if force.shape != (len(features), 3) or not np.isfinite(force).all():
        raise ValueError("force_xyz must be finite with shape (T,3)")
    regularizer = np.eye(features.shape[1], dtype=np.float64) * float(ridge)
    regularizer[0, 0] = 0.0
    weights = np.linalg.solve(
        features.T @ features + regularizer,
        features.T @ force,
    )
    return MotionBaselineModel(weights=weights)


def fit_ar1_noise(residual_blocks: list[np.ndarray]) -> AR1NoiseModel:
    """Fit a diagonal AR state and full innovation covariance by trajectory."""

    valid = [np.asarray(block, dtype=np.float64) for block in residual_blocks if len(block) >= 3]
    if not valid or any(block.ndim != 2 or block.shape[1] != 3 for block in valid):
        raise ValueError("AR(1) fitting requires at least one finite Nx3 residual block")
    if not all(np.isfinite(block).all() for block in valid):
        raise ValueError("AR(1) residual blocks contain NaN or Inf")
    previous = np.concatenate([block[:-1] for block in valid])
    current = np.concatenate([block[1:] for block in valid])
    denominator = np.sum(previous * previous, axis=0)
    phi = np.divide(
        np.sum(previous * current, axis=0),
        denominator,
        out=np.zeros(3, dtype=np.float64),
        where=denominator > 1.0e-12,
    )
    phi = np.clip(phi, -0.98, 0.98)
    innovations = current - previous * phi
    covariance = np.cov(innovations, rowvar=False, bias=False)
    covariance = np.atleast_2d(covariance)
    covariance += np.eye(3, dtype=np.float64) * 1.0e-12
    return AR1NoiseModel(phi=phi, innovation_covariance=covariance)
