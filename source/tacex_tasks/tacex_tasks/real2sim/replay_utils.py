"""Pure NumPy helpers for real-trajectory Cartesian replay.

The simulator uses scalar-first quaternions (w, x, y, z).  Keeping these
helpers independent from Isaac makes the replay math testable without opening
an Isaac Sim application.
"""

from __future__ import annotations

import math

import numpy as np


EPS = 1.0e-10


def normalize_quaternion(quaternion: np.ndarray, *, name: str = "quaternion") -> np.ndarray:
    value = np.asarray(quaternion, dtype=np.float64)
    if value.shape != (4,):
        raise ValueError(f"{name} must have shape (4,), got {value.shape}")
    norm = float(np.linalg.norm(value))
    if not np.isfinite(norm) or norm < EPS:
        raise ValueError(f"{name} must be finite and non-zero")
    return value / norm


def normalize_quaternions(quaternions: np.ndarray, *, name: str = "quaternions") -> np.ndarray:
    value = np.asarray(quaternions, dtype=np.float64)
    if value.ndim != 2 or value.shape[1] != 4:
        raise ValueError(f"{name} must have shape (N,4), got {value.shape}")
    norms = np.linalg.norm(value, axis=1, keepdims=True)
    if not np.isfinite(value).all() or np.any(norms < EPS):
        raise ValueError(f"{name} must contain finite non-zero quaternions")
    return value / norms


def make_quaternion_sign_continuous(quaternions: np.ndarray) -> np.ndarray:
    result = normalize_quaternions(quaternions).copy()
    for index in range(1, len(result)):
        if float(np.dot(result[index - 1], result[index])) < 0.0:
            result[index] *= -1.0
    return result


def quaternion_conjugate(quaternion: np.ndarray) -> np.ndarray:
    value = np.asarray(quaternion, dtype=np.float64)
    return np.array([value[0], -value[1], -value[2], -value[3]], dtype=np.float64)


def quaternion_multiply(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    w1, x1, y1, z1 = np.asarray(first, dtype=np.float64)
    w2, x2, y2, z2 = np.asarray(second, dtype=np.float64)
    return np.array(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ],
        dtype=np.float64,
    )


def quaternion_rotate(quaternion: np.ndarray, vector: np.ndarray) -> np.ndarray:
    q = normalize_quaternion(quaternion)
    vq = np.array([0.0, *np.asarray(vector, dtype=np.float64)], dtype=np.float64)
    return quaternion_multiply(quaternion_multiply(q, vq), quaternion_conjugate(q))[1:]


def quaternion_slerp(first: np.ndarray, second: np.ndarray, fraction: float) -> np.ndarray:
    q0 = normalize_quaternion(first)
    q1 = normalize_quaternion(second)
    t = float(np.clip(fraction, 0.0, 1.0))
    dot = float(np.dot(q0, q1))
    if dot < 0.0:
        q1 = -q1
        dot = -dot
    if dot > 1.0 - 1.0e-8:
        return normalize_quaternion((1.0 - t) * q0 + t * q1)
    angle = math.acos(float(np.clip(dot, -1.0, 1.0)))
    sine = math.sin(angle)
    return normalize_quaternion(
        (math.sin((1.0 - t) * angle) / sine) * q0 + (math.sin(t * angle) / sine) * q1
    )


def pose_to_matrix(pose: np.ndarray) -> np.ndarray:
    value = np.asarray(pose, dtype=np.float64)
    if value.shape != (7,):
        raise ValueError(f"pose must have shape (7,), got {value.shape}")
    position = value[:3]
    quaternion = normalize_quaternion(value[3:])
    w, x, y, z = quaternion
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ]
    )
    matrix[:3, 3] = position
    return matrix


def matrix_to_pose(matrix: np.ndarray) -> np.ndarray:
    value = np.asarray(matrix, dtype=np.float64)
    if value.shape != (4, 4):
        raise ValueError(f"matrix must have shape (4,4), got {value.shape}")
    rotation = value[:3, :3]
    trace = float(np.trace(rotation))
    if trace > 0.0:
        scale = 2.0 * math.sqrt(trace + 1.0)
        quaternion = np.array(
            [0.25 * scale, (rotation[2, 1] - rotation[1, 2]) / scale,
             (rotation[0, 2] - rotation[2, 0]) / scale,
             (rotation[1, 0] - rotation[0, 1]) / scale],
            dtype=np.float64,
        )
    else:
        diagonal = np.diag(rotation)
        axis = int(np.argmax(diagonal))
        if axis == 0:
            scale = 2.0 * math.sqrt(max(1.0 + rotation[0, 0] - rotation[1, 1] - rotation[2, 2], EPS))
            quaternion = np.array(
                [(rotation[2, 1] - rotation[1, 2]) / scale, 0.25 * scale,
                 (rotation[0, 1] + rotation[1, 0]) / scale,
                 (rotation[0, 2] + rotation[2, 0]) / scale], dtype=np.float64
            )
        elif axis == 1:
            scale = 2.0 * math.sqrt(max(1.0 + rotation[1, 1] - rotation[0, 0] - rotation[2, 2], EPS))
            quaternion = np.array(
                [(rotation[0, 2] - rotation[2, 0]) / scale,
                 (rotation[0, 1] + rotation[1, 0]) / scale, 0.25 * scale,
                 (rotation[1, 2] + rotation[2, 1]) / scale], dtype=np.float64
            )
        else:
            scale = 2.0 * math.sqrt(max(1.0 + rotation[2, 2] - rotation[0, 0] - rotation[1, 1], EPS))
            quaternion = np.array(
                [(rotation[1, 0] - rotation[0, 1]) / scale,
                 (rotation[0, 2] + rotation[2, 0]) / scale,
                 (rotation[1, 2] + rotation[2, 1]) / scale, 0.25 * scale], dtype=np.float64
            )
    return np.concatenate((value[:3, 3], normalize_quaternion(quaternion)))


def compose_pose(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    return matrix_to_pose(pose_to_matrix(first) @ pose_to_matrix(second))


def inverse_pose(pose: np.ndarray) -> np.ndarray:
    return matrix_to_pose(np.linalg.inv(pose_to_matrix(pose)))


def relative_pose(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    """Return the body-frame transform taking ``first`` to ``second``."""
    return compose_pose(inverse_pose(first), second)


def interpolate_pose(first: np.ndarray, second: np.ndarray, fraction: float) -> np.ndarray:
    first_value = np.asarray(first, dtype=np.float64)
    second_value = np.asarray(second, dtype=np.float64)
    return np.concatenate(
        (
            (1.0 - fraction) * first_value[:3] + fraction * second_value[:3],
            quaternion_slerp(first_value[3:], second_value[3:], fraction),
        )
    )


def robust_pose_mean(poses: np.ndarray) -> np.ndarray:
    values = np.asarray(poses, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 7 or len(values) == 0:
        raise ValueError(f"poses must have shape (N,7) with N>0, got {values.shape}")
    quaternions = make_quaternion_sign_continuous(values[:, 3:])
    mean_quaternion = normalize_quaternion(np.mean(quaternions, axis=0))
    return np.concatenate((np.median(values[:, :3], axis=0), mean_quaternion))


def validate_replay_inputs(
    joint_pos: np.ndarray,
    gripper: np.ndarray,
    timestamps: np.ndarray,
    *,
    max_joint_step: float = 1.0,
) -> None:
    q = np.asarray(joint_pos, dtype=np.float64)
    g = np.asarray(gripper, dtype=np.float64).reshape(-1)
    t = np.asarray(timestamps, dtype=np.float64).reshape(-1)
    if q.ndim != 2 or q.shape[1] != 7 or len(q) < 2:
        raise ValueError(f"joint_pos must have shape (N,7), N>=2; got {q.shape}")
    if g.shape != (len(q),) or t.shape != (len(q),):
        raise ValueError("joint_pos, gripper and timestamps must have the same frame count")
    if not all(np.isfinite(value).all() for value in (q, g, t)):
        raise ValueError("joint_pos, gripper and timestamps must be finite")
    if np.any(np.diff(t) <= 0.0):
        raise ValueError("timestamps must be strictly increasing")
    if np.any(np.abs(np.diff(q, axis=0)) > max_joint_step):
        raise ValueError(f"joint step exceeds safety limit {max_joint_step}")
