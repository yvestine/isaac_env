"""Replay real robot actions through the current Pi0 IsaacLab controller.

This is a policy-free Real -> Sim baseline.  It reads the recorded action
stream, sends each 8-D action through ``Pi0RealSimEnv._set_pi0_target`` and
``Pi0RealSimEnv._apply_action``, and compares the resulting robot trajectory
with the real trajectory.  The Pi0 network is never contacted.

The current dataset stores ``action/actual/arm`` as an absolute 7-joint
target.  The script still checks this relation at runtime and records the
result in ``action_semantics_report.txt`` instead of silently assuming a
Cartesian action convention.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np

from isaaclab.app import AppLauncher


REPO_ROOT = Path(__file__).resolve().parents[1]


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(
    "--data-root",
    type=Path,
    default=Path("real_data"),
    help="Directory containing traj_*/data.h5 (default: real_data).",
)
parser.add_argument(
    "--task",
    type=str,
    default="TacEx-RealSim-PegInsert-PI0-Direct-v0",
    help="IsaacLab task whose Pi0 scene/controller configuration is replayed.",
)
parser.add_argument(
    "--trajectories",
    type=int,
    nargs="+",
    default=None,
    help="Trajectory indices to replay, for example: --trajectories 0 1 2 3 4.",
)
parser.add_argument(
    "--max-trajectories",
    type=int,
    default=5,
    help="Number of sorted trajectories when --trajectories is omitted (default: 5).",
)
parser.add_argument(
    "--start-frame",
    type=int,
    default=0,
    help="First frame in every selected trajectory (default: 0).",
)
parser.add_argument(
    "--max-frames",
    type=int,
    default=None,
    help="Replay at most this many frames from start-frame; omit for the full trajectory.",
)
parser.add_argument(
    "--action-source",
    choices=("actual", "policy"),
    default="actual",
    help="Use action/actual/* or action/policy/* (default: actual).",
)
parser.add_argument(
    "--action-offset",
    type=int,
    choices=(0, 1),
    default=1,
    help=(
        "Action row used for transition t->t+1: 1 means action[t+1] and is "
        "the current Pi0/GT-next convention; 0 means action[t]."
    ),
)
parser.add_argument(
    "--timing",
    choices=("timestamps", "fixed"),
    default="timestamps",
    help=(
        "Hold each action for the measured H5 interval, or for the current "
        "IsaacLab control period (default: timestamps)."
    ),
)
parser.add_argument(
    "--enable-hole-collision",
    action="store_true",
    help="Keep fixed peg-hole colliders enabled. The default is free-space replay.",
)
parser.add_argument(
    "--output-dir",
    type=Path,
    default=Path("outputs/ground_truth_action_replay"),
    help="Output directory (default: outputs/ground_truth_action_replay).",
)
parser.add_argument(
    "--pose-alignment",
    choices=("fk", "raw"),
    default="fk",
    help=(
        "Compare EEF poses in the Pi0 simulator frame using each trajectory's "
        "existing ee_pose_sim_isaac6.csv FK calibration (default: fk); raw "
        "keeps the recorded H5 frame for debugging only."
    ),
)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

if args.max_trajectories <= 0:
    raise ValueError("--max-trajectories must be positive")
if args.start_frame < 0:
    raise ValueError("--start-frame must be non-negative")
if args.max_frames is not None and args.max_frames < 2:
    raise ValueError("--max-frames must be at least 2")

# Some launch shells export an empty CUDA_VISIBLE_DEVICES.  Isaac Sim can
# still use the host GPU in that case, so treat an empty value as unset.
if os.environ.get("CUDA_VISIBLE_DEVICES", "").strip() == "":
    os.environ.pop("CUDA_VISIBLE_DEVICES", None)

# Keep the same Isaac Sim 6 workaround used by the existing replay scripts.
_kit_args = (
    "--/log/level=warning --/log/outputStreamLevel=warning "
    "--/rtx-transient/resourcemanager/enableGeometryStreaming=false"
)
args.kit_args = f"{getattr(args, 'kit_args', '')} {_kit_args}".strip()

simulation_app = AppLauncher(args).app

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import tacex_tasks  # noqa: E402,F401
import torch  # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402
from pxr import Sdf, Usd, UsdPhysics  # noqa: E402

import tacex_tasks.real2sim.pi0_env as pi0_env_module  # noqa: E402
from tacex_tasks.real2sim.pi0_env import Pi0RealSimEnv  # noqa: E402


@dataclass
class RealTrajectory:
    index: int
    path: Path
    fields: list[str]
    timestamps: np.ndarray
    joint_pos: np.ndarray
    ee_pose: np.ndarray
    gripper_width_m: np.ndarray
    action_arm: np.ndarray
    action_gripper: np.ndarray
    gripper_source: str


def resolve_repo_path(path: Path) -> Path:
    return path if path.is_absolute() else REPO_ROOT / path


def _read_dataset(h5: h5py.File, name: str, *, required: bool = True) -> np.ndarray | None:
    if name not in h5:
        if required:
            raise KeyError(f"Missing dataset {name}")
        return None
    return np.asarray(h5[name][:])


def _flatten_column(value: np.ndarray, name: str, length: int) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if result.size != length:
        raise ValueError(f"{name} must contain {length} values, got {result.shape}")
    return result.reshape(length)


def load_trajectory(path: Path, index: int) -> RealTrajectory:
    with h5py.File(path, "r") as h5:
        fields: list[str] = []

        def collect_fields(name, obj):
            if isinstance(obj, h5py.Dataset):
                fields.append(name)

        h5.visititems(collect_fields)
        timestamps = np.asarray(_read_dataset(h5, "timestamps"), dtype=np.float64).reshape(-1)
        joint_pos = np.asarray(_read_dataset(h5, "obs/state/joint_pos"), dtype=np.float64)
        ee_pose = np.asarray(_read_dataset(h5, "obs/state/ee_pose"), dtype=np.float64)

        actual_arm = np.asarray(
            _read_dataset(h5, "action/actual/arm"), dtype=np.float64
        )
        if args.action_source == "policy":
            action_arm = np.asarray(
                _read_dataset(h5, "action/policy/arm"), dtype=np.float64
            )
            action_gripper_raw = _read_dataset(
                h5, "action/policy/gripper", required=False
            )
            gripper_source = "action/policy/gripper"
        else:
            action_arm = actual_arm
            action_gripper_raw = _read_dataset(
                h5, "action/actual/gripper", required=False
            )
            gripper_source = "action/actual/gripper"

        length = len(timestamps)
        if joint_pos.shape != (length, 7):
            raise ValueError(f"{path}: joint_pos must have shape ({length},7), got {joint_pos.shape}")
        if ee_pose.shape != (length, 7):
            raise ValueError(f"{path}: ee_pose must have shape ({length},7), got {ee_pose.shape}")
        if action_arm.shape != (length, 7):
            raise ValueError(f"{path}: action arm must have shape ({length},7), got {action_arm.shape}")

        width_raw = _read_dataset(h5, "obs/state/gripper_width_m", required=False)
        if width_raw is not None:
            gripper_width_m = _flatten_column(width_raw, "gripper_width_m", length)
            width_source = "obs/state/gripper_width_m"
        else:
            gripper_pos_raw = _read_dataset(h5, "obs/state/gripper_pos", required=False)
            if gripper_pos_raw is None:
                gripper_width_m = np.full(length, np.nan, dtype=np.float64)
                width_source = "unavailable"
            else:
                gripper_pos = _flatten_column(gripper_pos_raw, "gripper_pos", length)
                gripper_width_m = 0.08 * gripper_pos
                width_source = "0.08 * obs/state/gripper_pos"

        if action_gripper_raw is None:
            action_gripper = gripper_width_m.copy()
            gripper_source = f"{width_source} (action field missing)"
        else:
            action_gripper = _flatten_column(
                action_gripper_raw, gripper_source, length
            )

    if length < 2:
        raise ValueError(f"{path}: trajectory must have at least two frames")
    arrays = (timestamps, joint_pos, ee_pose, gripper_width_m, action_arm, action_gripper)
    if not all(np.isfinite(value).all() for value in arrays if value is not None):
        raise ValueError(f"{path}: trajectory contains NaN or Inf")
    if np.any(np.diff(timestamps) <= 0.0):
        raise ValueError(f"{path}: timestamps must be strictly increasing")
    if timestamps[0] != 0.0:
        timestamps = timestamps - timestamps[0]

    start = args.start_frame
    stop = len(timestamps) if args.max_frames is None else min(
        len(timestamps), start + args.max_frames
    )
    if stop - start < 2:
        raise ValueError(f"{path}: selected frame range contains fewer than two frames")

    return RealTrajectory(
        index=index,
        path=path,
        fields=sorted(fields),
        timestamps=timestamps[start:stop].copy(),
        joint_pos=joint_pos[start:stop].copy(),
        ee_pose=ee_pose[start:stop].copy(),
        gripper_width_m=gripper_width_m[start:stop].copy(),
        action_arm=action_arm[start:stop].copy(),
        action_gripper=action_gripper[start:stop].copy(),
        gripper_source=gripper_source,
    )


def _mae(first: np.ndarray, second: np.ndarray) -> float:
    return float(np.mean(np.abs(np.asarray(first) - np.asarray(second))))


def _format_vector(value: np.ndarray, precision: int = 6) -> str:
    return np.array2string(np.asarray(value), precision=precision, separator=", ")


def fit_rigid_transform(source: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """Fit ``target = rotation @ source + translation`` without scaling."""
    source_center = np.asarray(source, dtype=np.float64).mean(axis=0)
    target_center = np.asarray(target, dtype=np.float64).mean(axis=0)
    source_zero = np.asarray(source, dtype=np.float64) - source_center
    target_zero = np.asarray(target, dtype=np.float64) - target_center
    u, _, vt = np.linalg.svd(source_zero.T @ target_zero)
    rotation = vt.T @ u.T
    if np.linalg.det(rotation) < 0.0:
        vt[-1, :] *= -1.0
        rotation = vt.T @ u.T
    translation = target_center - rotation @ source_center
    aligned = np.asarray(source, dtype=np.float64) @ rotation.T + translation
    rmse = float(np.sqrt(np.mean(np.sum(np.square(aligned - target), axis=1))))
    return rotation, translation, rmse


def rotation_matrix_to_quaternion_xyzw(rotation: np.ndarray) -> np.ndarray:
    """Convert a proper 3x3 rotation matrix to an XYZW quaternion."""
    matrix = np.asarray(rotation, dtype=np.float64)
    trace = float(np.trace(matrix))
    if trace > 0.0:
        scale = 2.0 * np.sqrt(trace + 1.0)
        w = 0.25 * scale
        x = (matrix[2, 1] - matrix[1, 2]) / scale
        y = (matrix[0, 2] - matrix[2, 0]) / scale
        z = (matrix[1, 0] - matrix[0, 1]) / scale
    elif matrix[0, 0] > matrix[1, 1] and matrix[0, 0] > matrix[2, 2]:
        scale = 2.0 * np.sqrt(max(1.0 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2], 1.0e-12))
        w = (matrix[2, 1] - matrix[1, 2]) / scale
        x = 0.25 * scale
        y = (matrix[0, 1] + matrix[1, 0]) / scale
        z = (matrix[0, 2] + matrix[2, 0]) / scale
    elif matrix[1, 1] > matrix[2, 2]:
        scale = 2.0 * np.sqrt(max(1.0 + matrix[1, 1] - matrix[0, 0] - matrix[2, 2], 1.0e-12))
        w = (matrix[0, 2] - matrix[2, 0]) / scale
        x = (matrix[0, 1] + matrix[1, 0]) / scale
        y = 0.25 * scale
        z = (matrix[1, 2] + matrix[2, 1]) / scale
    else:
        scale = 2.0 * np.sqrt(max(1.0 + matrix[2, 2] - matrix[0, 0] - matrix[1, 1], 1.0e-12))
        w = (matrix[1, 0] - matrix[0, 1]) / scale
        x = (matrix[0, 2] + matrix[2, 0]) / scale
        y = (matrix[1, 2] + matrix[2, 1]) / scale
        z = 0.25 * scale
    quaternion = np.asarray((x, y, z, w), dtype=np.float64)
    return quaternion / np.linalg.norm(quaternion).clip(min=1.0e-12)


def quaternion_multiply_xyzw(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    """Multiply XYZW quaternions with broadcasting over the leading axis."""
    first = np.asarray(first, dtype=np.float64)
    second = np.asarray(second, dtype=np.float64)
    x1, y1, z1, w1 = np.moveaxis(first, -1, 0)
    x2, y2, z2, w2 = np.moveaxis(second, -1, 0)
    return np.stack(
        (
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        ),
        axis=-1,
    )


def align_real_pose(trajectory: RealTrajectory) -> tuple[np.ndarray, dict[str, object]]:
    """Put the recorded pose into the current Pi0 IsaacLab EEF frame.

    The H5 pose is in the real-robot frame and uses XYZW.  Existing
    ``ee_pose_sim_isaac6.csv`` files contain the same trajectory's legacy
    direct-FK pose.  Their position-only rigid calibration is reused here;
    the current Pi0 scene differs from that legacy scene by the documented
    180-degree X root rotation.  No current replay result is used to fit this
    transform, so it cannot hide controller tracking error.
    """
    pose = np.asarray(trajectory.ee_pose, dtype=np.float64).copy()
    if args.pose_alignment == "raw":
        return pose, {"mode": "raw", "quaternion_order": "xyzw"}

    fk_path = trajectory.path.parent / "ee_pose_sim_isaac6.csv"
    if not fk_path.is_file():
        raise FileNotFoundError(
            f"Pose alignment requires the existing FK file {fk_path}; "
            "use --pose-alignment raw only for frame-debugging."
        )
    legacy_fk_full = np.loadtxt(fk_path, delimiter=",", skiprows=1, ndmin=2)
    if legacy_fk_full.ndim != 2 or legacy_fk_full.shape[1] != 7 or not np.isfinite(legacy_fk_full).all():
        raise ValueError(f"Expected finite FK pose shape (N, 7), got {legacy_fk_full.shape}: {fk_path}")
    with h5py.File(trajectory.path, "r") as h5:
        real_pose_full = np.asarray(_read_dataset(h5, "obs/state/ee_pose"), dtype=np.float64)
    if real_pose_full.shape != legacy_fk_full.shape or not np.isfinite(real_pose_full).all():
        raise ValueError(
            f"Real H5 pose shape {real_pose_full.shape} does not match FK shape "
            f"{legacy_fk_full.shape}: {trajectory.path} / {fk_path}"
        )
    fk_start = args.start_frame
    fk_stop = fk_start + len(pose)
    if fk_stop > len(legacy_fk_full):
        raise ValueError(
            f"Selected FK frame range [{fk_start}:{fk_stop}] exceeds {len(legacy_fk_full)} rows: {fk_path}"
        )
    legacy_fk = legacy_fk_full[fk_start:fk_stop]

    real_to_legacy_rotation, real_to_legacy_translation, fit_rmse = fit_rigid_transform(
        real_pose_full[:, :3], legacy_fk_full[:, :3]
    )
    legacy_to_pi0_root_rotation = np.diag((1.0, -1.0, -1.0))
    real_to_pi0_rotation = legacy_to_pi0_root_rotation @ real_to_legacy_rotation
    real_to_pi0_translation = legacy_to_pi0_root_rotation @ real_to_legacy_translation
    pose[:, :3] = pose[:, :3] @ real_to_pi0_rotation.T + real_to_pi0_translation

    frame_quaternion = rotation_matrix_to_quaternion_xyzw(real_to_pi0_rotation)
    pose[:, 3:] = quaternion_multiply_xyzw(frame_quaternion, pose[:, 3:])
    pose[:, 3:] /= np.linalg.norm(pose[:, 3:], axis=1, keepdims=True).clip(min=1.0e-12)
    return pose, {
        "mode": "existing_trajectory_fk",
        "quaternion_order": "xyzw",
        "fk_path": str(fk_path.resolve()),
        "real_to_legacy_rotation": real_to_legacy_rotation.tolist(),
        "real_to_legacy_translation_m": real_to_legacy_translation.tolist(),
        "legacy_to_pi0_root_rotation": legacy_to_pi0_root_rotation.tolist(),
        "real_to_pi0_rotation": real_to_pi0_rotation.tolist(),
        "real_to_pi0_translation_m": real_to_pi0_translation.tolist(),
        "frame_quaternion_xyzw": frame_quaternion.tolist(),
        "fk_position_fit_rmse_m": fit_rmse,
    }


def write_action_semantics_report(
    path: Path,
    trajectories: list[RealTrajectory],
    cfg,
    *,
    physics_dt: float,
    action_source: str,
    timing_mode: str,
    action_offset: int,
) -> None:
    lines = [
        "Real -> Sim Ground-Truth Action Replay / System Identification",
        "==================================================================",
        "",
        "Purpose: replay recorded real actions through the current Pi0 IsaacLab",
        "action adapter and controller, with Pi0 inference disabled.",
        "This is a baseline measurement; no Kp/Kd, action scale, or timing tuning",
        "is performed by this script.",
        "",
        "Dataset/action semantics",
        "------------------------",
        f"action source used: action/{action_source}/*",
        "action dimension: 8 = 7 arm joints + 1 gripper field",
        "arm action representation: absolute joint-position target, not Cartesian delta",
        "arm action units: radians; coordinates: the robot's 7 joint coordinates",
        "EEF rotation representation in obs/state/ee_pose: XYZW quaternion (x,y,z,w)",
        "EEF position units in obs/state/ee_pose: metres",
        "EEF pose is a recorded reference only; it is not fed to the controller",
        "gripper: raw recorded action field; current Pi0 config holds physical gripper joints",
        "",
        "Timing/control path",
        "------------------",
        f"replay action offset: {action_offset} (transition t->t+1 uses action[t+{action_offset}]),",
        "the default offset=1 matches the existing Pi0 GT-next diagnostic.",
        "The absolute action target at each state timestamp is not treated as a Cartesian delta.",
        f"replay timing mode: {timing_mode}",
        f"IsaacLab physics dt: {physics_dt:.9f} s ({1.0 / physics_dt:.3f} Hz)",
        f"IsaacLab control dt: {float(cfg.decimation) * physics_dt:.9f} s",
        f"IsaacLab control Hz: {1.0 / (float(cfg.decimation) * physics_dt):.3f}",
        f"decimation: {int(cfg.decimation)}",
        "controller route: Pi0RealSimEnv._set_pi0_target -> _apply_action",
        f"use_taskspace_controller: {bool(getattr(cfg.pi0_policy_cfg, 'use_taskspace_controller', True))}",
        f"use_implicit_position_controller: {bool(getattr(cfg.pi0_policy_cfg, 'use_implicit_position_controller', True))}",
        f"hold_gripper: {bool(getattr(cfg.pi0_policy_cfg, 'hold_gripper', True))}",
        "Pi0 inference/network: disabled; the environment receives an offline null policy",
        f"peg-hole collision: {'enabled' if args.enable_hole_collision else 'disabled (free-space baseline)'}",
        "",
        "Runtime checks per trajectory",
        "------------------------------",
    ]
    for trajectory in trajectories:
        same_frame_mae = _mae(trajectory.action_arm, trajectory.joint_pos)
        next_frame_mae = _mae(trajectory.action_arm[:-1], trajectory.joint_pos[1:])
        action_increment_mae = _mae(
            np.diff(trajectory.action_arm, axis=0),
            np.diff(trajectory.joint_pos, axis=0),
        )
        inferred = "absolute target at the same timestamp" if same_frame_mae <= max(
            1.0e-5, 0.02 * max(next_frame_mae, 1.0e-8)
        ) else "not conclusively absolute-from-state; inspect the values below"
        dt = np.diff(trajectory.timestamps)
        lines.extend(
            [
                f"{trajectory.path.name} ({trajectory.path.parent.name})",
                f"  selected frames: {len(trajectory.timestamps)}",
                f"  action arm shape: {trajectory.action_arm.shape}",
                f"  action vs state[t] MAE: {same_frame_mae:.9g} rad",
                f"  action[t] vs state[t+1] MAE: {next_frame_mae:.9g} rad",
                f"  action increment vs joint increment MAE: {action_increment_mae:.9g} rad",
                f"  inference: {inferred}",
                f"  timestamp median dt: {float(np.median(dt)):.9f} s ({1.0 / np.median(dt):.3f} Hz)",
                f"  timestamp p01/p99 dt: {float(np.percentile(dt, 1)):.9f} / {float(np.percentile(dt, 99)):.9f} s",
                f"  gripper source: {trajectory.gripper_source}",
                f"  available H5 fields: {', '.join(trajectory.fields)}",
            ]
        )
    lines.extend(
        [
            "",
            "Interpretation for the current data",
            "------------------------------------",
            "The action stream is checked against the recorded joint state instead",
            "of being treated as a pose/Euler/axis-angle action. Thus the baseline",
            "tests the real absolute-joint command sequence against the same Pi0",
            "rate limiter plus implicit joint-position controller used in deployment.",
            "",
            "EEF pose comparison",
            "-------------------",
            f"comparison mode: {args.pose_alignment}",
            "H5 ee_pose is XYZW in the real-robot frame; sim pose is XYZW in the Pi0 IsaacLab frame.",
            "The default existing_trajectory_fk mode reuses the pre-existing FK file for each trajectory",
            "to remove only the fixed frame/tool transform; it does not fit against the current replay.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


class _OfflineNullPi0Policy:
    """Satisfy Pi0RealSimEnv's lifecycle without opening a network client."""

    def __init__(self, _cfg):
        pass

    def reset(self):
        return None

    def predict_action_chunk(self, _observation):
        return np.empty((0, 8), dtype=np.float32)

    def close(self):
        return None


class GroundTruthReplayEnv(Pi0RealSimEnv):
    """Pi0 scene/controller with the remote policy replaced by a null policy."""

    def __init__(self, cfg, **kwargs):
        # DirectRLEnv may invoke task hooks while the parent is still building
        # the scene.  Initialize the replay lifecycle first so the base task
        # cannot mistake construction for a completed data-collection run.
        self.collect_data = False
        self.immediate_stop = False
        self.save_failed_trajectory = False
        self.num_trajectories = 1_000_000
        self.cur_num_traj = 0
        # This baseline compares state trajectories only; camera sensors are
        # unnecessary and can block CPU-only Isaac Sim startup.
        cfg.tiled_camera = None
        cfg.wrist_camera = None
        original_policy_class = pi0_env_module.PI0RemoteJointPolicy
        pi0_env_module.PI0RemoteJointPolicy = _OfflineNullPi0Policy
        try:
            super().__init__(cfg, **kwargs)
        finally:
            pi0_env_module.PI0RemoteJointPolicy = original_policy_class

    def _setup_scene(self) -> None:
        # Pi0RealSimEnv enables cameras for policy inference before calling
        # this hook.  Ground-truth replay is state-only, so disable them at
        # the last point before RealSimEnv can create sensor instances.
        self.cfg.enable_cameras = False
        self.cfg.tiled_camera = None
        self.cfg.wrist_camera = None
        super()._setup_scene()
        # The referenced visual background can contain its own PhysicsScene.
        # IsaacLab already owns /physicsScene; remove the nested copy before
        # the scene is initialized by PhysX.
        nested_physics_scenes = [
            prim.GetPath()
            for prim in self.sim.stage.Traverse()
            if prim.IsA(UsdPhysics.Scene) and str(prim.GetPath()).startswith("/World/envs/")
        ]
        for scene_path in nested_physics_scenes:
            self.sim.stage.RemovePrim(scene_path)
            print(f"[GTReplay] removed nested PhysicsScene: {scene_path}", flush=True)


def set_asset_collision_enabled(env: GroundTruthReplayEnv, asset, enabled: bool) -> int:
    """Toggle every USD collider below one spawned asset root."""
    prim_paths = list(asset.root_physx_view.prim_paths)
    if len(prim_paths) != 1:
        raise RuntimeError(f"Expected one asset root, got {prim_paths}")
    root_path = Sdf.Path(prim_paths[0])
    changed = 0
    while root_path != Sdf.Path.absoluteRootPath:
        root_prim = env.sim.stage.GetPrimAtPath(root_path)
        colliders = []
        if root_prim.IsValid():
            for prim in Usd.PrimRange(root_prim):
                collision_api = UsdPhysics.CollisionAPI.Get(env.sim.stage, prim.GetPath())
                if collision_api:
                    colliders.append(collision_api)
        if colliders:
            for collision_api in colliders:
                collision_api.GetCollisionEnabledAttr().Set(bool(enabled))
            changed = len(colliders)
            break
        root_path = root_path.GetParentPath()
    if changed == 0:
        raise RuntimeError(f"No colliders found below {prim_paths[0]}")
    env.sim.forward()
    return changed


def configure_replay_cfg(task: str, first_q: np.ndarray):
    use_fabric = not str(args.device).lower().startswith("cpu")
    cfg = parse_env_cfg(task, device=args.device, num_envs=1, use_fabric=use_fabric)
    cfg.scene.num_envs = 1
    cfg.policy_cfg = None
    if hasattr(cfg, "teacher_policy_cfg"):
        cfg.teacher_policy_cfg = None
    cfg.enable_cameras = False
    cfg.teacher_eval_only = True
    cfg.episode_length_s = 3600.0
    if hasattr(cfg, "data_collect_cfg"):
        cfg.data_collect_cfg["collect_data"] = False
        cfg.data_collect_cfg["immediate_stop"] = False
        cfg.data_collect_cfg["save_failed_trajectory"] = False
        cfg.data_collect_cfg["num_trajectories"] = 1_000_000
    if hasattr(cfg, "ctrl") and hasattr(cfg.ctrl, "reset_joints"):
        cfg.ctrl.reset_joints = first_q.tolist()
    if hasattr(cfg, "task"):
        for name in (
            "fixed_asset_init_pos_noise",
            "hand_init_pos_noise",
            "hand_init_orn_noise",
            "held_asset_pos_noise",
        ):
            if hasattr(cfg.task, name):
                value = getattr(cfg.task, name)
                if isinstance(value, (list, tuple)):
                    setattr(cfg.task, name, [0.0] * len(value))
        for name in ("fixed_asset_init_orn_range_deg", "fixed_asset_init_orn_deg"):
            if hasattr(cfg.task, name):
                setattr(cfg.task, name, 0.0)
        if hasattr(cfg.task, "held_asset") and hasattr(cfg.task.held_asset, "spawn"):
            cfg.task.held_asset.spawn.visible = True
    return cfg


def set_initial_joint_state(
    env: GroundTruthReplayEnv,
    q_real: np.ndarray,
    gripper_width_m: float,
) -> None:
    """Write the corresponding initial state once, before action replay."""
    full_q = env.joint_pos.detach().clone()
    full_q[0, :7] = torch.as_tensor(q_real, dtype=torch.float32, device=env.device)
    if np.isfinite(gripper_width_m):
        lower, upper = env._joint_limits()
        finger = torch.tensor(
            float(gripper_width_m) / 2.0, dtype=torch.float32, device=env.device
        )
        full_q[0, 7:9] = torch.clamp(
            finger.expand(2), lower[0, 7:9], upper[0, 7:9]
        )
    zero_velocity = torch.zeros_like(full_q)
    env._robot.write_joint_state_to_sim(full_q, zero_velocity)
    env.ctrl_target_joint_pos[:] = full_q
    env._robot.set_joint_position_target(full_q)
    env._robot.set_joint_effort_target(torch.zeros_like(full_q))
    env.scene.write_data_to_sim()
    env.sim.forward()
    env.scene.update(dt=env.physics_dt)
    env._compute_intermediate_values(dt=env.physics_dt)

    current = env._current_pi0_state()
    env._pi0_command_q = current[:, :7].detach().clone()
    env._pi0_command_gripper = current[:, 7].detach().clone()
    env._pi0_finger_hold_target = full_q[:, 7:9].detach().clone()
    env._pi0_wrist_hold_target = full_q[:, 4:7].detach().clone()
    env._pi0_target = current.detach().clone()
    env._pi0_skip_advance_once = False
    env.pi0_target_clipped = 0


def capture_state(env: GroundTruthReplayEnv) -> dict[str, np.ndarray | float]:
    pose = np.concatenate(
        (
            env.fingertip_midpoint_pos[0].detach().cpu().numpy(),
            env.fingertip_midpoint_quat[0].detach().cpu().numpy(),
        )
    ).astype(np.float64)
    return {
        "joint_pos": env.joint_pos[0, :7].detach().cpu().numpy().astype(np.float64),
        "gripper_width_m": float(env.joint_pos[0, 7:9].sum().detach().cpu()),
        "ee_pose": pose,
    }


def step_physics(env: GroundTruthReplayEnv, physics_steps: int) -> None:
    for _ in range(physics_steps):
        env._sim_step_counter += 1
        env._apply_action()
        env.scene.write_data_to_sim()
        env.sim.step(render=False)
        env.scene.update(dt=env.physics_dt)
    env._compute_intermediate_values(dt=env.physics_dt)


def normalize_quaternion_rows(value: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(value, axis=1, keepdims=True)
    if np.any(norm < 1.0e-10) or not np.isfinite(value).all():
        raise ValueError("Quaternion rows must be finite and non-zero")
    return value / norm


def quaternion_error_deg(real_quat: np.ndarray, sim_quat: np.ndarray) -> np.ndarray:
    real_q = normalize_quaternion_rows(real_quat)
    sim_q = normalize_quaternion_rows(sim_quat)
    dot = np.sum(real_q * sim_q, axis=1)
    return np.rad2deg(2.0 * np.arccos(np.clip(np.abs(dot), 0.0, 1.0)))


def quaternion_increment_deg(quaternion_rows: np.ndarray) -> np.ndarray:
    value = np.asarray(quaternion_rows, dtype=np.float64)
    if len(value) < 2:
        return np.empty(0, dtype=np.float64)
    return quaternion_error_deg(value[:-1], value[1:])


def rms(value: np.ndarray, axis=0) -> np.ndarray:
    return np.sqrt(np.mean(np.square(np.asarray(value, dtype=np.float64)), axis=axis))


def max_abs_or_zero(value: np.ndarray, width: int) -> np.ndarray:
    value = np.asarray(value, dtype=np.float64)
    return np.zeros(width, dtype=np.float64) if len(value) == 0 else np.max(np.abs(value), axis=0)


def normalized_correlation(first: np.ndarray, second: np.ndarray) -> float:
    first = np.asarray(first, dtype=np.float64)
    second = np.asarray(second, dtype=np.float64)
    if len(first) < 2 or len(second) < 2:
        return 0.0
    first = first - np.mean(first)
    second = second - np.mean(second)
    denominator = float(np.linalg.norm(first) * np.linalg.norm(second))
    return 0.0 if denominator < 1.0e-10 else float(np.dot(first, second) / denominator)


def estimate_lag_steps(real_delta: np.ndarray, sim_delta: np.ndarray, max_lag: int = 20):
    real_delta = np.asarray(real_delta, dtype=np.float64)
    sim_delta = np.asarray(sim_delta, dtype=np.float64)
    max_lag = min(max_lag, max(len(real_delta) - 1, 0), max(len(sim_delta) - 1, 0))
    if max_lag <= 0:
        return 0, 0.0, 0.0
    candidates = []
    for lag in range(-max_lag, max_lag + 1):
        if lag > 0:
            real_part = real_delta[:-lag]
            sim_part = sim_delta[lag:]
        elif lag < 0:
            real_part = real_delta[-lag:]
            sim_part = sim_delta[:lag]
        else:
            real_part = real_delta
            sim_part = sim_delta
        correlations = [
            normalized_correlation(real_part[:, axis], sim_part[:, axis])
            for axis in range(real_part.shape[1])
        ]
        candidates.append((float(np.mean(correlations)), lag, correlations))
    best = max(candidates, key=lambda item: item[0])
    return int(best[1]), float(best[0]), float(np.mean(np.abs(best[2])))


def compute_diagnosis(
    real_pose: np.ndarray,
    sim_pose: np.ndarray,
    real_delta_ee: np.ndarray,
    sim_delta_ee: np.ndarray,
    action_delta_q: np.ndarray,
    real_delta_q: np.ndarray,
    sim_delta_q: np.ndarray,
    adapter_rate_limited: np.ndarray,
    action_clipped: np.ndarray,
    position_rmse_xyz: np.ndarray,
    final_position_error: float,
    orientation_mean_deg: float,
    joint_rmse: np.ndarray,
    lag_steps: int,
) -> list[str]:
    diagnosis: list[str] = []
    real_rms = rms(real_delta_ee)
    sim_rms = rms(sim_delta_ee)
    eef_gain = np.divide(sim_rms, real_rms, out=np.full(3, np.nan), where=real_rms > 1.0e-5)
    corr_matrix = np.array(
        [
            [normalized_correlation(real_delta_ee[:, i], sim_delta_ee[:, j]) for j in range(3)]
            for i in range(3)
        ]
    )
    diagonal_corr = np.diag(corr_matrix)
    valid_gain = np.isfinite(eef_gain)
    valid_corr = diagonal_corr[np.isfinite(diagonal_corr)]
    mean_abs_corr = float(np.mean(np.abs(valid_corr))) if len(valid_corr) else 0.0
    if np.any(((eef_gain < 0.8) | (eef_gain > 1.2)) & valid_gain) and mean_abs_corr > 0.7:
        diagnosis.append(
            "constant scale/response mismatch candidate: "
            f"EEF sim/real RMS gain={_format_vector(eef_gain)}"
        )
    for axis, value in enumerate(diagonal_corr):
        if value < -0.7:
            diagnosis.append(f"sign inversion candidate on EEF axis {axis}: correlation={value:.3f}")
    for axis in range(3):
        best_axis = int(np.argmax(np.abs(corr_matrix[axis])))
        if best_axis != axis and abs(corr_matrix[axis, best_axis]) > 0.7:
            diagnosis.append(
                f"axis ordering/frame candidate: real EEF axis {axis} maps to sim axis "
                f"{best_axis} (correlation={corr_matrix[axis, best_axis]:.3f})"
            )
    if lag_steps > 1:
        diagnosis.append(f"temporal lag candidate: sim trails real by about {lag_steps} action steps")
    elif lag_steps < -1:
        diagnosis.append(f"temporal alignment candidate: sim leads real by about {-lag_steps} action steps")
    error_norm = np.linalg.norm(sim_pose[:, :3] - real_pose[:, :3], axis=1)
    quarter = max(1, len(error_norm) // 4)
    if error_norm[-1] > 1.5 * max(float(np.median(error_norm[:quarter])), 1.0e-4):
        diagnosis.append("open-loop drift candidate: position error grows over the trajectory")
    if np.any(adapter_rate_limited):
        diagnosis.append(
            f"velocity-limit activity: {int(np.sum(adapter_rate_limited))} / {len(adapter_rate_limited)} transitions"
        )
    if np.any(action_clipped):
        diagnosis.append(
            f"joint-limit clipping activity: {int(np.sum(action_clipped))} / {len(action_clipped)} transitions"
        )
    command_rms = rms(action_delta_q)
    motion_rms = rms(sim_delta_q)
    deadband_axes = np.where(
        (command_rms > 1.0e-3)
        & (motion_rms / np.maximum(command_rms, 1.0e-8) < 0.1)
    )[0]
    if len(deadband_axes):
        diagnosis.append(f"deadband/low-response candidate on joint axes {deadband_axes.tolist()}")
    if orientation_mean_deg > 2.0 and float(np.mean(position_rmse_xyz)) < 0.005:
        diagnosis.append(
            "orientation-only mismatch candidate: position is relatively close but "
            f"mean orientation error is {orientation_mean_deg:.3f} deg"
        )
    if float(np.mean(joint_rmse)) > 0.05 and final_position_error < 0.01:
        diagnosis.append(
            "joint redundancy/controller candidate: joint trajectory differs while EEF remains close"
        )
    if not diagnosis:
        diagnosis.append("no dominant automatic failure signature; inspect the plots and metrics")
    return diagnosis


def run_trajectory(
    env: GroundTruthReplayEnv,
    trajectory: RealTrajectory,
    *,
    timing_mode: str,
    action_offset: int,
) -> dict:
    env.reset()
    set_initial_joint_state(env, trajectory.joint_pos[0], trajectory.gripper_width_m[0])

    real_pose_raw = np.asarray(trajectory.ee_pose, dtype=np.float64)
    real_pose, pose_alignment = align_real_pose(trajectory)
    real_q = trajectory.joint_pos
    sim_states = [capture_state(env)]
    adapter_targets = [env._pi0_target[0].detach().cpu().numpy().astype(np.float64)]
    real_dt = np.diff(trajectory.timestamps)
    if timing_mode == "fixed":
        hold_dt = np.full_like(real_dt, float(env.step_dt))
    else:
        hold_dt = real_dt.copy()

    physics_dt = float(env.physics_dt)
    physics_steps = np.maximum(1, np.rint(hold_dt / physics_dt).astype(int))
    sim_time = [0.0]
    action_clipped = []
    adapter_rate_limited = []
    action_indices = np.arange(len(trajectory.timestamps) - 1, dtype=int) + action_offset
    if np.any(action_indices < 0) or np.any(action_indices >= len(trajectory.timestamps)):
        raise ValueError(
            f"action offset {action_offset} cannot provide one action for every scored transition"
        )
    replay_action_arm = []
    replay_action_gripper = []
    replay_action_indices = []
    for frame, action_index in enumerate(action_indices):
        action = np.concatenate(
            (
                trajectory.action_arm[action_index],
                np.asarray([trajectory.action_gripper[action_index]]),
            )
        )
        replay_action_arm.append(trajectory.action_arm[action_index])
        replay_action_gripper.append(trajectory.action_gripper[action_index])
        replay_action_indices.append(action_index)
        lower, upper = env._joint_limits()
        lower_q = lower[0, :7].detach().cpu().numpy()
        upper_q = upper[0, :7].detach().cpu().numpy()
        clipped_q = np.clip(action[:7], lower_q, upper_q)
        action_clipped.append(bool(np.any(np.abs(clipped_q - action[:7]) > 1.0e-7)))
        env._set_pi0_target(action)
        # Mirror RealSimEnv.step(): the external Gym action is ignored by the
        # Pi0 path, but the normal pre-step bookkeeping still runs.
        env._pre_physics_step(torch.zeros_like(env.actions))
        target = env._pi0_target[0].detach().cpu().numpy().astype(np.float64)
        adapter_targets.append(target)
        adapter_rate_limited.append(bool(np.any(np.abs(target[:7] - clipped_q) > 1.0e-6)))
        step_physics(env, int(physics_steps[frame]))
        sim_states.append(capture_state(env))
        sim_time.append(sim_time[-1] + int(physics_steps[frame]) * physics_dt)

    sim_q = np.stack([state["joint_pos"] for state in sim_states])
    sim_width = np.asarray([state["gripper_width_m"] for state in sim_states], dtype=np.float64)
    sim_pose = np.stack([state["ee_pose"] for state in sim_states])
    adapter_targets = np.stack(adapter_targets)
    sim_time = np.asarray(sim_time, dtype=np.float64)
    raw_position_error = sim_pose[:, :3] - real_pose_raw[:, :3]
    raw_orientation_error = quaternion_error_deg(real_pose_raw[:, 3:], sim_pose[:, 3:])
    real_delta_ee = np.diff(real_pose[:, :3], axis=0)
    sim_delta_ee = np.diff(sim_pose[:, :3], axis=0)
    real_delta_q = np.diff(real_q, axis=0)
    sim_delta_q = np.diff(sim_q, axis=0)
    action_delta_q = np.diff(trajectory.action_arm, axis=0)
    real_orientation_delta = quaternion_increment_deg(real_pose[:, 3:])
    sim_orientation_delta = quaternion_increment_deg(sim_pose[:, 3:])
    orientation_error = quaternion_error_deg(real_pose[:, 3:], sim_pose[:, 3:])
    position_error = sim_pose[:, :3] - real_pose[:, :3]
    joint_error = sim_q - real_q
    sim_dt = np.diff(sim_time)
    real_joint_velocity = real_delta_q / np.maximum(real_dt[:, None], 1.0e-8)
    sim_joint_velocity = sim_delta_q / np.maximum(sim_dt[:, None], 1.0e-8)
    real_joint_acceleration = np.diff(real_joint_velocity, axis=0) / np.maximum(
        real_dt[1:, None], 1.0e-8
    )
    sim_joint_acceleration = np.diff(sim_joint_velocity, axis=0) / np.maximum(
        sim_dt[1:, None], 1.0e-8
    )
    lag_steps, lag_corr, lag_abs_corr = estimate_lag_steps(real_delta_ee, sim_delta_ee)
    position_rmse_xyz = rms(position_error)
    final_position_error = float(np.linalg.norm(position_error[-1]))
    joint_rmse = rms(joint_error)
    diagnosis = compute_diagnosis(
        real_pose,
        sim_pose,
        real_delta_ee,
        sim_delta_ee,
        action_delta_q,
        real_delta_q,
        sim_delta_q,
        np.asarray(adapter_rate_limited, dtype=bool),
        np.asarray(action_clipped, dtype=bool),
        position_rmse_xyz,
        final_position_error,
        float(np.mean(orientation_error)),
        joint_rmse,
        lag_steps,
    )

    metrics = {
        "trajectory": trajectory.path.parent.name,
        "h5": str(trajectory.path.resolve()),
        "frames": int(len(trajectory.timestamps)),
        "real_duration_s": float(trajectory.timestamps[-1]),
        "real_dataset_hz": float(1.0 / np.median(real_dt)),
        "timing_mode": timing_mode,
        "action_offset": int(action_offset),
        "physics_dt_s": physics_dt,
        "physics_hz": float(1.0 / physics_dt),
        "control_dt_s": float(env.step_dt),
        "control_hz": float(1.0 / env.step_dt),
        "decimation": int(env.cfg.decimation),
        "sim_duration_s": float(sim_time[-1]),
        "timing_error_final_s": float(sim_time[-1] - trajectory.timestamps[-1]),
        "timing_error_max_abs_s": float(np.max(np.abs(sim_time - trajectory.timestamps))),
        "pose_alignment": pose_alignment,
        "raw_unaligned_position_rmse_3d_m": float(
            np.sqrt(np.mean(np.sum(np.square(raw_position_error), axis=1)))
        ),
        "raw_unaligned_orientation_mean_deg": float(np.mean(raw_orientation_error)),
        "position_rmse_xyz_m": position_rmse_xyz.tolist(),
        "position_rmse_xyz_mm": (1000.0 * position_rmse_xyz).tolist(),
        "position_rmse_3d_m": float(np.sqrt(np.mean(np.sum(position_error**2, axis=1)))),
        "final_position_error_m": final_position_error,
        "final_position_error_mm": 1000.0 * final_position_error,
        "orientation_mean_deg": float(np.mean(orientation_error)),
        "orientation_final_deg": float(orientation_error[-1]),
        "orientation_max_deg": float(np.max(orientation_error)),
        "joint_rmse_rad": joint_rmse.tolist(),
        "joint_rmse_deg": np.rad2deg(joint_rmse).tolist(),
        "joint_rmse_mean_rad": float(np.mean(joint_rmse)),
        "joint_final_abs_error_rad": np.abs(joint_error[-1]).tolist(),
        "temporal_lag_steps": lag_steps,
        "temporal_lag_seconds_at_real_median_dt": float(lag_steps * np.median(real_dt)),
        "temporal_lag_mean_correlation": lag_corr,
        "temporal_lag_mean_abs_axis_correlation": lag_abs_corr,
        "action_increment_rms_rad": rms(action_delta_q).tolist(),
        "real_joint_motion_rms_rad": rms(real_delta_q).tolist(),
        "sim_joint_motion_rms_rad": rms(sim_delta_q).tolist(),
        "real_joint_velocity_max_abs_rad_s": np.max(np.abs(real_joint_velocity), axis=0).tolist(),
        "sim_joint_velocity_max_abs_rad_s": np.max(np.abs(sim_joint_velocity), axis=0).tolist(),
        "real_joint_acceleration_max_abs_rad_s2": max_abs_or_zero(
            real_joint_acceleration, 7
        ).tolist(),
        "sim_joint_acceleration_max_abs_rad_s2": max_abs_or_zero(
            sim_joint_acceleration, 7
        ).tolist(),
        "real_joint_gain": np.divide(
            rms(real_delta_q), rms(action_delta_q), out=np.full(7, np.nan), where=rms(action_delta_q) > 1.0e-8
        ).tolist(),
        "sim_joint_gain": np.divide(
            rms(sim_delta_q), rms(action_delta_q), out=np.full(7, np.nan), where=rms(action_delta_q) > 1.0e-8
        ).tolist(),
        "eef_sim_over_real_rms_gain_xyz": np.divide(
            rms(sim_delta_ee), rms(real_delta_ee), out=np.full(3, np.nan), where=rms(real_delta_ee) > 1.0e-8
        ).tolist(),
        "real_orientation_increment_rms_deg": float(rms(real_orientation_delta, axis=None)),
        "sim_orientation_increment_rms_deg": float(rms(sim_orientation_delta, axis=None)),
        "sim_over_real_orientation_increment_gain": float(
            np.divide(
                rms(sim_orientation_delta, axis=None),
                rms(real_orientation_delta, axis=None),
                out=np.array(np.nan),
                where=rms(real_orientation_delta, axis=None) > 1.0e-8,
            )
        ),
        "adapter_velocity_limited_transitions": int(np.sum(adapter_rate_limited)),
        "action_joint_clipped_transitions": int(np.sum(action_clipped)),
        "hole_collision_enabled": bool(args.enable_hole_collision),
        "diagnosis": diagnosis,
    }
    return {
        "trajectory": trajectory,
        "metrics": metrics,
        "real_pose": real_pose,
        "real_pose_raw": real_pose_raw,
        "real_q": real_q,
        "real_width": trajectory.gripper_width_m,
        "real_action_arm": trajectory.action_arm,
        "real_action_gripper": trajectory.action_gripper,
        "replay_action_arm": np.asarray(replay_action_arm, dtype=np.float64),
        "replay_action_gripper": np.asarray(replay_action_gripper, dtype=np.float64),
        "replay_action_indices": np.asarray(replay_action_indices, dtype=int),
        "sim_pose": sim_pose,
        "sim_q": sim_q,
        "sim_width": sim_width,
        "adapter_targets": adapter_targets,
        "real_time": trajectory.timestamps,
        "sim_time": sim_time,
        "hold_dt": hold_dt,
        "physics_steps": physics_steps,
        "real_delta_ee": real_delta_ee,
        "sim_delta_ee": sim_delta_ee,
        "real_delta_q": real_delta_q,
        "sim_delta_q": sim_delta_q,
        "action_delta_q": action_delta_q,
        "real_orientation_delta": real_orientation_delta,
        "sim_orientation_delta": sim_orientation_delta,
        "orientation_error": orientation_error,
        "position_error": position_error,
        "action_clipped": np.asarray(action_clipped, dtype=bool),
        "adapter_rate_limited": np.asarray(adapter_rate_limited, dtype=bool),
    }


def write_csv(path: Path, header: list[str], rows: list[list[object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)


def write_real_reference(output_dir: Path, result: dict) -> None:
    trajectory: RealTrajectory = result["trajectory"]
    header = ["frame_index", "time_s"]
    header += [f"eef_{axis}" for axis in ("x", "y", "z", "qx", "qy", "qz", "qw")]
    header += [f"recorded_eef_{axis}" for axis in ("x", "y", "z", "qx", "qy", "qz", "qw")]
    header += [f"joint_q{index}" for index in range(7)]
    header += ["gripper_width_m"]
    header += [f"action_arm_{index}" for index in range(7)] + ["action_gripper"]
    rows = []
    for index in range(len(trajectory.timestamps)):
        rows.append(
            [
                index,
                trajectory.timestamps[index],
                *result["real_pose"][index],
                *result["real_pose_raw"][index],
                *trajectory.joint_pos[index],
                trajectory.gripper_width_m[index],
                *trajectory.action_arm[index],
                trajectory.action_gripper[index],
            ]
        )
    write_csv(output_dir / "real_reference.csv", header, rows)


def write_sim_replay(output_dir: Path, result: dict) -> None:
    trajectory: RealTrajectory = result["trajectory"]
    header = [
        "state_frame_index",
        "action_frame_index",
        "real_time_s",
        "sim_time_s",
        "held_dt_s",
        "physics_steps",
    ]
    header += [f"action_arm_{index}" for index in range(7)] + ["action_gripper"]
    header += [f"adapter_target_q{index}" for index in range(7)] + ["adapter_target_gripper"]
    header += [f"sim_joint_q{index}" for index in range(7)] + ["sim_gripper_width_m"]
    header += [f"sim_eef_{axis}" for axis in ("x", "y", "z", "qx", "qy", "qz", "qw")]
    header += [f"real_delta_eef_{axis}" for axis in ("x", "y", "z")]
    header += [f"sim_delta_eef_{axis}" for axis in ("x", "y", "z")]
    header += ["real_delta_eef_orientation_deg", "sim_delta_eef_orientation_deg"]
    header += [f"real_delta_joint_q{index}" for index in range(7)]
    header += [f"sim_delta_joint_q{index}" for index in range(7)]
    header += [f"action_increment_joint_q{index}" for index in range(7)]
    header += ["action_joint_clipped", "adapter_velocity_limited"]
    rows = []
    for index in range(len(trajectory.timestamps) - 1):
        rows.append(
            [
                index + 1,
                result["replay_action_indices"][index],
                result["real_time"][index + 1],
                result["sim_time"][index + 1],
                result["hold_dt"][index],
                result["physics_steps"][index],
                *result["replay_action_arm"][index],
                result["replay_action_gripper"][index],
                *result["adapter_targets"][index + 1, :7],
                result["adapter_targets"][index + 1, 7],
                *result["sim_q"][index + 1],
                result["sim_width"][index + 1],
                *result["sim_pose"][index + 1],
                *result["real_delta_ee"][index],
                *result["sim_delta_ee"][index],
                result["real_orientation_delta"][index],
                result["sim_orientation_delta"][index],
                *result["real_delta_q"][index],
                *result["sim_delta_q"][index],
                *result["action_delta_q"][index],
                int(result["action_clipped"][index]),
                int(result["adapter_rate_limited"][index]),
            ]
        )
    write_csv(output_dir / "sim_replay.csv", header, rows)


def save_plots(output_dir: Path, result: dict) -> None:
    trajectory: RealTrajectory = result["trajectory"]
    time_s = result["real_time"]
    real_pose = result["real_pose"]
    sim_pose = result["sim_pose"]
    metrics = result["metrics"]

    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    for axis, name, column in zip(axes, ("x", "y", "z"), range(3)):
        axis.plot(time_s, real_pose[:, column], label="real", linewidth=1.5)
        axis.plot(time_s, sim_pose[:, column], label="sim", linewidth=1.2)
        axis.set_ylabel(f"EEF {name} (m)")
        axis.grid(True, alpha=0.3)
    axes[0].legend()
    axes[-1].set_xlabel("time (s)")
    fig.suptitle(f"{trajectory.path.parent.name}: EEF XYZ trajectory")
    fig.tight_layout()
    fig.savefig(output_dir / "01_xyz_trajectory.png", dpi=150)
    plt.close(fig)

    position_error = result["position_error"]
    fig, axis = plt.subplots(figsize=(10, 4))
    axis.plot(time_s, np.linalg.norm(position_error, axis=1) * 1000.0)
    axis.set_xlabel("time (s)")
    axis.set_ylabel("|p_sim - p_real| (mm)")
    axis.set_title(f"Position error; final={metrics['final_position_error_mm']:.3f} mm")
    axis.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / "02_position_error.png", dpi=150)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(10, 4))
    axis.plot(time_s, result["orientation_error"])
    axis.set_xlabel("time (s)")
    axis.set_ylabel("orientation geodesic error (deg)")
    axis.set_title(f"Orientation error; mean={metrics['orientation_mean_deg']:.3f} deg")
    axis.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / "03_orientation_error.png", dpi=150)
    plt.close(fig)

    real_q = result["real_q"]
    sim_q = result["sim_q"]
    fig, axes = plt.subplots(4, 2, figsize=(12, 12), sharex=True)
    for joint in range(7):
        axis = axes.flat[joint]
        axis.plot(time_s, real_q[:, joint], label="real", linewidth=1.2)
        axis.plot(time_s, sim_q[:, joint], label="sim", linewidth=1.0)
        axis.set_ylabel(f"q{joint} (rad)")
        axis.grid(True, alpha=0.3)
    axes.flat[7].axis("off")
    axes[0, 0].legend()
    axes[-1, 0].set_xlabel("time (s)")
    axes[-1, 1].set_xlabel("time (s)")
    fig.suptitle(f"{trajectory.path.parent.name}: joint trajectory")
    fig.tight_layout()
    fig.savefig(output_dir / "04_joint_trajectory.png", dpi=150)
    plt.close(fig)

    action_delta_q = result["action_delta_q"]
    real_delta_q = result["real_delta_q"]
    sim_delta_q = result["sim_delta_q"]
    transition_time = time_s[1:]
    fig, axes = plt.subplots(4, 2, figsize=(12, 12), sharex=True)
    for joint in range(7):
        axis = axes.flat[joint]
        axis.plot(transition_time, action_delta_q[:, joint], label="action increment", linewidth=1.0)
        axis.plot(transition_time, real_delta_q[:, joint], label="real actual", linewidth=1.0)
        axis.plot(transition_time, sim_delta_q[:, joint], label="sim actual", linewidth=1.0)
        axis.set_ylabel(f"q{joint} delta")
        axis.grid(True, alpha=0.3)
    axes.flat[7].axis("off")
    axes[0, 0].legend(fontsize=8)
    axes[-1, 0].set_xlabel("time (s)")
    axes[-1, 1].set_xlabel("time (s)")
    fig.suptitle(f"{trajectory.path.parent.name}: action increment vs actual joint motion")
    fig.tight_layout()
    fig.savefig(output_dir / "05_action_vs_actual_motion.png", dpi=150)
    plt.close(fig)

    fig = plt.figure(figsize=(8, 7))
    axis = fig.add_subplot(111, projection="3d")
    axis.plot(real_pose[:, 0], real_pose[:, 1], real_pose[:, 2], label="real", linewidth=1.5)
    axis.plot(sim_pose[:, 0], sim_pose[:, 1], sim_pose[:, 2], label="sim", linewidth=1.2)
    axis.scatter(*real_pose[0, :3], marker="o", label="start")
    axis.scatter(*real_pose[-1, :3], marker="x", label="real end")
    axis.set_xlabel("x (m)")
    axis.set_ylabel("y (m)")
    axis.set_zlabel("z (m)")
    axis.set_title(f"{trajectory.path.parent.name}: EEF 3D trajectory")
    axis.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "06_eef_3d_trajectory.png", dpi=150)
    plt.close(fig)


def write_json(path: Path, value) -> None:
    def clean(obj):
        if isinstance(obj, (np.floating, np.integer)):
            obj = obj.item()
        if isinstance(obj, float):
            return obj if np.isfinite(obj) else None
        if isinstance(obj, np.ndarray):
            return clean(obj.tolist())
        if isinstance(obj, dict):
            return {str(key): clean(item) for key, item in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [clean(item) for item in obj]
        return obj

    path.write_text(
        json.dumps(clean(value), indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def write_aggregate_summary(output_dir: Path, results: list[dict]) -> None:
    all_position_errors = np.concatenate([result["position_error"] for result in results], axis=0)
    all_orientation_errors = np.concatenate([result["orientation_error"] for result in results], axis=0)
    all_joint_errors = np.concatenate(
        [result["sim_q"] - result["real_q"] for result in results], axis=0
    )
    summary = {
        "trajectory_count": len(results),
        "total_aligned_frames": int(sum(len(result["real_time"]) for result in results)),
        "position_rmse_xyz_m": rms(all_position_errors).tolist(),
        "position_rmse_xyz_mm": (1000.0 * rms(all_position_errors)).tolist(),
        "position_rmse_3d_m": float(np.sqrt(np.mean(np.sum(all_position_errors**2, axis=1)))),
        "final_position_error_m": [result["metrics"]["final_position_error_m"] for result in results],
        "orientation_mean_deg": float(np.mean(all_orientation_errors)),
        "orientation_final_deg": [result["metrics"]["orientation_final_deg"] for result in results],
        "joint_rmse_rad": rms(all_joint_errors).tolist(),
        "joint_rmse_mean_rad": float(np.mean(rms(all_joint_errors))),
        "temporal_lag_steps": [result["metrics"]["temporal_lag_steps"] for result in results],
        "temporal_lag_seconds": [
            result["metrics"]["temporal_lag_seconds_at_real_median_dt"] for result in results
        ],
        "diagnosis": [
            {"trajectory": result["metrics"]["trajectory"], "items": result["metrics"]["diagnosis"]}
            for result in results
        ],
    }
    write_json(output_dir / "aggregate_metrics.json", summary)

    names = [result["metrics"]["trajectory"] for result in results]
    x = np.arange(len(names))
    final_position_mm = [result["metrics"]["final_position_error_mm"] for result in results]
    mean_position_mm = [float(np.mean(result["metrics"]["position_rmse_xyz_mm"])) for result in results]
    mean_joint_deg = [float(np.mean(result["metrics"]["joint_rmse_deg"])) for result in results]
    lag_seconds = [result["metrics"]["temporal_lag_seconds_at_real_median_dt"] for result in results]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes[0, 0].bar(x - 0.18, mean_position_mm, width=0.36, label="XYZ RMSE mean")
    axes[0, 0].bar(x + 0.18, final_position_mm, width=0.36, label="final error")
    axes[0, 0].set_ylabel("mm")
    axes[0, 0].set_title("Position error")
    axes[0, 0].legend(fontsize=8)
    axes[0, 1].bar(x, [result["metrics"]["orientation_mean_deg"] for result in results])
    axes[0, 1].set_ylabel("deg")
    axes[0, 1].set_title("Mean orientation error")
    axes[1, 0].bar(x, mean_joint_deg)
    axes[1, 0].set_ylabel("deg")
    axes[1, 0].set_title("Mean joint RMSE")
    axes[1, 1].bar(x, lag_seconds)
    axes[1, 1].axhline(0.0, color="black", linewidth=0.8)
    axes[1, 1].set_ylabel("s")
    axes[1, 1].set_title("Estimated temporal lag")
    for axis in axes.flat:
        axis.set_xticks(x)
        axis.set_xticklabels(names, rotation=35, ha="right")
        axis.grid(True, axis="y", alpha=0.3)
    fig.suptitle("Ground-truth action replay aggregate baseline")
    fig.tight_layout()
    fig.savefig(output_dir / "aggregate_summary.png", dpi=150)
    plt.close(fig)


def main() -> None:
    data_root = resolve_repo_path(args.data_root)
    output_dir = resolve_repo_path(args.output_dir)
    paths = sorted(data_root.glob("traj_*/data.h5"))
    if not paths:
        raise FileNotFoundError(f"No traj_*/data.h5 files under {data_root}")

    if args.trajectories is None:
        selected_paths = paths[: args.max_trajectories]
    else:
        by_index = {int(path.parent.name.split("_")[-1]): path for path in paths}
        missing = [index for index in args.trajectories if index not in by_index]
        if missing:
            raise FileNotFoundError(f"Requested trajectories not found: {missing}")
        selected_paths = [by_index[index] for index in args.trajectories]
    if not selected_paths:
        raise ValueError("No trajectories selected")

    trajectories = [
        load_trajectory(path, int(path.parent.name.split("_")[-1]))
        for path in selected_paths
    ]
    output_dir.mkdir(parents=True, exist_ok=True)
    first_q = trajectories[0].joint_pos[0]
    cfg = configure_replay_cfg(args.task, first_q)

    env = None
    try:
        env = GroundTruthReplayEnv(cfg, render_mode="rgb_array", output_dir=str(output_dir))
        if env.num_envs != 1:
            raise RuntimeError(f"Ground-truth replay requires one environment, got {env.num_envs}")
        if not args.enable_hole_collision:
            collider_count = set_asset_collision_enabled(env, env._fixed_asset, False)
            print(f"[GTReplay] disabled {collider_count} fixed-asset colliders", flush=True)

        write_action_semantics_report(
            output_dir / "action_semantics_report.txt",
            trajectories,
            cfg,
            physics_dt=float(env.physics_dt),
            action_source=args.action_source,
            timing_mode=args.timing,
            action_offset=args.action_offset,
        )
        results = []
        for trajectory in trajectories:
            print(
                f"[GTReplay] {trajectory.path.parent.name}: "
                f"{len(trajectory.timestamps)} frames, "
                f"{1.0 / np.median(np.diff(trajectory.timestamps)):.2f} Hz",
                flush=True,
            )
            result = run_trajectory(
                env,
                trajectory,
                timing_mode=args.timing,
                action_offset=args.action_offset,
            )
            trajectory_output = output_dir / trajectory.path.parent.name
            trajectory_output.mkdir(parents=True, exist_ok=True)
            write_real_reference(trajectory_output, result)
            write_sim_replay(trajectory_output, result)
            write_json(trajectory_output / "metrics.json", result["metrics"])
            save_plots(trajectory_output, result)
            results.append(result)
            print(
                f"[GTReplay] {trajectory.path.parent.name}: "
                f"XYZ_RMSE={result['metrics']['position_rmse_3d_m'] * 1000.0:.3f} mm, "
                f"final={result['metrics']['final_position_error_mm']:.3f} mm, "
                f"ori={result['metrics']['orientation_mean_deg']:.3f} deg, "
                f"lag={result['metrics']['temporal_lag_steps']} steps",
                flush=True,
            )
        write_aggregate_summary(output_dir, results)
        write_json(
            output_dir / "run_config.json",
            {
                "task": args.task,
                "data_root": str(data_root),
                "trajectories": [trajectory.path.parent.name for trajectory in trajectories],
                "action_source": args.action_source,
                "action_offset": args.action_offset,
                "timing": args.timing,
                "pose_alignment": args.pose_alignment,
                "hole_collision_enabled": args.enable_hole_collision,
                "pi0_inference": False,
                "controller": "Pi0RealSimEnv._set_pi0_target -> _apply_action",
                "output_dir": str(output_dir),
            },
        )
        print(f"[GTReplay] outputs written to {output_dir}", flush=True)
    finally:
        if env is not None:
            env.close()
        simulation_app.close()


if __name__ == "__main__":
    main()
