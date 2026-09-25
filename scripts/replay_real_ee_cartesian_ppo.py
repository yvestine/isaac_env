
"""Replay real EE XYZ with the original PPO Franka articulation and cameras.

This entry point intentionally uses the same RealSimEnv scene as PPO:
``/franka_env/Robot/franka`` is the controlled articulation, the existing
front camera is kept, and the wrist camera is the PPO camera attached to
panda_hand. The complete trajectory is prepared with DLS IK, then replayed
through PhysX with PPO's task-space controller, independent HeldAsset, and
closed-gripper contact semantics.
"""

from __future__ import annotations

import argparse
import atexit
import csv
import json
import os
import tempfile
from pathlib import Path

import cv2
import h5py
import numpy as np
import torch
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--h5", type=Path, default=Path("real_data/traj_0/data.h5"))
parser.add_argument("--task", type=str, default="TacEx-RealSim-PegInsert-Direct-v0")
parser.add_argument("--output-dir", type=Path, default=Path("outputs/traj0_ppo_cartesian_rollout"))
parser.add_argument("--endpoint-window", type=int, default=3)
parser.add_argument("--fps", type=float, default=10.0)
parser.add_argument(
    "--wrist-pitch-deg",
    type=float,
    default=0.0,
    help="Local OpenGL-camera X rotation applied after the migrated PPO wrist calibration.",
)
parser.add_argument(
    "--wrist-forward-offset",
    type=float,
    default=0.0,
    help="Move the PPO wrist camera along its local OpenGL optical axis, in metres.",
)
parser.add_argument(
    "--wrist-yaw-deg",
    type=float,
    default=0.0,
    help="Local OpenGL-camera Y rotation applied to align the real camera principal ray.",
)
parser.add_argument(
    "--hole-yaw-deg",
    type=float,
    default=None,
    help="Optional override of the Pi0 hole yaw in degrees; default uses joint replay calibration.",
)
parser.add_argument("--hole-offset-x", type=float, default=0.0)
parser.add_argument("--hole-offset-y", type=float, default=0.0)
parser.add_argument("--hole-offset-z", type=float, default=0.0)
parser.add_argument(
    "--link7-cutoff-hz",
    type=float,
    default=1.2,
    help="Causal low-pass cutoff for the panda_link7 wrench after de-spiking.",
)
parser.add_argument(
    "--contact-feedback-cutoff-hz",
    type=float,
    default=0.35,
    help="Causal low-pass cutoff for the peg-hole insertion feedback at output FPS.",
)
parser.add_argument(
    "--link7-force-gain",
    type=float,
    default=10.0,
    help=(
        "Global post-filter gain for the panda_link7 wrench. Raw and unscaled "
        "filtered streams remain available for diagnosis."
    ),
)
parser.add_argument(
    "--link7-frame-calibration",
    type=Path,
    default=None,
    help=(
        "result.json from diagnose_wrench_sign.py with panda_link7 used as both "
        "the applied-force and incoming-wrench body. When supplied, also export "
        "the incoming wrench in robot-base coordinates with torque referenced at K."
    ),
)
parser.add_argument("--jam-force-threshold-n", type=float, default=2.0)
parser.add_argument("--jam-min-duration-s", type=float, default=0.5)
parser.add_argument(
    "--disable-hole-collision",
    action="store_true",
    help="Keep the hole pose/visual unchanged but disable its colliders for a force-baseline rollout.",
)
parser.add_argument(
    "--record-contact-pair",
    action="store_true",
    help="Record explicit PhysX peg-hole normal and friction forces without changing replay control.",
)
parser.add_argument(
    "--sim-fk-csv",
    type=Path,
    default=None,
    help="Optional simulator FK CSV used for real->Pi0 pose calibration (source quaternion columns are WXYZ).",
)
parser.add_argument(
    "--pose-alignment",
    choices=("fk", "raw"),
    default="fk",
    help=(
        "Align H5 EE pose to the current Isaac/Pi0 frame using the existing "
        "ee_pose_sim_isaac6.csv (default: fk); raw is only for debugging."
    ),
)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if args.link7_cutoff_hz <= 0.0:
    raise ValueError("--link7-cutoff-hz must be positive")
if args.contact_feedback_cutoff_hz <= 0.0:
    raise ValueError("--contact-feedback-cutoff-hz must be positive")
if args.link7_force_gain <= 0.0:
    raise ValueError("--link7-force-gain must be positive")
# Keep the same Isaac Sim 6 startup workaround as the verified joint replay:
# RTX geometry streaming can leave Fabric/USD render meshes out of sync.
_kit_args = (
    "--/log/level=warning --/log/outputStreamLevel=warning "
    "--/rtx-transient/resourcemanager/enableGeometryStreaming=false"
)
args.kit_args = f"{getattr(args, 'kit_args', '')} {_kit_args}".strip()
# This script always captures the PPO front and wrist cameras.  The old
# working replay enables the Isaac Sim camera extensions before AppLauncher;
# setting only cfg.enable_cameras is too late and can make startup terminate
# after simulation reset without a Python traceback.
if hasattr(args, "enable_cameras"):
    args.enable_cameras = True

simulation_app = AppLauncher(args).app
from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics  # noqa: E402

# Isaac Sim 6 keeps this legacy Python module in an extension directory that
# is not on isaacsim.__path__ by default. TacEx still imports XFormPrim from it.
import isaacsim  # noqa: E402
_ISAACSIM_EXTS = Path(isaacsim.__file__).resolve().parent / "exts"
for _extension_name in (
    "isaacsim.core.prims",
    "isaacsim.core.utils",
    "isaacsim.core.api",
    "isaacsim.storage.native",
):
    _extension_pkg = _ISAACSIM_EXTS / _extension_name / "isaacsim"
    if _extension_pkg.is_dir() and str(_extension_pkg) not in isaacsim.__path__:
        isaacsim.__path__.append(str(_extension_pkg))

import tacex_tasks  # noqa: E402,F401

from tacex_tasks import torch_compat as torch_utils  # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402
from isaaclab_tasks.direct.factory import factory_control, factory_utils  # noqa: E402
from isaaclab.utils import math as isaaclab_math  # noqa: E402
from tacex_tasks.real2sim.pi0_env import (  # noqa: E402
    _composite_pi0_cable_grommets,
    _composite_pi0_green_table_front,
    _composite_pi0_reference_cylinder,
    _match_pi0_wrist_appearance,
    _pi0_wrist_camera_rotation,
    _resize_pi0_rgb,
)
from tacex_tasks.real2sim.pi0_env_cfg import RealSimPi0PegInsertCfg  # noqa: E402
from tacex_tasks.real2sim.realsim_env import RealSimEnv, _write_h264_mp4  # noqa: E402


def create_physics_free_background_usd() -> Path:
    """Reference the camera background without its embedded PhysicsScene."""
    source = Path(__file__).resolve().parents[1] / "franka_env_background_edit" / "franka_env.usd"
    if not source.is_file():
        raise FileNotFoundError(f"Background USD not found: {source}")
    fd, filename = tempfile.mkstemp(prefix="tacex_visual_background_", suffix=".usd")
    os.close(fd)
    wrapper_path = Path(filename)
    stage = Usd.Stage.CreateNew(str(wrapper_path))
    root = UsdGeom.Xform.Define(stage, "/World")
    root.GetPrim().GetReferences().AddReference(str(source), "/World")
    stage.SetDefaultPrim(root.GetPrim())
    # The source PhysicsScene is valid in standalone Isaac Sim, but must not
    # be composed into an IsaacLab SimulationContext that owns /physicsScene.
    stage.OverridePrim("/World/PhysicsScene").SetActive(False)
    stage.GetRootLayer().Save()
    return wrapper_path


_VISUAL_BACKGROUND_USD = create_physics_free_background_usd()
atexit.register(lambda: _VISUAL_BACKGROUND_USD.unlink(missing_ok=True))

class PPOReplayEnv(RealSimEnv):
    """PPO-compatible bootstrap with one complete Franka articulation."""

    def __init__(self, cfg, **kwargs):
        # DirectRLEnv can invoke _apply_action while the parent is building
        # the scene. Prepare only the whole-articulation position target;
        # no USD link is ever moved directly.
        # FactoryEnv defaults to collection mode during parent construction;
        # set the replay lifecycle before super() so bootstrap cannot exit.
        self.collect_data = False
        self.immediate_stop = False
        self.save_failed_trajectory = False
        self.num_trajectories = 1_000_000
        self.cur_num_traj = 0
        self._replay_gripper_open_width_m = 0.04
        self._gripper_force_n = 5.0
        self._couple_contact_reaction = False
        self._physical_load_cell = False
        self._kinematic_held_asset = False
        self._grasp_constraint = False
        self._replay_target_pos = None
        self._replay_target_quat = None
        self._replay_target_gripper = 0.0
        super().__init__(cfg, **kwargs)

    def _setup_scene(self) -> None:
        super()._setup_scene()
        # The migrated visual background contains its own /World/PhysicsScene.
        # Once referenced under each environment it conflicts with IsaacLab's
        # simulation scene and Kit stops on the second sim.step(). Keep only
        # the SimulationContext-owned scene; do not edit the source USD.
        nested_physics_scenes = [
            prim.GetPath()
            for prim in self.sim.stage.Traverse()
            if prim.IsA(UsdPhysics.Scene) and str(prim.GetPath()).startswith("/World/envs/")
        ]
        for scene_path in nested_physics_scenes:
            self.sim.stage.RemovePrim(scene_path)
            print(f"[ReplayInit] removed background PhysicsScene: {scene_path}", flush=True)

    def prepare_replay_state(self) -> None:
        """Initialize the articulation without the legacy task-reset IK/step."""
        env_ids = torch.arange(self.num_envs, dtype=torch.long, device=self.device)
        print("[ReplayInit] scene.reset", flush=True)
        self.scene.reset(env_ids)
        print("[ReplayInit] assets", flush=True)
        self._set_assets_to_default_pose(env_ids)

        joint_pos = self._robot.data.default_joint_pos[env_ids].clone()
        joint_pos[:, 7:] = float(self.cfg.task.held_asset_cfg.diameter) / 2.0 * 1.25
        joint_pos[:, :7] = torch.as_tensor(
            self.cfg.ctrl.reset_joints, dtype=torch.float32, device=self.device
        ).view(1, 7)
        joint_vel = torch.zeros_like(joint_pos)
        self.ctrl_target_joint_pos[env_ids] = joint_pos
        self._robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)
        self._robot.set_joint_position_target(self.ctrl_target_joint_pos[env_ids], env_ids=env_ids)
        self._robot.set_joint_effort_target(torch.zeros_like(joint_pos), env_ids=env_ids)
        print("[ReplayInit] robot state written", flush=True)

        # Initialize through the lightweight replay path. Physical stepping is
        # performed explicitly after IK targets and the grasp are prepared.
        self.scene.write_data_to_sim()
        print("[ReplayInit] scene written", flush=True)
        # Commit the initial articulation state once so the RTX/USD view is
        # synchronized with the PhysX tensors.
        self.sim.step(render=False)
        print("[ReplayInit] bootstrap step", flush=True)
        self.sim.forward()
        print("[ReplayInit] sim forward", flush=True)
        self.scene.update(dt=self.physics_dt)
        print("[ReplayInit] scene updated", flush=True)
        self._compute_intermediate_values(dt=self.physics_dt)
        # The lightweight bootstrap intentionally bypasses RealSimEnv.reset(),
        # so initialize the same task-space gains used by PPO before the first
        # physical replay step.
        self.task_prop_gains = self.default_gains
        self.task_deriv_gains = factory_utils.get_deriv_gains(self.default_gains)
        print("[ReplayInit] values computed", flush=True)

        fixed_tip_local = torch.zeros((self.num_envs, 3), device=self.device)
        fixed_tip_local[:, 2] = (
            float(self.cfg.task.fixed_asset_cfg.height)
            + float(self.cfg.task.fixed_asset_cfg.base_height)
        )
        identity = torch.tensor([0.0, 0.0, 0.0, 1.0], device=self.device).view(1, 4)
        fixed_tip_pos, _ = isaaclab_math.combine_frame_transforms(
            self.fixed_pos,
            self.fixed_quat,
            fixed_tip_local,
            identity.expand(self.num_envs, -1),
        )
        self.fixed_pos_obs_frame[:] = fixed_tip_pos
        self.init_fixed_pos_obs_noise.zero_()
        print("[ReplayInit] fixed frame", flush=True)
        # The task-history buffers below are only consumed by the inherited
        # reset/step path. Replay drives the PPO task-space controller
        # directly, so initializing them here can re-enter Isaac Sim 6's
        # legacy task state machinery and terminate Kit during startup.

    def _apply_action(self) -> None:
        # This is the same low-level task-space control call used by the PPO
        # branch in RealSimEnv._apply_action. The measured joint state is
        # never written to PhysX during rollout.
        if self._replay_target_pos is None or self._replay_target_quat is None:
            return
        if self.last_update_timestamp < self._robot._data._sim_timestamp:
            self._compute_intermediate_values(dt=self.physics_dt)
        if not getattr(self, "_printed_servo_debug", False):
            print(
                "[ReplayDebug] first-servo "
                f"ee={self.fingertip_midpoint_pos[0].detach().cpu().tolist()} "
                f"target={self._replay_target_pos[0].detach().cpu().tolist()}",
                flush=True,
            )
            self._printed_servo_debug = True
        self.ctrl_target_fingertip_midpoint_pos = self._replay_target_pos
        self.ctrl_target_fingertip_midpoint_quat = self._replay_target_quat
        self.ctrl_target_gripper_dof_pos = 0.0
        self.generate_ctrl_signals(
            ctrl_target_fingertip_midpoint_pos=self._replay_target_pos,
            ctrl_target_fingertip_midpoint_quat=self._replay_target_quat,
            ctrl_target_gripper_dof_pos=0.0,
        )

def load_real_data(path: Path):
    with h5py.File(path, "r") as h5:
        q = np.asarray(h5["obs/state/joint_pos"][:], dtype=np.float32)
        ee_pose = np.asarray(h5["obs/state/ee_pose"][:], dtype=np.float32)
        gripper_width = np.asarray(h5["obs/state/gripper_width_m"][:], dtype=np.float32).reshape(-1)
        timestamps = np.asarray(h5["timestamps"][:], dtype=np.float64)
    if q.ndim != 2 or q.shape[1] != 7:
        raise ValueError(f"joint_pos must be (N,7), got {q.shape}")
    if ee_pose.ndim != 2 or ee_pose.shape[0] != len(q) or ee_pose.shape[1] != 7:
        raise ValueError(f"ee_pose must be (N,7), got {ee_pose.shape}")
    if timestamps.shape != (len(q),):
        raise ValueError("timestamps and trajectory length differ")
    if gripper_width.shape != (len(q),):
        raise ValueError("gripper_width_m and trajectory length differ")
    if not all(np.isfinite(x).all() for x in (q, ee_pose, gripper_width, timestamps)):
        raise ValueError("real trajectory contains NaN or Inf")
    if len(timestamps) > 1 and np.any(np.diff(timestamps) <= 0.0):
        raise ValueError("timestamps must be strictly increasing for physical replay")
    return q, ee_pose, gripper_width, timestamps

def fit_real_to_legacy(real_xyz: np.ndarray, legacy_xyz: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """Fit ``legacy = rotation @ real + translation`` without scaling."""
    real_xyz = np.asarray(real_xyz, dtype=np.float64)
    legacy_xyz = np.asarray(legacy_xyz, dtype=np.float64)
    real_center = real_xyz.mean(axis=0)
    legacy_center = legacy_xyz.mean(axis=0)
    real_zero = real_xyz - real_center
    legacy_zero = legacy_xyz - legacy_center
    u, _, vt = np.linalg.svd(real_zero.T @ legacy_zero)
    rotation = vt.T @ u.T
    if np.linalg.det(rotation) < 0.0:
        vt[-1, :] *= -1.0
        rotation = vt.T @ u.T
    translation = legacy_center - rotation @ real_center
    aligned = real_xyz @ rotation.T + translation
    rmse = float(np.sqrt(np.mean(np.sum(np.square(aligned - legacy_xyz), axis=1))))
    return rotation, translation, rmse


def rotation_matrix_to_quaternion_xyzw(rotation: np.ndarray) -> np.ndarray:
    """Convert a proper rotation matrix to IsaacLab's XYZW quaternion."""
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
    """Multiply IsaacLab XYZW quaternions over the leading dimensions."""
    x1, y1, z1, w1 = np.moveaxis(np.asarray(first, dtype=np.float64), -1, 0)
    x2, y2, z2, w2 = np.moveaxis(np.asarray(second, dtype=np.float64), -1, 0)
    return np.stack(
        (
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        ),
        axis=-1,
    )


def align_real_pose(
    ee_pose: np.ndarray,
    h5_path: Path,
    mode: str,
    sim_fk_csv: Path | None = None,
) -> tuple[np.ndarray, dict[str, object]]:
    """Align the recorded real-frame pose with the original PPO Isaac frame.

    ``obs/state/ee_pose`` is XYZW in the real H5 contract.  The existing FK CSV is
    the old direct-joint replay frame and stores its quaternion columns as
    WXYZ; only its XYZ columns are used for the rigid calibration.  The
    continuous replay now keeps that same original PPO frame, so no additional
    Pi0 root rotation is applied.
    """
    pose = np.asarray(ee_pose, dtype=np.float64).copy()
    if pose.ndim != 2 or pose.shape[1] != 7:
        raise ValueError(f"EE pose must have shape (N,7), got {pose.shape}")
    pose[:, 3:] /= np.linalg.norm(pose[:, 3:], axis=1, keepdims=True).clip(min=1.0e-12)
    if mode == "raw":
        return pose.astype(np.float32), {
            "mode": "raw",
            "quaternion_order": "xyzw",
        }

    fk_path = sim_fk_csv or h5_path.parent / "ee_pose_sim_isaac6.csv"
    if not fk_path.is_file():
        raise FileNotFoundError(
            f"Pose alignment requires the existing FK file {fk_path}; "
            "use --pose-alignment raw only for frame debugging."
        )
    legacy_fk = np.loadtxt(fk_path, delimiter=",", skiprows=1, dtype=np.float64, ndmin=2)
    if legacy_fk.shape != pose.shape or not np.isfinite(legacy_fk).all():
        raise ValueError(f"Expected finite FK shape {pose.shape}, got {legacy_fk.shape}: {fk_path}")

    real_to_legacy_rotation, real_to_legacy_translation, fit_rmse = fit_real_to_legacy(
        pose[:, :3], legacy_fk[:, :3]
    )
    pose[:, :3] = pose[:, :3] @ real_to_legacy_rotation.T + real_to_legacy_translation
    frame_quaternion = rotation_matrix_to_quaternion_xyzw(real_to_legacy_rotation)
    pose[:, 3:] = quaternion_multiply_xyzw(frame_quaternion, pose[:, 3:])
    pose[:, 3:] /= np.linalg.norm(pose[:, 3:], axis=1, keepdims=True).clip(min=1.0e-12)
    return pose.astype(np.float32), {
        "mode": "existing_trajectory_fk",
        "quaternion_order": "xyzw",
        "fk_path": str(fk_path.resolve()),
        "fk_source_quaternion_order": "wxyz",
        "real_to_legacy_rotation": real_to_legacy_rotation.tolist(),
        "real_to_legacy_translation_m": real_to_legacy_translation.tolist(),
        "target_frame": "original RealSim/PPO frame",
        "frame_quaternion_xyzw": frame_quaternion.tolist(),
        "fk_position_fit_rmse_m": fit_rmse,
    }

def write_video(path: Path, frames: list[np.ndarray], fps: float) -> None:
    # Use the same FFmpeg/libx264 writer as the PPO/RealSim pipeline.
    # OpenCV's mp4v output is MPEG-4 Part 2, not H.264.
    _write_h264_mp4(path, frames, fps)

def wrench_row(env: RealSimEnv, attribute: str) -> np.ndarray:
    """Read one already-computed PPO wrench stream without stepping physics."""
    value = getattr(env, attribute, None)
    if value is None:
        raise RuntimeError(f"PPO wrench attribute is unavailable: {attribute}")
    row = value[0].detach().cpu().numpy().astype(np.float64, copy=False).reshape(-1)
    if row.shape != (6,) or not np.isfinite(row).all():
        raise RuntimeError(f"PPO wrench {attribute} must be finite with shape (6,), got {row.shape}")
    return row.copy()


def incoming_wrench_all_bodies(env: RealSimEnv) -> np.ndarray:
    """Return the raw incoming-joint wrench for every articulation body."""

    view = getattr(env._robot, "root_physx_view", None)
    getter = getattr(view, "get_link_incoming_joint_force", None)
    if not callable(getter):
        raise RuntimeError("PhysX does not expose get_link_incoming_joint_force")
    values = getter()
    values = values.torch if hasattr(values, "torch") else values
    values = torch.as_tensor(values, dtype=torch.float32, device=env.device)
    if values.ndim != 3 or values.shape[0] != 1 or values.shape[2] != 6:
        raise RuntimeError(f"invalid all-body incoming wrench shape: {tuple(values.shape)}")
    if values.shape[1] != len(env._robot.body_names):
        raise RuntimeError("incoming wrench body count does not match articulation body names")
    return values[0].detach().cpu().numpy().astype(np.float64, copy=True)


def load_link7_frame_calibration(path: Path | None) -> dict[str, object] | None:
    """Load a passed directed-force calibration for panda_link7."""
    if path is None:
        return None
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(f"link7 frame calibration is missing: {path}")
    result = json.loads(path.read_text(encoding="utf-8"))
    if result.get("incoming_wrench_read_body") != "panda_link7":
        raise ValueError("link7 frame calibration was not measured at panda_link7")
    if result.get("force_application_body") != "panda_link7":
        raise ValueError("link7 frame calibration did not apply force at panda_link7")
    if not bool(result.get("summary", {}).get("passed", False)):
        raise RuntimeError("panda_link7 directed-force calibration did not pass")
    frame = result.get("frame_calibration")
    if not isinstance(frame, dict):
        raise ValueError("link7 frame calibration has no fitted frame transform")
    rotation = np.asarray(
        frame.get("child_joint_to_parent_rotation_row_vector"), dtype=np.float64
    )
    sign = float(frame.get("raw_global_sign"))
    if rotation.shape != (3, 3) or not np.isfinite(rotation).all():
        raise ValueError("link7 frame calibration rotation must be finite 3x3")
    if sign not in (-1.0, 1.0):
        raise ValueError("link7 frame calibration sign must be +/-1")
    determinant = float(np.linalg.det(rotation))
    if abs(determinant - 1.0) > 1.0e-3:
        raise ValueError(f"link7 frame calibration is not a proper rotation: det={determinant}")
    return {
        "path": str(path),
        "raw_global_sign": sign,
        "child_joint_to_link7_rotation_row": rotation,
        "fit_rmse_n": float(frame.get("fit_rmse_n", np.nan)),
        "validation_rmse_n": float(frame.get("validation_rmse_n", np.nan)),
    }


def link7_incoming_wrench_base_at_k(
    env: RealSimEnv,
    wrench_raw: np.ndarray,
    link7_body_index: int,
    calibration: dict[str, object],
) -> np.ndarray:
    """Transform calibrated link7 incoming wrench to Franka O_F_ext_hat_K semantics.

    The directed +/-XYZ test identifies the fixed child-joint-frame rotation
    and the action/reaction sign. Force and torque are then expressed in the
    robot base frame. Torque is transported from the panda_link7 origin to the
    current fingertip stiffness-frame K origin.
    """
    raw = torch.as_tensor(
        np.asarray(wrench_raw, dtype=np.float64),
        dtype=torch.float32,
        device=env.device,
    ).view(1, 6)
    rotation = torch.as_tensor(
        calibration["child_joint_to_link7_rotation_row"],
        dtype=torch.float32,
        device=env.device,
    ).view(3, 3)
    sign = float(calibration["raw_global_sign"])
    force_link7 = sign * torch.matmul(raw[:, :3], rotation)
    torque_link7 = sign * torch.matmul(raw[:, 3:], rotation)

    body_pos_w = env._robot.data.body_pos_w
    body_quat_w = env._robot.data.body_quat_w
    root_quat_w = env._robot.data.root_quat_w
    body_pos_w = body_pos_w.torch if hasattr(body_pos_w, "torch") else body_pos_w
    body_quat_w = body_quat_w.torch if hasattr(body_quat_w, "torch") else body_quat_w
    root_quat_w = root_quat_w.torch if hasattr(root_quat_w, "torch") else root_quat_w
    link7_pos_w = torch.as_tensor(
        body_pos_w[:, link7_body_index], dtype=torch.float32, device=env.device
    )
    link7_quat_w = torch.as_tensor(
        body_quat_w[:, link7_body_index], dtype=torch.float32, device=env.device
    )
    root_quat_w = torch.as_tensor(root_quat_w, dtype=torch.float32, device=env.device)
    k_pos_w = torch.as_tensor(
        env.fingertip_midpoint_pos, dtype=torch.float32, device=env.device
    )

    force_w = torch_utils.quat_apply(link7_quat_w, force_link7)
    torque_w_link7 = torch_utils.quat_apply(link7_quat_w, torque_link7)
    torque_w_k = torque_w_link7 + torch.cross(link7_pos_w - k_pos_w, force_w, dim=-1)
    world_to_base = torch_utils.quat_conjugate(root_quat_w)
    force_base = torch_utils.quat_apply(world_to_base, force_w)
    torque_base_k = torch_utils.quat_apply(world_to_base, torque_w_k)
    result = torch.cat((force_base, torque_base_k), dim=-1)
    if not torch.isfinite(result).all():
        raise FloatingPointError("transformed panda_link7 wrench contains NaN or Inf")
    return result[0].detach().cpu().numpy().astype(np.float64, copy=True)


def articulation_load_path_topology(env: RealSimEnv) -> dict[str, object]:
    """Inspect whether force_sensor is in series with the gripper load path."""

    root_paths = list(env._robot.root_physx_view.prim_paths)
    if len(root_paths) != 1:
        raise RuntimeError(f"expected one robot articulation root, got {root_paths}")
    root_path = str(root_paths[0])
    root_prim = env.sim.stage.GetPrimAtPath(root_path)
    if not root_prim.IsValid():
        raise RuntimeError(f"robot articulation prim is missing: {root_path}")
    # Isaac Sim 6 may expose the articulation view at its root fixed joint.
    # Traverse the owning robot prim so all authored joints are inspected.
    if root_prim.IsA(UsdPhysics.Joint):
        root_prim = root_prim.GetParent()
        root_path = str(root_prim.GetPath())

    body_names = list(env._robot.body_names)

    def body_name(path: Sdf.Path) -> str | None:
        text = str(path)
        # Fixed links such as panda_hand may be collapsed into panda_link7 by
        # PhysX while their authored prim remains nested below it. Map an
        # authored path to the deepest surviving articulation body segment.
        matches = [
            (text.rfind(f"/{name}"), name)
            for name in body_names
            if f"/{name}" in text
        ]
        return max(matches)[1] if matches else None

    joints = []
    children: dict[str, list[str]] = {name: [] for name in body_names}
    for prim in Usd.PrimRange(root_prim):
        if not prim.IsA(UsdPhysics.Joint):
            continue
        joint = UsdPhysics.Joint(prim)
        body0_targets = joint.GetBody0Rel().GetTargets()
        body1_targets = joint.GetBody1Rel().GetTargets()
        body0_path = str(body0_targets[0]) if body0_targets else None
        body1_path = str(body1_targets[0]) if body1_targets else None
        parent = body_name(body0_targets[0]) if body0_targets else None
        child = body_name(body1_targets[0]) if body1_targets else None
        if parent is not None and child is not None and parent != child:
            children[parent].append(child)
        joints.append(
            {
                "joint": str(prim.GetPath()),
                "type": prim.GetTypeName(),
                "body0_path": body0_path,
                "body1_path": body1_path,
                "parent_body": parent,
                "child_body": child,
            }
        )

    def descendants(start: str) -> list[str]:
        found = []
        pending = list(children.get(start, []))
        while pending:
            current = pending.pop(0)
            if current in found:
                continue
            found.append(current)
            pending.extend(children.get(current, []))
        return found

    gripper_load_bodies = [
        name
        for name in ("panda_leftfinger", "panda_rightfinger", "panda_fingertip_centered")
        if name in body_names
    ]
    sensor_descendants = descendants("force_sensor") if "force_sensor" in body_names else []
    sensor_in_series = bool(gripper_load_bodies) and all(
        body in sensor_descendants for body in gripper_load_bodies
    )

    wrist_candidates = []
    for name in ("panda_hand", "panda_link7", "force_sensor"):
        if name not in body_names:
            continue
        downstream = descendants(name)
        wrist_candidates.append(
            {
                "body": name,
                "descendants": downstream,
                "contains_all_gripper_load_bodies": bool(gripper_load_bodies)
                and all(body in downstream for body in gripper_load_bodies),
            }
        )
    recommended = next(
        (
            item["body"]
            for item in wrist_candidates
            if item["contains_all_gripper_load_bodies"]
        ),
        None,
    )
    return {
        "robot_root": root_path,
        "body_names": body_names,
        "gripper_load_bodies": gripper_load_bodies,
        "force_sensor_descendants": sensor_descendants,
        "force_sensor_is_series_ancestor_of_gripper": sensor_in_series,
        "recommended_topological_wrist_body": recommended,
        "wrist_candidates": wrist_candidates,
        "joint_edges": joints,
        "interpretation": (
            "force_sensor can represent gripper load only if it is an ancestor of both fingers; "
            "otherwise read the incoming wrench at the nearest common wrist ancestor"
        ),
    }


def incoming_load_path_report(
    body_names: list[str],
    values: np.ndarray,
    contact_force_base: np.ndarray,
    baseline_frames: int = 30,
) -> dict[str, object]:
    """Rank articulation joints by their response to peg-hole contact."""

    baseline_frames = min(max(1, baseline_frames), len(values))
    corrected = values - np.median(values[:baseline_frames], axis=0, keepdims=True)
    body_force_norm = np.linalg.norm(corrected[:, :, :3], axis=2)
    contact_norm = np.linalg.norm(contact_force_base, axis=1)
    contact_mask = contact_norm > 1.0e-4
    free_mask = ~contact_mask

    def corr(first: np.ndarray, second: np.ndarray) -> float | None:
        if len(first) < 2 or np.std(first) < 1.0e-12 or np.std(second) < 1.0e-12:
            return None
        value = float(np.corrcoef(first, second)[0, 1])
        return value if np.isfinite(value) else None

    rows = []
    contact_p95 = float(np.percentile(contact_norm[contact_mask], 95)) if np.any(contact_mask) else 0.0
    for index, name in enumerate(body_names):
        norms = body_force_norm[:, index]
        free_p95 = float(np.percentile(norms[free_mask], 95)) if np.any(free_mask) else 0.0
        body_contact_p95 = (
            float(np.percentile(norms[contact_mask], 95)) if np.any(contact_mask) else 0.0
        )
        correlation = corr(contact_norm, norms)
        response_delta = max(0.0, body_contact_p95 - free_p95)
        score = response_delta * max(0.0, correlation or 0.0)
        rows.append(
            {
                "body": name,
                "body_index": index,
                "free_force_p95_n": free_p95,
                "contact_force_p95_n": body_contact_p95,
                "contact_minus_free_p95_n": response_delta,
                "correlation_with_peg_hole_contact": correlation,
                "transmission_ratio_vs_contact": (
                    body_contact_p95 / contact_p95 if contact_p95 > 1.0e-12 else None
                ),
                "selection_score": score,
            }
        )
    rows.sort(key=lambda item: item["selection_score"], reverse=True)
    force_sensor = next((item for item in rows if item["body"] == "force_sensor"), None)
    recommended = rows[0] if rows else None
    force_sensor_valid = bool(
        force_sensor is not None
        and force_sensor["correlation_with_peg_hole_contact"] is not None
        and force_sensor["correlation_with_peg_hole_contact"] >= 0.3
        and force_sensor["transmission_ratio_vs_contact"] is not None
        and force_sensor["transmission_ratio_vs_contact"] >= 0.5
    )
    return {
        "baseline_frames": baseline_frames,
        "peg_hole_contact_frames": int(np.count_nonzero(contact_mask)),
        "peg_hole_contact_force_p95_n": contact_p95,
        "force_sensor_on_effective_load_path": force_sensor_valid,
        "recommended_incoming_wrench_body": None if recommended is None else recommended["body"],
        "recommended_body_metrics": recommended,
        "force_sensor_metrics": force_sensor,
        "all_body_metrics_ranked": rows,
    }


def _sensor_tensor(value) -> torch.Tensor:
    """Convert an IsaacLab sensor ProxyArray to a torch tensor."""
    value = value.torch if hasattr(value, "torch") else value
    return torch.as_tensor(value, dtype=torch.float32)


def _contact_array(value) -> np.ndarray:
    """Convert PhysX tensor-view output (NumPy/Torch/Warp) to NumPy."""
    if hasattr(value, "numpy"):
        value = value.numpy()
    elif hasattr(value, "torch"):
        value = value.torch
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=np.float64)


class PhysxContactProbe:
    """Direct IsaacLab PhysX contact view for the peg-hole pair."""

    def __init__(self, env: RealSimEnv) -> None:
        def contact_report_paths(root_path: str) -> list[str]:
            root = env.sim.stage.GetPrimAtPath(root_path)
            return [
                str(prim.GetPath())
                for prim in Usd.PrimRange(root)
                if "PhysxContactReportAPI" in prim.GetAppliedSchemas()
            ]

        held_paths = contact_report_paths("/World/envs/env_0/HeldAsset")
        fixed_paths = contact_report_paths("/World/envs/env_0/FixedAsset")
        robot_paths = contact_report_paths("/World/envs/env_0/franka_env/Robot/franka")
        if not held_paths or not fixed_paths:
            raise RuntimeError(
                "Could not find PhysxContactReportAPI bodies for peg/hole: "
                f"held={held_paths}, fixed={fixed_paths}"
            )
        physics_view = env.sim.physics_manager.get_physics_sim_view()
        if physics_view is None:
            raise RuntimeError("IsaacLab PhysX simulation view is unavailable")
        self._contact_view = physics_view.create_rigid_contact_view(
            held_paths,
            filter_patterns=[fixed_paths for _ in held_paths],
            max_contact_data_count=256,
        )
        self._net_contact_view = physics_view.create_rigid_contact_view(
            held_paths,
            max_contact_data_count=0,
        )
        self._robot_hole_view = physics_view.create_rigid_contact_view(
            robot_paths,
            filter_patterns=[fixed_paths for _ in robot_paths],
            max_contact_data_count=256,
        )
        self._debug_reported = False
        print(
            f"[ReplayForce] direct PhysX contact view ready: "
            f"held={held_paths}, fixed={fixed_paths}, robot_bodies={len(robot_paths)}",
            flush=True,
        )

    def get_contact_force_matrix(self, dt: float):
        return self._contact_view.get_contact_force_matrix(dt=dt)

    def get_friction_data(self, dt: float):
        return self._contact_view.get_friction_data(dt=dt)

    def get_net_contact_forces(self, dt: float):
        return self._contact_view.get_net_contact_forces(dt=dt)

    def get_unfiltered_net_contact_forces(self, dt: float):
        return self._net_contact_view.get_net_contact_forces(dt=dt)

    def get_robot_hole_contact_force_matrix(self, dt: float):
        return self._robot_hole_view.get_contact_force_matrix(dt=dt)


def contact_pair_rows(env: RealSimEnv, contact_probe: PhysxContactProbe) -> dict[str, np.ndarray]:
    """Read filtered peg-hole normal/friction forces in world and robot-base frames.

    The ContactSensor is deliberately diagnostic-only.  The replay controller
    continues to use the existing PPO wrench stream, so adding this sensor
    cannot alter IK targets or joint commands.
    """
    contact_dt = float(env.cfg.sim.dt)
    force_matrix = _contact_array(contact_probe.get_contact_force_matrix(dt=contact_dt))
    net_forces = _contact_array(contact_probe.get_net_contact_forces(dt=contact_dt))
    unfiltered_net_forces = _contact_array(contact_probe.get_unfiltered_net_contact_forces(dt=contact_dt))
    robot_hole_matrix = _contact_array(contact_probe.get_robot_hole_contact_force_matrix(dt=contact_dt))
    friction_forces, _friction_points, friction_counts, _friction_starts = contact_probe.get_friction_data(
        dt=contact_dt
    )
    friction_forces = _contact_array(friction_forces)
    friction_counts = _contact_array(friction_counts).reshape(-1)
    friction_count = int(np.maximum(friction_counts, 0).sum())
    if not contact_probe._debug_reported:
        print(
            f"[ReplayForce] contact debug: matrix_shape={force_matrix.shape}, "
            f"net_shape={net_forces.shape}, friction_shape={friction_forces.shape}, "
            f"friction_counts={friction_counts.tolist()}, "
            f"matrix_max={float(np.abs(force_matrix).max(initial=0.0))}, "
            f"net_max={float(np.abs(net_forces).max(initial=0.0))}, "
            f"unfiltered_net_max={float(np.abs(unfiltered_net_forces).max(initial=0.0))}, "
            f"robot_hole_matrix_shape={robot_hole_matrix.shape}, "
            f"robot_hole_max={float(np.abs(robot_hole_matrix).max(initial=0.0))}",
            flush=True,
        )
        contact_probe._debug_reported = True
    # The probe has one filter (FixedAsset).  Sum all sensor-body/filter
    # entries for normal force and all valid friction anchors for that pair.
    normal_world_np = force_matrix[0].sum(axis=0) if force_matrix.ndim == 3 else force_matrix.sum(axis=0)
    friction_world_np = friction_forces[:friction_count].sum(axis=0) if friction_count else np.zeros(3)
    normal_world = torch.as_tensor(normal_world_np, dtype=torch.float32, device=env.device)
    friction_world = torch.as_tensor(friction_world_np, dtype=torch.float32, device=env.device)
    total_world = normal_world + friction_world

    root_quat_w = env._robot.data.root_quat_w[0]
    root_quat_w = root_quat_w.torch if hasattr(root_quat_w, "torch") else root_quat_w
    root_quat_w = torch.as_tensor(root_quat_w, dtype=torch.float32, device=env.device).view(1, 4)
    world_to_base = torch_utils.quat_conjugate(root_quat_w)

    def to_base(vector_world: torch.Tensor) -> torch.Tensor:
        return torch_utils.quat_apply(world_to_base, vector_world.view(1, 3))[0]

    values = {
        "normal_world": normal_world,
        "friction_world": friction_world,
        "total_world": total_world,
        "normal_base": to_base(normal_world),
        "friction_base": to_base(friction_world),
        "total_base": to_base(total_world),
    }
    return {
        key: value.detach().cpu().numpy().astype(np.float64, copy=True)
        for key, value in values.items()
    }

def write_matrix_csv(path: Path, header: list[str], rows: list[np.ndarray]) -> None:
    """Write one final numeric stream; no intermediate replay files are created."""
    values = np.asarray(rows, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != len(header):
        raise ValueError(f"CSV data/header shape mismatch for {path}: {values.shape} vs {len(header)}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(values.tolist())

def write_scalar_csv(path: Path, header: str, values: list[float]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow([header])
        writer.writerows([[float(value)] for value in values])

def write_asset_pose(env: RealSimEnv, asset, pos: torch.Tensor, quat: torch.Tensor) -> None:
    state = asset.data.root_state_w.clone()
    state[:, :3] = pos + env.scene.env_origins
    state[:, 3:7] = quat
    state[:, 7:] = 0.0
    asset.write_root_pose_to_sim(state[:, :7])
    asset.write_root_velocity_to_sim(state[:, 7:])
    asset.reset()

def set_asset_collision_enabled(env: RealSimEnv, asset, enabled: bool) -> int:
    """Toggle every USD collider below one spawned articulation root."""
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

def held_peg_pose(env: RealSimEnv) -> tuple[torch.Tensor, torch.Tensor]:
    # PPO 5.1 used WXYZ [0, 0, 1, 0] to flip the gripper Z direction
    # (180 deg about Y).  IsaacLab 6 frame math uses XYZW, so the equivalent
    # quaternion is [0, 1, 0, 0].  Copying the four numbers unchanged rotates
    # about Z instead and displaces the peg root by twice the grasp offset.
    flip_z_quat = torch.tensor([0.0, 1.0, 0.0, 0.0], device=env.device).view(1, 4)
    zero_pos = torch.zeros((1, 3), device=env.device)
    flipped_pos, flipped_quat = isaaclab_math.combine_frame_transforms(
        env.fingertip_midpoint_pos,
        env.fingertip_midpoint_quat,
        zero_pos,
        flip_z_quat,
    )
    relative_pos = torch.zeros((1, 3), device=env.device)
    relative_pos[:, 2] = (
        float(env.cfg.task.held_asset_cfg.height)
        - float(env.cfg.task.robot_cfg.franka_fingerpad_length)
    )
    identity = torch.tensor([0.0, 0.0, 0.0, 1.0], device=env.device).view(1, 4)
    held_pos, held_quat = isaaclab_math.combine_frame_transforms(
        flipped_pos,
        flipped_quat,
        -relative_pos,
        identity,
    )
    return held_quat, held_pos

def ppo_hand_down_quat(env: RealSimEnv) -> torch.Tensor:
    euler = torch.as_tensor(env.cfg.task.hand_init_orn, dtype=torch.float32, device=env.device)
    euler = euler.view(1, 3)
    return isaaclab_math.quat_from_euler_xyz(euler[:, 0], euler[:, 1], euler[:, 2])


def pose_error_xyzw(
    current_pos: torch.Tensor,
    current_quat: torch.Tensor,
    target_pos: torch.Tensor,
    target_quat: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute PPO's pose error using IsaacLab's XYZW convention."""
    pos_error = target_pos - current_pos
    quat_dot = (target_quat * current_quat).sum(dim=1, keepdim=True)
    target_quat = torch.where(quat_dot >= 0.0, target_quat, -target_quat)
    quat_error = isaaclab_math.quat_mul(
        target_quat, isaaclab_math.quat_conjugate(current_quat)
    )
    quat_error = torch.where(quat_error[:, 3:4] >= 0.0, quat_error, -quat_error)
    rot_error = isaaclab_math.axis_angle_from_quat(quat_error)
    return pos_error, rot_error


def move_ee_with_ppo_ik(
    env: RealSimEnv,
    xyz: np.ndarray,
    quat: torch.Tensor,
    seed_q: np.ndarray | None = None,
) -> float:
    target_pos = torch.as_tensor(xyz, dtype=torch.float32, device=env.device).view(1, 3)
    target_quat = quat.view(1, 4)
    if seed_q is not None:
        # Select the measured-trajectory IK branch before solving.  The seed
        # is not the output: DLS still solves the 6-D pose target below.
        seed_robot_for_fk(env, seed_q)
    for _ in range(80):
        env._compute_intermediate_values(dt=env.physics_dt)
        pos_error, rot_error = pose_error_xyzw(
            current_pos=env.fingertip_midpoint_pos,
            current_quat=env.fingertip_midpoint_quat,
            target_pos=target_pos,
            target_quat=target_quat,
        )
        if torch.linalg.norm(pos_error, dim=1).max() < 1e-4 and torch.linalg.norm(rot_error, dim=1).max() < 1e-4:
            break
        delta_dof_pos = factory_control.get_delta_dof_pos(
            delta_pose=torch.cat((pos_error, rot_error), dim=-1),
            ik_method="dls",
            jacobian=env.fingertip_midpoint_jacobian,
            device=env.device,
        )
        env.joint_pos[:, :7] += delta_dof_pos[:, :7]
        lower, upper = env._joint_position_limits()
        env.joint_pos[:, :7] = torch.clamp(env.joint_pos[:, :7], lower, upper)
        env.joint_vel.zero_()
        env.ctrl_target_joint_pos[:, :7] = env.joint_pos[:, :7]
        env._robot.write_joint_state_to_sim(env.joint_pos, env.joint_vel)
        env._robot.set_joint_position_target(env.ctrl_target_joint_pos)
        env.scene.write_data_to_sim()
        env.sim.forward()
        env.scene.update(dt=env.physics_dt)
    env._compute_intermediate_values(dt=env.physics_dt)
    return float(torch.linalg.norm(pos_error, dim=1).max().detach().cpu()) + float(
        torch.linalg.norm(rot_error, dim=1).max().detach().cpu()
    )


def seed_robot_for_fk(env: RealSimEnv, seed_q: np.ndarray) -> torch.Tensor:
    """Write one measured joint state and return its current FK quaternion."""
    seed = torch.as_tensor(seed_q, dtype=torch.float32, device=env.device).view(1, 7)
    env.joint_pos[:, :7] = seed
    env.joint_vel.zero_()
    env.ctrl_target_joint_pos[:, :7] = seed
    env._robot.write_joint_state_to_sim(env.joint_pos, env.joint_vel)
    env._robot.set_joint_position_target(env.ctrl_target_joint_pos)
    env.scene.write_data_to_sim()
    env.sim.forward()
    env.scene.update(dt=env.physics_dt)
    env._compute_intermediate_values(dt=env.physics_dt)
    return env.fingertip_midpoint_quat.clone()

def _interpolate_pose_xyzw(
    start_pos: torch.Tensor,
    start_quat: torch.Tensor,
    target_pos: torch.Tensor,
    target_quat: torch.Tensor,
    alpha: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Linearly interpolate position and shortest-arc normalized quaternion."""
    target_quat = target_quat.clone()
    if bool(torch.any((start_quat * target_quat).sum(dim=1, keepdim=True) < 0.0)):
        target_quat = torch.where(
            (start_quat * target_quat).sum(dim=1, keepdim=True) >= 0.0,
            target_quat,
            -target_quat,
        )
    quat = torch.nn.functional.normalize(torch.lerp(start_quat, target_quat, alpha), dim=1)
    return torch.lerp(start_pos, target_pos, alpha), quat


def step_ppo_physics(
    env: PPOReplayEnv,
    target_pos: torch.Tensor,
    target_quat: torch.Tensor,
    duration_s: float,
    incoming_body_index: int | None = None,
    incoming_frame_calibration: dict[str, object] | None = None,
) -> dict[str, np.ndarray] | None:
    """Advance PhysX with 120 Hz interpolated PPO task-space targets."""
    if duration_s < 0.0:
        raise ValueError("Physics duration cannot be negative")
    target_pos = target_pos.view(1, 3).to(device=env.device, dtype=torch.float32)
    target_quat = torch.nn.functional.normalize(
        target_quat.view(1, 4).to(device=env.device, dtype=torch.float32), dim=1
    )
    start_pos = env.fingertip_midpoint_pos.clone()
    start_quat = torch.nn.functional.normalize(env.fingertip_midpoint_quat.clone(), dim=1)
    step_count = max(1, int(round(duration_s / env.physics_dt)))
    incoming_samples = [] if incoming_body_index is not None else None
    incoming_base_at_k_samples = (
        []
        if incoming_body_index is not None and incoming_frame_calibration is not None
        else None
    )
    for step_index in range(step_count):
        alpha = float(step_index + 1) / float(step_count)
        env._replay_target_pos, env._replay_target_quat = _interpolate_pose_xyzw(
            start_pos,
            start_quat,
            target_pos,
            target_quat,
            alpha,
        )
        env._apply_action()
        env.scene.write_data_to_sim()
        env.sim.step(render=False)
        env.scene.update(dt=env.physics_dt)
        if incoming_samples is not None:
            raw = incoming_wrench_all_bodies(env)[incoming_body_index]
            incoming_samples.append(raw)
            if incoming_base_at_k_samples is not None:
                incoming_base_at_k_samples.append(
                    link7_incoming_wrench_base_at_k(
                        env,
                        raw,
                        incoming_body_index,
                        incoming_frame_calibration,
                    )
                )
    env._replay_target_pos = target_pos
    env._replay_target_quat = target_quat
    env._compute_intermediate_values(dt=env.physics_dt)
    if incoming_samples is None:
        return None
    streams = {"native": np.asarray(incoming_samples, dtype=np.float64)}
    if incoming_base_at_k_samples is not None:
        streams["base_at_K"] = np.asarray(incoming_base_at_k_samples, dtype=np.float64)
    return streams


def causal_lowpass(values: np.ndarray, sample_dt: float, cutoff_hz: float) -> tuple[np.ndarray, float]:
    """Apply a first-order causal low-pass to a finite NxD stream."""

    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 2 or not len(values):
        raise ValueError(f"low-pass input must be non-empty NxD, got {values.shape}")
    if sample_dt <= 0.0 or cutoff_hz <= 0.0:
        raise ValueError("sample_dt and cutoff_hz must be positive")
    alpha = 1.0 - np.exp(-2.0 * np.pi * cutoff_hz * sample_dt)
    filtered = np.empty_like(values)
    filtered[0] = values[0]
    for index in range(1, len(values)):
        filtered[index] = alpha * values[index] + (1.0 - alpha) * filtered[index - 1]
    return filtered, float(alpha)


def causal_median_filter(values: np.ndarray, window: int = 3) -> np.ndarray:
    """Remove isolated causal impulse samples without using future data."""

    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 2 or not len(values):
        raise ValueError(f"median-filter input must be non-empty NxD, got {values.shape}")
    if window <= 0 or window % 2 == 0:
        raise ValueError("median-filter window must be a positive odd integer")
    filtered = np.empty_like(values)
    for index in range(len(values)):
        start = max(0, index - window + 1)
        filtered[index] = np.median(values[start : index + 1], axis=0)
    return filtered


def sustained_threshold(values: np.ndarray, threshold: float, minimum_frames: int) -> np.ndarray:
    """Return a causal flag after a threshold remains active for N frames."""

    result = np.zeros(len(values), dtype=np.float64)
    run_length = 0
    for index, value in enumerate(values):
        run_length = run_length + 1 if value > threshold else 0
        result[index] = float(run_length >= minimum_frames)
    return result


def reset_ppo_velocity_history(
    env: PPOReplayEnv, *, write_zero_joint_velocity: bool = True
) -> None:
    """Remove FK/bootstrap jumps from PPO's finite-difference history."""
    env._compute_intermediate_values(dt=env.physics_dt)
    if write_zero_joint_velocity:
        # This is allowed only during offline preparation, before the physical
        # grasp and recorded rollout begin.
        joint_pos = env.joint_pos.clone()
        joint_vel = torch.zeros_like(env.joint_vel)
        env._robot.write_joint_state_to_sim(joint_pos, joint_vel)
        env.scene.write_data_to_sim()
        env.sim.forward()
        env.scene.update(dt=env.physics_dt)
        env._compute_intermediate_values(dt=env.physics_dt)
    env.prev_fingertip_pos = env.fingertip_midpoint_pos.clone()
    env.prev_fingertip_quat = env.fingertip_midpoint_quat.clone()
    env.prev_joint_pos = env.joint_pos[:, :7].clone()
    for name in ("ee_linvel_fd", "ee_angvel_fd", "joint_vel_fd"):
        value = getattr(env, name, None)
        if value is not None:
            value.zero_()


def establish_ppo_grasp(
    env: PPOReplayEnv,
    peg_pos: torch.Tensor,
    peg_quat: torch.Tensor,
    target_pos: torch.Tensor,
    target_quat: torch.Tensor,
) -> None:
    """Place the independent peg once, then reproduce PPO close/snap/settle."""
    write_asset_pose(env, env._held_asset, peg_pos, peg_quat)

    close_time_s = float(getattr(env.cfg.task, "grasp_close_time_s", 0.6))
    reset_gains = torch.as_tensor(
        env.cfg.ctrl.reset_task_prop_gains, dtype=torch.float32, device=env.device
    ).view(1, 6).repeat(env.num_envs, 1)
    env.task_prop_gains = reset_gains
    env.task_deriv_gains = factory_utils.get_deriv_gains(
        reset_gains, env.cfg.ctrl.reset_rot_deriv_scale
    )
    step_ppo_physics(env, target_pos, target_quat, close_time_s)

    # RealSim PPO uses this one post-close correction to remove reset noise.
    # This is the final explicit peg pose write; rollout frames are physical.
    if bool(getattr(env.cfg.task, "snap_held_asset_after_grasp", False)):
        snapped_quat, snapped_pos = held_peg_pose(env)
        write_asset_pose(env, env._held_asset, snapped_pos, snapped_quat)
        step_ppo_physics(env, target_pos, target_quat, env.physics_dt)

    settle_time_s = float(getattr(env.cfg.task, "grasp_settle_time_s", 0.2))
    if settle_time_s > 0.0:
        step_ppo_physics(env, target_pos, target_quat, settle_time_s)
    env.task_prop_gains = env.default_gains
    env.task_deriv_gains = factory_utils.get_deriv_gains(env.default_gains)
    # The physical grasp has already started. From this point onward, reset
    # only software histories and never write articulation state to PhysX.
    reset_ppo_velocity_history(env, write_zero_joint_velocity=False)
    print(
        "[ReplayDebug] post-grasp "
        f"ee={env.fingertip_midpoint_pos[0].detach().cpu().tolist()} "
        f"target={target_pos.view(-1).detach().cpu().tolist()} "
        f"q={env.joint_pos[0, :7].detach().cpu().tolist()}",
        flush=True,
    )


def physical_pose_error(
    env: PPOReplayEnv, target_pos: torch.Tensor, target_quat: torch.Tensor
) -> float:
    pos_error, rot_error = pose_error_xyzw(
        env.fingertip_midpoint_pos,
        env.fingertip_midpoint_quat,
        target_pos.view(1, 3),
        target_quat.view(1, 4),
    )
    return float(torch.linalg.norm(pos_error, dim=1).max().detach().cpu()) + float(
        torch.linalg.norm(rot_error, dim=1).max().detach().cpu()
    )

def capture_front(env: RealSimEnv) -> np.ndarray:
    """Capture the PPO front camera without a replay-only visual pipeline."""
    front, _ = capture_ppo_cameras(env)
    return front


def capture_ppo_cameras(env: RealSimEnv) -> tuple[np.ndarray, np.ndarray]:
    """Capture exactly like the rollout that produced ``sim_force``."""
    if getattr(env.cfg, "replay_left_background_visual_shell", False):
        current_q = env.joint_pos[0, :7].detach().cpu().numpy()
        gripper_open = float(env.joint_pos[0, 7:9].mean().detach().cpu())
        env._pose_background_robot_visual(
            env.sim.stage,
            "/World/envs/env_0/franka_env/LeftRobotVisual/Robot/franka",
            current_q,
            gripper_open,
            log=False,
        )
    env.sim.render()
    env.tiled_camera.update(env.physics_dt, force_recompute=True)
    env.wrist_tiled_camera.update(env.physics_dt, force_recompute=True)
    front = env.tiled_camera.data.output["rgb"][0].detach().cpu().numpy().copy()
    wrist = env.wrist_tiled_camera.data.output["rgb"][0].detach().cpu().numpy().copy()

    front = _resize_pi0_rgb(front, env.cfg)
    front = _composite_pi0_green_table_front(front, env.cfg)
    front = _composite_pi0_green_table_front(
        front,
        env.cfg,
        polygon_attr="pi0_green_table_right_patch_image_polygon",
        enabled_attr="pi0_green_table_right_edge_front_repair_enabled",
        alpha_attr="pi0_green_table_right_edge_front_repair_alpha",
        apply_hole_cutout=False,
    )
    front = _composite_pi0_green_table_front(
        front,
        env.cfg,
        polygon_attr="pi0_green_table_border_image_polygon",
        enabled_attr="pi0_green_table_border_front_repair_enabled",
        alpha_attr="pi0_green_table_border_front_repair_alpha",
        apply_hole_cutout=False,
    )
    front = _composite_pi0_reference_cylinder(front, env.cfg, "rgb")
    front = _composite_pi0_cable_grommets(front, env.cfg, "rgb")
    front = match_sim_force_front_table(front, env.cfg)
    wrist = _resize_pi0_rgb(wrist, env.cfg)
    wrist = _match_pi0_wrist_appearance(wrist, env.cfg, "rgb")
    return (
        cv2.cvtColor(np.ascontiguousarray(front[:, :, :3]), cv2.COLOR_RGB2BGR),
        cv2.cvtColor(np.ascontiguousarray(wrist[:, :, :3]), cv2.COLOR_RGB2BGR),
    )


def match_sim_force_front_table(frame: np.ndarray, cfg) -> np.ndarray:
    """Match only green tabletop pixels to the old sim_force rollout."""
    image = np.ascontiguousarray(frame).copy()
    rgb = image[:, :, :3]
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)

    polygon = np.asarray(
        getattr(cfg, "pi0_green_table_overlay_image_polygon", ()),
        dtype=np.float32,
    )
    if polygon.ndim != 2 or polygon.shape[0] < 3 or polygon.shape[1] != 2:
        raise ValueError("pi0_green_table_overlay_image_polygon is invalid")
    scale_x = rgb.shape[1] / float(getattr(cfg, "pi0_visual_annotation_width", 640))
    scale_y = rgb.shape[0] / float(getattr(cfg, "pi0_visual_annotation_height", 480))
    polygon[:, 0] *= scale_x
    polygon[:, 1] *= scale_y
    polygon_mask = np.zeros(rgb.shape[:2], dtype=np.uint8)
    cv2.fillPoly(polygon_mask, [np.rint(polygon).astype(np.int32)], 255)

    # Orange peg is outside this hue range; the gray/brown hole is excluded
    # by saturation, so task-object colors are unchanged.
    table_mask = (
        (polygon_mask > 0)
        & (hsv[:, :, 0] >= 35)
        & (hsv[:, :, 0] <= 110)
        & (hsv[:, :, 1] >= 30)
        & (hsv[:, :, 2] >= 25)
    ).astype(np.float32)
    table_mask = cv2.GaussianBlur(table_mask, (0, 0), sigmaX=0.8)
    blend = np.clip(0.94 * table_mask, 0.0, 0.94)[:, :, None]
    target = np.asarray((24.0, 75.0, 65.0), dtype=np.float32).reshape(1, 1, 3)
    image[:, :, :3] = np.rint(
        rgb.astype(np.float32) * (1.0 - blend) + target * blend
    ).astype(np.uint8)
    return np.ascontiguousarray(image)


def camera_calibration_row(
    camera, output_width: int | None = None, output_height: int | None = None
) -> np.ndarray:
    """Return camera pose and intrinsics scaled to the saved video resolution."""
    def as_tensor(value):
        return value.torch if hasattr(value, "torch") else value

    pos = as_tensor(camera.data.pos_w)[0].detach().cpu().numpy().reshape(3)
    quat = as_tensor(camera.data.quat_w_opengl)[0].detach().cpu().numpy().reshape(4)
    intrinsic = (
        as_tensor(camera.data.intrinsic_matrices)[0]
        .detach()
        .cpu()
        .numpy()
        .reshape(3, 3)
        .copy()
    )
    if output_width is not None and output_height is not None:
        source_width = float(camera.cfg.width)
        source_height = float(camera.cfg.height)
        intrinsic[0, :] *= float(output_width) / source_width
        intrinsic[1, :] *= float(output_height) / source_height
    intrinsic = intrinsic.reshape(9)
    return np.concatenate((pos, quat, intrinsic)).astype(np.float64, copy=False)


def configure_aligned_visual_scene(cfg) -> None:
    """Port only the verified joint-rollout visual configuration."""
    visual_cfg = RealSimPi0PegInsertCfg()
    for name in dir(visual_cfg):
        if name.startswith("pi0_"):
            setattr(cfg, name, getattr(visual_cfg, name))

    cfg.override_held_asset_color = visual_cfg.override_held_asset_color
    cfg.held_asset_visual_color = visual_cfg.held_asset_visual_color
    cfg.override_fixed_asset_color = visual_cfg.override_fixed_asset_color
    cfg.fixed_asset_visual_color = visual_cfg.fixed_asset_visual_color
    for name in (
        "background_right_robot_offset_pos",
        "background_right_robot_joint_pos",
        "background_right_robot_gripper_open",
        "background_right_robot_endpoint_reference_offset_pos",
        "background_right_robot_endpoint_target_rot",
    ):
        setattr(cfg, name, getattr(visual_cfg, name))

    # Preserve the continuous PPO articulation and its controller/Jacobian
    # frame. Visual matching below must not change this scene composition.
    cfg.remove_background_robot = False
    cfg.background_robot_visual_only = False
    cfg.replay_left_background_visual_shell = False
    cfg.background_right_robot_visual_only = True
    cfg.background_fr3v2_right_robot_visual_only = False
    cfg.background_right_robot_copy_active_visual = False
    cfg.remove_authored_right_robot = True

    cfg.pi0_green_table_overlay_enabled = True
    cfg.pi0_green_table_overlay_z_offset = -0.005
    cfg.pi0_green_table_front_composite_enabled = False
    cfg.pi0_green_table_right_patch_enabled = False
    cfg.pi0_green_table_right_edge_front_repair_enabled = True
    cfg.pi0_green_table_right_edge_front_repair_alpha = 1.0
    cfg.pi0_green_table_right_edge_front_repair_soft_mask_enabled = True
    cfg.pi0_green_table_right_edge_front_repair_soft_hue_margin = 12.0
    cfg.pi0_green_table_right_edge_front_repair_soft_saturation_floor = 2.0
    cfg.pi0_green_table_right_edge_front_repair_soft_saturation_full = 24.0
    cfg.pi0_green_table_right_edge_front_repair_soft_blur_sigma = 8.0
    cfg.pi0_green_table_right_patch_image_polygon = (
        (540.0, 237.0),
        (625.0, 237.0),
        (680.0, 342.0),
        (680.0, 500.0),
        (540.0, 500.0),
    )
    cfg.pi0_green_table_right_patch_z_offset = 0.0018
    cfg.pi0_green_table_hole_cutout_xyxy = ()
    cfg.pi0_green_table_border_front_repair_enabled = True
    cfg.pi0_green_table_border_front_repair_alpha = 1.0
    cfg.pi0_green_table_border_front_repair_soft_mask_enabled = True
    cfg.pi0_green_table_border_front_repair_soft_hue_margin = 12.0
    cfg.pi0_green_table_border_front_repair_soft_saturation_floor = 2.0
    cfg.pi0_green_table_border_front_repair_soft_saturation_full = 24.0
    cfg.pi0_green_table_border_front_repair_soft_blur_sigma = 0.8
    cfg.pi0_green_table_border_front_repair_soft_close_kernel_size = 3
    cfg.pi0_green_table_border_image_polygon = (
        (70.0, 231.0),
        (613.0, 229.0),
        (619.0, 242.0),
        (72.0, 244.0),
    )
    cfg.pi0_white_table_overlay_enabled = True
    cfg.pi0_white_table_overlay_image_polygon = (
        (171.0, 105.0),
        (472.0, 103.0),
        (566.0, 250.0),
        (84.0, 250.0),
    )
    cfg.pi0_white_table_overlay_z_offset = -0.032
    # The default linear value (0.23) clips close to pure white under this
    # force-preserving scene's lighting.  Lower only the visual table material
    # to the light gray seen in the old sim_force rollout; camera exposure and
    # robot/object materials remain unchanged.
    cfg.pi0_white_table_overlay_color = (0.145, 0.145, 0.140)
    cfg.pi0_black_gripper_visual = False

    render_width = int(getattr(cfg, "pi0_camera_render_width", 1280))
    render_height = int(getattr(cfg, "pi0_camera_render_height", 960))
    cfg.tiled_camera.width = render_width
    cfg.tiled_camera.height = render_height
    cfg.wrist_camera.width = render_width
    cfg.wrist_camera.height = render_height
    if cfg.wrist_camera.spawn is not None:
        cfg.wrist_camera.spawn.focal_length = float(cfg.pi0_wrist_camera_focal_length)
        cfg.wrist_camera.spawn.vertical_aperture_offset = float(
            cfg.pi0_wrist_camera_vertical_aperture_offset
        )
        cfg.wrist_camera.spawn.clipping_range = tuple(cfg.pi0_wrist_camera_clipping_range)
    cfg.wrist_camera.prim_path = (
        "/World/envs/env_.*/franka_env/Robot/franka/"
        "panda_link7/panda_link8/panda_hand/wrist_camera"
    )
    cfg.wrist_camera.offset.pos = tuple(cfg.pi0_wrist_camera_offset_pos)
    cfg.wrist_camera.offset.rot = _pi0_wrist_camera_rotation(cfg)
    cfg.wrist_camera.offset.convention = "opengl"


def configure_ppo_scene(cfg) -> None:
    """Keep the original PPO scene and controller configuration."""
    cfg.enable_cameras = True
    # PPO's operational-space controller emits effort commands. Keep implicit
    # actuator gains disabled exactly as in the PPO environment.
    for actuator_name in ("panda_arm1", "panda_arm2"):
        actuator = cfg.robot.actuators.get(actuator_name)
        if actuator is not None:
            actuator.stiffness = 0.0
            actuator.damping = 0.0
            actuator.effort_limit_sim = 100.0


def main() -> None:
    q, ee_pose, gripper_width, timestamps = load_real_data(args.h5)
    link7_frame_calibration = load_link7_frame_calibration(
        args.link7_frame_calibration
    )
    sim_pose, alignment = align_real_pose(
        ee_pose,
        args.h5,
        args.pose_alignment,
        args.sim_fk_csv,
    )
    sim_xyz = sim_pose[:, :3]
    if args.endpoint_window < 1:
        raise ValueError("endpoint-window must be positive")
    window = min(args.endpoint_window, len(sim_xyz))

    # Match the PPO/old replay bootstrap.  The scene and camera sensors were
    # authored for the Fabric path; the non-Fabric path can leave the physics
    # scene stepping subscription invalid during startup.
    cfg = parse_env_cfg(args.task, device=args.device, num_envs=1, use_fabric=True)
    print("[ReplayStage] configuration parsed", flush=True)
    cfg.scene.num_envs = 1
    cfg.policy_cfg = None
    cfg.teacher_policy_cfg = None
    cfg.enable_cameras = True
    cfg.data_collect_cfg["collect_data"] = False
    cfg.data_collect_cfg["immediate_stop"] = False
    cfg.data_collect_cfg["save_failed_trajectory"] = False
    cfg.data_collect_cfg["num_trajectories"] = 1_000_000
    cfg.teacher_eval_only = True
    cfg.episode_length_s = max(30.0, float(timestamps[-1] - timestamps[0] + 2.0))
    # Warm-start DLS from the same joint branch used to generate the simulator
    # FK targets.  Starting from the PPO default posture can converge to a
    # different redundant Franka IK branch while still reporting a small
    # Cartesian error, which makes the rendered arm follow the wrong motion.
    cfg.ctrl.reset_joints = q[0].tolist()
    configure_ppo_scene(cfg)
    configure_aligned_visual_scene(cfg)
    print("[ReplayStage] PPO physics and aligned visual scene configured", flush=True)
    # Use the task's original visible independent HeldAsset. It already has
    # PPO's rigid-body, mass, collision and contact-sensor configuration.
    cfg.task.held_asset.spawn.visible = True
    cfg.task.fixed_asset_init_pos_noise = [0.0, 0.0, 0.0]
    cfg.task.hand_init_pos_noise = [0.0, 0.0, 0.0]
    cfg.task.hand_init_orn_noise = [0.0, 0.0, 0.0]
    cfg.task.held_asset_pos_noise = [0.0, 0.0, 0.0]
    cfg.task.fixed_asset_init_orn_range_deg = 0.0
    # Keep the joint-replay Pi0 wrist calibration as the base and preserve
    # optional per-run local corrections from this Cartesian entry point.
    wrist_base_quat = torch.as_tensor(
        cfg.wrist_camera.offset.rot, dtype=torch.float32
    ).view(1, 4)
    wrist_pitch = torch.tensor([np.deg2rad(args.wrist_pitch_deg)], dtype=torch.float32)
    wrist_yaw = torch.tensor([np.deg2rad(args.wrist_yaw_deg)], dtype=torch.float32)
    wrist_zero = torch.zeros_like(wrist_pitch)
    wrist_correction_quat = isaaclab_math.quat_from_euler_xyz(
        wrist_pitch, wrist_yaw, wrist_zero
    )
    wrist_quat = isaaclab_math.quat_mul(wrist_base_quat, wrist_correction_quat)[0]
    wrist_base_pos = torch.as_tensor(
        cfg.wrist_camera.offset.pos, dtype=torch.float32
    ).view(1, 3)
    wrist_forward_local = isaaclab_math.quat_apply(
        wrist_quat.view(1, 4), torch.tensor([[0.0, 0.0, -1.0]], dtype=torch.float32)
    )
    wrist_pos = wrist_base_pos + float(args.wrist_forward_offset) * wrist_forward_local
    cfg.wrist_camera.offset.pos = tuple(float(value) for value in wrist_pos[0])
    cfg.wrist_camera.offset.rot = tuple(float(value) for value in wrist_quat)
    cfg.sim.render_interval = 4

    args.output_dir.mkdir(parents=True, exist_ok=True)
    previous_background_usd = os.environ.get("TACEX_BACKGROUND_USD")
    os.environ["TACEX_BACKGROUND_USD"] = str(_VISUAL_BACKGROUND_USD)
    print("[ReplayStage] constructing continuous PPO environment", flush=True)
    try:
        env = PPOReplayEnv(cfg, render_mode="rgb_array", output_dir=str(args.output_dir))
    finally:
        if previous_background_usd is None:
            os.environ.pop("TACEX_BACKGROUND_USD", None)
        else:
            os.environ["TACEX_BACKGROUND_USD"] = previous_background_usd
    print("[ReplayStage] continuous PPO environment constructed", flush=True)
    try:
        env.prepare_replay_state()
        topology_report = articulation_load_path_topology(env)
        (args.output_dir / "articulation_load_path_topology.json").write_text(
            json.dumps(topology_report, indent=2), encoding="utf-8"
        )
        print(
            "[LoadPathTopology] "
            f"force_sensor_in_series={topology_report['force_sensor_is_series_ancestor_of_gripper']} "
            f"recommended={topology_report['recommended_topological_wrist_body']}",
            flush=True,
        )
        contact_probe = None
        if args.record_contact_pair:
            # Create the low-level PhysX contact view only after IsaacLab has
            # started the simulation.  Both assets already enable contact
            # reporting in their USD spawners, so no scene or collider is
            # changed here.
            contact_probe = PhysxContactProbe(env)
        env.sim.forward()
        print("[ReplayMain] prepared", flush=True)
        print("[ReplaySim] playing=", env.sim.is_playing(), "stopped=", env.sim.is_stopped(), flush=True)
        # Solve the complete IK trajectory before grasping to select the
        # correct Franka branch and infer ideal peg roots.  Position targets
        # come from the calibrated recorded EE pose; orientation targets come
        # from FK of the same recorded joints.  This preserves the exact
        # visual branch of joint replay while letting PPO control the motion
        # continuously in task space.
        ik_errors = []
        ideal_peg_roots = []
        target_quats = []
        for index, (xyz, seed_q) in enumerate(zip(sim_xyz, q)):
            seed_fk_quat = seed_robot_for_fk(env, seed_q)
            target_quat = seed_fk_quat[0].detach().clone()
            target_quats.append(target_quat)
            error = move_ee_with_ppo_ik(env, xyz, target_quat.view(1, 4), seed_q)
            ik_errors.append(error)
            _, ideal_peg_pos = held_peg_pose(env)
            ideal_peg_roots.append(ideal_peg_pos[0].detach().cpu().numpy())
            if index == 0 or index + 1 == len(sim_xyz) or (index + 1) % 50 == 0:
                print(
                    f"[ReplayIK] target {index + 1}/{len(sim_xyz)} error={error:.6f}",
                    flush=True,
                )
        init_error = ik_errors[0]
        target_quats = torch.stack(target_quats, dim=0)

        # The hole remains the PPO FixedAsset. Its USD contains a fixed joint,
        # so this one reset-time pose write does not make it kinematic.
        hole_pos = np.median(np.asarray(ideal_peg_roots[-window:]), axis=0)
        hole_pos += np.asarray(
            [args.hole_offset_x, args.hole_offset_y, args.hole_offset_z], dtype=np.float32
        )
        # Keep the original PPO fixed-asset orientation unless explicitly
        # overridden for a diagnostic run.
        if args.hole_yaw_deg is None:
            hole_quat = torch.as_tensor(
                cfg.task.fixed_asset.init_state.rot,
                dtype=torch.float32,
                device=env.device,
            ).view(1, 4)
        else:
            hole_yaw = torch.tensor(
                [np.deg2rad(args.hole_yaw_deg)], dtype=torch.float32, device=env.device
            )
            hole_quat = isaaclab_math.quat_from_euler_xyz(
                torch.zeros_like(hole_yaw), torch.zeros_like(hole_yaw), hole_yaw
            )
        hole_pos_t = torch.as_tensor(hole_pos, dtype=torch.float32, device=env.device).view(1, 3)
        write_asset_pose(env, env._fixed_asset, hole_pos_t, hole_quat)
        cfg.task.fixed_asset.init_state.pos = tuple(float(value) for value in hole_pos)
        cfg.task.fixed_asset.init_state.rot = tuple(float(value) for value in hole_quat[0].detach().cpu().tolist())
        # The old rollout already authors the physical hole at the calibrated
        # -30 degree yaw, so render that same mesh directly. A second visual
        # proxy would rotate it twice and can drift from the contact geometry.
        visual_hole_proxy_path = (
            "/World/envs/env_0/FixedAsset/forge_hole_8mm/forge_hole_8mm/mesh"
        )
        if args.disable_hole_collision:
            collider_count = set_asset_collision_enabled(env, env._fixed_asset, False)
            print(f"[ReplayForce] disabled {collider_count} hole colliders", flush=True)

        # Return to the first target and reproduce PPO's physical grasp:
        # independent HeldAsset, finger target zero, close/snap/settle under
        # PhysX. The peg pose is never explicitly written during rollout.
        move_ee_with_ppo_ik(env, sim_xyz[0], target_quats[0:1], q[0])
        peg_quat, peg_pos = held_peg_pose(env)
        reset_ppo_velocity_history(env)
        establish_ppo_grasp(
            env,
            peg_pos,
            peg_quat,
            torch.as_tensor(sim_xyz[0], dtype=torch.float32, device=env.device),
            target_quats[0],
        )
        front_frames = []
        wrist_frames = []
        front_camera_calibration = []
        wrist_camera_calibration = []
        sim_ee = []
        replay_q = []
        held_poses = []
        contact_pair_streams = {
            key: []
            for key in (
                "normal_world",
                "friction_world",
                "total_world",
                "normal_base",
                "friction_base",
                "total_base",
            )
        }
        incoming_body_names = list(env._robot.body_names)
        if "panda_link7" not in incoming_body_names:
            raise RuntimeError(f"panda_link7 missing from articulation bodies: {incoming_body_names}")
        link7_body_index = incoming_body_names.index("panda_link7")
        incoming_body_stream = []
        link7_highrate_raw = []
        link7_highrate_base_at_k = []
        link7_highrate_times = []
        link7_frame_slices = []
        print(f"[LoadPath] articulation bodies={incoming_body_names}", flush=True)
        replay_timestamps = []
        physical_pose_errors = []
        for index, xyz in enumerate(sim_xyz):
            target_quat = target_quats[index].view(1, 4)
            interval_link7 = None
            if index > 0:
                frame_dt = float(timestamps[index] - timestamps[index - 1])
                interval_link7 = step_ppo_physics(
                    env,
                    torch.as_tensor(xyz, dtype=torch.float32, device=env.device),
                    target_quat,
                    frame_dt,
                    incoming_body_index=link7_body_index,
                    incoming_frame_calibration=link7_frame_calibration,
                )
            target_pos = torch.as_tensor(xyz, dtype=torch.float32, device=env.device)
            error = physical_pose_error(env, target_pos, target_quat)
            physical_pose_errors.append(error)
            incoming_all_bodies = incoming_wrench_all_bodies(env)
            incoming_body_stream.append(incoming_all_bodies)
            slice_start = len(link7_highrate_raw)
            if index == 0:
                first_link7_raw = incoming_all_bodies[link7_body_index]
                link7_highrate_raw.append(first_link7_raw)
                if link7_frame_calibration is not None:
                    link7_highrate_base_at_k.append(
                        link7_incoming_wrench_base_at_k(
                            env,
                            first_link7_raw,
                            link7_body_index,
                            link7_frame_calibration,
                        )
                    )
                link7_highrate_times.append(0.0)
            else:
                if interval_link7 is None or not len(interval_link7["native"]):
                    raise RuntimeError(f"no panda_link7 substep samples for frame {index}")
                link7_highrate_raw.extend(interval_link7["native"])
                if link7_frame_calibration is not None:
                    if "base_at_K" not in interval_link7:
                        raise RuntimeError("link7 base/K stream missing after frame calibration")
                    link7_highrate_base_at_k.extend(interval_link7["base_at_K"])
                interval_times = np.linspace(
                    float(timestamps[index - 1] - timestamps[0]),
                    float(timestamps[index] - timestamps[0]),
                    len(interval_link7["native"]) + 1,
                    dtype=np.float64,
                )[1:]
                link7_highrate_times.extend(interval_times.tolist())
            link7_frame_slices.append((slice_start, len(link7_highrate_raw)))
            if contact_probe is not None:
                contact_rows = contact_pair_rows(env, contact_probe)
                for key, value in contact_rows.items():
                    contact_pair_streams[key].append(value)
            replay_timestamps.append(float(timestamps[index] - timestamps[0]))
            front, wrist = capture_ppo_cameras(env)
            front_frames.append(front)
            wrist_frames.append(wrist)
            output_width = int(cfg.pi0_camera_output_width)
            output_height = int(cfg.pi0_camera_output_height)
            front_camera_calibration.append(
                camera_calibration_row(env.tiled_camera, output_width, output_height)
            )
            wrist_camera_calibration.append(
                camera_calibration_row(env.wrist_tiled_camera, output_width, output_height)
            )
            sim_ee.append(
                np.concatenate(
                    (
                        env.fingertip_midpoint_pos[0].detach().cpu().numpy(),
                        env.fingertip_midpoint_quat[0].detach().cpu().numpy(),
                    )
                )
            )
            replay_q.append(env.joint_pos[0, :7].detach().cpu().numpy())
            held_poses.append(
                np.concatenate(
                    (
                        env.held_pos[0].detach().cpu().numpy(),
                        env.held_quat[0].detach().cpu().numpy(),
                    )
                )
            )
            if index == 0 or index + 1 == len(sim_xyz) or (index + 1) % 10 == 0:
                print(f"[Replay] frame {index + 1}/{len(sim_xyz)} error={error:.6f}", flush=True)

        comparison_index = len(front_frames) // 2
        cv2.imwrite(
            str(args.output_dir / "front_workspace_matched.png"),
            front_frames[comparison_index],
        )
        write_video(args.output_dir / "front_camera.mp4", front_frames, args.fps)
        write_video(args.output_dir / "wrist_camera.mp4", wrist_frames, args.fps)
        with (args.output_dir / "ee_pose_sim.csv").open("w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["x", "y", "z", "qx", "qy", "qz", "qw"])
            writer.writerows(np.asarray(sim_ee).tolist())
        with (args.output_dir / "joint_pos_sim.csv").open("w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow([f"q{i}" for i in range(7)])
            writer.writerows(np.asarray(replay_q).tolist())
        with (args.output_dir / "peg_pose_sim.csv").open("w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["x", "y", "z", "qx", "qy", "qz", "qw"])
            writer.writerows(np.asarray(held_poses).tolist())
        wrench_header = ["Fx", "Fy", "Fz", "Tx", "Ty", "Tz"]

        link7_highrate_raw_array = np.asarray(link7_highrate_raw, dtype=np.float64)
        link7_highrate_base_at_k_array = (
            np.asarray(link7_highrate_base_at_k, dtype=np.float64)
            if link7_frame_calibration is not None
            else None
        )
        link7_highrate_time_array = np.asarray(link7_highrate_times, dtype=np.float64)
        if (
            link7_highrate_base_at_k_array is not None
            and len(link7_highrate_base_at_k_array) != len(link7_highrate_raw_array)
        ):
            raise RuntimeError("native and base/K link7 high-rate streams are not aligned")
        baseline_mask = link7_highrate_time_array <= 3.0
        if not np.any(baseline_mask):
            raise RuntimeError("panda_link7 stream has no samples in the 3 second baseline window")
        link7_baseline = np.median(link7_highrate_raw_array[baseline_mask], axis=0)
        link7_highrate_corrected = link7_highrate_raw_array - link7_baseline
        link7_highrate_despiked = causal_median_filter(
            link7_highrate_corrected,
            window=3,
        )
        link7_highrate_filtered, link7_filter_alpha = causal_lowpass(
            link7_highrate_despiked,
            float(env.physics_dt),
            float(args.link7_cutoff_hz),
        )
        link7_frame_filtered = np.asarray(
            [
                np.mean(link7_highrate_filtered[start:stop], axis=0)
                for start, stop in link7_frame_slices
            ],
            dtype=np.float64,
        )
        link7_frame_rms = np.asarray(
            [
                np.sqrt(np.mean(np.sum(link7_highrate_corrected[start:stop, :3] ** 2, axis=1)))
                for start, stop in link7_frame_slices
            ],
            dtype=np.float64,
        )
        link7_force_gain = float(args.link7_force_gain)
        link7_frame_calibrated = link7_frame_filtered * link7_force_gain
        link7_frame_rms_calibrated = link7_frame_rms * link7_force_gain
        link7_force_norm = np.linalg.norm(link7_frame_calibrated[:, :3], axis=1)
        jam_minimum_frames = max(1, int(np.ceil(float(args.jam_min_duration_s) * args.fps)))
        link7_jam = sustained_threshold(
            link7_force_norm,
            float(args.jam_force_threshold_n),
            jam_minimum_frames,
        )
        link7_highrate_table = np.column_stack(
            (
                link7_highrate_time_array,
                link7_highrate_raw_array,
                link7_highrate_corrected,
                link7_highrate_filtered,
            )
        )
        write_matrix_csv(
            args.output_dir / "panda_link7_wrench_120hz.csv",
            ["timestamp"]
            + [f"raw_{name}" for name in wrench_header]
            + [f"corrected_{name}" for name in wrench_header]
            + [f"filtered_{name}" for name in wrench_header],
            link7_highrate_table,
        )
        if link7_highrate_base_at_k_array is not None:
            write_matrix_csv(
                args.output_dir / "panda_link7_wrench_base_at_K_120hz.csv",
                ["timestamp", *wrench_header],
                np.column_stack(
                    (link7_highrate_time_array, link7_highrate_base_at_k_array)
                ),
            )
        write_matrix_csv(
            args.output_dir / "panda_link7_wrench_filtered.csv",
            wrench_header,
            link7_frame_filtered,
        )
        write_matrix_csv(
            args.output_dir / "panda_link7_wrench_calibrated.csv",
            wrench_header,
            link7_frame_calibrated,
        )
        write_matrix_csv(
            args.output_dir / "panda_link7_force_feedback.csv",
            ["calibrated_force_norm_n", "calibrated_substep_rms_force_n", "jam_binary"],
            np.column_stack((link7_force_norm, link7_frame_rms_calibrated, link7_jam)),
        )
        print(
            "[Link7Force] "
            f"samples_120hz={len(link7_highrate_raw_array)} cutoff_hz={args.link7_cutoff_hz:.3f} "
            f"alpha={link7_filter_alpha:.6f} gain={link7_force_gain:.3f} "
            f"p95={np.percentile(link7_force_norm, 95):.6f}N "
            f"jam_frames={int(np.count_nonzero(link7_jam))}",
            flush=True,
        )
        if args.record_contact_pair:
            contact_header = ["Fx", "Fy", "Fz"]
            for key in contact_pair_streams:
                write_matrix_csv(
                    args.output_dir / f"contact_{key}.csv",
                    contact_header,
                    contact_pair_streams[key],
                )
            contact_total_base = np.asarray(contact_pair_streams["total_base"], dtype=np.float64)
            incoming_body_array = np.asarray(incoming_body_stream, dtype=np.float64)
            incoming_header = [
                f"{body}_{component}"
                for body in incoming_body_names
                for component in wrench_header
            ]
            write_matrix_csv(
                args.output_dir / "incoming_wrench_all_bodies.csv",
                incoming_header,
                incoming_body_array.reshape(len(incoming_body_array), -1),
            )
            load_path_report = incoming_load_path_report(
                incoming_body_names,
                incoming_body_array,
                contact_total_base,
            )
            (args.output_dir / "wrist_load_path_report.json").write_text(
                json.dumps(load_path_report, indent=2), encoding="utf-8"
            )
            contact_norm = np.linalg.norm(contact_total_base, axis=1)
            feedback_ema_stream, feedback_alpha = causal_lowpass(
                contact_norm[:, None],
                1.0 / float(args.fps),
                float(args.contact_feedback_cutoff_hz),
            )
            feedback_ema = feedback_ema_stream[:, 0]
            insertion_feedback = np.column_stack(
                (contact_norm, feedback_ema, (contact_norm > 1.0e-4).astype(np.float64))
            )
            write_matrix_csv(
                args.output_dir / "insertion_feedback.csv",
                ["contact_force_norm_n", "causal_ema_force_n", "contact_binary"],
                insertion_feedback,
            )
            print(
                "[LoadPath] "
                f"force_sensor_valid={load_path_report['force_sensor_on_effective_load_path']} "
                f"recommended={load_path_report['recommended_incoming_wrench_body']} "
                f"report={args.output_dir / 'wrist_load_path_report.json'}",
                flush=True,
            )
        write_scalar_csv(args.output_dir / "timestamps.csv", "timestamp", replay_timestamps)
        camera_header = [
            "x", "y", "z", "qx", "qy", "qz", "qw",
            "k00", "k01", "k02", "k10", "k11", "k12", "k20", "k21", "k22",
        ]
        write_matrix_csv(
            args.output_dir / "front_camera_calibration.csv",
            camera_header,
            front_camera_calibration,
        )
        write_matrix_csv(
            args.output_dir / "wrist_camera_calibration.csv",
            camera_header,
            wrist_camera_calibration,
        )
        metadata = {
            "controller": "120 Hz interpolated end-effector targets -> PPO generate_ctrl_signals operational-space controller -> PhysX",
            "robot_articulation_path": "/World/envs/env_.*/franka_env/Robot/franka",
            "base_pose_unchanged": True,
            "real_to_sim_xyz": alignment,
            "first_frame_ik_error": init_error,
            "position_target_source": "H5 obs/state/ee_pose rigidly aligned to existing joint-replay FK frame",
            "orientation_target_source": "FK of the recorded joint_pos in the current IsaacLab scene",
            "arm_actuator": {
                "mode": "PPO task-space generate_ctrl_signals with effort control",
                "stiffness": 0.0,
                "damping": 0.0,
                "effort_limit_sim": 100.0,
            },
            "physical_pose_error_m": {
                "first": float(physical_pose_errors[0]),
                "median": float(np.median(physical_pose_errors)),
                "max": float(np.max(physical_pose_errors)),
                "last": float(physical_pose_errors[-1]),
            },
            "endpoint_window": window,
            "hole_position_sim_m": hole_pos.tolist(),
            "hole_orientation_xyzw": hole_quat[0].detach().cpu().tolist(),
            "hole_yaw_override_deg": (
                float(args.hole_yaw_deg) if args.hole_yaw_deg is not None else None
            ),
            "hole_position_correction_m": [
                float(args.hole_offset_x),
                float(args.hole_offset_y),
                float(args.hole_offset_z),
            ],
            "gripper_width_source": "PPO task-space controller closed-gripper target",
            "gripper_joint_target_m": 0.0,
            "grasp_close_time_s": float(cfg.task.grasp_close_time_s),
            "grasp_settle_time_s": float(cfg.task.grasp_settle_time_s),
            "snap_held_asset_after_grasp": bool(cfg.task.snap_held_asset_after_grasp),
            "recorded_gripper_width_source": "obs/state/gripper_width_m",
            "gripper_width_m_min_max": [float(gripper_width.min()), float(gripper_width.max())],
            "wrist_camera_pitch_correction_deg": float(args.wrist_pitch_deg),
            "wrist_camera_yaw_correction_deg": float(args.wrist_yaw_deg),
            "wrist_camera_forward_correction_m": float(args.wrist_forward_offset),
            "wrist_camera_offset_position_m": list(cfg.wrist_camera.offset.pos),
            "wrist_camera_offset_xyzw": list(cfg.wrist_camera.offset.rot),
            "visual_contract": "continuous PPO physics with the verified sim_force rollout background",
            "camera_rendering": {
                "mode": "2x supersampled PPO sensors with joint-rollout Pi0 visual matching",
                "front_gaussian_prim": "/World/envs/env_0/franka_env/lab_0/gauss/gauss",
                "per_camera_visibility_switching": False,
                "front_camera_extrinsics_source": "verified sim_force rollout background pose",
                "wrist_state_source": "physical rollout state with calibrated Pi0 hand mount",
                "green_table_overlay": bool(cfg.pi0_green_table_overlay_enabled),
                "white_table_overlay": bool(cfg.pi0_white_table_overlay_enabled),
                "green_table_right_patch": False,
                "green_table_border_repair": True,
                "static_right_arm_visual": True,
                "left_arm_visual": "pose-synchronised sim_force visual shell",
                "visual_hole_proxy": {
                    "prim_path": visual_hole_proxy_path,
                    "mode": "physical hole mesh rendered directly",
                    "yaw_deg_relative_to_physical_hole": 0.0,
                    "collision_enabled": not bool(args.disable_hole_collision),
                    "physical_hole_render_hidden": False,
                    "material": "sim_force fixed-asset material",
                },
                "render_resolution": [
                    int(cfg.tiled_camera.width),
                    int(cfg.tiled_camera.height),
                ],
                "video_resolution": [
                    int(cfg.pi0_camera_output_width),
                    int(cfg.pi0_camera_output_height),
                ],
                "front_comparison_frame_index": comparison_index,
            },
            "front_camera_calibration": {
                "method": "background USD camera, queried after scene creation",
                "position_m": front_camera_calibration[0][:3].tolist(),
                "orientation_opengl_xyzw": front_camera_calibration[0][3:7].tolist(),
            },
            "peg_mode": "PPO independent dynamic HeldAsset",
            "peg_root_prim": "/World/envs/env_.*/HeldAsset",
            "independent_held_asset_hidden": False,
            "per_frame_peg_pose_write": False,
            "peg_pose_writes": "initial placement and PPO post-close snap only",
            "physical_peg_pose_file": "peg_pose_sim.csv",
            "physics_dt_s": float(cfg.sim.dt),
            "physics_contact_response": True,
            "hole_collision_enabled": not bool(args.disable_hole_collision),
            "peg_diameter_m": float(cfg.task.held_asset_cfg.diameter),
            "hole_diameter_m": float(cfg.task.fixed_asset_cfg.diameter),
            "frame_count": len(sim_xyz),
            "fps": float(args.fps),
            "panda_link7_force_feedback": {
                "source": "PhysX get_link_incoming_joint_force at panda_link7",
                "source_frame": "PhysX native incoming-joint frame",
                "base_at_K_file": (
                    "panda_link7_wrench_base_at_K_120hz.csv"
                    if link7_highrate_base_at_k_array is not None
                    else None
                ),
                "base_at_K_semantics": (
                    "external wrench expressed in robot base; torque transported from "
                    "panda_link7 origin to fingertip stiffness frame K"
                    if link7_highrate_base_at_k_array is not None
                    else None
                ),
                "directed_frame_calibration": (
                    {
                        "path": link7_frame_calibration["path"],
                        "raw_global_sign": link7_frame_calibration["raw_global_sign"],
                        "child_joint_to_link7_rotation_row": np.asarray(
                            link7_frame_calibration[
                                "child_joint_to_link7_rotation_row"
                            ]
                        ).tolist(),
                        "fit_rmse_n": link7_frame_calibration["fit_rmse_n"],
                        "validation_rmse_n": link7_frame_calibration[
                            "validation_rmse_n"
                        ],
                    }
                    if link7_frame_calibration is not None
                    else None
                ),
                "sample_rate_hz": float(1.0 / env.physics_dt),
                "baseline_seconds": 3.0,
                "baseline": link7_baseline.tolist(),
                "causal_lowpass_cutoff_hz": float(args.link7_cutoff_hz),
                "causal_lowpass_alpha": link7_filter_alpha,
                "causal_median_window": 3,
                "post_filter_force_gain": link7_force_gain,
                "downsample": "mean of filtered 120 Hz samples in each output interval",
                "jam_force_threshold_n": float(args.jam_force_threshold_n),
                "jam_min_duration_s": float(args.jam_min_duration_s),
                "jam_minimum_frames": jam_minimum_frames,
                "insertion_feedback_cutoff_hz": float(args.contact_feedback_cutoff_hz),
                "insertion_feedback_alpha": float(feedback_alpha)
                if args.record_contact_pair
                else None,
            },
            "force_collection": {
                "primary_file": "panda_link7_wrench_calibrated.csv",
                "primary_source": (
                    "120 Hz filtered panda_link7 incoming joint wrench with fixed "
                    f"post-filter gain {link7_force_gain:.6g}"
                ),
                "raw_filtered_file": "panda_link7_wrench_filtered.csv",
                "force_gain": link7_force_gain,
                "force_sensor_exported": False,
                "component_order": wrench_header,
                "units": ["N", "N", "N", "N*m", "N*m", "N*m"],
                "collection_note": (
                    "panda_link7 is sampled at every PhysX substep; force_sensor streams "
                    "are retained only for diagnosis and are not training inputs"
                ),
            },
            "contact_pair_collection": {
                "enabled": bool(args.record_contact_pair),
                "sensor": "IsaacLab PhysX SimulationView.create_rigid_contact_view",
                "sensor_prim_path": "/World/envs/env_.*/HeldAsset",
                "filter_prim_path": "/World/envs/env_.*/FixedAsset",
                "normal_force_files": ["contact_normal_world.csv", "contact_normal_base.csv"],
                "friction_force_files": ["contact_friction_world.csv", "contact_friction_base.csv"],
                "total_force_files": ["contact_total_world.csv", "contact_total_base.csv"],
                "used_by_controller": False,
                "route_control_unchanged": True,
            },
        }
        (args.output_dir / "replay_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        print(f"[DONE] front: {args.output_dir / 'front_camera.mp4'}", flush=True)
        print(f"[DONE] wrist: {args.output_dir / 'wrist_camera.mp4'}", flush=True)
    except BaseException as exc:
        print(f"[ReplayFatal] {type(exc).__name__}: {exc!r}", flush=True)
        raise
    finally:
        env.close()

try:
    main()
except SystemExit as exc:
    # Isaac/legacy environment code occasionally uses exit(0) as a lifecycle
    # signal. For an offline replay that has not saved its required streams,
    # that is a failure and must not let the packaging stage continue.
    expected = [
        args.output_dir / "panda_link7_wrench_filtered.csv",
        args.output_dir / "panda_link7_wrench_calibrated.csv",
        args.output_dir / "panda_link7_force_feedback.csv",
    ]
    if args.link7_frame_calibration is not None:
        expected.append(args.output_dir / "panda_link7_wrench_base_at_K_120hz.csv")
    if exc.code in (None, 0) and not all(path.is_file() for path in expected):
        print(
            "[ReplayFatal] premature SystemExit(0) before rollout outputs were saved",
            flush=True,
        )
        raise RuntimeError("continuous PPO replay exited before saving wrench streams") from exc
    raise
finally:
    simulation_app.close()
