"""Render a Pi0-style paired traj_0 first frame and compare it to real data.

This is a visualization-only diagnostic.  It does not instantiate the Pi0
WebSocket client or send anything to the policy server.  The simulator uses
the paired replay joint state, the configured Pi0 static right arm, the rollout
black gripper visual, and the Pi0 active-left-base white material override.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import h5py
import numpy as np
import torch
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--h5", type=Path, default=Path("real_data/traj_0/data.h5"))
parser.add_argument(
    "--paired-dir",
    type=Path,
    default=Path("outputs/paired_rollouts_40_fk/traj_0"),
)
parser.add_argument(
    "--real-video",
    type=Path,
    default=Path("real_data/traj_0/front_camera.mp4"),
)
parser.add_argument(
    "--real-wrist-video",
    type=Path,
    default=Path("real_data/traj_0/wrist_camera.mp4"),
)
parser.add_argument(
    "--output-dir",
    type=Path,
    default=Path("outputs/traj0_real_vs_sim_pi0_visual"),
)
parser.add_argument(
    "--task",
    type=str,
    default="TacEx-RealSim-PegInsert-PI0-Direct-v0",
)
parser.add_argument(
    "--hide-right-arm",
    action="store_true",
    help="Hide only the static right-arm visual for tabletop-boundary diagnosis.",
)
parser.add_argument("--wrist-focal-length", type=float, default=None)
parser.add_argument("--wrist-offset-x", type=float, default=None)
parser.add_argument("--wrist-offset-y", type=float, default=None)
parser.add_argument("--wrist-offset-z", type=float, default=None)
parser.add_argument("--wrist-pitch-deg", type=float, default=None)
parser.add_argument("--wrist-yaw-deg", type=float, default=None)
parser.add_argument("--wrist-roll-deg", type=float, default=None)
parser.add_argument("--hole-yaw-deg", type=float, default=None)
parser.add_argument(
    "--comparisons-only",
    action="store_true",
    help="Save only the front and wrist real/sim comparison PNGs.",
)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if hasattr(args, "enable_cameras"):
    args.enable_cameras = True

simulation_app = AppLauncher(args).app

# Isaac Sim 6 keeps these legacy modules in extension directories that are not
# always exposed through isaacsim.__path__.
import isaacsim  # noqa: E402

_ISAACSIM_EXTS = Path(isaacsim.__file__).resolve().parent / "exts"
for _extension_name in ("isaacsim.core.prims", "isaacsim.core.utils"):
    _extension_pkg = _ISAACSIM_EXTS / _extension_name / "isaacsim"
    if _extension_pkg.is_dir() and str(_extension_pkg) not in isaacsim.__path__:
        isaacsim.__path__.append(str(_extension_pkg))

import tacex_tasks  # noqa: E402,F401
from tacex_tasks import torch_compat as torch_utils  # noqa: E402
from isaaclab.utils import math as math_utils  # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402
from tacex_tasks.real2sim.pi0_env import (  # noqa: E402
    Pi0RealSimEnv,
    apply_pi0_visual_contract,
    capture_pi0_observation,
    configure_pi0_cameras,
    pi0_visual_contract_metadata,
    sync_pi0_left_visual_shell,
)
from tacex_tasks.real2sim.pi0_static_right_arm import solve_pose_ik  # noqa: E402
from tacex_tasks.real2sim.realsim_env import RealSimEnv  # noqa: E402


def _load_inputs(h5_path: Path, paired_dir: Path) -> tuple[np.ndarray, float, float]:
    with h5py.File(h5_path, "r") as data:
        q_real = np.asarray(data["obs/state/joint_pos"][:], dtype=np.float32)
        gripper = float(np.asarray(data["obs/state/gripper_pos"][0]).reshape(-1)[0])
        gripper_width_m = float(
            np.asarray(data["obs/state/gripper_width_m"][0]).reshape(-1)[0]
        )

    q_sim = np.loadtxt(paired_dir / "joint_pos_sim.csv", delimiter=",", skiprows=1).astype(
        np.float32
    )
    if q_real.ndim != 2 or q_real.shape[1] != 7:
        raise ValueError(f"real joint_pos must be (N, 7), got {q_real.shape}")
    if q_sim.ndim == 1:
        q_sim = q_sim.reshape(1, -1)
    if q_sim.shape != q_real.shape:
        raise ValueError(f"paired q shape {q_sim.shape} != real q shape {q_real.shape}")
    if not np.isfinite(q_sim[0]).all():
        raise ValueError("paired traj_0 first joint state contains NaN or Inf")
    return q_sim, gripper, gripper_width_m


def _read_first_bgr(video_path: Path) -> np.ndarray:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Cannot open real video: {video_path}")
    try:
        ok, frame = capture.read()
    finally:
        capture.release()
    if not ok or frame is None:
        raise RuntimeError(f"Cannot read first frame from: {video_path}")
    return np.ascontiguousarray(frame)


def _set_pi0_visual_cfg(cfg) -> None:
    """Apply the Pi0 environment's visual-only setup without Pi0 inference."""

    cfg.policy_cfg = None
    cfg.teacher_policy_cfg = None
    apply_pi0_visual_contract(cfg)
    cfg.data_collect_cfg["collect_data"] = False
    cfg.data_collect_cfg["immediate_stop"] = False
    cfg.data_collect_cfg["save_failed_trajectory"] = False
    cfg.data_collect_cfg["num_trajectories"] = 1_000_000
    cfg.teacher_eval_only = True



def _solve_right_pose_from_pi0_cfg(cfg) -> None:
    """Apply the same fixed-pose IK used by Pi0RealSimEnv."""

    from pxr import Usd

    endpoint_pos = np.asarray(
        cfg.background_right_robot_endpoint_reference_offset_pos,
        dtype=np.float64,
    ).reshape(3)
    endpoint_rot = np.asarray(
        cfg.background_right_robot_endpoint_target_rot,
        dtype=np.float64,
    ).reshape(3, 3)
    q_seed = np.asarray(cfg.background_right_robot_joint_pos, dtype=np.float64).reshape(7)
    source_usd = Path(__file__).resolve().parents[1] / "franka_env_background_edit" / "franka_env.usd"
    stage = Usd.Stage.Open(str(source_usd))
    if stage is None:
        raise RuntimeError(f"Could not open right-arm pose-IK source: {source_usd}")
    q_pose, pos_residual, rot_residual = solve_pose_ik(
        stage,
        "/World/Robot/franka",
        endpoint_pos,
        endpoint_rot,
        q_seed,
        max_iterations=500,
        nominal_weight=1.0e-5,
    )
    cfg.background_right_robot_joint_pos = tuple(float(value) for value in q_pose)
    print(
        "[Compare] right endpoint IK: "
        f"target={tuple(endpoint_pos)} q={np.array2string(q_pose, precision=7)} "
        f"pos_residual={pos_residual:.6f} m rot_residual={rot_residual:.6f} rad",
        flush=True,
    )


def _set_robot_q(
    env: RealSimEnv,
    q: np.ndarray,
    gripper: float,
    visual_gripper_width_m: float | None = None,
) -> None:
    q_tensor = torch.as_tensor(q, dtype=torch.float32, device=env.device).view(1, 7)
    full_q = env.joint_pos.clone()
    full_q[:, :7] = q_tensor
    # The recorded Pi0 gripper field is the raw training-domain value.  For
    # the paired visualization, prefer the measured physical width directly;
    # an optional visual adapter can then separate the imported fingers a bit
    # without changing the state value sent to Pi0.
    if visual_gripper_width_m is None:
        visual_gripper_width_m = float(gripper) * 0.08
    full_q[:, 7:9] = float(visual_gripper_width_m) * 0.5
    zero_velocity = torch.zeros_like(full_q)
    env._robot.write_joint_state_to_sim(full_q, zero_velocity)
    env.ctrl_target_joint_pos[:] = full_q
    env._robot.set_joint_position_target(full_q)
    env._robot.set_joint_effort_target(torch.zeros_like(full_q))
    env.scene.write_data_to_sim()
    env.sim.forward()
    env.scene.update(dt=env.physics_dt)
    env._compute_intermediate_values(dt=env.physics_dt)


def _held_peg_pose(env: RealSimEnv) -> tuple[torch.Tensor, torch.Tensor]:
    # Match Pi0RealSimEnv.randomize_initial_state exactly: IsaacLab 6 uses
    # XYZW and the peg is rotated 180 degrees around Y.  The root offset is
    # the rollout black-gripper calibration (32.392 mm).
    flip_y_quat = torch.tensor([0.0, 1.0, 0.0, 0.0], device=env.device).view(1, 4)
    identity_quat = torch.tensor([0.0, 0.0, 0.0, 1.0], device=env.device).view(1, 4)
    zero_pos = torch.zeros((1, 3), device=env.device)
    flipped_pos, flipped_quat = math_utils.combine_frame_transforms(
        env.fingertip_midpoint_pos,
        env.fingertip_midpoint_quat,
        zero_pos,
        flip_y_quat,
    )
    relative_pos = torch.zeros((1, 3), device=env.device)
    relative_pos[:, 2] = 0.050 - 0.017608 + float(
        getattr(env.cfg, "pi0_peg_mount_depth_adjust_m", 0.0)
    )
    held_pos, held_quat = math_utils.combine_frame_transforms(
        flipped_pos,
        flipped_quat,
        -relative_pos,
        identity_quat,
    )
    return held_quat, held_pos


def _write_asset_pose(asset, pos: torch.Tensor, quat: torch.Tensor) -> None:
    state = asset.data.root_state_w.clone()
    state[:, :3] = pos
    state[:, 3:7] = quat
    state[:, 7:] = 0.0
    asset.write_root_pose_to_sim(state[:, :7])
    asset.write_root_velocity_to_sim(state[:, 7:])
    asset.reset()


def _force_pi0_hole_orientation(env: RealSimEnv, cfg) -> None:
    """Undo the legacy RealSim reset quaternion before capturing the frame."""

    state = env._fixed_asset.data.root_state_w.clone()
    state[:, 3:7] = torch.as_tensor(
        cfg.pi0_hole_init_rot,
        dtype=torch.float32,
        device=env.device,
    ).view(1, 4)
    state[:, 7:] = 0.0
    env._fixed_asset.write_root_pose_to_sim(state[:, :7])
    env._fixed_asset.write_root_velocity_to_sim(state[:, 7:])
    env._fixed_asset.reset()
    env.scene.write_data_to_sim()
    env.sim.forward()
    env.scene.update(dt=env.physics_dt)
    env._compute_intermediate_values(dt=env.physics_dt)


def _capture_cameras(env: RealSimEnv) -> tuple[np.ndarray, np.ndarray]:
    sim_front_rgb, sim_wrist_rgb = capture_pi0_observation(env)
    return (
        cv2.cvtColor(np.ascontiguousarray(sim_front_rgb[:, :, :3]), cv2.COLOR_RGB2BGR),
        cv2.cvtColor(np.ascontiguousarray(sim_wrist_rgb[:, :, :3]), cv2.COLOR_RGB2BGR),
    )


def _capture_front(env: RealSimEnv) -> np.ndarray:
    return _capture_cameras(env)[0]


def _hide_static_right_arm(env: RealSimEnv) -> bool:
    """Hide the Pi0-only static right arm without changing the scene config."""

    from pxr import UsdGeom

    hidden = False
    for prim_path in (
        "/World/envs/env_0/franka_env/RightRobot",
        "/World/envs/env_0/franka_env/fr3v2_01",
    ):
        prim = env.sim.stage.GetPrimAtPath(prim_path)
        if not prim.IsValid() or not prim.IsA(UsdGeom.Imageable):
            continue
        UsdGeom.Imageable(prim).MakeInvisible()
        hidden = True
        print(f"[Compare] hidden static right arm: {prim_path}", flush=True)
    return hidden


def _make_comparison(
    real_bgr: np.ndarray,
    sim_bgr: np.ndarray,
    real_label: str = "REAL traj_0 frame 0",
    sim_label: str = "SIM paired traj_0 frame 0 | Pi0 visuals",
) -> np.ndarray:
    target_h, target_w = real_bgr.shape[:2]
    sim_bgr = cv2.resize(sim_bgr, (target_w, target_h), interpolation=cv2.INTER_LANCZOS4)
    banner_h = 46
    canvas = np.full((target_h + banner_h, target_w * 2 + 4, 3), 245, dtype=np.uint8)
    canvas[banner_h:, :target_w] = real_bgr
    canvas[banner_h:, target_w + 4:] = sim_bgr
    cv2.rectangle(canvas, (0, banner_h), (target_w - 1, target_h + banner_h - 1), (30, 30, 30), 2)
    cv2.rectangle(
        canvas,
        (target_w + 4, banner_h),
        (target_w * 2 + 3, target_h + banner_h - 1),
        (30, 30, 30),
        2,
    )
    cv2.putText(canvas, real_label, (14, 31), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (20, 20, 20), 2, cv2.LINE_AA)
    cv2.putText(
        canvas,
        sim_label,
        (target_w + 18, 31),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.62,
        (20, 20, 20),
        2,
        cv2.LINE_AA,
    )
    return canvas


def main() -> None:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    q_sim, gripper, real_gripper_width_m = _load_inputs(args.h5, args.paired_dir)
    real_bgr = _read_first_bgr(args.real_video)
    real_wrist_bgr = _read_first_bgr(args.real_wrist_video)

    cfg = parse_env_cfg(args.task, device=args.device, num_envs=1, use_fabric=True)
    cfg.scene.num_envs = 1
    cfg.episode_length_s = 30.0
    cfg.ctrl.reset_joints = q_sim[0].tolist()
    _set_pi0_visual_cfg(cfg)
    if args.wrist_focal_length is not None:
        cfg.pi0_wrist_camera_focal_length = float(args.wrist_focal_length)
    if any(
        value is not None
        for value in (args.wrist_offset_x, args.wrist_offset_y, args.wrist_offset_z)
    ):
        wrist_pos = list(cfg.pi0_wrist_camera_offset_pos)
        if args.wrist_offset_x is not None:
            wrist_pos[0] = float(args.wrist_offset_x)
        if args.wrist_offset_y is not None:
            wrist_pos[1] = float(args.wrist_offset_y)
        if args.wrist_offset_z is not None:
            wrist_pos[2] = float(args.wrist_offset_z)
        cfg.pi0_wrist_camera_offset_pos = tuple(wrist_pos)
    wrist_correction = list(cfg.pi0_wrist_camera_rotation_correction_deg)
    if args.wrist_pitch_deg is not None:
        wrist_correction[0] = float(args.wrist_pitch_deg)
    if args.wrist_yaw_deg is not None:
        wrist_correction[1] = float(args.wrist_yaw_deg)
    if args.wrist_roll_deg is not None:
        wrist_correction[2] = float(args.wrist_roll_deg)
    cfg.pi0_wrist_camera_rotation_correction_deg = tuple(wrist_correction)
    if args.hole_yaw_deg is not None:
        half_yaw = np.deg2rad(float(args.hole_yaw_deg)) * 0.5
        cfg.pi0_hole_init_rot = (
            0.0,
            0.0,
            float(np.sin(half_yaw)),
            float(np.cos(half_yaw)),
        )
    # Re-apply only the camera object values after optional diagnostic
    # overrides; the scene/material contract itself remains unchanged.
    configure_pi0_cameras(cfg)
    _solve_right_pose_from_pi0_cfg(cfg)

    env = RealSimEnv(cfg, render_mode="rgb_array", output_dir=str(args.output_dir))
    try:
        print("[Compare] resetting Pi0-style visualization scene", flush=True)
        env.reset()
        # Reuse the exact material-binding implementation from Pi0RealSimEnv;
        # this does not create a policy client or communicate with the server.
        Pi0RealSimEnv._override_pi0_active_robot_base_gray_to_white(env)
        _force_pi0_hole_orientation(env, cfg)

        visual_gripper_width_m = real_gripper_width_m + float(
            cfg.pi0_visual_gripper_extra_width_m
        )
        _set_robot_q(env, q_sim[0], gripper, visual_gripper_width_m)
        sync_pi0_left_visual_shell(env)
        peg_quat, peg_pos = _held_peg_pose(env)
        _write_asset_pose(env._held_asset, peg_pos, peg_quat)
        env.scene.write_data_to_sim()
        env.sim.forward()
        env.scene.update(dt=env.physics_dt)
        env._compute_intermediate_values(dt=env.physics_dt)

        right_arm_hidden = _hide_static_right_arm(env) if args.hide_right_arm else False
        sim_bgr, sim_wrist_bgr = _capture_cameras(env)
        real_path = args.output_dir / "real_traj0_first.png"
        sim_path = args.output_dir / "sim_traj0_first_pi0_visuals.png"
        compare_path = args.output_dir / "traj0_real_vs_sim_front_first.png"
        real_wrist_path = args.output_dir / "real_traj0_first_left_wrist.png"
        sim_wrist_path = args.output_dir / "sim_traj0_first_left_wrist.png"
        wrist_compare_path = args.output_dir / "traj0_real_vs_sim_left_wrist_first.png"
        cv2.imwrite(str(compare_path), _make_comparison(real_bgr, sim_bgr))
        cv2.imwrite(
            str(wrist_compare_path),
            _make_comparison(
                real_wrist_bgr,
                sim_wrist_bgr,
                real_label="REAL wrist traj_0 frame 0",
                sim_label="SIM left wrist traj_0 frame 0",
            ),
        )

        if args.comparisons_only:
            print(f"[DONE] front comparison: {compare_path.resolve()}", flush=True)
            print(f"[DONE] wrist comparison: {wrist_compare_path.resolve()}", flush=True)
            return

        cv2.imwrite(str(real_path), real_bgr)
        cv2.imwrite(str(sim_path), sim_bgr)
        cv2.imwrite(str(real_wrist_path), real_wrist_bgr)
        cv2.imwrite(str(sim_wrist_path), sim_wrist_bgr)

        report = {
            "real_source": str(args.real_video.resolve()),
            "real_wrist_source": str(args.real_wrist_video.resolve()),
            "sim_joint_source": str((args.paired_dir / "joint_pos_sim.csv").resolve()),
            "sim_task": args.task,
            "pi0_visual_contract": pi0_visual_contract_metadata(cfg),
            "left_wrist_camera": {
                "offset_pos_m": list(cfg.pi0_wrist_camera_offset_pos),
                "offset_rot_xyzw": list(cfg.wrist_camera.offset.rot),
                "rotation_correction_deg": list(
                    cfg.pi0_wrist_camera_rotation_correction_deg
                ),
            },
            "sim_first_q": q_sim[0].tolist(),
            "real_first_gripper_raw_training_domain": gripper,
            "real_first_gripper_width_m": real_gripper_width_m,
            "sim_visual_gripper_width_m": visual_gripper_width_m,
            "right_arm": {
                "visual_only": True,
                "base_offset_m": list(cfg.background_right_robot_offset_pos),
                "control_or_inference": False,
                "hidden_for_preview": right_arm_hidden,
            },
            "active_left_arm": {
                "base_material": "white via Pi0 material-binding override",
                "black_gripper_visual": bool(cfg.pi0_black_gripper_visual),
            },
            "hole_pose_source": "Pi0 scene config",
            "server_contacted": False,
            "outputs": {
                "real_first": str(real_path),
                "sim_first": str(sim_path),
                "comparison": str(compare_path),
                "real_left_wrist_first": str(real_wrist_path),
                "sim_left_wrist_first": str(sim_wrist_path),
                "left_wrist_comparison": str(wrist_compare_path),
            },
        }
        (args.output_dir / "report.json").write_text(
            json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        print(f"[DONE] comparison: {compare_path.resolve()}", flush=True)
    finally:
        env.close()


try:
    main()
finally:
    simulation_app.close()
