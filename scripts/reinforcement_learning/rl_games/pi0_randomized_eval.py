"""Evaluate Pi0 or TAVLA over randomized resets in one Isaac Sim process."""

from __future__ import annotations

import argparse
import atexit
import csv
import json
import logging
import os
import tempfile
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm
from isaaclab.app import AppLauncher


PI0_TASK = "TacEx-RealSim-PegInsert-PI0-Direct-v0"
TAVLA_TASK = "TacEx-RealSim-PegInsert-TAVLA-Teacher-v0"

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(
    "--policy",
    choices=("pi0", "tavla"),
    default=None,
    help="Policy backend. If omitted, infer it from --task; the default is Pi0.",
)
parser.add_argument("--task", type=str, default=None)
parser.add_argument("--pi0-host", type=str, default="114.214.164.36")
parser.add_argument("--pi0-port", type=int, default=8000)
parser.add_argument("--tavla-host", type=str, default="114.214.164.36")
parser.add_argument("--tavla-port", type=int, choices=(8000, 8001), default=8000)
parser.add_argument("--tavla-action-start-index", type=int, default=5)
parser.add_argument(
    "--tavla-force-gate",
    action="store_true",
    help="Gate TAVLA effort until direct PhysX peg-hole contact is confirmed.",
)
parser.add_argument("--tavla-force-gate-threshold-n", type=float, default=0.5)
parser.add_argument("--tavla-force-gate-confirm-steps", type=int, default=2)
parser.add_argument("--tavla-force-gate-release-steps", type=int, default=3)
parser.add_argument(
    "--tavla-force-gate-precontact-wrench",
    type=float,
    nargs=6,
    metavar=("Fx", "Fy", "Fz", "Tx", "Ty", "Tz"),
    default=(0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
    help="Six-value effort baseline sent before peg-hole contact is confirmed.",
)
parser.add_argument("--teacher-hold-steps", type=int, default=None)
parser.add_argument(
    "--save-policy-input-video",
    action="store_true",
    help="Save the exact transformed front/wrist frames sent to the policy.",
)
parser.add_argument("--episodes", type=int, default=100)
parser.add_argument("--episode-length-s", type=float, default=60.0)
parser.add_argument(
    "--replan-actions",
    type=int,
    default=None,
    help="Number of actions held per inference; defaults to 3 for Pi0 and 10 for TAVLA.",
)
parser.add_argument("--eval-seed", type=int, default=42)
parser.add_argument(
    "--reset-schedule",
    type=Path,
    default=None,
    help="CSV with fixed per-episode hole/hand offsets; overrides random sampling.",
)
parser.add_argument(
    "--tavla-pi0-reset-dir",
    type=Path,
    default=Path("outputs/pi0-eval/pi0-Cotraining"),
    help="PI0 output directory containing episode_*/joint_states.csv for direct TAVLA resets.",
)
parser.add_argument(
    "--hole-position-range-m",
    type=float,
    nargs=3,
    metavar=("X", "Y", "Z"),
    default=(0.02, 0.02, 0.0),
    help="Symmetric half-range for random hole-base XYZ offsets in meters.",
)
parser.add_argument(
    "--hand-position-range-m",
    type=float,
    nargs=3,
    metavar=("X", "Y", "Z"),
    default=(0.02, 0.02, 0.01),
    help="Symmetric half-range for random initial fingertip XYZ offsets relative to the hole.",
)
parser.add_argument(
    "--output-dir",
    type=Path,
    default=Path("outputs/pi0_randomized_100_eval"),
)
parser.add_argument(
    "--flat-output",
    action="store_true",
    help="Save episodes directly under --output-dir instead of a run_TIMESTAMP child.",
)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

if args.policy is None:
    args.policy = "tavla" if args.task and "TAVLA" in args.task.upper() else "pi0"
if args.task is None:
    args.task = TAVLA_TASK if args.policy == "tavla" else PI0_TASK
if args.replan_actions is None:
    args.replan_actions = 10 if args.policy == "tavla" else 3
if args.tavla_action_start_index < 1:
    raise ValueError("--tavla-action-start-index must be >= 1")
if args.teacher_hold_steps is not None and args.teacher_hold_steps < 1:
    raise ValueError("--teacher-hold-steps must be >= 1")
if args.tavla_force_gate_threshold_n <= 0.0:
    raise ValueError("--tavla-force-gate-threshold-n must be positive")
if args.tavla_force_gate_confirm_steps < 1:
    raise ValueError("--tavla-force-gate-confirm-steps must be >= 1")
if args.tavla_force_gate_release_steps < 0:
    raise ValueError("--tavla-force-gate-release-steps must be >= 0")
if not np.isfinite(args.tavla_force_gate_precontact_wrench).all():
    raise ValueError("--tavla-force-gate-precontact-wrench must be finite")
is_tavla = args.policy == "tavla"

if args.episodes <= 0:
    raise ValueError("--episodes must be positive")
if args.episode_length_s <= 0.0:
    raise ValueError("--episode-length-s must be positive")
if args.replan_actions <= 0:
    raise ValueError("--replan-actions must be positive")
if any(value < 0.0 for value in args.hole_position_range_m):
    raise ValueError("--hole-position-range-m values must be non-negative")
if any(value < 0.0 for value in args.hand_position_range_m):
    raise ValueError("--hand-position-range-m values must be non-negative")
if hasattr(args, "enable_cameras"):
    args.enable_cameras = True

# Keep Isaac Sim's console focused on warnings/errors.  In particular, RTX
# geometry streaming emits one INFO line per update; the rollout progress bar
# remains the only regular per-step output.
_kit_args = "--/log/level=warning --/log/outputStreamLevel=warning"
args.kit_args = f"{getattr(args, 'kit_args', '')} {_kit_args}".strip()

# IsaacLab installs a scoped INFO stream handler during Kit startup. Silence
# only the RTX streaming progress logger; warnings and the rollout tqdm remain.
logging.getLogger(
    "isaaclab_physx.renderers.isaac_rtx_renderer_utils"
).setLevel(logging.WARNING)
simulation_app = AppLauncher(args).app

import gymnasium as gym  # noqa: E402
import tacex_tasks  # noqa: E402,F401
from isaaclab_tasks.direct.factory import factory_utils  # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402
from pxr import Usd, UsdGeom  # noqa: E402


def _create_physics_free_background_usd() -> Path:
    source = Path(__file__).resolve().parents[3] / "franka_env_background_edit" / "franka_env.usd"
    if not source.is_file():
        raise FileNotFoundError(f"Background USD not found: {source}")
    fd, filename = tempfile.mkstemp(prefix="tacex_pi0_random_eval_", suffix=".usd")
    os.close(fd)
    wrapper_path = Path(filename)
    stage = Usd.Stage.CreateNew(str(wrapper_path))
    root = UsdGeom.Xform.Define(stage, "/World")
    root.GetPrim().GetReferences().AddReference(str(source), "/World")
    stage.SetDefaultPrim(root.GetPrim())
    stage.OverridePrim("/World/PhysicsScene").SetActive(False)
    stage.GetRootLayer().Save()
    return wrapper_path


_PI0_BACKGROUND_USD = _create_physics_free_background_usd()
atexit.register(lambda: _PI0_BACKGROUND_USD.unlink(missing_ok=True))


def _scalar(value) -> float:
    if torch.is_tensor(value):
        return float(value.detach().reshape(-1)[0].cpu())
    return float(value)


def _load_reset_schedule(path: Path | None, episodes: int) -> list[dict[str, list[float]]] | None:
    if path is None:
        return None
    path = path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Reset schedule CSV not found: {path}")
    required = (
        "hole_dx_m", "hole_dy_m", "hole_dz_m",
        "hand_dx_m", "hand_dy_m", "hand_dz_m",
    )
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) < episodes:
        raise ValueError(
            f"Reset schedule has {len(rows)} rows, but --episodes={episodes} was requested"
        )
    missing = [name for name in required if name not in (rows[0] if rows else {})]
    if missing:
        raise ValueError(f"Reset schedule is missing columns: {missing}")
    schedule = []
    for index, row in enumerate(rows[:episodes]):
        try:
            schedule.append(
                {
                    "hole_offset_m": [float(row[name]) for name in required[:3]],
                    "hand_offset_m": [float(row[name]) for name in required[3:]],
                }
            )
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid reset schedule row {index + 2} in {path}") from exc
    return schedule


def _load_pi0_reset_joint_schedule(path: Path, episodes: int) -> list[list[float]]:
    """Load PI0's first recorded arm joint state for each reset episode."""
    path = path.expanduser().resolve()
    schedule: list[list[float]] = []
    for episode_index in range(episodes):
        joint_path = path / f"episode_{episode_index}" / "joint_states.csv"
        if not joint_path.is_file():
            raise FileNotFoundError(
                f"PI0 reset joint state not found for episode {episode_index}: {joint_path}"
            )
        with joint_path.open(newline="", encoding="utf-8") as handle:
            row = next(csv.DictReader(handle), None)
        if row is None:
            raise ValueError(f"PI0 joint state file is empty: {joint_path}")
        try:
            joints = [float(row[f"joint_{joint_index}"]) for joint_index in range(7)]
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"Invalid PI0 arm joint state in {joint_path}") from exc
        if not np.isfinite(joints).all():
            raise ValueError(f"PI0 arm joint state contains NaN/Inf: {joint_path}")
        schedule.append(joints)
    return schedule


def _insertion_metrics(raw_env) -> dict[str, float | bool]:
    held_base_pos, _ = factory_utils.get_held_base_pose(
        raw_env.held_pos,
        raw_env.held_quat,
        raw_env.cfg_task.name,
        raw_env.cfg_task.fixed_asset_cfg,
        raw_env.num_envs,
        raw_env.device,
    )
    target_base_pos, _ = factory_utils.get_target_held_base_pose(
        raw_env.fixed_pos,
        raw_env.fixed_quat,
        raw_env.cfg_task.name,
        raw_env.cfg_task.fixed_asset_cfg,
        raw_env.num_envs,
        raw_env.device,
    )
    delta = target_base_pos[0] - held_base_pos[0]
    success = raw_env._get_curr_successes(
        success_threshold=raw_env.cfg_task.success_threshold,
        check_rot=False,
    )[0]
    xy_error_m = float(torch.linalg.vector_norm(delta[:2]).detach().cpu())
    z_disp_m = float((held_base_pos[0, 2] - target_base_pos[0, 2]).detach().cpu())

    # A peg is considered partially inserted when its XY center is aligned and
    # its base has crossed the hole top, but it has not yet reached the strict
    # full-insertion depth used by FactoryEnv.  For the 8 mm peg/hole task,
    # these limits are 3 mm XY, 25 mm entry depth, and 3 mm full depth.
    xy_threshold_m = float(getattr(raw_env.cfg_task, "success_xy_threshold", 0.003))
    hole_height_m = float(getattr(raw_env.cfg_task.fixed_asset_cfg, "height", 0.0))
    full_depth_threshold_m = hole_height_m * float(
        getattr(raw_env.cfg_task, "success_threshold", 0.12)
    )
    partial_insertion = (
        not bool(success.detach().cpu())
        and xy_error_m < xy_threshold_m
        and full_depth_threshold_m <= z_disp_m < hole_height_m
    )
    return {
        "xy_error_m": xy_error_m,
        "z_disp_m": z_disp_m,
        "geometric_success": bool(success.detach().cpu()),
        "partial_insertion": partial_insertion,
    }


def _tensor_row(value, default=(0.0, 0.0, 0.0)) -> list[float]:
    if value is None:
        return [float(item) for item in default]
    if torch.is_tensor(value):
        value = value[0].detach().cpu().numpy()
    return np.asarray(value, dtype=np.float64).reshape(-1).tolist()


def _reset_offset(raw_env, name: str) -> list[float]:
    """Read the reset offset exposed by either the Pi0 or TAVLA env."""
    for policy_name in ("tavla", "pi0"):
        value = getattr(raw_env, f"{policy_name}_last_{name}_position_offset_m", None)
        if value is not None:
            return _tensor_row(value)
    return _tensor_row(None)


def _reset_snapshot(raw_env) -> dict[str, object]:
    metrics = _insertion_metrics(raw_env)
    return {
        "hole_offset_m": _reset_offset(raw_env, "hole"),
        "hand_offset_m": _reset_offset(raw_env, "hand"),
        "hole_position_m": _tensor_row(raw_env.fixed_pos),
        "fingertip_position_m": _tensor_row(raw_env.fingertip_midpoint_pos),
        "initial_xy_error_m": metrics["xy_error_m"],
        "initial_z_disp_m": metrics["z_disp_m"],
    }


def _policy_runtime_stats(raw_env, tavla: bool) -> dict[str, float | int]:
    if tavla:
        return {
            "inference": int(getattr(raw_env, "teacher_inference_count", 0)),
            "failures": int(getattr(raw_env, "teacher_failures", 0)),
            "timeouts": int(getattr(raw_env, "teacher_timeouts", 0)),
            "force_abort": int(getattr(raw_env, "tavla_force_abort_count", 0)),
            "latency": float(getattr(raw_env, "teacher_inference_latency_s", 0.0)),
        }
    return {
        "inference": int(getattr(raw_env, "pi0_inference_count", 0)),
        "failures": int(getattr(raw_env, "pi0_failures", 0)),
        "timeouts": 0,
        "force_abort": 0,
        "latency": float(getattr(raw_env, "pi0_last_latency_s", 0.0)),
    }


def _create_run_directory(output_root: Path, flat: bool = False) -> Path:
    """Create an evaluation directory, optionally without a timestamp child."""
    output_root.mkdir(parents=True, exist_ok=True)
    if flat:
        return output_root
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    candidate = output_root / f"run_{timestamp}"
    suffix = 1
    while candidate.exists():
        candidate = output_root / f"run_{timestamp}_{suffix:02d}"
        suffix += 1
    candidate.mkdir(parents=False)
    return candidate


def _write_episode_csv(
    path: Path,
    rows: list[dict[str, object]],
    policy: str,
) -> None:
    fields = [
        "episode",
        "episode_dir",
        "success",
        "steps",
        "return",
        "timed_out",
        "hole_dx_m",
        "hole_dy_m",
        "hole_dz_m",
        "hand_dx_m",
        "hand_dy_m",
        "hand_dz_m",
        "hole_x_m",
        "hole_y_m",
        "hole_z_m",
        "initial_fingertip_x_m",
        "initial_fingertip_y_m",
        "initial_fingertip_z_m",
        "initial_xy_error_m",
        "initial_z_disp_m",
        "best_xy_error_m",
        "minimum_z_disp_m",
        "partial_insertion",
        "partial_insertion_first_step",
    ]
    if policy == "tavla":
        fields.extend(
            [
                "tavla_inference_count_peak",
                "tavla_failures_peak",
                "tavla_timeouts_peak",
                "tavla_force_abort_count_peak",
            ]
        )
    else:
        fields.extend(["pi0_inference_count_peak", "pi0_failures_peak"])
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _load_completed_rows(path: Path, episodes: int) -> list[dict[str, object]]:
    """Load completed flat-output episodes so an interrupted run can resume."""
    if not path.is_file():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) > episodes:
        raise ValueError(
            f"Existing {path} contains {len(rows)} episodes, but --episodes={episodes}"
        )
    for index, row in enumerate(rows):
        expected_episode = str(index + 1)
        expected_dir = f"episode_{index}"
        if row.get("episode") != expected_episode or row.get("episode_dir") != expected_dir:
            raise ValueError(
                f"Existing {path} is not contiguous at row {index + 1}: "
                f"episode={row.get('episode')!r}, episode_dir={row.get('episode_dir')!r}"
            )
        # Older runs predate the partial-insertion columns. Keep those rows
        # readable; newly completed rows will contain the new values.
        row.setdefault("partial_insertion", "")
        row.setdefault("partial_insertion_first_step", "")
    return rows


def main() -> None:
    reset_schedule = _load_reset_schedule(args.reset_schedule, args.episodes)
    tavla_direct_reset_joints_schedule = (
        _load_pi0_reset_joint_schedule(args.tavla_pi0_reset_dir, args.episodes)
        if is_tavla
        else None
    )
    env_cfg = parse_env_cfg(
        args.task,
        device=args.device,
        num_envs=1,
        use_fabric=not getattr(args, "disable_fabric", False),
    )
    env_cfg.seed = int(args.eval_seed)
    env_cfg.episode_length_s = float(args.episode_length_s)
    # RealSimEnv contains a legacy process-level exit guard for data
    # collection.  This evaluator owns the episode loop and must be allowed
    # to write its aggregate CSV/JSON before closing Isaac Sim.
    env_cfg.teacher_eval_only = True
    env_cfg.data_collect_cfg["immediate_stop"] = True
    env_cfg.data_collect_cfg["collect_data"] = True
    env_cfg.data_collect_cfg["save_failed_trajectory"] = True
    env_cfg.data_collect_cfg["minimal_output"] = False
    # Keep one continuous video source. For input-comparison probes, save the
    # exact transformed tensors sent to the Pi0/TAVLA server and skip raw RGB.
    env_cfg.data_collect_cfg["save_policy_input_video"] = bool(
        args.save_policy_input_video
    )
    env_cfg.data_collect_cfg["save_raw_camera_video"] = not bool(
        args.save_policy_input_video
    )
    env_cfg.data_collect_cfg["save_reward_video"] = False
    # These rollouts are evaluation-only; keep the lightweight CSV/MP4
    # diagnostics, but do not export the large training-style HDF5 dataset.
    env_cfg.data_collect_cfg["save_tavla_hdf5"] = False
    # RealSim normally exits the process when this count is reached. Keep the
    # limit one larger so this script can write its aggregate summary itself.
    env_cfg.data_collect_cfg["num_trajectories"] = int(args.episodes) + 1
    if is_tavla:
        env_cfg.scene.num_envs = 1
        env_cfg.teacher_eval_only = True
        env_cfg.teacher_prompt = "peg-in-hole"
        env_cfg.teacher_control_mode = "aligned_joint"
        env_cfg.teacher_state_alignment = False
        env_cfg.teacher_action_state_alignment = False
        env_cfg.teacher_action_start_index = int(args.tavla_action_start_index)
        env_cfg.teacher_replan_actions = int(args.replan_actions)
        env_cfg.teacher_action_interpolation = True
        env_cfg.tavla_force_gate_enabled = bool(args.tavla_force_gate)
        env_cfg.tavla_force_gate_threshold_n = float(args.tavla_force_gate_threshold_n)
        env_cfg.tavla_force_gate_confirm_steps = int(args.tavla_force_gate_confirm_steps)
        env_cfg.tavla_force_gate_release_steps = int(args.tavla_force_gate_release_steps)
        env_cfg.tavla_force_gate_precontact_wrench = [
            float(value) for value in args.tavla_force_gate_precontact_wrench
        ]
        # The deployed TAVLA server consumes ten six-dimensional wrench
        # frames at offsets [-36, -32, ..., 0].  Set these explicitly after
        # parse_env_cfg because configclass copies can retain the old default.
        env_cfg.teacher_policy_cfg.num_history_steps = 10
        env_cfg.teacher_policy_cfg.history_step_interval = 4
        env_cfg.teacher_policy_cfg.history_idx = list(range(-37, 0, 4))
        env_cfg.task.success_threshold = 0.12
        env_cfg.task.success_xy_threshold = 0.003
        env_cfg.task.skip_reset_ik = True
        env_cfg.task.use_pi0_reset_pose = True
        # Use the same reset ranges as the Pi0 evaluator when no explicit
        # schedule is supplied.  TAVLA stores these on cfg.task.
        env_cfg.task.fixed_asset_init_pos_noise = list(args.hole_position_range_m)
        env_cfg.task.hand_init_pos_noise = list(args.hand_position_range_m)
        if reset_schedule is not None:
            env_cfg.task.fixed_asset_init_orn_range_deg = 0.0
            env_cfg.task.hand_init_orn_noise = [0.0, 0.0, 0.0]
        if args.teacher_hold_steps is not None:
            env_cfg.teacher_hold_steps = int(args.teacher_hold_steps)
        env_cfg.teacher_policy_cfg.host_ip = args.tavla_host
        env_cfg.teacher_policy_cfg.host_port = args.tavla_port
        env_cfg.wait_for_textures = False
    else:
        env_cfg.pi0_policy_cfg.host_ip = args.pi0_host
        env_cfg.pi0_policy_cfg.host_port = args.pi0_port
        # Use the fifth future action as the temporary forward-label validation.
        env_cfg.pi0_policy_cfg.action_start_index = 5
        env_cfg.pi0_action_start_index = 5
        env_cfg.pi0_replan_actions = int(args.replan_actions)
        env_cfg.pi0_hole_position_noise_m = tuple(args.hole_position_range_m)
        env_cfg.pi0_hand_position_noise_m = tuple(args.hand_position_range_m)
    # Fixed reset-schedule evaluations are deliberately written directly into
    # the requested directory so the 80 episode folders remain episode_0..79.
    run_output_dir = _create_run_directory(
        args.output_dir,
        flat=args.flat_output or args.reset_schedule is not None,
    )
    episode_csv_path = run_output_dir / "episodes.csv"
    summary_path = run_output_dir / "summary.json"
    completed_rows = _load_completed_rows(episode_csv_path, args.episodes)
    if completed_rows:
        print(
            f"[{args.policy}RandomEval] resuming at episode {len(completed_rows)} "
            f"of {args.episodes}",
            flush=True,
        )
    if len(completed_rows) >= args.episodes:
        print(
            f"[{args.policy}RandomEval] all requested episodes already exist; nothing to run",
            flush=True,
        )
        return
    print(f"[{args.policy}RandomEval] output_dir={run_output_dir}", flush=True)

    env = None
    rows: list[dict[str, object]] = completed_rows
    successes_so_far = sum(int(row.get("success") or 0) for row in rows)
    completed_count = len(rows)
    started = time.perf_counter()
    try:
        previous_background_usd = os.environ.get("TACEX_BACKGROUND_USD")
        os.environ["TACEX_BACKGROUND_USD"] = str(_PI0_BACKGROUND_USD)
        try:
            env = gym.make(args.task, cfg=env_cfg, output_dir=str(run_output_dir))
            raw_env = env.unwrapped
            if is_tavla:
                raw_env.tavla_direct_reset_joints_schedule = tavla_direct_reset_joints_schedule
            if reset_schedule is not None:
                if is_tavla:
                    raw_env.tavla_reset_schedule = reset_schedule
                    raw_env.tavla_reset_schedule_index = completed_count
                else:
                    raw_env.pi0_reset_schedule = reset_schedule
                    raw_env.pi0_reset_schedule_index = completed_count
        finally:
            if previous_background_usd is None:
                os.environ.pop("TACEX_BACKGROUND_USD", None)
            else:
                os.environ["TACEX_BACKGROUND_USD"] = previous_background_usd

        env.reset()
        raw_env = env.unwrapped
        previous_successes = int(getattr(raw_env, "success_times", 0))
        previous_total = int(getattr(raw_env, "total_times", 0))
        reset_info = _reset_snapshot(raw_env)
        episode_steps = 0
        episode_return = 0.0
        best_xy_error = float(reset_info["initial_xy_error_m"])
        minimum_z_disp = float(reset_info["initial_z_disp_m"])
        partial_insertion_seen = False
        partial_insertion_first_step = None
        runtime_stats = _policy_runtime_stats(raw_env, is_tavla)
        inference_peak = int(runtime_stats["inference"])
        failure_peak = int(runtime_stats["failures"])
        timeout_peak = int(runtime_stats["timeouts"])
        force_abort_peak = int(runtime_stats["force_abort"])
        progress = tqdm(
            total=int(round(args.episode_length_s * 10.0)) * args.episodes,
            desc=f"{args.policy.upper()} rollout",
            unit="step",
            dynamic_ncols=True,
        )
        progress.update(int(round(args.episode_length_s * 10.0)) * completed_count)
        atexit.register(progress.close)

        while len(rows) < args.episodes:
            if not simulation_app.is_running():
                raise RuntimeError("Isaac Sim stopped before all episodes completed")

            before_step = _insertion_metrics(raw_env)
            best_xy_error = min(best_xy_error, float(before_step["xy_error_m"]))
            minimum_z_disp = min(minimum_z_disp, float(before_step["z_disp_m"]))
            if bool(before_step["partial_insertion"]):
                if not partial_insertion_seen:
                    partial_insertion_first_step = episode_steps
                partial_insertion_seen = True
            action = torch.zeros(
                (1, raw_env.cfg.action_space),
                dtype=torch.float32,
                device=raw_env.device,
            )
            with torch.inference_mode():
                _, reward, terminated, truncated, _ = env.step(action)
            episode_steps += 1
            episode_return += _scalar(reward)
            runtime_stats = _policy_runtime_stats(raw_env, is_tavla)
            inference_peak = max(inference_peak, int(runtime_stats["inference"]))
            failure_peak = max(failure_peak, int(runtime_stats["failures"]))
            timeout_peak = max(timeout_peak, int(runtime_stats["timeouts"]))
            force_abort_peak = max(force_abort_peak, int(runtime_stats["force_abort"]))
            progress.update(1)
            progress.set_postfix(
                infer=inference_peak,
                latency=f"{float(runtime_stats['latency']):.3f}s",
                failures=failure_peak,
            )

            done = bool(_scalar(terminated)) or bool(_scalar(truncated))
            if not done:
                continue

            current_successes = int(getattr(raw_env, "success_times", 0))
            current_total = int(getattr(raw_env, "total_times", 0))
            if current_total != previous_total + 1:
                raise RuntimeError(
                    f"Expected one completed episode, total changed {previous_total} -> {current_total}"
                )
            success = current_successes == previous_successes + 1
            if current_successes not in (previous_successes, previous_successes + 1):
                raise RuntimeError(
                    f"Unexpected success counter change {previous_successes} -> {current_successes}"
                )

            hole_offset = reset_info["hole_offset_m"]
            hand_offset = reset_info["hand_offset_m"]
            hole_position = reset_info["hole_position_m"]
            fingertip_position = reset_info["fingertip_position_m"]
            row = {
                "episode": len(rows) + 1,
                "episode_dir": f"episode_{len(rows)}",
                "success": int(success),
                "steps": episode_steps,
                "return": episode_return,
                "timed_out": int(bool(_scalar(truncated))),
                "hole_dx_m": hole_offset[0],
                "hole_dy_m": hole_offset[1],
                "hole_dz_m": hole_offset[2],
                "hand_dx_m": hand_offset[0],
                "hand_dy_m": hand_offset[1],
                "hand_dz_m": hand_offset[2],
                "hole_x_m": hole_position[0],
                "hole_y_m": hole_position[1],
                "hole_z_m": hole_position[2],
                "initial_fingertip_x_m": fingertip_position[0],
                "initial_fingertip_y_m": fingertip_position[1],
                "initial_fingertip_z_m": fingertip_position[2],
                "initial_xy_error_m": reset_info["initial_xy_error_m"],
                "initial_z_disp_m": reset_info["initial_z_disp_m"],
                "best_xy_error_m": best_xy_error,
                "minimum_z_disp_m": minimum_z_disp,
                "partial_insertion": int(partial_insertion_seen),
                "partial_insertion_first_step": partial_insertion_first_step,
            }
            if is_tavla:
                row.update(
                    {
                        "tavla_inference_count_peak": inference_peak,
                        "tavla_failures_peak": failure_peak,
                        "tavla_timeouts_peak": timeout_peak,
                        "tavla_force_abort_count_peak": force_abort_peak,
                    }
                )
            else:
                row.update(
                    {
                        "pi0_inference_count_peak": inference_peak,
                        "pi0_failures_peak": failure_peak,
                    }
                )
            rows.append(row)
            _write_episode_csv(episode_csv_path, rows, args.policy)
            successes_so_far += int(success)
            print(
                f"[{args.policy}RandomEval] "
                f"episode={len(rows)}/{args.episodes} success={success} "
                f"partial_insertion={bool(partial_insertion_seen)} "
                f"steps={episode_steps} cumulative={successes_so_far}/{len(rows)} "
                f"hole_offset={hole_offset} hand_offset={hand_offset}",
                flush=True,
            )

            previous_successes = current_successes
            previous_total = current_total
            if len(rows) >= args.episodes:
                break

            # DirectRLEnv has already reset the same environment internally.
            reset_info = _reset_snapshot(raw_env)
            episode_steps = 0
            episode_return = 0.0
            best_xy_error = float(reset_info["initial_xy_error_m"])
            minimum_z_disp = float(reset_info["initial_z_disp_m"])
            partial_insertion_seen = False
            partial_insertion_first_step = None
            runtime_stats = _policy_runtime_stats(raw_env, is_tavla)
            inference_peak = int(runtime_stats["inference"])
            failure_peak = int(runtime_stats["failures"])
            timeout_peak = int(runtime_stats["timeouts"])
            force_abort_peak = int(runtime_stats["force_abort"])

        progress.close()
        success_count = sum(int(row["success"]) for row in rows)
        partial_insertion_count = sum(int(row.get("partial_insertion") or 0) for row in rows)
        unique_resets = len(
            {
                tuple(round(float(row[name]), 9) for name in (
                    "hole_dx_m",
                    "hole_dy_m",
                    "hole_dz_m",
                    "hand_dx_m",
                    "hand_dy_m",
                    "hand_dz_m",
                ))
                for row in rows
            }
        )
        if is_tavla:
            server = {"host": args.tavla_host, "port": args.tavla_port}
            action_start_index = args.tavla_action_start_index
        else:
            server = {"host": args.pi0_host, "port": args.pi0_port}
            action_start_index = 5
        summary = {
            "task": args.task,
            "policy": args.policy,
            "server": server,
            "episodes": len(rows),
            "successes": success_count,
            "failures": len(rows) - success_count,
            "success_rate": success_count / len(rows),
            "success_rate_percent": 100.0 * success_count / len(rows),
            "partial_insertions": partial_insertion_count,
            "partial_insertion_rate": partial_insertion_count / len(rows),
            "partial_insertion_rate_percent": 100.0 * partial_insertion_count / len(rows),
            "partial_insertion_definition": {
                "xy_error_lt_m": 0.003,
                "z_disp_range_m": [0.003, 0.025],
                "meaning": "peg center aligned and peg base entered the 25 mm hole, but not fully inserted",
            },
            "unique_random_resets": unique_resets,
            "episode_length_s": args.episode_length_s,
            "action_start_index": action_start_index,
            "replan_actions": args.replan_actions,
            "seed": args.eval_seed,
            "reset_schedule": (
                str(args.reset_schedule.expanduser().resolve())
                if args.reset_schedule is not None
                else None
            ),
            "hole_position_half_range_m": list(args.hole_position_range_m),
            "hand_position_half_range_m": list(args.hand_position_range_m),
            "mean_episode_steps": float(np.mean([row["steps"] for row in rows])),
            "total_wall_time_s": time.perf_counter() - started,
            "immediate_stop_on_success": True,
            "single_simulation_process": True,
            "data_collection_enabled": True,
            "saved_failed_episodes": True,
            "output_dir": str(run_output_dir.resolve()),
            "episode_directories": [str(row["episode_dir"]) for row in rows],
        }
        if is_tavla:
            summary.update(
                {
                    "teacher_hold_steps": int(getattr(env_cfg, "teacher_hold_steps", 3)),
                    "force_input": "gated_wrench_final" if args.tavla_force_gate else "wrench_final",
                    "force_gate_enabled": bool(args.tavla_force_gate),
                    "force_gate_threshold_n": float(args.tavla_force_gate_threshold_n),
                    "force_gate_confirm_steps": int(args.tavla_force_gate_confirm_steps),
                    "force_gate_release_steps": int(args.tavla_force_gate_release_steps),
                    "force_gate_precontact_wrench": [
                        float(value) for value in args.tavla_force_gate_precontact_wrench
                    ],
                    "force_component_order": ["Fx", "Fy", "Fz", "Tx", "Ty", "Tz"],
                    "force_data_files": [
                        "wrench_raw.csv",
                        "wrench_anchor.csv",
                        "wrench_base.csv",
                        "wrench_corrected.csv",
                        "wrench_final.csv",
                        "tavla_wrench_base.csv",
                        "tavla_wrench_final.csv",
                        "tavla_server_effort.csv",
                        "tavla_server_effort_matches_final.csv",
                        "tavla_policy_wrench.csv",
                        "tavla_force_gate_active.csv",
                        "tavla_force_gate_contact_norm.csv",
                    ],
                }
            )
        summary_path.write_text(
            json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        print(
            f"[{args.policy}RandomEvalSummary] successes={success_count}/{len(rows)} "
            f"success_rate={summary['success_rate_percent']:.2f}% "
            f"unique_resets={unique_resets}",
            flush=True,
        )
    finally:
        if env is not None:
            env.close()
        simulation_app.close()


if __name__ == "__main__":
    main()
