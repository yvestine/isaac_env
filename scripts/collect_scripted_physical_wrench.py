#!/usr/bin/env python3
"""Collect a scripted peg-in-hole trajectory with three independent wrench sources.

This entry point is deliberately independent of the real-trajectory rollout.
The robot is advanced only by the normal task-space effort controller and
PhysX; no joint or asset pose is written while the trajectory is running.
"""

from __future__ import annotations

import argparse
import csv
import json
import traceback
from pathlib import Path

import numpy as np
import torch
from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task", default="TacEx-RealSim-PegInsert-Direct-v0")
parser.add_argument("--output-dir", type=Path, default=Path("outputs/scripted_physical_wrench/traj_0"))
parser.add_argument("--fps", type=float, default=10.0)
parser.add_argument("--hold-seconds", type=float, default=3.0)
parser.add_argument("--move-seconds", type=float, default=3.0)
parser.add_argument("--descend-seconds", type=float, default=3.0)
parser.add_argument("--insert-seconds", type=float, default=3.0)
parser.add_argument("--wiggle-seconds", type=float, default=3.0)
parser.add_argument("--retract-seconds", type=float, default=3.0)
parser.add_argument("--approach-height-m", type=float, default=0.030)
parser.add_argument("--precontact-height-m", type=float, default=0.003)
parser.add_argument("--lateral-offset-m", type=float, default=0.0003)
parser.add_argument("--wiggle-amplitude-m", type=float, default=0.0003)
parser.add_argument(
    "--assembly-z-offset-m",
    type=float,
    default=0.0,
    help="Offset from the task's known fully assembled peg pose; keep at zero first.",
)
parser.add_argument("--ema-alpha", type=float, default=0.20)
parser.add_argument("--dls-lambda", type=float, default=0.05)
parser.add_argument("--contact-force-limit-n", type=float, default=30.0)
parser.add_argument("--seed", type=int, default=0)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

# This script always records both cameras.
if hasattr(args, "enable_cameras"):
    args.enable_cameras = True
args.kit_args = (
    f"{getattr(args, 'kit_args', '')} "
    "--/log/level=warning --/log/outputStreamLevel=warning "
    "--/rtx-transient/resourcemanager/enableGeometryStreaming=false"
).strip()

simulation_app = AppLauncher(args).app

# Isaac Sim 6 compatibility modules must be exposed after AppLauncher starts.
import isaacsim  # noqa: E402

_ISAACSIM_EXTS = Path(isaacsim.__file__).resolve().parent / "exts"
for _extension_name in ("isaacsim.core.prims", "isaacsim.core.utils"):
    _extension_pkg = _ISAACSIM_EXTS / _extension_name / "isaacsim"
    if _extension_pkg.is_dir() and str(_extension_pkg) not in isaacsim.__path__:
        isaacsim.__path__.append(str(_extension_pkg))

import h5py  # noqa: E402
from pxr import Usd  # noqa: E402

import tacex_tasks  # noqa: E402,F401
from isaaclab_tasks.direct.factory import factory_utils  # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402
from tacex_tasks import torch_compat as torch_utils  # noqa: E402
from tacex_tasks.real2sim.realsim_env import RealSimEnv, _write_h264_mp4  # noqa: E402


WRENCH_COLUMNS = ("Fx", "Fy", "Fz", "Tx", "Ty", "Tz")


def as_torch(value, *, device: str) -> torch.Tensor:
    value = value.torch if hasattr(value, "torch") else value
    return torch.as_tensor(value, dtype=torch.float32, device=device)


def as_numpy(value) -> np.ndarray:
    if hasattr(value, "numpy"):
        value = value.numpy()
    elif hasattr(value, "torch"):
        value = value.torch
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=np.float64)


def smoothstep(value: float) -> float:
    value = float(np.clip(value, 0.0, 1.0))
    return value * value * (3.0 - 2.0 * value)


def ema_filter(values: np.ndarray, alpha: float) -> np.ndarray:
    if not 0.0 < alpha <= 1.0:
        raise ValueError("--ema-alpha must be in (0, 1]")
    result = np.empty_like(values)
    result[0] = values[0]
    for index in range(1, len(values)):
        result[index] = alpha * values[index] + (1.0 - alpha) * result[index - 1]
    return result


def write_matrix_csv(path: Path, values: np.ndarray, columns: tuple[str, ...]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(columns)
        writer.writerows(np.asarray(values).tolist())


def capture_cameras(env: RealSimEnv) -> tuple[np.ndarray, np.ndarray]:
    env.sim.render()
    env.tiled_camera.update(env.physics_dt, force_recompute=True)
    env.wrist_tiled_camera.update(env.physics_dt, force_recompute=True)

    def bgr(camera) -> np.ndarray:
        rgb = camera.data.output["rgb"][0].detach().cpu().numpy()[..., :3]
        return np.ascontiguousarray(rgb[..., ::-1], dtype=np.uint8)

    return bgr(env.tiled_camera), bgr(env.wrist_tiled_camera)


class PegHoleContactProbe:
    """Diagnostic peg-hole contact wrench expressed in the robot base at K."""

    def __init__(self, env: RealSimEnv) -> None:
        def report_paths(root_path: str) -> list[str]:
            root = env.sim.stage.GetPrimAtPath(root_path)
            return [
                str(prim.GetPath())
                for prim in Usd.PrimRange(root)
                if "PhysxContactReportAPI" in prim.GetAppliedSchemas()
            ]

        held_paths = report_paths("/World/envs/env_0/HeldAsset")
        fixed_paths = report_paths("/World/envs/env_0/FixedAsset")
        if not held_paths or not fixed_paths:
            raise RuntimeError(
                f"peg/hole contact-report bodies missing: held={held_paths}, fixed={fixed_paths}"
            )
        physics_view = env.sim.physics_manager.get_physics_sim_view()
        if physics_view is None:
            raise RuntimeError("PhysX simulation tensor view is unavailable")
        self.view = physics_view.create_rigid_contact_view(
            held_paths,
            filter_patterns=[fixed_paths for _ in held_paths],
            max_contact_data_count=256,
        )

    @staticmethod
    def active_indices(counts, starts, capacity: int) -> np.ndarray:
        counts = as_numpy(counts).reshape(-1).astype(np.int64)
        starts = as_numpy(starts).reshape(-1).astype(np.int64)
        indices: list[int] = []
        for count, start in zip(counts, starts, strict=False):
            if count > 0 and 0 <= start < capacity:
                indices.extend(range(int(start), min(int(start + count), capacity)))
        return np.asarray(indices, dtype=np.int64)

    def read(self, env: RealSimEnv, dt: float) -> tuple[np.ndarray, int]:
        matrix = as_numpy(self.view.get_contact_force_matrix(dt=dt))
        entries = matrix[0] if matrix.ndim == 3 else matrix
        entries = np.asarray(entries, dtype=np.float64).reshape(-1, 3)
        normal_world = entries.sum(axis=0) if len(entries) else np.zeros(3)

        normal_mag, normal_points, normal_dirs, _, normal_counts, normal_starts = (
            self.view.get_contact_data(dt=dt)
        )
        friction, friction_points, friction_counts, friction_starts = self.view.get_friction_data(dt=dt)
        normal_mag = as_numpy(normal_mag).reshape(-1)
        normal_points = as_numpy(normal_points).reshape(-1, 3)
        normal_dirs = as_numpy(normal_dirs).reshape(-1, 3)
        friction = as_numpy(friction).reshape(-1, 3)
        friction_points = as_numpy(friction_points).reshape(-1, 3)

        normal_ids = self.active_indices(
            normal_counts,
            normal_starts,
            min(len(normal_mag), len(normal_points), len(normal_dirs)),
        )
        friction_ids = self.active_indices(
            friction_counts,
            friction_starts,
            min(len(friction), len(friction_points)),
        )
        friction_world = friction[friction_ids].sum(axis=0) if len(friction_ids) else np.zeros(3)
        total_world = normal_world + friction_world
        k_world = env.fingertip_midpoint_pos[0].detach().cpu().numpy().astype(np.float64)
        torque_world = np.zeros(3, dtype=np.float64)
        if len(normal_ids):
            point_forces = normal_mag[normal_ids, None] * normal_dirs[normal_ids]
            if np.dot(point_forces.sum(axis=0), normal_world) < 0.0:
                point_forces *= -1.0
            torque_world += np.cross(normal_points[normal_ids] - k_world, point_forces).sum(axis=0)
        if len(friction_ids):
            point_forces = friction[friction_ids]
            if np.dot(point_forces.sum(axis=0), friction_world) < 0.0:
                point_forces *= -1.0
            torque_world += np.cross(friction_points[friction_ids] - k_world, point_forces).sum(axis=0)

        root_quat = as_torch(env._robot.data.root_quat_w, device=env.device)
        world_to_base = torch_utils.quat_conjugate(root_quat)

        def to_base(vector: np.ndarray) -> np.ndarray:
            tensor = torch.as_tensor(vector, dtype=torch.float32, device=env.device).view(1, 3)
            return torch_utils.quat_apply(world_to_base, tensor)[0].detach().cpu().numpy()

        wrench = np.concatenate((to_base(total_world), to_base(torque_world))).astype(np.float64)
        count = int(np.count_nonzero(np.linalg.norm(entries, axis=1) > 1.0e-8))
        return wrench, count


class FrankaTorqueObserver:
    """Franka-style model residual mapped from joint torque to a K-frame wrench."""

    def __init__(self, env: RealSimEnv, damping: float) -> None:
        self.env = env
        self.view = env._robot.root_physx_view
        self.damping = float(damping)
        if self.damping <= 0.0:
            raise ValueError("--dls-lambda must be positive")

    def read(self) -> dict[str, np.ndarray]:
        env = self.env
        device = env.device
        mass = as_torch(self.view.get_generalized_mass_matrices(), device=device)[:, :7, :7]
        coriolis = as_torch(
            self.view.get_coriolis_and_centrifugal_compensation_forces(), device=device
        )[:, :7]
        gravity = as_torch(self.view.get_gravity_compensation_forces(), device=device)[:, :7]
        actuation = as_torch(self.view.get_dof_actuation_forces(), device=device)[:, :7]
        projected = as_torch(self.view.get_dof_projected_joint_forces(), device=device)[:, :7]
        acceleration = as_torch(env._robot.data.joint_acc, device=device)[:, :7]

        model = torch.bmm(mass, acceleration.unsqueeze(-1)).squeeze(-1) + coriolis + gravity
        # Fixed-base rigid-body balance: M qdd + C + G = tau_act + tau_ext.
        tau_external = model - actuation
        jacobian = env.fingertip_midpoint_jacobian[:, :6, :7]
        identity = torch.eye(6, dtype=torch.float32, device=device).unsqueeze(0)
        lhs = torch.bmm(jacobian, jacobian.transpose(1, 2)) + self.damping**2 * identity
        rhs = torch.bmm(jacobian, tau_external.unsqueeze(-1))
        wrench_world = torch.linalg.solve(lhs, rhs).squeeze(-1)

        root_quat = as_torch(env._robot.data.root_quat_w, device=device)
        world_to_base = torch_utils.quat_conjugate(root_quat)
        force_base = torch_utils.quat_apply(world_to_base, wrench_world[:, :3])
        torque_base = torch_utils.quat_apply(world_to_base, wrench_world[:, 3:])
        wrench_base = torch.cat((force_base, torque_base), dim=-1)

        env._update_wrench()
        incoming = env.wrench_base

        def row(value: torch.Tensor) -> np.ndarray:
            return value[0].detach().cpu().numpy().astype(np.float64, copy=True)

        return {
            "wrench": row(wrench_base),
            "tau_external": row(tau_external),
            "tau_model": row(model),
            "tau_actuation": row(actuation),
            "tau_projected": row(projected),
            "incoming": row(incoming),
        }


def phase_target(
    phase: str,
    fraction: float,
    start: np.ndarray,
    end: np.ndarray,
    wiggle_amplitude: float,
) -> np.ndarray:
    if phase == "wiggle":
        angle = 4.0 * np.pi * fraction
        target = end.copy()
        target[0] += wiggle_amplitude * np.sin(angle)
        target[1] += wiggle_amplitude * np.cos(angle)
        return target
    return start + smoothstep(fraction) * (end - start)


def save_h5(output_path: Path, data: dict[str, np.ndarray], metadata: dict) -> None:
    with h5py.File(output_path, "w") as handle:
        handle.attrs["format"] = "scripted_physical_wrench_v1"
        handle.attrs["metadata_json"] = json.dumps(metadata, ensure_ascii=False)
        for path, value in data.items():
            parent, name = path.rsplit("/", 1)
            group = handle.require_group(parent)
            if value.dtype.kind in {"U", "O"}:
                group.create_dataset(name, data=value.astype(h5py.string_dtype("utf-8")))
            else:
                group.create_dataset(name, data=value, compression="gzip", compression_opts=4)


def make_plots(output_dir: Path, timestamps: np.ndarray, phases: np.ndarray, streams: dict[str, np.ndarray]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    observer = streams["observer"]
    incoming = streams["incoming"]
    contact = streams["contact"]
    figure, axes = plt.subplots(3, 1, figsize=(13, 9), sharex=True)
    for index, label in enumerate(("Fx", "Fy", "Fz")):
        axes[index].plot(timestamps, observer[:, index], label="joint-torque observer", linewidth=1.8)
        axes[index].plot(timestamps, incoming[:, index], label="incoming joint wrench", alpha=0.7)
        axes[index].plot(timestamps, contact[:, index], label="peg-hole contact", alpha=0.7)
        axes[index].axhline(0.0, color="black", linewidth=0.6)
        axes[index].set_ylabel(f"{label} [N]")
        axes[index].grid(alpha=0.25)
    axes[0].legend(ncol=3, fontsize=8)
    axes[-1].set_xlabel("time [s]")
    figure.suptitle("Scripted physical insertion: three wrench sources (robot base, at K)")
    figure.tight_layout()
    figure.savefig(output_dir / "force_components.png", dpi=170)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(13, 4.5))
    for values, label in ((observer, "joint-torque observer"), (incoming, "incoming"), (contact, "contact")):
        axis.plot(timestamps, np.linalg.norm(values[:, :3], axis=1), label=label)
    boundaries = np.flatnonzero(phases[1:] != phases[:-1]) + 1
    for index in boundaries:
        axis.axvline(timestamps[index], color="#888888", linewidth=0.7, alpha=0.6)
        axis.text(timestamps[index], axis.get_ylim()[1], phases[index], rotation=90, va="top", fontsize=7)
    axis.set_xlabel("time [s]")
    axis.set_ylabel("|F| [N]")
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_dir / "force_norm_contact_timeline.png", dpi=170)
    plt.close(figure)


def make_wrist_force_video(
    path: Path,
    wrist_frames: list[np.ndarray],
    timestamps: np.ndarray,
    phases: np.ndarray,
    observer: np.ndarray,
    fps: float,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_agg import FigureCanvasAgg

    frames: list[np.ndarray] = []
    y_limit = max(float(np.percentile(np.abs(observer[:, :3]), 99.0)), 1.0)
    for index, wrist_bgr in enumerate(wrist_frames):
        figure, axes = plt.subplots(1, 2, figsize=(12.8, 4.8), gridspec_kw={"width_ratios": [1.0, 1.25]})
        axes[0].imshow(wrist_bgr[..., ::-1])
        axes[0].set_title(f"wrist | {phases[index]} | {timestamps[index]:.2f}s")
        axes[0].axis("off")
        for component, label in enumerate(("Fx", "Fy", "Fz")):
            axes[1].plot(timestamps[: index + 1], observer[: index + 1, component], label=label)
        axes[1].axhline(0.0, color="black", linewidth=0.6)
        axes[1].set_xlim(timestamps[0], timestamps[-1])
        axes[1].set_ylim(-y_limit, y_limit)
        axes[1].set_xlabel("time [s]")
        axes[1].set_ylabel("joint-torque observer [N]")
        axes[1].grid(alpha=0.25)
        axes[1].legend(loc="upper left")
        figure.tight_layout()
        canvas = FigureCanvasAgg(figure)
        canvas.draw()
        rgba = np.asarray(canvas.buffer_rgba())
        frames.append(np.ascontiguousarray(rgba[..., :3][..., ::-1]))
        plt.close(figure)
    _write_h264_mp4(path, frames, fps)


def main() -> None:
    if args.fps <= 0.0:
        raise ValueError("--fps must be positive")
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    cfg = parse_env_cfg(args.task, device=args.device, num_envs=1, use_fabric=True)
    cfg.scene.num_envs = 1
    cfg.enable_cameras = True
    cfg.policy_cfg = None
    cfg.teacher_policy_cfg = None
    cfg.teacher_eval_only = True
    cfg.data_collect_cfg["collect_data"] = False
    cfg.data_collect_cfg["save_tavla_hdf5"] = False
    cfg.data_collect_cfg["num_trajectories"] = 1_000_000
    cfg.episode_length_s = 60.0
    cfg.task.fixed_asset_init_pos_noise = [0.0, 0.0, 0.0]
    cfg.task.hand_init_pos_noise = [0.0, 0.0, 0.0]
    cfg.task.hand_init_orn_noise = [0.0, 0.0, 0.0]
    cfg.task.held_asset_pos_noise = [0.0, 0.0, 0.0]
    cfg.task.fixed_asset_init_orn_range_deg = 0.0
    cfg.sim.render_interval = max(int(round(1.0 / (args.fps * float(cfg.sim.dt)))), 1)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    env = None
    try:
        env = RealSimEnv(cfg, render_mode="rgb_array", output_dir=str(output_dir))
        env.reset(seed=args.seed)
        env._compute_intermediate_values(dt=env.physics_dt)
        contact_probe = PegHoleContactProbe(env)
        observer = FrankaTorqueObserver(env, args.dls_lambda)
        physics_dt = float(env.sim.get_physics_dt())
        sample_stride = max(int(round(1.0 / (args.fps * physics_dt))), 1)

        current_ee = env.fingertip_midpoint_pos[0].detach().cpu().numpy().astype(np.float64)
        target_quat = env.fingertip_midpoint_quat.detach().clone()
        held_base, _ = factory_utils.get_held_base_pose(
            env.held_pos,
            env.held_quat,
            env.cfg_task.name,
            env.cfg_task.fixed_asset_cfg,
            env.num_envs,
            env.device,
        )
        target_held_base, _ = factory_utils.get_target_held_base_pose(
            env.fixed_pos,
            env.fixed_quat,
            env.cfg_task.name,
            env.cfg_task.fixed_asset_cfg,
            env.num_envs,
            env.device,
        )
        assembly_delta = (target_held_base - held_base)[0].detach().cpu().numpy().astype(np.float64)
        assembly_ee = current_ee + assembly_delta
        assembly_ee[2] += args.assembly_z_offset_m
        lateral = np.array([args.lateral_offset_m, 0.0, 0.0], dtype=np.float64)
        preinsert = assembly_ee + lateral + np.array([0.0, 0.0, args.approach_height_m])
        precontact = assembly_ee + lateral + np.array([0.0, 0.0, args.precontact_height_m])
        assembled = assembly_ee.copy()
        retract = assembly_ee + np.array([0.0, 0.0, args.approach_height_m])
        phase_specs = [
            ("hold", args.hold_seconds, current_ee, current_ee),
            ("approach", args.move_seconds, current_ee, preinsert),
            ("descend", args.descend_seconds, preinsert, precontact),
            ("insert", args.insert_seconds, precontact, assembled),
            ("wiggle", args.wiggle_seconds, assembled, assembled),
            ("retract", args.retract_seconds, assembled, retract),
        ]
        gripper_target = env.joint_pos[:, 7:9].mean(dim=1, keepdim=True).detach().clone()

        rows: dict[str, list] = {
            "time": [], "phase": [], "target": [], "ee": [], "peg": [], "hole": [],
            "q": [], "qd": [], "observer": [], "incoming": [], "contact": [], "count": [],
            "tau_external": [], "tau_model": [], "tau_actuation": [], "tau_projected": [],
        }
        front_frames: list[np.ndarray] = []
        wrist_frames: list[np.ndarray] = []
        accum: dict[str, list[np.ndarray]] = {key: [] for key in (
            "observer", "incoming", "contact", "tau_external", "tau_model", "tau_actuation", "tau_projected"
        )}
        accum_count: list[int] = []
        elapsed = 0.0
        global_step = 0
        safety_latched = False
        safety_z = float("-inf")

        for phase, duration, start, end in phase_specs:
            steps = max(int(round(duration / physics_dt)), 1)
            for local_step in range(steps):
                fraction = (local_step + 1) / steps
                target_np = phase_target(
                    phase, fraction, np.asarray(start), np.asarray(end), args.wiggle_amplitude_m
                )
                if safety_latched and phase in {"descend", "insert", "wiggle", "retract"}:
                    target_np[2] = max(target_np[2], safety_z)
                target = torch.as_tensor(target_np, dtype=torch.float32, device=env.device).view(1, 3)
                env.generate_ctrl_signals(target, target_quat, gripper_target)
                env.scene.write_data_to_sim()
                env.sim.step(render=False)
                env.scene.update(dt=physics_dt)
                env._compute_intermediate_values(dt=physics_dt)

                torque_sample = observer.read()
                contact_wrench, contact_count = contact_probe.read(env, physics_dt)
                for key in ("wrench", "incoming", "tau_external", "tau_model", "tau_actuation", "tau_projected"):
                    destination = "observer" if key == "wrench" else key
                    accum[destination].append(torque_sample[key])
                accum["contact"].append(contact_wrench)
                accum_count.append(contact_count)

                if (
                    not safety_latched
                    and phase in {"descend", "insert", "wiggle"}
                    and np.linalg.norm(contact_wrench[:3]) > args.contact_force_limit_n
                ):
                    safety_latched = True
                    safety_z = float(env.fingertip_midpoint_pos[0, 2].detach().cpu())
                    print(
                        f"[Safety] contact force exceeded {args.contact_force_limit_n:.1f} N; "
                        f"downward target clamped at z={safety_z:.6f} m",
                        flush=True,
                    )

                elapsed += physics_dt
                global_step += 1
                if global_step % sample_stride != 0:
                    continue

                rows["time"].append(elapsed)
                rows["phase"].append(phase)
                rows["target"].append(target_np.copy())
                rows["ee"].append(np.concatenate((
                    env.fingertip_midpoint_pos[0].detach().cpu().numpy(),
                    env.fingertip_midpoint_quat[0].detach().cpu().numpy(),
                )))
                rows["peg"].append(np.concatenate((
                    env.held_pos[0].detach().cpu().numpy(), env.held_quat[0].detach().cpu().numpy()
                )))
                rows["hole"].append(np.concatenate((
                    env.fixed_pos[0].detach().cpu().numpy(), env.fixed_quat[0].detach().cpu().numpy()
                )))
                rows["q"].append(env.joint_pos[0, :7].detach().cpu().numpy().copy())
                rows["qd"].append(env.joint_vel[0, :7].detach().cpu().numpy().copy())
                for key in accum:
                    rows[key].append(np.mean(np.stack(accum[key]), axis=0))
                    accum[key].clear()
                rows["count"].append(max(accum_count))
                accum_count.clear()
                front, wrist = capture_cameras(env)
                front_frames.append(front)
                wrist_frames.append(wrist)
                print(
                    f"[Collect] t={elapsed:6.2f}s phase={phase:<8} "
                    f"|F_contact|={np.linalg.norm(rows['contact'][-1][:3]):8.3f}N",
                    flush=True,
                )

        arrays = {key: np.asarray(value) for key, value in rows.items()}
        hold_mask = arrays["phase"] == "hold"
        if hold_mask.sum() < 2:
            raise RuntimeError("not enough hold samples for wrench baseline")
        baseline = np.median(arrays["observer"][hold_mask], axis=0)
        observer_unfiltered = arrays["observer"] - baseline
        observer_filtered = ema_filter(observer_unfiltered, args.ema_alpha)
        incoming_baseline = np.median(arrays["incoming"][hold_mask], axis=0)
        incoming_corrected = arrays["incoming"] - incoming_baseline

        metadata = {
            "source": "standalone scripted physical trajectory; existing rollout scripts untouched",
            "training_wrench": "joint torque residual: M*qdd+C+G-tau_act, DLS mapped through J_K",
            "incoming_wrench": "diagnostic only",
            "contact_wrench": "diagnostic only",
            "coordinate_contract": "robot base expressed, moment referenced at fingertip stiffness frame K",
            "physics_dt_s": physics_dt,
            "sample_fps": args.fps,
            "dls_lambda": args.dls_lambda,
            "ema_alpha": args.ema_alpha,
            "observer_baseline": baseline.tolist(),
            "incoming_diagnostic_baseline": incoming_baseline.tolist(),
            "known_assembly_ee_position_m": assembly_ee.tolist(),
            "safety_latched": safety_latched,
            "no_runtime_pose_writes": True,
        }
        h5_data = {
            "time/timestamp_s": arrays["time"].astype(np.float64),
            "time/phase": arrays["phase"],
            "obs/state/ee_pose": arrays["ee"].astype(np.float32),
            "obs/state/joint_pos": arrays["q"].astype(np.float32),
            "obs/state/joint_vel": arrays["qd"].astype(np.float32),
            "obs/state/ee_wrench_base": observer_filtered.astype(np.float32),
            "action/target_ee_position": arrays["target"].astype(np.float32),
            "sim/pose/peg": arrays["peg"].astype(np.float32),
            "sim/pose/hole": arrays["hole"].astype(np.float32),
            "sim/force/joint_torque_observer_raw": arrays["observer"].astype(np.float32),
            "sim/force/joint_torque_observer_baseline_corrected": observer_unfiltered.astype(np.float32),
            "sim/force/joint_torque_observer_filtered": observer_filtered.astype(np.float32),
            "sim/force/incoming_joint_wrench_base_at_K_raw": arrays["incoming"].astype(np.float32),
            "sim/force/incoming_joint_wrench_base_at_K": incoming_corrected.astype(np.float32),
            "sim/force/contact_wrench_base_at_K": arrays["contact"].astype(np.float32),
            "sim/force/contact_count": arrays["count"].astype(np.int32),
            "sim/joint/tau_external_residual": arrays["tau_external"].astype(np.float32),
            "sim/joint/tau_model": arrays["tau_model"].astype(np.float32),
            "sim/joint/tau_actuation": arrays["tau_actuation"].astype(np.float32),
            "sim/joint/tau_projected": arrays["tau_projected"].astype(np.float32),
        }
        save_h5(output_dir / "data.h5", h5_data, metadata)
        write_matrix_csv(output_dir / "O_F_ext_hat_K_sim.csv", observer_filtered, WRENCH_COLUMNS)
        write_matrix_csv(output_dir / "wrench_observer_raw.csv", arrays["observer"], WRENCH_COLUMNS)
        write_matrix_csv(output_dir / "incoming_joint_wrench_base_at_K_raw.csv", arrays["incoming"], WRENCH_COLUMNS)
        write_matrix_csv(output_dir / "incoming_joint_wrench_base_at_K.csv", incoming_corrected, WRENCH_COLUMNS)
        write_matrix_csv(output_dir / "contact_wrench_base_at_K.csv", arrays["contact"], WRENCH_COLUMNS)
        (output_dir / "metadata.json").write_text(
            json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        make_plots(
            output_dir,
            arrays["time"],
            arrays["phase"],
            {"observer": observer_filtered, "incoming": incoming_corrected, "contact": arrays["contact"]},
        )
        _write_h264_mp4(output_dir / "front_camera.mp4", front_frames, args.fps)
        _write_h264_mp4(output_dir / "wrist_camera.mp4", wrist_frames, args.fps)
        make_wrist_force_video(
            output_dir / "wrist_force.mp4",
            wrist_frames,
            arrays["time"],
            arrays["phase"],
            observer_filtered,
            args.fps,
        )
        summary = {
            "frames": int(len(arrays["time"])),
            "observer_force_p95_n": float(np.percentile(np.linalg.norm(observer_filtered[:, :3], axis=1), 95)),
            "incoming_force_p95_n": float(np.percentile(np.linalg.norm(incoming_corrected[:, :3], axis=1), 95)),
            "contact_force_p95_n": float(np.percentile(np.linalg.norm(arrays["contact"][:, :3], axis=1), 95)),
            "contact_frames": int(np.count_nonzero(arrays["count"])),
            "safety_latched": safety_latched,
        }
        (output_dir / "summary.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        print(f"[DONE] data: {output_dir / 'data.h5'}", flush=True)
        print(f"[DONE] visualization: {output_dir / 'force_components.png'}", flush=True)
        print(f"[DONE] video: {output_dir / 'wrist_force.mp4'}", flush=True)
    except BaseException:
        traceback.print_exc()
        raise
    finally:
        if env is not None:
            env.close()
        simulation_app.close()


if __name__ == "__main__":
    main()
