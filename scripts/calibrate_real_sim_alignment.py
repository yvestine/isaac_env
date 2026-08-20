"""Fit and evaluate a rigid pose alignment from real_data against Isaac Sim FK.

This script is deliberately a kinematic check.  It writes each recorded joint
configuration directly to the robot articulation, calls Isaac Sim FK, and
does not run the controller, contact replay, or PPO policy.

The fitted position transform is

    p_real = R_sim_to_real @ p_sim + t_sim_to_real

The orientation offset is reported separately because the recorded real EE
pose and the simulator fingertip pose can use different tool-frame origins.
Quaternions use Isaac Sim's wxyz convention.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--data-dir", type=Path, default=Path("real_data"))
parser.add_argument("--output-dir", type=Path, default=Path("outputs/real_data_alignment"))
parser.add_argument("--task", type=str, default="TacEx-RealSim-PegInsert-Direct-v0")
parser.add_argument("--joint-offsets", type=str, default="0,0,0,0,0,0,0")
parser.add_argument("--joint-signs", type=str, default="1,1,1,1,1,1,1")
parser.add_argument("--max-trajectories", type=int, default=0, help="0 means all traj_* directories")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

simulation_app = AppLauncher(args).app

import h5py  # noqa: E402
import torch  # noqa: E402
import tacex_tasks  # noqa: E402,F401
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

from tacex_tasks.real2sim.realsim_env import RealSimEnv  # noqa: E402


def parse_vector(value: str, expected: int, name: str) -> np.ndarray:
    result = np.asarray([float(part) for part in value.split(",")], dtype=np.float64)
    if result.shape != (expected,):
        raise ValueError(f"{name} must contain {expected} comma-separated values, got {value!r}")
    return result


def quat_normalize(q: np.ndarray) -> np.ndarray:
    return q / np.linalg.norm(q, axis=-1, keepdims=True).clip(min=1.0e-12)


def quat_conjugate(q: np.ndarray) -> np.ndarray:
    result = q.copy()
    result[..., 1:] *= -1.0
    return result


def quat_multiply(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    w1, x1, y1, z1 = np.moveaxis(q1, -1, 0)
    w2, x2, y2, z2 = np.moveaxis(q2, -1, 0)
    return np.stack(
        (
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ),
        axis=-1,
    )


def quaternion_mean(quaternions: np.ndarray) -> np.ndarray:
    """Markley mean, after making quaternion signs consistent."""
    q = quat_normalize(np.asarray(quaternions, dtype=np.float64))
    reference = q[0]
    q = np.where((q @ reference)[:, None] < 0.0, -q, q)
    accumulator = q.T @ q
    eigenvalues, eigenvectors = np.linalg.eigh(accumulator)
    mean = eigenvectors[:, np.argmax(eigenvalues)]
    if mean[0] < 0.0:
        mean = -mean
    return quat_normalize(mean)


def quat_angle_deg(q: np.ndarray) -> np.ndarray:
    q = quat_normalize(q)
    return np.rad2deg(2.0 * np.arctan2(np.linalg.norm(q[..., 1:], axis=-1), np.abs(q[..., 0])))


def fit_rigid_transform(source: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return R,t minimizing ||target - (source @ R.T + t)||."""
    source_center = source.mean(axis=0)
    target_center = target.mean(axis=0)
    source_zero = source - source_center
    target_zero = target - target_center
    covariance = source_zero.T @ target_zero
    u, _, vt = np.linalg.svd(covariance)
    rotation = vt.T @ u.T
    if np.linalg.det(rotation) < 0.0:
        vt[-1, :] *= -1.0
        rotation = vt.T @ u.T
    translation = target_center - rotation @ source_center
    return rotation, translation


def scalar_metrics(values: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    return {
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "rmse": float(np.sqrt(np.mean(values**2))),
        "p95": float(np.percentile(values, 95.0)),
        "max": float(np.max(values)),
    }


def load_trajectory(path: Path) -> dict[str, np.ndarray]:
    with h5py.File(path / "data.h5", "r") as h5:
        q = np.asarray(h5["obs/state/joint_pos"][:], dtype=np.float64)
        real_pose = np.asarray(h5["obs/state/ee_pose"][:], dtype=np.float64)
        timestamps = np.asarray(h5["timestamps"][:], dtype=np.float64)
        wrench = np.asarray(h5["obs/state/ee_wrench_base"][:], dtype=np.float64)

    if q.ndim != 2 or q.shape[1] != 7:
        raise ValueError(f"{path}: expected joint_pos shape (N,7), got {q.shape}")
    if real_pose.shape != (len(q), 7):
        raise ValueError(f"{path}: expected ee_pose shape {(len(q), 7)}, got {real_pose.shape}")
    if timestamps.shape != (len(q),):
        raise ValueError(f"{path}: timestamps do not match q: {timestamps.shape} vs {q.shape}")
    if wrench.shape != (len(q), 6):
        raise ValueError(f"{path}: expected ee_wrench_base shape {(len(q), 6)}, got {wrench.shape}")
    if not all(np.isfinite(x).all() for x in (q, real_pose, timestamps, wrench)):
        raise ValueError(f"{path}: non-finite values found")

    real_pose[:, 3:] = quat_normalize(real_pose[:, 3:])
    return {"q": q, "real_pose": real_pose, "timestamps": timestamps, "wrench": wrench}


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    data_dirs = sorted(
        (path for path in args.data_dir.glob("traj_*") if (path / "data.h5").is_file()),
        key=lambda path: int(path.name.split("_")[-1]),
    )
    if args.max_trajectories > 0:
        data_dirs = data_dirs[: args.max_trajectories]
    if not data_dirs:
        raise FileNotFoundError(f"No traj_*/data.h5 found under {args.data_dir}")

    joint_offsets = parse_vector(args.joint_offsets, 7, "joint-offsets")
    joint_signs = parse_vector(args.joint_signs, 7, "joint-signs")
    if not np.all(np.isin(joint_signs, (-1.0, 1.0))):
        raise ValueError("joint-signs must contain only -1 or 1")

    trajectories = [(path.name, load_trajectory(path)) for path in data_dirs]
    all_q = np.concatenate([item["q"] for _, item in trajectories], axis=0)
    all_real_pose = np.concatenate([item["real_pose"] for _, item in trajectories], axis=0)

    env_cfg = parse_env_cfg(args.task, device=args.device, num_envs=1, use_fabric=True)
    env_cfg.scene.num_envs = 1
    env_cfg.policy_cfg = None
    env_cfg.data_collect_cfg["collect_data"] = False
    env_cfg.data_collect_cfg["immediate_stop"] = False
    env_cfg.data_collect_cfg["save_failed_trajectory"] = False
    env_cfg.episode_length_s = 3600.0
    env_cfg.ctrl.reset_joints = (joint_signs * (all_q[0] - joint_offsets)).tolist()

    env = RealSimEnv(env_cfg, render_mode=None, output_dir=str(args.output_dir))
    env.reset()

    sim_poses: list[np.ndarray] = []
    for index, q_real in enumerate(all_q):
        q_sim = joint_signs * (q_real - joint_offsets)
        q_tensor = torch.as_tensor(q_sim, dtype=torch.float32, device=env.device).view(1, 7)
        full_q = env.joint_pos.clone()
        full_q[:, :7] = q_tensor
        zero_velocity = torch.zeros_like(full_q)
        env._robot.write_joint_state_to_sim(full_q, zero_velocity)
        env.ctrl_target_joint_pos[:] = full_q
        env.scene.write_data_to_sim()
        env.sim.forward()
        env.scene.update(dt=env.physics_dt)
        env._compute_intermediate_values(dt=env.physics_dt)
        pose = torch.cat(
            (env.fingertip_midpoint_pos[0], env.fingertip_midpoint_quat_aligned[0]), dim=0
        ).detach().cpu().numpy()
        sim_poses.append(pose)
        if (index + 1) % 500 == 0:
            print(f"[FK] {index + 1}/{len(all_q)} frames")
    env.close()

    all_sim_pose = np.asarray(sim_poses, dtype=np.float64)
    all_sim_pose[:, 3:] = quat_normalize(all_sim_pose[:, 3:])
    sim_position = all_sim_pose[:, :3]
    real_position = all_real_pose[:, :3]
    sim_quaternion = all_sim_pose[:, 3:]
    real_quaternion = all_real_pose[:, 3:]

    rotation, translation = fit_rigid_transform(sim_position, real_position)
    aligned_position = sim_position @ rotation.T + translation
    position_error = np.linalg.norm(aligned_position - real_position, axis=1)

    orientation_offsets = quat_multiply(real_quaternion, quat_conjugate(sim_quaternion))
    orientation_offset = quaternion_mean(orientation_offsets)
    aligned_quaternion = quat_multiply(
        np.broadcast_to(orientation_offset, sim_quaternion.shape), sim_quaternion
    )
    orientation_error = quat_angle_deg(quat_multiply(aligned_quaternion, quat_conjugate(real_quaternion)))

    raw_position_error = np.linalg.norm(sim_position - real_position, axis=1)
    raw_orientation_error = quat_angle_deg(quat_multiply(sim_quaternion, quat_conjugate(real_quaternion)))

    per_trajectory_rows: list[dict[str, object]] = []
    per_trajectory_transforms: list[dict[str, object]] = []
    frame_start = 0
    for name, trajectory in trajectories:
        frame_count = len(trajectory["q"])
        frame_end = frame_start + frame_count
        sim_pos_i = sim_position[frame_start:frame_end]
        real_pos_i = real_position[frame_start:frame_end]
        sim_q_i = sim_quaternion[frame_start:frame_end]
        real_q_i = real_quaternion[frame_start:frame_end]

        rotation_i, translation_i = fit_rigid_transform(sim_pos_i, real_pos_i)
        pos_global_i = position_error[frame_start:frame_end]
        pos_local_i = np.linalg.norm(sim_pos_i @ rotation_i.T + translation_i - real_pos_i, axis=1)
        offset_i = quaternion_mean(quat_multiply(real_q_i, quat_conjugate(sim_q_i)))
        quat_global_i = orientation_error[frame_start:frame_end]
        quat_local_i = quat_angle_deg(
            quat_multiply(
                quat_multiply(np.broadcast_to(offset_i, sim_q_i.shape), sim_q_i),
                quat_conjugate(real_q_i),
            )
        )
        timestamps = trajectory["timestamps"]
        hz = 1.0 / np.median(np.diff(timestamps)) if len(timestamps) > 1 else float("nan")
        row = {
            "trajectory": name,
            "frames": frame_count,
            "duration_s": float(timestamps[-1] - timestamps[0]) if len(timestamps) else 0.0,
            "sample_hz": float(hz),
            "global_pos_rmse_mm": scalar_metrics(pos_global_i)["rmse"] * 1000.0,
            "global_pos_p95_mm": scalar_metrics(pos_global_i)["p95"] * 1000.0,
            "global_pos_max_mm": scalar_metrics(pos_global_i)["max"] * 1000.0,
            "local_pos_rmse_mm": scalar_metrics(pos_local_i)["rmse"] * 1000.0,
            "local_pos_p95_mm": scalar_metrics(pos_local_i)["p95"] * 1000.0,
            "local_pos_max_mm": scalar_metrics(pos_local_i)["max"] * 1000.0,
            "global_ori_mean_deg": scalar_metrics(quat_global_i)["mean"],
            "global_ori_p95_deg": scalar_metrics(quat_global_i)["p95"],
            "global_ori_max_deg": scalar_metrics(quat_global_i)["max"],
            "local_ori_mean_deg": scalar_metrics(quat_local_i)["mean"],
            "local_ori_p95_deg": scalar_metrics(quat_local_i)["p95"],
            "local_ori_max_deg": scalar_metrics(quat_local_i)["max"],
            "local_t_x_m": float(translation_i[0]),
            "local_t_y_m": float(translation_i[1]),
            "local_t_z_m": float(translation_i[2]),
        }
        per_trajectory_rows.append(row)
        per_trajectory_transforms.append(
            {"trajectory": name, "rotation": rotation_i.tolist(), "translation_m": translation_i.tolist()}
        )
        frame_start = frame_end

    translations = np.asarray([item["translation_m"] for item in per_trajectory_transforms], dtype=np.float64)
    transform_drift = {
        "median_m": np.median(translations, axis=0).tolist(),
        "min_m": np.min(translations, axis=0).tolist(),
        "max_m": np.max(translations, axis=0).tolist(),
        "range_m": (np.max(translations, axis=0) - np.min(translations, axis=0)).tolist(),
        "std_m": np.std(translations, axis=0).tolist(),
    }

    report = {
        "input": {
            "data_dir": str(args.data_dir.resolve()),
            "trajectory_count": len(trajectories),
            "total_frames": int(len(all_q)),
            "joint_offsets_rad": joint_offsets.tolist(),
            "joint_signs": joint_signs.tolist(),
            "sim_task": args.task,
            "quaternion_order": "wxyz",
        },
        "raw_before_alignment": {
            "position_norm_m": scalar_metrics(raw_position_error),
            "orientation_angle_deg": scalar_metrics(raw_orientation_error),
        },
        "global_position_alignment": {
            "equation": "p_real = R_sim_to_real @ p_sim + t_sim_to_real",
            "rotation": rotation.tolist(),
            "translation_m": translation.tolist(),
            "position_error_m": scalar_metrics(position_error),
            "position_error_mm": {key: value * 1000.0 for key, value in scalar_metrics(position_error).items()},
        },
        "global_orientation_alignment": {
            "equation": "q_real ~= q_offset * q_sim",
            "offset_wxyz": orientation_offset.tolist(),
            "orientation_error_deg": scalar_metrics(orientation_error),
        },
        "per_trajectory_translation_drift": transform_drift,
        "per_trajectory_count": len(per_trajectory_rows),
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "calibration_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (args.output_dir / "sim_to_real_calibration.json").write_text(
        json.dumps(
            {
                "quaternion_order": "wxyz",
                "position": {"rotation": rotation.tolist(), "translation_m": translation.tolist()},
                "orientation": {"offset_wxyz": orientation_offset.tolist()},
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    write_csv(args.output_dir / "per_trajectory_metrics.csv", per_trajectory_rows)
    np.savez(
        args.output_dir / "aligned_pose_arrays.npz",
        sim_pose=all_sim_pose,
        real_pose=all_real_pose,
        aligned_sim_position=aligned_position,
        aligned_sim_quaternion=aligned_quaternion,
        position_error_m=position_error,
        orientation_error_deg=orientation_error,
    )

    print(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"[OK] Wrote alignment report to {args.output_dir.resolve()}")


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
