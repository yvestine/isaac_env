"""Apply saved global/per-trajectory pose calibration and report residuals.

This is a post-processing step for the FK result produced by
``calibrate_real_sim_alignment.py``.  It does not start Isaac Sim and does not
modify the source HDF5 trajectories.
"""

from __future__ import annotations

import csv
import argparse
import json
from pathlib import Path

import h5py
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
INPUT_DIR = ROOT / "outputs" / "real_data_alignment"
DATA_DIR = ROOT / "real_data"
OUTPUT_DIR = ROOT / "outputs" / "real_data_alignment"


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
    q = quat_normalize(quaternions)
    reference = q[0]
    q = np.where((q @ reference)[:, None] < 0.0, -q, q)
    eigenvalues, eigenvectors = np.linalg.eigh(q.T @ q)
    result = eigenvectors[:, np.argmax(eigenvalues)]
    if result[0] < 0.0:
        result = -result
    return quat_normalize(result)


def quat_angle_deg(q: np.ndarray) -> np.ndarray:
    q = quat_normalize(q)
    return np.rad2deg(2.0 * np.arctan2(np.linalg.norm(q[..., 1:], axis=-1), np.abs(q[..., 0])))


def fit_rigid_transform(source: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    source_center = source.mean(axis=0)
    target_center = target.mean(axis=0)
    covariance = (source - source_center).T @ (target - target_center)
    u, _, vt = np.linalg.svd(covariance)
    rotation = vt.T @ u.T
    if np.linalg.det(rotation) < 0.0:
        vt[-1, :] *= -1.0
        rotation = vt.T @ u.T
    return rotation, target_center - rotation @ source_center


def metrics(values: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    return {
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "rmse": float(np.sqrt(np.mean(values**2))),
        "p95": float(np.percentile(values, 95.0)),
        "max": float(np.max(values)),
    }


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=INPUT_DIR)
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()
    input_dir = args.input_dir
    data_dir = args.data_dir
    output_dir = args.output_dir

    arrays = np.load(input_dir / "aligned_pose_arrays.npz")
    sim_pose = np.asarray(arrays["sim_pose"], dtype=np.float64)
    real_pose = np.asarray(arrays["real_pose"], dtype=np.float64)
    sim_pose[:, 3:] = quat_normalize(sim_pose[:, 3:])
    real_pose[:, 3:] = quat_normalize(real_pose[:, 3:])

    with (input_dir / "sim_to_real_calibration.json").open() as file:
        global_calibration = json.load(file)
    global_rotation = np.asarray(global_calibration["position"]["rotation"], dtype=np.float64)
    global_translation = np.asarray(global_calibration["position"]["translation_m"], dtype=np.float64)
    global_offset = np.asarray(global_calibration["orientation"]["offset_wxyz"], dtype=np.float64)

    global_position = sim_pose[:, :3] @ global_rotation.T + global_translation
    global_quaternion = quat_multiply(
        np.broadcast_to(global_offset, sim_pose[:, 3:].shape), sim_pose[:, 3:]
    )
    global_position_error = np.linalg.norm(global_position - real_pose[:, :3], axis=1)
    global_orientation_error = quat_angle_deg(
        quat_multiply(global_quaternion, quat_conjugate(real_pose[:, 3:]))
    )

    trajectories = sorted(
        (path for path in data_dir.glob("traj_*") if (path / "data.h5").is_file()),
        key=lambda path: int(path.name.split("_")[-1]),
    )
    local_position = np.empty_like(global_position)
    local_quaternion = np.empty_like(global_quaternion)
    rows: list[dict[str, object]] = []
    transforms: list[dict[str, object]] = []
    start = 0
    for trajectory in trajectories:
        with h5py.File(trajectory / "data.h5", "r") as h5:
            frame_count = int(h5["obs/state/joint_pos"].shape[0])
        end = start + frame_count
        sim_pos = sim_pose[start:end, :3]
        real_pos = real_pose[start:end, :3]
        sim_quat = sim_pose[start:end, 3:]
        real_quat = real_pose[start:end, 3:]

        rotation, translation = fit_rigid_transform(sim_pos, real_pos)
        offset = quaternion_mean(quat_multiply(real_quat, quat_conjugate(sim_quat)))
        local_position[start:end] = sim_pos @ rotation.T + translation
        local_quaternion[start:end] = quat_multiply(
            np.broadcast_to(offset, sim_quat.shape), sim_quat
        )

        local_pos_error = np.linalg.norm(local_position[start:end] - real_pos, axis=1)
        local_ori_error = quat_angle_deg(
            quat_multiply(local_quaternion[start:end], quat_conjugate(real_quat))
        )
        global_pos_error = global_position_error[start:end]
        global_ori_error = global_orientation_error[start:end]
        with h5py.File(trajectory / "data.h5", "r") as h5:
            timestamps = np.asarray(h5["timestamps"][:], dtype=np.float64)

        rows.append(
            {
                "trajectory": trajectory.name,
                "frames": frame_count,
                "global_pos_rmse_mm": metrics(global_pos_error)["rmse"] * 1000.0,
                "global_pos_p95_mm": metrics(global_pos_error)["p95"] * 1000.0,
                "global_pos_max_mm": metrics(global_pos_error)["max"] * 1000.0,
                "local_pos_rmse_mm": metrics(local_pos_error)["rmse"] * 1000.0,
                "local_pos_p95_mm": metrics(local_pos_error)["p95"] * 1000.0,
                "local_pos_max_mm": metrics(local_pos_error)["max"] * 1000.0,
                "global_ori_mean_deg": metrics(global_ori_error)["mean"],
                "global_ori_p95_deg": metrics(global_ori_error)["p95"],
                "local_ori_mean_deg": metrics(local_ori_error)["mean"],
                "local_ori_p95_deg": metrics(local_ori_error)["p95"],
                "local_ori_max_deg": metrics(local_ori_error)["max"],
                "local_t_x_m": float(translation[0]),
                "local_t_y_m": float(translation[1]),
                "local_t_z_m": float(translation[2]),
                "sample_hz": float(1.0 / np.median(np.diff(timestamps))),
            }
        )
        transforms.append(
            {
                "trajectory": trajectory.name,
                "rotation": rotation.tolist(),
                "translation_m": translation.tolist(),
                "orientation_offset_wxyz": offset.tolist(),
            }
        )
        start = end

    if start != len(sim_pose):
        raise ValueError(f"Trajectory frame count {start} does not match pose array {len(sim_pose)}")

    local_position_error = np.linalg.norm(local_position - real_pose[:, :3], axis=1)
    local_orientation_error = quat_angle_deg(
        quat_multiply(local_quaternion, quat_conjugate(real_pose[:, 3:]))
    )
    local_translations = np.asarray([item["translation_m"] for item in transforms])
    report = {
        "source": str((input_dir / "aligned_pose_arrays.npz").resolve()),
        "trajectory_count": len(trajectories),
        "total_frames": int(len(sim_pose)),
        "global_calibration": {
            "position_error_mm": {key: value * 1000.0 for key, value in metrics(global_position_error).items()},
            "orientation_error_deg": metrics(global_orientation_error),
        },
        "per_trajectory_calibration": {
            "position_error_mm": {key: value * 1000.0 for key, value in metrics(local_position_error).items()},
            "orientation_error_deg": metrics(local_orientation_error),
            "translation_range_m": (
                np.max(local_translations, axis=0) - np.min(local_translations, axis=0)
            ).tolist(),
        },
        "interpretation": "Per-trajectory calibration removes the between-run base/fixture drift; it is suitable for replay comparisons when the real setup is repositioned between runs.",
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "per_trajectory_alignment_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (output_dir / "per_trajectory_calibrations.json").write_text(
        json.dumps(transforms, indent=2), encoding="utf-8"
    )
    write_csv(output_dir / "per_trajectory_recalibrated_metrics.csv", rows)
    np.savez(
        output_dir / "per_trajectory_aligned_pose_arrays.npz",
        sim_pose=sim_pose,
        real_pose=real_pose,
        aligned_position=local_position,
        aligned_quaternion=local_quaternion,
        position_error_m=local_position_error,
        orientation_error_deg=local_orientation_error,
        global_position_error_m=global_position_error,
        global_orientation_error_deg=global_orientation_error,
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
