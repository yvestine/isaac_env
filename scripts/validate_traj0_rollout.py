#!/usr/bin/env python3
"""Integrity and force-quality gate for each incremental traj0 rollout."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import h5py
import numpy as np


def correlation(first: np.ndarray, second: np.ndarray) -> float | None:
    if len(first) < 2 or np.std(first) < 1.0e-12 or np.std(second) < 1.0e-12:
        return None
    return float(np.corrcoef(first, second)[0, 1])


def first_sustained(values: np.ndarray, threshold: float, count: int = 3) -> int | None:
    active = np.asarray(values) > threshold
    for index in range(max(0, len(active) - count + 1)):
        if active[index : index + count].all():
            return index
    return None


def video_motion(path: Path) -> dict[str, float | int]:
    capture = cv2.VideoCapture(str(path))
    previous = None
    differences: list[float] = []
    frame_count = 0
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        frame_count += 1
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
        if previous is not None:
            differences.append(float(np.mean(np.abs(gray - previous))))
        previous = gray
    capture.release()
    if frame_count < 2:
        raise RuntimeError(f"Video contains fewer than two frames: {path}")
    values = np.asarray(differences, dtype=np.float64)
    return {
        "frame_count": frame_count,
        "adjacent_difference_mean": float(np.mean(values)),
        "adjacent_difference_p95": float(np.percentile(values, 95.0)),
        "adjacent_difference_max": float(np.max(values)),
    }


def read_wrench_csv(path: Path) -> np.ndarray | None:
    if not path.is_file():
        return None
    values = np.loadtxt(path, delimiter=",", skiprows=1)
    values = np.asarray(values, dtype=np.float64).reshape(-1, 6)
    return values


def read_vector_csv(path: Path, width: int) -> np.ndarray | None:
    if not path.is_file():
        return None
    values = np.loadtxt(path, delimiter=",", skiprows=1)
    return np.asarray(values, dtype=np.float64).reshape(-1, width)


def causal_ema(values: np.ndarray, alpha: float) -> np.ndarray:
    filtered = np.empty_like(values)
    filtered[0] = values[0]
    for index in range(1, len(values)):
        filtered[index] = alpha * values[index] + (1.0 - alpha) * filtered[index - 1]
    return filtered


def best_filter_diagnostic(
    real_force: np.ndarray,
    raw_sim_force: np.ndarray,
) -> dict[str, float | int] | None:
    """Score candidate causal filters on traj0 without applying them to output."""
    best = None
    for alpha in (1.0, 0.75, 0.5, 0.35, 0.25, 0.15, 0.1, 0.05):
        filtered = causal_ema(raw_sim_force, alpha)
        for delay_frames in range(9):
            if delay_frames:
                real_aligned = real_force[delay_frames:]
                sim_aligned = filtered[:-delay_frames]
            else:
                real_aligned = real_force
                sim_aligned = filtered
            real_norm = np.linalg.norm(real_aligned, axis=1)
            sim_norm = np.linalg.norm(sim_aligned, axis=1)
            score = correlation(real_norm, sim_norm)
            if score is None or (best is not None and score <= best["norm_correlation"]):
                continue
            sim_p95 = float(np.percentile(sim_norm, 95.0))
            real_p95 = float(np.percentile(real_norm, 95.0))
            best = {
                "ema_alpha": float(alpha),
                "delay_frames": int(delay_frames),
                "delay_seconds_at_10hz": float(delay_frames / 10.0),
                "norm_correlation": float(score),
                "diagnostic_positive_gain": (
                    float(real_p95 / sim_p95) if sim_p95 > 1.0e-12 else 0.0
                ),
            }
    return best


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--real-h5", type=Path, default=Path("real_data/traj_0/data.h5"))
    parser.add_argument("--sim-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--expected-replay-mode",
        choices=("direct", "servo"),
        default=None,
        help="Fail integrity if replay_metadata.json reports a different mode.",
    )
    parser.add_argument(
        "--joint-error-limit",
        type=float,
        default=1.0e-4,
        help="Direct-replay baseline limit; change explicitly for later physical-servo steps.",
    )
    args = parser.parse_args()

    required = (
        args.sim_dir / "data.h5",
        args.sim_dir / "joint_pos_sim.csv",
        args.sim_dir / "front_camera.mp4",
        args.sim_dir / "wrist_camera.mp4",
        args.sim_dir / "replay_metadata.json",
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing rollout outputs: {missing}")

    with h5py.File(args.real_h5, "r") as data:
        real_q = np.asarray(data["obs/state/joint_pos"], dtype=np.float64)
        real_wrench = np.asarray(data["obs/state/ee_wrench_base"], dtype=np.float64)
        timestamps = np.asarray(data["timestamps"], dtype=np.float64)
    timestamps -= timestamps[0]

    with h5py.File(args.sim_dir / "data.h5", "r") as data:
        sim_wrench = np.asarray(data["obs/state/ee_wrench_base"], dtype=np.float64)
        force_source = str(data.attrs.get("sim_force_training_source", "unknown"))

    metadata_path = args.sim_dir / "replay_metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    replay_mode = str(metadata.get("replay_mode", "direct"))

    sim_q = np.loadtxt(args.sim_dir / "joint_pos_sim.csv", delimiter=",", skiprows=1)
    incoming_wrench = read_wrench_csv(
        args.sim_dir / "incoming_joint_wrench_base_at_K.csv"
    )
    if incoming_wrench is None:
        incoming_wrench = read_wrench_csv(args.sim_dir / "wrench_final.csv")
    contact_wrench = read_wrench_csv(args.sim_dir / "contact_wrench_base.csv")
    contact_normal = read_vector_csv(args.sim_dir / "contact_normal_base.csv", 3)
    contact_friction = read_vector_csv(args.sim_dir / "contact_friction_base.csv", 3)
    contact_total_raw = read_vector_csv(
        args.sim_dir / "contact_total_base_raw.csv", 3
    )
    source_lengths = {
        "real_q": len(real_q),
        "real_wrench": len(real_wrench),
        "sim_q": len(sim_q),
        "sim_wrench": len(sim_wrench),
        "timestamps": len(timestamps),
    }
    frame_count = min(source_lengths.values())
    real_q = real_q[:frame_count]
    sim_q = sim_q[:frame_count]
    real_wrench = real_wrench[:frame_count]
    sim_wrench = sim_wrench[:frame_count]
    timestamps = timestamps[:frame_count]

    joint_error = np.abs(sim_q - real_q)
    joint_range = np.ptp(sim_q, axis=0)
    real_force_norm = np.linalg.norm(real_wrench[:, :3], axis=1)
    sim_force_norm = np.linalg.norm(sim_wrench[:, :3], axis=1)
    sim_torque_norm = np.linalg.norm(sim_wrench[:, 3:], axis=1)
    onset_index = first_sustained(sim_force_norm, 3.0)

    front_motion = video_motion(args.sim_dir / "front_camera.mp4")
    wrist_motion = video_motion(args.sim_dir / "wrist_camera.mp4")
    integrity_checks = {
        "all_frame_counts_match": len(set(source_lengths.values())) == 1,
        "all_values_finite": bool(
            np.isfinite(sim_q).all() and np.isfinite(sim_wrench).all()
        ),
        "robot_joint_trajectory_moves": bool(np.max(joint_range) > 0.01),
        "joint_tracking_within_limit": bool(np.max(joint_error) <= args.joint_error_limit),
        "front_video_changes": front_motion["adjacent_difference_p95"] > 0.05,
        "wrist_video_changes": wrist_motion["adjacent_difference_p95"] > 0.05,
    }
    force_safety_checks = {
        "training_force_has_nonzero_frames": bool(
            np.count_nonzero(sim_force_norm > 1.0e-4) >= 5
        ),
        "training_force_below_200_n": bool(np.max(sim_force_norm) < 200.0),
    }
    if args.expected_replay_mode is not None:
        integrity_checks["expected_replay_mode"] = replay_mode == args.expected_replay_mode

    component_correlation = [
        correlation(real_wrench[:, index], sim_wrench[:, index]) for index in range(3)
    ]
    force_norm_correlation = correlation(real_force_norm, sim_force_norm)
    real_p95 = float(np.percentile(real_force_norm, 95.0))
    sim_p95 = float(np.percentile(sim_force_norm, 95.0))
    p95_ratio = sim_p95 / real_p95 if real_p95 > 1.0e-12 else None
    quality_targets = {
        "force_norm_correlation_at_least_0_6": bool(
            force_norm_correlation is not None and force_norm_correlation >= 0.6
        ),
        "p95_ratio_between_0_5_and_2": bool(
            p95_ratio is not None and 0.5 <= p95_ratio <= 2.0
        ),
        "six_axis_torque_is_nonzero": bool(np.count_nonzero(sim_torque_norm > 1.0e-6) >= 5),
    }
    incoming_diagnostic = None
    if incoming_wrench is not None:
        incoming_wrench = incoming_wrench[:frame_count]
        incoming_force_norm = np.linalg.norm(incoming_wrench[:, :3], axis=1)
        incoming_torque_norm = np.linalg.norm(incoming_wrench[:, 3:], axis=1)
        incoming_diagnostic = {
            "nonzero_force_frames": int(
                np.count_nonzero(incoming_force_norm > 1.0e-4)
            ),
            "force_norm_p95_n": float(np.percentile(incoming_force_norm, 95.0)),
            "force_norm_max_n": float(np.max(incoming_force_norm)),
            "nonzero_torque_frames": int(
                np.count_nonzero(incoming_torque_norm > 1.0e-6)
            ),
            "torque_norm_p95_nm": float(np.percentile(incoming_torque_norm, 95.0)),
        }
    contact_wrench_diagnostic = None
    if contact_wrench is not None:
        contact_wrench = contact_wrench[:frame_count]
        contact_force_norm = np.linalg.norm(contact_wrench[:, :3], axis=1)
        contact_torque_norm = np.linalg.norm(contact_wrench[:, 3:], axis=1)
        contact_wrench_diagnostic = {
            "nonzero_force_frames": int(
                np.count_nonzero(contact_force_norm > 1.0e-4)
            ),
            "force_norm_p95_n": float(np.percentile(contact_force_norm, 95.0)),
            "force_norm_max_n": float(np.max(contact_force_norm)),
            "force_norm_correlation_with_real": correlation(
                real_force_norm, contact_force_norm
            ),
            "nonzero_torque_frames": int(
                np.count_nonzero(contact_torque_norm > 1.0e-6)
            ),
            "torque_norm_p95_nm": float(np.percentile(contact_torque_norm, 95.0)),
            "torque_norm_max_nm": float(np.max(contact_torque_norm)),
            "torque_norm_correlation_with_real": correlation(
                np.linalg.norm(real_wrench[:, 3:], axis=1), contact_torque_norm
            ),
        }
    contact_component_diagnostic = None
    if contact_normal is not None and contact_friction is not None:
        normal_norm = np.linalg.norm(contact_normal[:frame_count], axis=1)
        friction_norm = np.linalg.norm(contact_friction[:frame_count], axis=1)
        contact_component_diagnostic = {
            "normal_p95_n": float(np.percentile(normal_norm, 95.0)),
            "normal_norm_correlation_with_real": correlation(
                real_force_norm, normal_norm
            ),
            "friction_p95_n": float(np.percentile(friction_norm, 95.0)),
            "friction_norm_correlation_with_real": correlation(
                real_force_norm, friction_norm
            ),
        }
    filter_diagnostic = None
    if contact_total_raw is not None:
        filter_diagnostic = best_filter_diagnostic(
            real_wrench[:, :3], contact_total_raw[:frame_count]
        )

    report = {
        "real_h5": str(args.real_h5.resolve()),
        "sim_dir": str(args.sim_dir.resolve()),
        "force_source": force_source,
        "replay_mode": replay_mode,
        "frame_count": int(frame_count),
        "source_lengths": source_lengths,
        "integrity_pass": bool(all(integrity_checks.values())),
        "integrity_checks": integrity_checks,
        "force_safety_pass": bool(all(force_safety_checks.values())),
        "force_safety_checks": force_safety_checks,
        "joint": {
            "max_abs_error_rad": float(np.max(joint_error)),
            "mean_abs_error_rad": float(np.mean(joint_error)),
            "per_joint_range_rad": joint_range.tolist(),
        },
        "video": {"front": front_motion, "wrist": wrist_motion},
        "force": {
            "nonzero_frames": int(np.count_nonzero(sim_force_norm > 1.0e-4)),
            "max_norm_n": float(np.max(sim_force_norm)),
            "real_p95_norm_n": real_p95,
            "sim_p95_norm_n": sim_p95,
            "p95_ratio": p95_ratio,
            "component_correlation": component_correlation,
            "norm_correlation": force_norm_correlation,
            "first_sustained_above_3n_frame": onset_index,
            "first_sustained_above_3n_time_s": (
                None if onset_index is None else float(timestamps[onset_index])
            ),
            "nonzero_torque_frames": int(np.count_nonzero(sim_torque_norm > 1.0e-6)),
        },
        "incoming_wrench_diagnostic": incoming_diagnostic,
        "contact_wrench_diagnostic": contact_wrench_diagnostic,
        "contact_component_diagnostic": contact_component_diagnostic,
        "filter_diagnostic_traj0_only_not_applied": filter_diagnostic,
        "quality_pass": bool(all(quality_targets.values())),
        "quality_targets": quality_targets,
    }
    output = args.output or args.sim_dir / "validation_report.json"
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"[VALIDATE] integrity_pass={report['integrity_pass']}")
    print(f"[VALIDATE] force_safety_pass={report['force_safety_pass']}")
    print(f"[VALIDATE] quality_pass={report['quality_pass']}")
    ratio_text = "none" if p95_ratio is None else f"{p95_ratio:.4f}"
    correlation_text = (
        "none" if force_norm_correlation is None else f"{force_norm_correlation:.4f}"
    )
    print(
        "[VALIDATE] "
        f"q_max={report['joint']['max_abs_error_rad']:.6g} rad, "
        f"force_nonzero={report['force']['nonzero_frames']}/{frame_count}, "
        f"force_p95_ratio={ratio_text}, "
        f"force_norm_corr={correlation_text}"
    )
    print(f"[VALIDATE] report={output.resolve()}")
    if incoming_diagnostic is not None:
        print(
            "[VALIDATE] incoming_wrench "
            f"nonzero={incoming_diagnostic['nonzero_force_frames']}/{frame_count}, "
            f"force_p95={incoming_diagnostic['force_norm_p95_n']:.4f} N, "
            f"torque_nonzero={incoming_diagnostic['nonzero_torque_frames']}/{frame_count}"
        )
    if contact_wrench_diagnostic is not None:
        contact_corr = contact_wrench_diagnostic["force_norm_correlation_with_real"]
        contact_corr_text = "none" if contact_corr is None else f"{contact_corr:.4f}"
        print(
            "[VALIDATE] contact_wrench "
            f"force_nonzero={contact_wrench_diagnostic['nonzero_force_frames']}/{frame_count}, "
            f"force_p95={contact_wrench_diagnostic['force_norm_p95_n']:.4f} N, "
            f"force_corr={contact_corr_text}, "
            f"torque_nonzero={contact_wrench_diagnostic['nonzero_torque_frames']}/{frame_count}, "
            f"torque_p95={contact_wrench_diagnostic['torque_norm_p95_nm']:.6f} Nm"
        )
    if contact_component_diagnostic is not None:
        normal_corr = contact_component_diagnostic[
            "normal_norm_correlation_with_real"
        ]
        friction_corr = contact_component_diagnostic[
            "friction_norm_correlation_with_real"
        ]
        normal_corr_text = "none" if normal_corr is None else f"{normal_corr:.4f}"
        friction_corr_text = (
            "none" if friction_corr is None else f"{friction_corr:.4f}"
        )
        print(
            "[VALIDATE] contact_components "
            f"normal_p95={contact_component_diagnostic['normal_p95_n']:.4f} N, "
            f"normal_corr={normal_corr_text}, "
            f"friction_p95={contact_component_diagnostic['friction_p95_n']:.4f} N, "
            f"friction_corr={friction_corr_text}"
        )
    if filter_diagnostic is not None:
        print(
            "[VALIDATE] filter_diagnostic_traj0_only "
            f"best_corr={filter_diagnostic['norm_correlation']:.4f}, "
            f"ema_alpha={filter_diagnostic['ema_alpha']:.2f}, "
            f"delay_frames={filter_diagnostic['delay_frames']}, "
            f"diagnostic_gain={filter_diagnostic['diagnostic_positive_gain']:.6f}"
        )
    if not report["integrity_pass"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
