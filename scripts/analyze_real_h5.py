"""Summarize PI0-relevant fields in paired real robot trajectories."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import h5py
import numpy as np


SKIP_STATS_TOKENS = ("force", "torque", "wrench")


def dataset_schema(h5: h5py.File) -> dict[str, tuple[tuple[int, ...], str]]:
    schema = {}

    def visit(name, obj):
        if isinstance(obj, h5py.Dataset):
            shape = (-1, *obj.shape[1:]) if obj.ndim else ()
            schema[name] = (shape, str(obj.dtype))

    h5.visititems(visit)
    return schema


def describe(values: np.ndarray) -> dict:
    values = np.asarray(values, dtype=np.float64)
    return {
        "min": np.min(values, axis=0).tolist(),
        "p01": np.percentile(values, 1, axis=0).tolist(),
        "mean": np.mean(values, axis=0).tolist(),
        "std": np.std(values, axis=0).tolist(),
        "p99": np.percentile(values, 99, axis=0).tolist(),
        "max": np.max(values, axis=0).tolist(),
    }


def video_info(path: Path) -> dict:
    capture = cv2.VideoCapture(str(path))
    result = {
        "exists": path.is_file(),
        "frames": int(capture.get(cv2.CAP_PROP_FRAME_COUNT)),
        "fps": float(capture.get(cv2.CAP_PROP_FPS)),
        "width": int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
    }
    capture.release()
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path, nargs="?", default=Path("real_data"))
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    paths = sorted(args.root.glob("traj_*/data.h5"))
    if not paths:
        raise FileNotFoundError(f"No traj_*/data.h5 files under {args.root}")

    values_by_field: dict[str, list[np.ndarray]] = defaultdict(list)
    schema_counts: Counter[str] = Counter()
    lengths = []
    dt_values = []
    video_mismatches = []
    action_actual_joint_mae = []
    action_policy_actual_mae = []
    joint_velocity_gradient_mae = []
    gripper_width_scale_mae = []

    for path in paths:
        with h5py.File(path, "r") as h5:
            schema_key = json.dumps(dataset_schema(h5), sort_keys=True)
            schema_counts[schema_key] += 1
            length = int(h5["timestamps"].shape[0])
            lengths.append(length)
            timestamps = np.asarray(h5["timestamps"][:], dtype=np.float64)
            if length > 1:
                dt_values.append(np.diff(timestamps))

            for name in (
                "obs/state/joint_pos",
                "obs/state/joint_vel",
                "obs/state/ee_pose",
                "obs/state/gripper_pos",
                "obs/state/gripper_width_m",
                "action/actual/arm",
                "action/actual/gripper",
                "action/policy/arm",
                "action/policy/gripper",
            ):
                values_by_field[name].append(np.asarray(h5[name][:]))

            action_actual_joint_mae.append(
                np.abs(h5["action/actual/arm"][:] - h5["obs/state/joint_pos"][:])
            )
            action_policy_actual_mae.append(
                np.abs(h5["action/policy/arm"][:] - h5["action/actual/arm"][:])
            )
            if length > 1:
                numerical_velocity = np.gradient(
                    h5["obs/state/joint_pos"][:], timestamps, axis=0
                )
                joint_velocity_gradient_mae.append(
                    np.abs(h5["obs/state/joint_vel"][:] - numerical_velocity)
                )
            gripper_width_scale_mae.append(
                np.abs(
                    h5["obs/state/gripper_width_m"][:]
                    - 0.08 * h5["obs/state/gripper_pos"][:]
                )
            )

        for camera in ("front_camera", "wrist_camera"):
            info = video_info(path.parent / f"{camera}.mp4")
            if not info["exists"] or info["frames"] != length:
                video_mismatches.append(
                    {"trajectory": path.parent.name, "camera": camera, **info, "h5_frames": length}
                )

    reference_schema = json.loads(schema_counts.most_common(1)[0][0])
    report = {
        "trajectory_count": len(paths),
        "frame_count": describe(np.asarray(lengths)[:, None]),
        "total_frames": int(sum(lengths)),
        "schema_variant_count": len(schema_counts),
        "reference_schema": reference_schema,
        "video_frame_mismatches": video_mismatches,
        "fields": {
            name: describe(np.concatenate(chunks, axis=0))
            for name, chunks in values_by_field.items()
        },
        "timestamp_dt_seconds": describe(np.concatenate(dt_values)[:, None]),
        "checks": {
            "action_actual_vs_joint_pos_mae": float(
                np.mean(np.concatenate(action_actual_joint_mae, axis=0))
            ),
            "action_policy_vs_actual_mae": float(
                np.mean(np.concatenate(action_policy_actual_mae, axis=0))
            ),
            "joint_velocity_vs_timestamp_gradient_mae_per_joint": np.mean(
                np.concatenate(joint_velocity_gradient_mae, axis=0), axis=0
            ).tolist(),
            "gripper_width_m_vs_gripper_pos_times_0.08_mae": float(
                np.mean(np.concatenate(gripper_width_scale_mae, axis=0))
            ),
        },
        "force_fields_note": "Force/torque/wrench fields are included in schema only; numerical statistics intentionally skipped.",
    }

    output = args.output or args.root / "h5_statistics.json"
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"[DONE] {output}")


if __name__ == "__main__":
    main()
