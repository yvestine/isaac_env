#!/usr/bin/env python3
"""Paired A/B/C/D diagnosis for the image/state-only Pi0 WebSocket policy.

The real and paired-simulation videos are already synchronized with their
trajectory states.  For selected frames this script sends four payloads:

    A: real image + real state
    B: sim  image + real state
    C: real image + sim  state
    D: sim  image + sim  state

The current WebSocket contract has no explicit noise/seed field.  Therefore
this script does not claim deterministic same-noise inference.  It supports
repeated calls and reports within-condition stochastic variation separately.
It never sends effort data.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path
from typing import Any

import cv2
import h5py
import numpy as np
from openpi_client import msgpack_numpy
import websockets.sync.client


CONDITION_NAMES = ("A_real_image_real_state", "B_sim_image_real_state", "C_real_image_sim_state", "D_sim_image_sim_state")


def _read_video_frames(path: Path, frame_indices: list[int]) -> dict[int, np.ndarray]:
    """Read selected BGR video frames and return contiguous RGB uint8 arrays."""
    wanted = sorted(set(int(index) for index in frame_indices))
    if not wanted:
        return {}
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"Cannot open video: {path}")
    frames: dict[int, np.ndarray] = {}
    next_position = 0
    try:
        for target in wanted:
            if target < next_position:
                capture.set(cv2.CAP_PROP_POS_FRAMES, target)
            else:
                while next_position < target:
                    ok, _ = capture.read()
                    if not ok:
                        raise RuntimeError(f"Cannot seek to frame {target} in {path}")
                    next_position += 1
            ok, frame = capture.read()
            if not ok:
                raise RuntimeError(f"Cannot read frame {target} from {path}")
            frames[target] = np.ascontiguousarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            next_position = target + 1
    finally:
        capture.release()
    return frames


def _load_csv(path: Path) -> np.ndarray:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.reader(handle))
    if len(rows) <= 1:
        raise ValueError(f"CSV has no data rows: {path}")
    return np.asarray(rows[1:], dtype=np.float32)


def _load_trajectory(
    real_root: Path,
    paired_root: Path,
    trajectory: int,
    frames_per_trajectory: int,
) -> dict[str, Any]:
    real_dir = real_root / f"traj_{trajectory}"
    sim_dir = paired_root / f"traj_{trajectory}"
    h5_path = real_dir / "data.h5"
    required = (
        h5_path,
        real_dir / "front_camera.mp4",
        real_dir / "wrist_camera.mp4",
        sim_dir / "front_camera.mp4",
        sim_dir / "wrist_camera.mp4",
        sim_dir / "joint_pos_sim.csv",
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing paired trajectory files:\n" + "\n".join(missing))

    with h5py.File(h5_path, "r") as data:
        q_real = np.asarray(data["obs/state/joint_pos"][:], dtype=np.float32)
        gripper = np.asarray(data["obs/state/gripper_pos"][:], dtype=np.float32).reshape(-1)
        timestamps = np.asarray(data["timestamps"][:], dtype=np.float64)
    q_sim = _load_csv(sim_dir / "joint_pos_sim.csv")

    if q_real.ndim != 2 or q_real.shape[1] != 7:
        raise ValueError(f"traj_{trajectory}: real q has invalid shape {q_real.shape}")
    if q_sim.ndim != 2 or q_sim.shape[1] != 7:
        raise ValueError(f"traj_{trajectory}: sim q has invalid shape {q_sim.shape}")
    if gripper.shape != (len(q_real),) or timestamps.shape != (len(q_real),):
        raise ValueError(f"traj_{trajectory}: H5 state/timestamp length mismatch")
    if len(q_sim) != len(q_real):
        raise ValueError(f"traj_{trajectory}: real/sim frame count mismatch: {len(q_real)} vs {len(q_sim)}")
    if not all(np.isfinite(value).all() for value in (q_real, q_sim, gripper, timestamps)):
        raise ValueError(f"traj_{trajectory}: NaN/Inf in state data")

    frame_count = len(q_real)
    selected_frames = _frame_indices(frame_count, frames_per_trajectory)
    real_front = _read_video_frames(real_dir / "front_camera.mp4", selected_frames)
    real_wrist = _read_video_frames(real_dir / "wrist_camera.mp4", selected_frames)
    sim_front = _read_video_frames(sim_dir / "front_camera.mp4", selected_frames)
    sim_wrist = _read_video_frames(sim_dir / "wrist_camera.mp4", selected_frames)

    return {
        "trajectory": trajectory,
        "q_real": q_real,
        "q_sim": q_sim,
        "gripper": gripper,
        "timestamps": timestamps,
        "frame_indices": selected_frames,
        "real_front": real_front,
        "real_wrist": real_wrist,
        "sim_front": sim_front,
        "sim_wrist": sim_wrist,
    }


def _normalize_action(value: Any) -> np.ndarray:
    action = np.asarray(value, dtype=np.float32)
    if action.ndim == 3 and action.shape[0] == 1:
        action = action[0]
    if action.ndim != 2 or action.shape[1] != 8:
        raise ValueError(f"Pi0 action must have shape (H, 8), got {action.shape}")
    if not np.isfinite(action).all():
        raise ValueError("Pi0 action contains NaN or Inf")
    return action


def _cosine(lhs: np.ndarray, rhs: np.ndarray) -> float:
    lhs = np.asarray(lhs, dtype=np.float64).reshape(-1)
    rhs = np.asarray(rhs, dtype=np.float64).reshape(-1)
    denominator = float(np.linalg.norm(lhs) * np.linalg.norm(rhs))
    if denominator <= 1.0e-12:
        return 1.0 if np.linalg.norm(lhs - rhs) <= 1.0e-12 else 0.0
    return float(np.dot(lhs, rhs) / denominator)


def _action_metrics(action: np.ndarray, state: np.ndarray, action_index: int) -> dict[str, Any]:
    index = min(max(int(action_index), 0), len(action) - 1)
    target = action[index]
    delta_q = target[:7] - state[:7]
    return {
        "selected_action_index": index,
        "selected_action": target.tolist(),
        "delta_q": delta_q.tolist(),
        "delta_gripper": float(target[7] - state[7]),
        "delta_q_norm": float(np.linalg.norm(delta_q)),
        "action_joint_min": action[:, :7].min(axis=0).tolist(),
        "action_joint_max": action[:, :7].max(axis=0).tolist(),
        "action_gripper_min": float(action[:, 7].min()),
        "action_gripper_max": float(action[:, 7].max()),
        "chunk_shape": list(action.shape),
    }


class Pi0WebSocket:
    def __init__(self, host: str, port: int, timeout_s: float) -> None:
        self.uri = f"ws://{host}:{port}"
        self.timeout_s = float(timeout_s)
        self.packer = msgpack_numpy.Packer()
        self.ws = websockets.sync.client.connect(
            self.uri,
            compression=None,
            max_size=None,
            open_timeout=self.timeout_s,
        )
        metadata = self.ws.recv(timeout=self.timeout_s)
        if isinstance(metadata, str):
            raise RuntimeError(f"Pi0 server sent text metadata: {metadata}")
        self.metadata = msgpack_numpy.unpackb(metadata)

    def reset(self) -> None:
        self.ws.send(self.packer.pack({"reset": True}))
        response = self.ws.recv(timeout=self.timeout_s)
        if isinstance(response, str):
            raise RuntimeError(f"Pi0 reset failed: {response}")
        result = msgpack_numpy.unpackb(response)
        if result != {"reset": True}:
            raise RuntimeError(f"Unexpected Pi0 reset response: {result!r}")

    def infer(self, front: np.ndarray, wrist: np.ndarray, state: np.ndarray) -> np.ndarray:
        front = np.asarray(front, dtype=np.uint8)
        wrist = np.asarray(wrist, dtype=np.uint8)
        state = np.asarray(state, dtype=np.float32).reshape(-1)
        if front.ndim != 3 or front.shape[-1] != 3:
            raise ValueError(f"front image must be HxWx3, got {front.shape}")
        if wrist.ndim != 3 or wrist.shape[-1] != 3:
            raise ValueError(f"wrist image must be HxWx3, got {wrist.shape}")
        if state.shape != (8,):
            raise ValueError(f"state must be (8,), got {state.shape}")
        payload = {
            "images": {
                "cam_high": np.ascontiguousarray(front),
                "cam_left_wrist": np.ascontiguousarray(wrist),
                "cam_right_wrist": np.ascontiguousarray(wrist.copy()),
            },
            "state": state,
            "prompt": "peg-in-hole",
        }
        self.ws.send(self.packer.pack(payload))
        response = self.ws.recv(timeout=self.timeout_s)
        if isinstance(response, str):
            raise RuntimeError(f"Pi0 inference failed: {response}")
        result = msgpack_numpy.unpackb(response)
        if "actions" not in result:
            raise KeyError(f"Pi0 response has no actions: {result!r}")
        return _normalize_action(result["actions"])

    def close(self) -> None:
        self.ws.close()


def _select_trajectories(root: Path, spec: str, max_count: int | None) -> list[int]:
    if spec.strip().lower() == "all":
        values = sorted(
            int(path.name.removeprefix("traj_"))
            for path in root.glob("traj_*")
            if path.is_dir() and path.name.removeprefix("traj_").isdigit()
        )
    else:
        values = [int(item) for item in spec.split(",") if item.strip()]
    if max_count is not None:
        values = values[: max(0, int(max_count))]
    if not values:
        raise ValueError("No trajectories selected")
    return values


def _frame_indices(frame_count: int, count: int) -> list[int]:
    if count <= 1:
        return [0]
    return sorted(set(int(round(value)) for value in np.linspace(0, frame_count - 1, count)))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--real-root", type=Path, default=Path("real_data"))
    parser.add_argument("--paired-root", type=Path, default=Path("outputs/paired_rollouts_40_fk"))
    parser.add_argument("--trajectories", type=str, default="0,1,2,3,4")
    parser.add_argument("--max-trajectories", type=int, default=None)
    parser.add_argument("--frames-per-trajectory", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=1, help="Repeated inference per condition; no fixed noise is available")
    parser.add_argument("--action-index", type=int, default=1)
    parser.add_argument("--host", type=str, default="114.214.164.112")
    parser.add_argument("--port", type=int, default=8001)
    parser.add_argument("--timeout-s", type=float, default=45.0)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/pi0_paired_diagnosis"))
    args = parser.parse_args()

    if args.repeats < 1:
        raise ValueError("--repeats must be >= 1")
    trajectories = _select_trajectories(args.paired_root, args.trajectories, args.max_trajectories)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    client = Pi0WebSocket(args.host, args.port, args.timeout_s)
    records: list[dict[str, Any]] = []
    start_time = time.perf_counter()
    try:
        client.reset()
        print(f"[Pi0Paired] connected to {client.uri}; metadata={client.metadata!r}", flush=True)
        for trajectory in trajectories:
            data = _load_trajectory(
                args.real_root,
                args.paired_root,
                trajectory,
                args.frames_per_trajectory,
            )
            indices = data["frame_indices"]
            for frame_index in indices:
                q_real = data["q_real"][frame_index]
                q_sim = data["q_sim"][frame_index]
                gripper = float(data["gripper"][frame_index])
                state_real = np.concatenate((q_real, [gripper])).astype(np.float32)
                state_sim = np.concatenate((q_sim, [gripper])).astype(np.float32)
                payloads = {
                    "A_real_image_real_state": (data["real_front"][frame_index], data["real_wrist"][frame_index], state_real),
                    "B_sim_image_real_state": (data["sim_front"][frame_index], data["sim_wrist"][frame_index], state_real),
                    "C_real_image_sim_state": (data["real_front"][frame_index], data["real_wrist"][frame_index], state_sim),
                    "D_sim_image_sim_state": (data["sim_front"][frame_index], data["sim_wrist"][frame_index], state_sim),
                }
                sample: dict[str, Any] = {
                    "trajectory": trajectory,
                    "frame": frame_index,
                    "timestamp_s": float(data["timestamps"][frame_index] - data["timestamps"][0]),
                    "q_real": q_real.tolist(),
                    "q_sim": q_sim.tolist(),
                    "q_sim_minus_real": (q_sim - q_real).tolist(),
                    "q_sim_error_max_rad": float(np.max(np.abs(q_sim - q_real))),
                    "gripper_state": gripper,
                    "conditions": {},
                }
                for condition in CONDITION_NAMES:
                    front, wrist, state = payloads[condition]
                    condition_runs = []
                    for repeat in range(args.repeats):
                        call_start = time.perf_counter()
                        action = client.infer(front, wrist, state)
                        elapsed = time.perf_counter() - call_start
                        metrics = _action_metrics(action, state, args.action_index)
                        metrics.update({"repeat": repeat, "latency_s": elapsed})
                        condition_runs.append({"action": action.tolist(), "metrics": metrics})
                    sample["conditions"][condition] = {
                        "state": state.tolist(),
                        "runs": condition_runs,
                    }
                records.append(sample)
                print(
                    f"[Pi0Paired] traj={trajectory:02d} frame={frame_index:03d} "
                    f"qerr={sample['q_sim_error_max_rad']:.6f} rad",
                    flush=True,
                )
    finally:
        client.close()

    records_path = args.output_dir / "records.json"
    records_path.write_text(json.dumps(records, indent=2), encoding="utf-8")

    # Aggregate over repeats first, then compare conditions within each paired frame.
    condition_actions: dict[str, list[np.ndarray]] = {name: [] for name in CONDITION_NAMES}
    pair_rows: list[dict[str, Any]] = []
    repeat_spreads: dict[str, list[float]] = {name: [] for name in CONDITION_NAMES}
    for sample in records:
        means: dict[str, np.ndarray] = {}
        for condition in CONDITION_NAMES:
            runs = sample["conditions"][condition]["runs"]
            actions = np.asarray([run["action"] for run in runs], dtype=np.float32)
            mean_action = actions.mean(axis=0)
            means[condition] = mean_action
            condition_actions[condition].append(mean_action)
            if len(actions) > 1:
                repeat_spreads[condition].append(float(np.mean(np.std(actions, axis=0))))
        real_state = np.asarray(sample["conditions"][CONDITION_NAMES[0]]["state"], dtype=np.float32)
        sim_state = np.asarray(sample["conditions"][CONDITION_NAMES[2]]["state"], dtype=np.float32)
        row = {"trajectory": sample["trajectory"], "frame": sample["frame"]}
        for lhs, rhs, label in (
            ("A_real_image_real_state", "B_sim_image_real_state", "visual_effect_real_state"),
            ("C_real_image_sim_state", "D_sim_image_sim_state", "visual_effect_sim_state"),
            ("A_real_image_real_state", "C_real_image_sim_state", "state_effect_real_image"),
            ("B_sim_image_real_state", "D_sim_image_sim_state", "state_effect_sim_image"),
        ):
            lhs_action = means[lhs]
            rhs_action = means[rhs]
            lhs_state = real_state if lhs.startswith(("A_", "B_")) else sim_state
            rhs_state = real_state if rhs.startswith(("A_", "B_")) else sim_state
            lhs_delta = lhs_action[args.action_index, :7] - lhs_state[:7]
            rhs_delta = rhs_action[args.action_index, :7] - rhs_state[:7]
            row[f"{label}_chunk_rmse"] = float(np.sqrt(np.mean(np.square(lhs_action - rhs_action))))
            row[f"{label}_action_index_mae"] = float(np.mean(np.abs(lhs_action[args.action_index] - rhs_action[args.action_index])))
            row[f"{label}_delta_q_cosine"] = _cosine(lhs_delta, rhs_delta)
            row[f"{label}_delta_q_lhs"] = lhs_delta.tolist()
            row[f"{label}_delta_q_rhs"] = rhs_delta.tolist()
        pair_rows.append(row)

    summary: dict[str, Any] = {
        "protocol": {
            "server": f"{args.host}:{args.port}",
            "fixed_noise": False,
            "noise_note": "Current WebSocket payload has no noise/seed field; repeated-call spread is reported separately.",
            "effort_sent": False,
            "prompt": "peg-in-hole",
            "conditions": list(CONDITION_NAMES),
        },
        "selection": {
            "trajectories": trajectories,
            "frames_per_trajectory": args.frames_per_trajectory,
            "repeats": args.repeats,
            "action_index": args.action_index,
            "paired_samples": len(records),
        },
        "state_alignment": {
            "mean_max_q_error_rad": float(np.mean([record["q_sim_error_max_rad"] for record in records])),
            "max_q_error_rad": float(np.max([record["q_sim_error_max_rad"] for record in records])),
            "mean_q_error_rad": float(np.mean([np.mean(np.abs(record["q_sim_minus_real"])) for record in records])),
        },
        "conditions": {},
        "pair_comparisons": pair_rows,
        "elapsed_s": time.perf_counter() - start_time,
        "records_file": str(records_path),
    }
    for condition in CONDITION_NAMES:
        values = np.asarray(condition_actions[condition], dtype=np.float32)
        summary["conditions"][condition] = {
            "samples": len(values),
            "mean_action_selected": values[:, min(max(args.action_index, 0), values.shape[1] - 1)].mean(axis=0).tolist(),
            "mean_action_joint_min": values[:, :, :7].min(axis=(0, 1)).tolist(),
            "mean_action_joint_max": values[:, :, :7].max(axis=(0, 1)).tolist(),
            "mean_selected_delta_q": (
                values[:, min(max(args.action_index, 0), values.shape[1] - 1), :7]
                - np.asarray([record["conditions"][condition]["state"][:7] for record in records], dtype=np.float32)
            ).mean(axis=0).tolist(),
            "repeat_mean_std_action": float(np.mean(repeat_spreads[condition])) if repeat_spreads[condition] else None,
        }

    summary_path = args.output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({"summary": str(summary_path), "records": str(records_path), "samples": len(records)}, indent=2))


if __name__ == "__main__":
    main()
