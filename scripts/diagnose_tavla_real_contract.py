"""Probe the remote TAVLA server with recorded real-data observations.

This is a read-only contract diagnostic. It does not modify checkpoints or
simulation state; it compares the server action chunk against the recorded
state/action at selected frames.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import h5py
from openpi_client import msgpack_numpy
import numpy as np
import websockets.sync.client


def _read_rgb(path: Path, frame_index: int) -> np.ndarray:
    capture = cv2.VideoCapture(str(path))
    capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ok, frame = capture.read()
    capture.release()
    if not ok:
        raise RuntimeError(f"Cannot read frame {frame_index} from {path}")
    return np.ascontiguousarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))


def _load_frame(traj_dir: Path, frame_index: int) -> tuple[dict, np.ndarray, np.ndarray]:
    with h5py.File(traj_dir / "data.h5", "r") as data:
        q = np.asarray(data["obs/state/joint_pos"][frame_index], dtype=np.float32)
        gripper = float(np.asarray(data["action/actual/gripper"][frame_index]).reshape(-1)[0])
        effort = np.asarray(data["obs/state/ee_wrench_base"][frame_index], dtype=np.float32).reshape(-1)
        action_q = np.asarray(data["action/actual/arm"][frame_index], dtype=np.float32)
        action_gripper = float(np.asarray(data["action/actual/gripper"][frame_index]).reshape(-1)[0])
    if q.shape != (7,) or effort.shape != (6,) or action_q.shape != (7,):
        raise ValueError(
            f"unexpected shapes q={q.shape}, effort={effort.shape}, action_q={action_q.shape}"
        )
    front = _read_rgb(traj_dir / "front_camera.mp4", frame_index)
    wrist = _read_rgb(traj_dir / "wrist_camera.mp4", frame_index)
    payload = {
        "images": {
            "cam_high": front,
            "cam_left_wrist": wrist,
            "cam_right_wrist": wrist.copy(),
        },
        "state": np.concatenate([q, [gripper]]).astype(np.float32),
        "effort": effort.reshape(1, 6),
        "prompt": "single arm manipulation",
    }
    recorded_action = np.concatenate([action_q, [action_gripper]]).astype(np.float32)
    return payload, recorded_action, effort


def _summary(chunk: np.ndarray, state: np.ndarray, recorded_action: np.ndarray) -> dict:
    chunk = np.asarray(chunk, dtype=np.float32)
    return {
        "shape": list(chunk.shape),
        "first": chunk[0].tolist(),
        "second": chunk[min(1, len(chunk) - 1)].tolist(),
        "last": chunk[-1].tolist(),
        "first_minus_state": (chunk[0] - state).tolist(),
        "first_minus_recorded_action": (chunk[0] - recorded_action).tolist(),
        "joint_min": chunk[:, :7].min(axis=0).tolist(),
        "joint_max": chunk[:, :7].max(axis=0).tolist(),
        "gripper_min_max": [float(chunk[:, 7].min()), float(chunk[:, 7].max())],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("real_data"))
    parser.add_argument("--trajectory", type=int, default=0)
    parser.add_argument("--frames", type=int, nargs="+", default=[0, 10, 30, 60])
    parser.add_argument("--host", type=str, default="10.0.40.113")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--output", type=Path, default=Path("outputs/tavla_real_contract_probe.json"))
    args = parser.parse_args()

    uri = f"ws://{args.host}:{args.port}"
    packer = msgpack_numpy.Packer()
    results = []
    with websockets.sync.client.connect(uri, compression=None, max_size=None, open_timeout=15.0) as ws:
        metadata = ws.recv(timeout=15.0)
        if isinstance(metadata, str):
            raise RuntimeError(f"server metadata is text: {metadata}")
        for frame_index in args.frames:
            payload, recorded_action, effort = _load_frame(
                args.data_dir / f"traj_{args.trajectory}", frame_index
            )
            ws.send(packer.pack(payload))
            response = ws.recv(timeout=30.0)
            if isinstance(response, str):
                raise RuntimeError(f"server returned an error: {response}")
            result = msgpack_numpy.unpackb(response)
            chunk = np.asarray(result["actions"], dtype=np.float32)
            if chunk.ndim == 3:
                chunk = chunk[0]
            if chunk.ndim != 2 or chunk.shape[-1] != 8:
                raise ValueError(f"unexpected server action shape {chunk.shape}")
            state = payload["state"]
            results.append(
                {
                    "frame": frame_index,
                    "state": state.tolist(),
                    "effort": effort.tolist(),
                    "recorded_action": recorded_action.tolist(),
                    "server": _summary(chunk, state, recorded_action),
                }
            )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(args.output), "results": results}, indent=2))


if __name__ == "__main__":
    main()
