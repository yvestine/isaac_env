"""Probe Pi0 on recorded traj_0 observations without moving the simulator.

For every selected real-data frame, this script sends the recorded RGB images
and 8-D state to the Pi0 WebSocket server.  It compares the predicted action
chunk with the following recorded joint states, which separates policy/action
quality from closed-loop simulator behavior.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import cv2
import h5py
import numpy as np
from openpi_client.websocket_client_policy import WebsocketClientPolicy


def _read_h5(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with h5py.File(path, "r") as data:
        qpos = np.asarray(data["obs/state/joint_pos"][:], dtype=np.float32)
        gripper = np.asarray(data["action/actual/gripper"][:], dtype=np.float32).reshape(-1)
        actual_arm = np.asarray(data["action/actual/arm"][:], dtype=np.float32)

    if qpos.ndim != 2 or qpos.shape[1] < 7:
        raise ValueError(f"Expected obs/state/joint_pos with shape (N,>=7), got {qpos.shape}")
    if actual_arm.shape != (len(qpos), 7):
        raise ValueError(f"Expected action/actual/arm with shape {(len(qpos), 7)}, got {actual_arm.shape}")
    if gripper.shape != (len(qpos),):
        raise ValueError(f"Expected action/actual/gripper with shape {(len(qpos),)}, got {gripper.shape}")
    qpos = qpos[:, :7]
    if not np.isfinite(qpos).all() or not np.isfinite(gripper).all() or not np.isfinite(actual_arm).all():
        raise ValueError("traj_0 state/action contains NaN or Inf")
    return qpos, gripper, actual_arm


def _rgb_frame(frame: np.ndarray) -> np.ndarray:
    frame = np.asarray(frame)
    if frame.ndim != 3:
        raise ValueError(f"Expected rank-3 camera frame, got {frame.shape}")
    if frame.shape[-1] != 3:
        raise ValueError(f"Expected HxWx3 camera frame, got {frame.shape}")
    return np.ascontiguousarray(frame)


def _read_next_rgb(capture: cv2.VideoCapture, path: Path, frame_index: int) -> np.ndarray:
    ok, bgr = capture.read()
    if not ok:
        raise RuntimeError(f"Cannot read frame {frame_index} from {path}")
    return np.ascontiguousarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))


def _normalize_chunk(result: dict) -> np.ndarray:
    if "actions" not in result:
        raise KeyError("Pi0 response does not contain 'actions'")
    actions = np.asarray(result["actions"], dtype=np.float32)
    if actions.ndim == 3 and actions.shape[0] == 1:
        actions = actions[0]
    if actions.ndim != 2 or actions.shape[1] != 8:
        raise ValueError(f"Expected actions with shape (H,8), got {actions.shape}")
    if not np.isfinite(actions).all():
        raise ValueError("Pi0 actions contain NaN or Inf")
    return actions


def _action_error(predicted: np.ndarray, target: np.ndarray) -> dict[str, object]:
    error = np.asarray(predicted, dtype=np.float64) - np.asarray(target, dtype=np.float64)
    return {
        "mae_rad": float(np.mean(np.abs(error))),
        "rmse_rad": float(np.sqrt(np.mean(error**2))),
        "max_abs_rad": float(np.max(np.abs(error))),
        "per_joint_abs_rad": np.abs(error).tolist(),
    }


def _annotate(frame_rgb: np.ndarray, lines: list[str]) -> np.ndarray:
    frame = np.ascontiguousarray(frame_rgb[..., ::-1].copy())
    for line_index, line in enumerate(lines):
        origin = (12, 28 + 24 * line_index)
        cv2.putText(frame, line, origin, cv2.FONT_HERSHEY_SIMPLEX, 0.58, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(frame, line, origin, cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 255, 255), 1, cv2.LINE_AA)
    return frame


def _open_writer(path: Path, width: int, height: int, fps: float) -> cv2.VideoWriter:
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"mp4v"), max(1.0, fps), (width, height)
    )
    if not writer.isOpened():
        raise RuntimeError(f"Cannot open video writer: {path}")
    return writer


def _close_client(client: WebsocketClientPolicy) -> None:
    """Close both current and older openpi-client client implementations."""
    close = getattr(client, "close", None)
    if callable(close):
        close()
        return
    websocket = getattr(client, "_ws", None)
    close_websocket = getattr(websocket, "close", None)
    if callable(close_websocket):
        close_websocket()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--h5", type=Path, default=Path("real_data/traj_0/data.h5"))
    parser.add_argument("--front-video", type=Path, default=None)
    parser.add_argument("--wrist-video", type=Path, default=None)
    parser.add_argument("--host", type=str, default="114.214.164.36")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--prompt", type=str, default="peg-in-hole")
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument(
        "--action-index",
        type=int,
        default=1,
        help="Action in each chunk used by the rollout; Pi0 rollout currently uses action[1].",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("outputs/pi0_offline_traj0_probe")
    )
    parser.add_argument("--save-video", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--video-fps", type=float, default=10.0)
    args = parser.parse_args()

    if args.stride < 1:
        raise ValueError("--stride must be >= 1")
    if args.action_index < 0:
        raise ValueError("--action-index must be >= 0")

    traj_dir = args.h5.parent
    front_path = args.front_video or traj_dir / "front_camera.mp4"
    wrist_path = args.wrist_video or traj_dir / "wrist_camera.mp4"
    qpos, gripper, actual_arm = _read_h5(args.h5)
    frame_count = len(qpos)
    selected = set(range(0, frame_count, args.stride))
    selected.add(frame_count - 1)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    client = WebsocketClientPolicy(host=args.host, port=args.port)
    front_capture = cv2.VideoCapture(str(front_path))
    wrist_capture = cv2.VideoCapture(str(wrist_path))
    if not front_capture.isOpened() or not wrist_capture.isOpened():
        raise RuntimeError(f"Cannot open videos: {front_path}, {wrist_path}")

    front_writer = None
    wrist_writer = None
    rows: list[dict[str, object]] = []
    video_error = ""
    try:
        metadata = client.get_server_metadata()
        client.reset()
        print(f"[Pi0Offline] server metadata: {metadata}", flush=True)

        for frame_index in range(frame_count):
            front = _read_next_rgb(front_capture, front_path, frame_index)
            wrist = _read_next_rgb(wrist_capture, wrist_path, frame_index)
            if frame_index not in selected:
                continue

            state = np.concatenate((qpos[frame_index], [gripper[frame_index]])).astype(np.float32)
            observation = {
                "images": {
                    "cam_high": front,
                    "cam_left_wrist": wrist,
                    "cam_right_wrist": wrist.copy(),
                },
                "state": state,
                "prompt": args.prompt,
            }
            result = client.infer(observation)
            chunk = _normalize_chunk(result)
            if args.action_index >= len(chunk):
                raise ValueError(
                    f"--action-index {args.action_index} is outside action chunk length {len(chunk)}"
                )

            pred = chunk[args.action_index, :7]
            next_index = min(frame_index + 1, frame_count - 1)
            next_gt = qpos[next_index]
            current_gt = qpos[frame_index]
            future_count = min(
                len(chunk) - args.action_index,
                frame_count - (frame_index + 1),
            )
            future_gt = qpos[frame_index + 1 : frame_index + 1 + future_count]
            horizon_pred = chunk[
                args.action_index : args.action_index + future_count,
                :7,
            ]
            row: dict[str, object] = {
                "frame": frame_index,
                "state8": state.tolist(),
                "gt_next_frame": next_index,
                "action_index": args.action_index,
                "predicted_action": chunk[args.action_index].tolist(),
                "gt_next_q": next_gt.tolist(),
                "gt_actual_action": np.concatenate(
                    (actual_arm[frame_index], [gripper[frame_index]])
                ).tolist(),
                "selected_vs_gt_next": _action_error(pred, next_gt),
                "selected_vs_current_state": _action_error(pred, current_gt),
                "selected_vs_recorded_actual": _action_error(pred, actual_arm[frame_index]),
                "chunk_vs_gt_future": (
                    _action_error(horizon_pred.reshape(-1), future_gt.reshape(-1))
                    if len(future_gt)
                    else None
                ),
                "chunk_length": int(len(chunk)),
            }
            if len(future_gt):
                horizon_error = horizon_pred - future_gt
                row["chunk_vs_gt_future_per_joint_mae_rad"] = np.mean(
                    np.abs(horizon_error), axis=0
                ).tolist()
            rows.append(row)

            selected_error = row["selected_vs_gt_next"]
            assert isinstance(selected_error, dict)
            lines = [
                f"traj_0 frame: {frame_index:03d}",
                f"selected action: [{args.action_index}]",
                f"pred vs GT next MAE: {1000.0 * float(selected_error['mae_rad']):.2f} mrad",
                f"pred vs GT next max: {1000.0 * float(selected_error['max_abs_rad']):.2f} mrad",
            ]
            if row["chunk_vs_gt_future"] is not None:
                chunk_error = row["chunk_vs_gt_future"]
                assert isinstance(chunk_error, dict)
                lines.append(
                    f"aligned chunk MAE: {1000.0 * float(chunk_error['mae_rad']):.2f} mrad"
                )

            if args.save_video:
                if front_writer is None:
                    height, width = front.shape[:2]
                    front_writer = _open_writer(
                        args.output_dir / "real_front_pi0_probe.mp4",
                        width,
                        height,
                        args.video_fps / args.stride,
                    )
                    height, width = wrist.shape[:2]
                    wrist_writer = _open_writer(
                        args.output_dir / "real_wrist_pi0_probe.mp4",
                        width,
                        height,
                        args.video_fps / args.stride,
                    )
                front_writer.write(_annotate(front, lines))
                wrist_writer.write(_annotate(wrist, lines))
    finally:
        front_capture.release()
        wrist_capture.release()
        if front_writer is not None:
            front_writer.release()
        if wrist_writer is not None:
            wrist_writer.release()
        _close_client(client)

    selected_mae = np.asarray(
        [row["selected_vs_gt_next"]["mae_rad"] for row in rows], dtype=np.float64
    )
    selected_max = np.asarray(
        [row["selected_vs_gt_next"]["max_abs_rad"] for row in rows], dtype=np.float64
    )
    per_joint = np.asarray(
        [row["selected_vs_gt_next"]["per_joint_abs_rad"] for row in rows], dtype=np.float64
    )
    horizon_rows = [row for row in rows if row["chunk_vs_gt_future"] is not None]
    summary = {
        "h5": str(args.h5.resolve()),
        "front_video": str(front_path.resolve()),
        "wrist_video": str(wrist_path.resolve()),
        "server": {"host": args.host, "port": args.port},
        "prompt": args.prompt,
        "trajectory_frames": frame_count,
        "probed_frames": len(rows),
        "stride": args.stride,
        "action_index": args.action_index,
        "selected_action_vs_gt_next": {
            "mean_mae_rad": float(selected_mae.mean()),
            "median_mae_rad": float(np.median(selected_mae)),
            "max_mae_rad": float(selected_mae.max()),
            "mean_max_abs_rad": float(selected_max.mean()),
            "per_joint_mean_abs_rad": per_joint.mean(axis=0).tolist(),
        },
        "chunk_vs_gt_future": {
            "probed_frames_with_future": len(horizon_rows),
            "mean_mae_rad": float(
                np.mean([row["chunk_vs_gt_future"]["mae_rad"] for row in horizon_rows])
            )
            if horizon_rows
            else None,
        },
        "note": (
            "GT-next compares action[action_index] with the next recorded joint state. "
            "This is an offline policy diagnostic, not a closed-loop success test."
        ),
        "video": {
            "enabled": bool(args.save_video),
            "front": str(args.output_dir / "real_front_pi0_probe.mp4") if args.save_video else "",
            "wrist": str(args.output_dir / "real_wrist_pi0_probe.mp4") if args.save_video else "",
            "error": video_error,
        },
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (args.output_dir / "frames.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    with (args.output_dir / "action_error.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "frame",
                "selected_mae_rad",
                "selected_max_abs_rad",
                *[f"joint_{index}_abs_error_rad" for index in range(7)],
            ]
        )
        for row in rows:
            error = row["selected_vs_gt_next"]
            writer.writerow(
                [
                    row["frame"],
                    error["mae_rad"],
                    error["max_abs_rad"],
                    *error["per_joint_abs_rad"],
                ]
            )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
