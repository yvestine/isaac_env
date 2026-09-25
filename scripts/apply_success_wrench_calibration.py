#!/usr/bin/env python3
"""Apply the global successful-insertion wrench calibration to one paired H5."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
from pathlib import Path

import h5py
import numpy as np


SOURCE_DATASET = "sim/force/panda_link7_wrench_contact_isolated_base_at_K_unscaled"


def causal_filter(values: np.ndarray, alpha: float, delay: int) -> np.ndarray:
    output = np.empty_like(values, dtype=np.float64)
    output[0] = values[0]
    for index in range(1, len(values)):
        output[index] = alpha * values[index] + (1.0 - alpha) * output[index - 1]
    if delay == 0:
        return output
    delayed = np.zeros_like(output)
    delayed[delay:] = output[:-delay]
    return delayed


def apply_model(raw: np.ndarray, calibration: dict) -> tuple[np.ndarray, np.ndarray]:
    if calibration.get("version") != "link7_success_wrench_calibration_v3":
        raise ValueError("unsupported calibration version")
    semantics = calibration.get("dataset_semantics", {})
    if semantics.get("all_40_real_trajectories_are_successful_insertions") is not True:
        raise ValueError("calibration does not declare the successful-insertion dataset contract")
    permutation = np.asarray(calibration["axis_permutation"], dtype=np.int64)
    sign = np.asarray(calibration["axis_sign"], dtype=np.float64)
    gain = np.asarray(calibration["axis_gain"], dtype=np.float64)
    bias = np.asarray(calibration["constant_bias"], dtype=np.float64)
    offset = np.asarray(calibration["torque_reference_offset_base_m"], dtype=np.float64)
    if not np.array_equal(permutation, np.arange(6)) or not np.all(sign == 1.0):
        raise ValueError("axis permutation/sign must remain fixed by the directed-load test")
    if gain.shape != (6,) or np.any(gain <= 0.0) or not np.isfinite(gain).all():
        raise ValueError("axis_gain must contain six finite positive values")
    force = raw[:, :3]
    torque = raw[:, 3:] + np.cross(np.broadcast_to(offset, force.shape), force)
    physical = np.concatenate((force, torque), axis=1)
    filtered = causal_filter(
        physical,
        float(calibration["causal_ema_alpha"]),
        int(calibration["causal_delay_frames"]),
    )
    unclipped = filtered * gain + bias
    clipped = np.clip(
        unclipped,
        np.asarray(calibration["clip_lower"], dtype=np.float64),
        np.asarray(calibration["clip_upper"], dtype=np.float64),
    )
    return unclipped, clipped


def replace_dataset(handle: h5py.File, path: str, values: np.ndarray) -> None:
    if path not in handle:
        raise KeyError(f"missing training field: {path}")
    dataset = handle[path]
    if dataset.shape != values.shape:
        raise ValueError(f"{path}: shape mismatch {dataset.shape} != {values.shape}")
    dataset[...] = values.astype(dataset.dtype, copy=False)


def write_csv(path: Path, values: np.ndarray) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(("Fx", "Fy", "Fz", "Tx", "Ty", "Tz"))
        writer.writerows(values.tolist())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sim-h5", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--output-h5", type=Path, required=True)
    parser.add_argument("--trajectory", type=int, required=True)
    args = parser.parse_args()

    calibration_path = args.calibration.expanduser().resolve()
    calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
    fitted = {int(value) for value in calibration.get("fitted_trajectories", [])}
    holdout = {int(value) for value in calibration.get("holdout_trajectories", [])}
    if args.trajectory == 0 and args.trajectory not in holdout:
        raise RuntimeError("traj0 is not declared as a calibration holdout")
    if args.trajectory == 0 and args.trajectory in fitted:
        raise RuntimeError("traj0 must not be used to fit its own calibration")

    source_path = args.sim_h5.expanduser().resolve()
    output_path = args.output_h5.expanduser().resolve()
    if source_path == output_path:
        raise ValueError("--output-h5 must differ from --sim-h5; raw paired H5 is preserved")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_path, output_path)
    calibration_hash = hashlib.sha256(calibration_path.read_bytes()).hexdigest()

    with h5py.File(output_path, "r+") as data:
        if SOURCE_DATASET not in data:
            raise KeyError(f"{source_path}: missing {SOURCE_DATASET}")
        raw = np.asarray(data[SOURCE_DATASET][:], dtype=np.float64)
        if raw.ndim != 2 or raw.shape[1] != 6 or not np.isfinite(raw).all():
            raise ValueError(f"{SOURCE_DATASET} must be finite Nx6")
        unclipped, calibrated = apply_model(raw, calibration)
        force_group = data["sim/force"]
        for name, values in (
            ("wrench_calibrated_unclipped", unclipped),
            ("wrench_calibrated", calibrated),
        ):
            if name in force_group:
                del force_group[name]
            dataset = force_group.create_dataset(name, data=values, compression="gzip")
            dataset.attrs["source"] = SOURCE_DATASET
            dataset.attrs["calibration_sha256"] = calibration_hash
        for suffix in ("", "_base", "_stiffness"):
            replace_dataset(data, f"obs/state/ee_wrench{suffix}", calibrated)
            replace_dataset(data, f"obs/state/ee_force{suffix}", calibrated[:, :3])
            replace_dataset(data, f"obs/state/ee_torque{suffix}", calibrated[:, 3:])
        data.attrs["sim_force_training_source"] = "sim/force/wrench_calibrated"
        data.attrs["sim_wrench_calibration_version"] = calibration["version"]
        data.attrs["sim_wrench_calibration_sha256"] = calibration_hash
        data.attrs["all_real_reference_trajectories_successful"] = True

    write_csv(output_path.parent / "O_F_ext_hat_K_global_calibrated.csv", calibrated)
    summary = {
        "trajectory": int(args.trajectory),
        "input_h5": str(source_path),
        "output_h5": str(output_path),
        "calibration": str(calibration_path),
        "calibration_sha256": calibration_hash,
        "force_p95_n": float(np.quantile(np.linalg.norm(calibrated[:, :3], axis=1), 0.95)),
        "force_max_n": float(np.max(np.linalg.norm(calibrated[:, :3], axis=1))),
    }
    (output_path.parent / "global_calibration_application.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
