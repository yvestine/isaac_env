#!/usr/bin/env python3
"""Validate synchronized real-robot wrench/pose data without robot or server access."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

try:
    import h5py
except ImportError:  # The IsaacLab Python environment normally provides this.
    h5py = None
import numpy as np

CHANNELS = ("Fx", "Fy", "Fz", "Tx", "Ty", "Tz")
REQUIRED_STATE_FIELDS = (
    "EE_T_K", "F_T_EE", "O_T_EE", "m_ee", "F_x_Cee", "I_ee",
    "m_load", "F_x_Cload", "I_load", "m_total",
)


def _find_dataset(data: h5py.File, name: str):
    found = []

    def visitor(path, item):
        if isinstance(item, h5py.Dataset) and path.rsplit("/", 1)[-1] == name:
            found.append(path)

    data.visititems(visitor)
    return found[0] if found else None


def _rotation_from_quaternion(quat: np.ndarray, order: str) -> np.ndarray:
    if order == "xyzw":
        x, y, z, w = np.moveaxis(quat, -1, 0)
    else:
        w, x, y, z = np.moveaxis(quat, -1, 0)
    return np.stack(
        [
            1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w),
            2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w),
            2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y),
        ], axis=-1
    ).reshape(-1, 3, 3)


def _pose_force_check(pose: np.ndarray, base: np.ndarray, stiffness: np.ndarray) -> dict:
    quat = pose[:, 3:7]
    results = {}
    for order in ("xyzw", "wxyz"):
        rotation = _rotation_from_quaternion(quat, order)
        predicted = np.einsum("nij,nj->ni", rotation.transpose(0, 2, 1), base[:, :3])
        error = predicted - stiffness[:, :3]
        results[order] = {
            "mean_rms_N": float(np.sqrt(np.mean(error * error))),
            "p95_abs_N": float(np.percentile(np.abs(error), 95)),
        }
    return results


def _check_file(path: Path) -> dict:
    result = {"file": str(path), "missing": [], "shape": {}, "finite": True}
    with h5py.File(path, "r") as data:
        fields = {
            "ee_pose": "obs/state/ee_pose",
            "ee_wrench_base": "obs/state/ee_wrench_base",
            "ee_wrench_stiffness": "obs/state/ee_wrench_stiffness",
        }
        arrays = {}
        for name, dataset_path in fields.items():
            if dataset_path not in data:
                result["missing"].append(dataset_path)
                continue
            arrays[name] = np.asarray(data[dataset_path], dtype=np.float64)
            result["shape"][name] = list(arrays[name].shape)
            result["finite"] = result["finite"] and bool(np.isfinite(arrays[name]).all())

        state_metadata = {}
        for field in REQUIRED_STATE_FIELDS:
            dataset_path = _find_dataset(data, field)
            state_metadata[field] = dataset_path
        result["robot_state_metadata_paths"] = state_metadata
        result["missing_robot_state_metadata"] = [
            field for field, dataset_path in state_metadata.items() if dataset_path is None
        ]

        if all(name in arrays for name in fields):
            pose = arrays["ee_pose"].reshape(-1, 7)
            base = arrays["ee_wrench_base"].reshape(-1, 6)
            stiffness = arrays["ee_wrench_stiffness"].reshape(-1, 6)
            count = min(len(pose), len(base), len(stiffness))
            result["aligned_frames"] = count
            result["pose_force_order_check"] = _pose_force_check(
                pose[:count], base[:count], stiffness[:count]
            )
            result["declared_pose_order"] = "[x,y,z,qx,qy,qz,qw] is not present in legacy HDF5 attrs"
            result["wrench_order"] = "[Fx,Fy,Fz,Tx,Ty,Tz]"
            result["wrench_units"] = "[N,N,N,N*m,N*m,N*m]"
        for key in ("ee_wrench_base", "ee_wrench_stiffness"):
            if key in arrays and arrays[key].shape[-1] != 6:
                result["missing"].append(f"{key} does not have six components")
    return result


def _plot(results: list[dict], output: Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    labels = ("xyzw", "wxyz")
    means = {label: [] for label in labels}
    names = []
    for result in results:
        check = result.get("pose_force_order_check")
        if not check:
            continue
        names.append(Path(result["file"]).parent.name)
        for label in labels:
            means[label].append(check[label]["mean_rms_N"])
    if not names:
        return
    figure, axis = plt.subplots(figsize=(max(7, len(names) * 0.7), 4.5))
    x = np.arange(len(names))
    for label, values in means.items():
        axis.plot(x, values, marker="o", label=label)
    axis.set_xticks(x, names, rotation=45, ha="right")
    axis.set_ylabel("force-frame mean RMS [N]")
    axis.set_title("Pose quaternion order check: R_ee^T F_base vs F_stiffness")
    axis.grid(True, alpha=0.3)
    axis.legend()
    figure.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=160)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("real_data"))
    parser.add_argument("--output", type=Path, default=Path("outputs/real_wrench_alignment/validation.json"))
    parser.add_argument("--plot", type=Path, default=Path("outputs/real_wrench_alignment/pose_force_order_check.png"))
    args = parser.parse_args()
    if h5py is None:
        raise SystemExit("h5py is required; run this script with the IsaacLab/Python environment")
    files = sorted(args.data_dir.glob("traj_*/data.h5"))
    results = [_check_file(path) for path in files]
    summary = {
        "data_dir": str(args.data_dir),
        "file_count": len(results),
        "results": results,
        "conclusion": "No sign/axis correction is inferred by this script; use known-direction contact tests.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    _plot(results, args.plot)
    print(json.dumps({
        "output": str(args.output),
        "plot": str(args.plot),
        "file_count": len(results),
        "files_missing_robot_state_metadata": sum(bool(r["missing_robot_state_metadata"]) for r in results),
    }, indent=2))


if __name__ == "__main__":
    main()
