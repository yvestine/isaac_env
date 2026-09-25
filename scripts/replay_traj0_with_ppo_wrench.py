#!/usr/bin/env python3
"""Run an isolated traj0 replay and expose the same wrench fields as PPO play.

The existing replay implementation is invoked without modification.  After
generation, this wrapper makes the force contract explicit:

* PPO HDF5 ``observations/effort`` equivalent: ``wrench_final`` (default).
* PPO CSV ``force_local.csv`` equivalent: ``wrench_model``.

Both streams originate from the RealSim environment's force-sensor incoming
joint wrench pipeline, exactly as in rl_games/play.py data collection.
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

import h5py
import numpy as np


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--h5", type=Path, default=Path("real_data/traj_0/data.h5"))
parser.add_argument(
    "--output-dir",
    type=Path,
    default=Path("outputs/traj0_ppo_wrench_rollout/traj_0"),
)
parser.add_argument("--task", default="TacEx-RealSim-PegInsert-Direct-v0")
parser.add_argument("--device", default="cuda:0")
parser.add_argument("--fps", type=float, default=10.0)
parser.add_argument("--max-frames", type=int, default=None)
parser.add_argument(
    "--primary-force",
    choices=("hdf5-effort", "force-local"),
    default="hdf5-effort",
    help=(
        "Training field selection. hdf5-effort matches play.py HDF5 observations/effort "
        "(wrench_final); force-local matches play.py force_local.csv (wrench_model)."
    ),
)
parser.add_argument("--collision-substeps", type=int, default=4)
parser.add_argument("--contact-force-smoothing-alpha", type=float, default=0.25)
parser.add_argument("--headless", action=argparse.BooleanOptionalAction, default=True)
args = parser.parse_args()


ROOT = Path(__file__).resolve().parents[1]
BASE_REPLAY = ROOT / "scripts" / "replay_real_joint_ppo.py"


def write_wrench_csv(path: Path, values: np.ndarray, prefix: str) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow([f"{prefix}_{index}" for index in range(6)])
        writer.writerows(np.asarray(values, dtype=np.float32).tolist())


def replace_dataset(handle: h5py.File, path: str, values: np.ndarray) -> None:
    if path not in handle:
        raise KeyError(f"paired rollout HDF5 is missing required dataset: {path}")
    dataset = handle[path]
    if dataset.shape != values.shape:
        raise ValueError(f"{path}: expected shape {dataset.shape}, got {values.shape}")
    dataset[...] = values.astype(dataset.dtype, copy=False)


def apply_ppo_force_contract(output_dir: Path, primary_force: str) -> dict:
    h5_path = output_dir / "data.h5"
    if not h5_path.is_file():
        raise FileNotFoundError(f"base replay did not generate {h5_path}")

    with h5py.File(h5_path, "r+") as handle:
        required = (
            "sim/force/wrench_raw",
            "sim/force/wrench_base",
            "sim/force/wrench_final",
            "sim/force/wrench_model_clean",
            "sim/force/wrench_model",
        )
        missing = [path for path in required if path not in handle]
        if missing:
            raise KeyError(f"base replay is missing PPO wrench streams: {missing}")

        streams = {
            name: np.asarray(handle[f"sim/force/{name}"], dtype=np.float32)
            for name in (
                "wrench_raw",
                "wrench_base",
                "wrench_final",
                "wrench_model_clean",
                "wrench_model",
            )
        }
        for name, values in streams.items():
            if values.ndim != 2 or values.shape[1] != 6 or not np.isfinite(values).all():
                raise ValueError(f"invalid PPO wrench stream {name}: shape={values.shape}")

        # This mirrors RealSimEnv._save_tavla_hdf5(): observations/effort is
        # wrench_final.  force_local.csv is the noisy wrench_model stream.
        hdf5_effort = streams["wrench_final"]
        force_local = streams["wrench_model"]
        primary = hdf5_effort if primary_force == "hdf5-effort" else force_local
        source_name = "wrench_final" if primary_force == "hdf5-effort" else "wrench_model"

        for suffix in ("", "_base", "_stiffness"):
            replace_dataset(handle, f"obs/state/ee_wrench{suffix}", primary)
            replace_dataset(handle, f"obs/state/ee_force{suffix}", primary[:, :3])
            replace_dataset(handle, f"obs/state/ee_torque{suffix}", primary[:, 3:])

        force_group = handle["sim/force"]
        aliases = {
            "ppo_observations_effort": hdf5_effort,
            "ppo_force_local_model": force_local,
        }
        for name, values in aliases.items():
            if name in force_group:
                del force_group[name]
            force_group.create_dataset(name, data=values, compression="gzip")
        force_group.attrs["training_contract_source"] = source_name
        force_group.attrs["ppo_hdf5_effort_source"] = "wrench_final"
        force_group.attrs["ppo_force_local_csv_source"] = "wrench_model"
        force_group.attrs["acquisition_source"] = (
            "force_sensor get_link_incoming_joint_force through RealSim PPO wrench pipeline"
        )
        force_group.attrs["contact_wrench_role"] = "diagnostic only"
        handle.attrs["sim_force_training_source"] = source_name
        handle.attrs["ppo_wrench_contract"] = True

    write_wrench_csv(output_dir / "ppo_observations_effort.csv", hdf5_effort, "ppo_effort")
    write_wrench_csv(output_dir / "ppo_force_local.csv", force_local, "force_local")
    write_wrench_csv(output_dir / "force_local.csv", force_local, "force")
    # Existing visualization tools use this filename for the selected training
    # stream. The metadata below records that it is a simulated PPO field, not
    # the proprietary Franka estimator.
    write_wrench_csv(output_dir / "O_F_ext_hat_K.csv", primary, "ppo_training_wrench")

    summary = {
        "primary_force": primary_force,
        "primary_sim_force_dataset": f"sim/force/{source_name}",
        "training_dataset": "obs/state/ee_wrench_base",
        "ppo_hdf5_effort": "sim/force/ppo_observations_effort",
        "ppo_force_local": "sim/force/ppo_force_local_model",
        "source": "force_sensor incoming joint wrench",
        "contact_force_used_for_training": False,
        "frame_count": int(len(primary)),
        "primary_force_p95_n": float(np.percentile(np.linalg.norm(primary[:, :3], axis=1), 95)),
        "primary_force_nonzero_frames": int(np.count_nonzero(np.linalg.norm(primary[:, :3], axis=1) > 1e-6)),
    }
    (output_dir / "ppo_wrench_contract.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    metadata_path = output_dir / "replay_metadata.json"
    if metadata_path.is_file():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        metadata["training_wrench_source"] = f"ppo-{source_name}"
        metadata["ppo_wrench_contract"] = summary
        metadata["contact_pair_collection"]["used_by_controller"] = False
        metadata["contact_pair_collection"]["used_for_training"] = False
        metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    if not BASE_REPLAY.is_file():
        raise FileNotFoundError(BASE_REPLAY)
    if not args.h5.is_file():
        raise FileNotFoundError(args.h5)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    command = [
        sys.executable,
        "-u",
        str(BASE_REPLAY),
        "--enable_cameras",
        "--device",
        args.device,
        "--task",
        args.task,
        "--h5",
        str(args.h5),
        "--output-dir",
        str(args.output_dir),
        "--fps",
        str(args.fps),
        "--save-sim-force",
        "--physical-wrist-force",
        "--training-wrench-source",
        "incoming-joint",
        "--incoming-wrench-body",
        "force_sensor",
        "--record-contact-pair",
        "--resolve-asset-collisions",
        "--collision-substeps",
        str(args.collision_substeps),
        "--contact-force-smoothing-alpha",
        str(args.contact_force_smoothing_alpha),
        "--hole-reference",
        "gt-final",
    ]
    if args.headless:
        command.append("--headless")
    if args.max_frames is not None:
        command.extend(("--max-frames", str(args.max_frames)))

    print("[PPOWrenchReplay] launching isolated base replay", flush=True)
    subprocess.run(command, cwd=ROOT, check=True)
    summary = apply_ppo_force_contract(args.output_dir, args.primary_force)
    print(
        "[PPOWrenchReplay] complete: "
        f"source={summary['primary_sim_force_dataset']} "
        f"nonzero={summary['primary_force_nonzero_frames']}/{summary['frame_count']} "
        f"p95={summary['primary_force_p95_n']:.6f} N",
        flush=True,
    )
    print(f"[PPOWrenchReplay] data={args.output_dir / 'data.h5'}", flush=True)


if __name__ == "__main__":
    main()
