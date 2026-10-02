#!/usr/bin/env python3
"""Package continuous traj0 with calibrated panda_link7 insertion force."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import h5py
import numpy as np


WRENCH_HEADER = ("Fx", "Fy", "Fz", "Tx", "Ty", "Tz")
STREAM_FILES = {}


def read_csv(path: Path, columns: int) -> np.ndarray:
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        next(reader, None)
        rows = [[float(value) for value in row] for row in reader if row]
    values = np.asarray(rows, dtype=np.float32)
    if values.ndim != 2 or values.shape[1] != columns:
        raise ValueError(f"{path}: expected Nx{columns}, got {values.shape}")
    if not np.isfinite(values).all():
        raise FloatingPointError(f"{path}: contains NaN or Inf")
    return values


def read_named_csv(path: Path) -> tuple[list[str], np.ndarray]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        header = next(reader, None)
        rows = [[float(value) for value in row] for row in reader if row]
    if not header:
        raise ValueError(f"{path}: missing CSV header")
    values = np.asarray(rows, dtype=np.float32)
    if values.ndim != 2 or values.shape[1] != len(header):
        raise ValueError(f"{path}: header/data shape mismatch {len(header)} vs {values.shape}")
    if not np.isfinite(values).all():
        raise FloatingPointError(f"{path}: contains NaN or Inf")
    return header, values


def write_csv(path: Path, values: np.ndarray, header=WRENCH_HEADER) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(np.asarray(values).tolist())


def causal_median_filter(values: np.ndarray, window: int = 3) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 2 or not len(values) or window <= 0 or window % 2 == 0:
        raise ValueError("causal median filter requires non-empty NxD data and an odd window")
    filtered = np.empty_like(values)
    for index in range(len(values)):
        start = max(0, index - window + 1)
        filtered[index] = np.median(values[start : index + 1], axis=0)
    return filtered


def causal_lowpass(
    values: np.ndarray,
    sample_dt: float,
    cutoff_hz: float,
) -> tuple[np.ndarray, float]:
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 2 or not len(values) or sample_dt <= 0.0 or cutoff_hz <= 0.0:
        raise ValueError("causal low-pass requires non-empty NxD data and positive timing")
    alpha = 1.0 - np.exp(-2.0 * np.pi * cutoff_hz * sample_dt)
    filtered = np.empty_like(values)
    filtered[0] = values[0]
    for index in range(1, len(values)):
        filtered[index] = alpha * values[index] + (1.0 - alpha) * filtered[index - 1]
    return filtered, float(alpha)


def interval_mean_to_frames(
    sample_time: np.ndarray,
    values: np.ndarray,
    frame_time: np.ndarray,
) -> np.ndarray:
    """Average 120 Hz samples over each recorded 10 Hz frame interval."""
    sample_time = np.asarray(sample_time, dtype=np.float64).reshape(-1)
    frame_time = np.asarray(frame_time, dtype=np.float64).reshape(-1)
    values = np.asarray(values, dtype=np.float64)
    if len(sample_time) != len(values) or values.ndim != 2:
        raise ValueError("high-rate timestamps and wrench values are not aligned")
    if len(frame_time) < 1 or np.any(np.diff(frame_time) <= 0.0):
        raise ValueError("frame timestamps must be strictly increasing")
    rows = []
    for index, timestamp in enumerate(frame_time):
        if index == 0:
            nearest = int(np.argmin(np.abs(sample_time - timestamp)))
            rows.append(values[nearest])
            continue
        start = frame_time[index - 1]
        mask = (sample_time > start + 1.0e-9) & (sample_time <= timestamp + 1.0e-9)
        if not np.any(mask):
            nearest = int(np.argmin(np.abs(sample_time - timestamp)))
            rows.append(values[nearest])
        else:
            rows.append(np.mean(values[mask], axis=0))
    return np.asarray(rows, dtype=np.float32)


def copy_real_schema(source_path: Path, output_path: Path, frame_count: int) -> None:
    with h5py.File(source_path, "r") as source, h5py.File(output_path, "w") as output:
        source_frames = int(source["timestamps"].shape[0])

        def copy_group(source_group, output_group) -> None:
            for key, value in source_group.attrs.items():
                output_group.attrs[key] = value
            for name, node in source_group.items():
                if isinstance(node, h5py.Group):
                    copy_group(node, output_group.create_group(name))
                    continue
                data = node[:frame_count] if node.ndim and node.shape[0] == source_frames else node[()]
                kwargs = {"compression": node.compression} if np.asarray(data).ndim and node.compression else {}
                dataset = output_group.create_dataset(name, data=data, dtype=node.dtype, **kwargs)
                for key, value in node.attrs.items():
                    dataset.attrs[key] = value

        copy_group(source, output)


def replace_dataset(handle: h5py.File, path: str, values: np.ndarray) -> None:
    if path not in handle:
        raise KeyError(f"required paired-data field is missing: {path}")
    dataset = handle[path]
    if dataset.shape != values.shape:
        raise ValueError(f"{path}: H5 shape {dataset.shape}, replay shape {values.shape}")
    dataset[...] = values.astype(dataset.dtype, copy=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--real-h5", type=Path, required=True)
    parser.add_argument("--sim-dir", type=Path, required=True)
    parser.add_argument(
        "--force-dir",
        type=Path,
        default=None,
        help=(
            "Optional continuous-physics force directory. When supplied, videos, "
            "joint replay and metadata stay in --sim-dir while only force streams "
            "are imported from this directory."
        ),
    )
    parser.add_argument(
        "--no-contact-force-dir",
        type=Path,
        default=None,
        help=(
            "Optional continuous replay of the same targets with hole collision disabled. "
            "When supplied, the base/K incoming wrench is subtracted sample-by-sample "
            "before causal filtering and becomes the training force."
        ),
    )
    parser.add_argument("--isolated-cutoff-hz", type=float, default=1.2)
    parser.add_argument(
        "--isolated-force-gain",
        type=float,
        default=1.0,
        help="Explicit global gain after counterfactual subtraction; default keeps physical units.",
    )
    args = parser.parse_args()
    if args.isolated_cutoff_hz <= 0.0:
        raise ValueError("--isolated-cutoff-hz must be positive")
    if args.isolated_force_gain <= 0.0:
        raise ValueError("--isolated-force-gain must be positive")
    force_dir = args.force_dir if args.force_dir is not None else args.sim_dir
    split_visual_force = force_dir.resolve() != args.sim_dir.resolve()
    aligned_force_files = {
        "clean": force_dir / "panda_link7_force_clean.csv",
        "model": force_dir / "panda_link7_force_model.csv",
        "highrate": force_dir / "panda_link7_force_alignment_120hz.csv",
        "config": force_dir / "force_alignment_config.json",
    }
    use_online_alignment = all(path.is_file() for path in aligned_force_files.values())
    use_counterfactual = args.no_contact_force_dir is not None and not use_online_alignment

    missing = [
        str(force_dir / filename)
        for filename in (
            *STREAM_FILES.values(),
            "panda_link7_wrench_120hz.csv",
            "panda_link7_wrench_filtered.csv",
            "panda_link7_wrench_calibrated.csv",
            "panda_link7_force_feedback.csv",
        )
        if not (force_dir / filename).is_file()
    ]
    missing.extend(
        str(args.sim_dir / filename)
        for filename in ("joint_pos_sim.csv", "timestamps.csv")
        if not (args.sim_dir / filename).is_file()
    )
    if use_counterfactual:
        for directory in (force_dir, args.no_contact_force_dir):
            path = directory / "panda_link7_wrench_base_at_K_120hz.csv"
            if not path.is_file():
                missing.append(str(path))
    if missing:
        raise RuntimeError(
            "continuous replay did not finish stage 1; missing raw outputs: "
            + ", ".join(missing)
        )
    streams = {
        name: read_csv(force_dir / filename, 6)
        for name, filename in STREAM_FILES.items()
    }
    replay_q = read_csv(args.sim_dir / "joint_pos_sim.csv", 7)
    timestamps = read_csv(args.sim_dir / "timestamps.csv", 1)[:, 0]
    link7_filtered = read_csv(force_dir / "panda_link7_wrench_filtered.csv", 6)
    link7_calibrated = read_csv(force_dir / "panda_link7_wrench_calibrated.csv", 6)
    aligned_clean = (
        read_csv(aligned_force_files["clean"], 6) if use_online_alignment else None
    )
    aligned_model = (
        read_csv(aligned_force_files["model"], 6) if use_online_alignment else None
    )
    link7_feedback = read_csv(force_dir / "panda_link7_force_feedback.csv", 3)
    link7_120hz = read_csv(force_dir / "panda_link7_wrench_120hz.csv", 19)
    frame_count = len(replay_q)
    lengths = {
        "joint_pos_sim": frame_count,
        "timestamps": len(timestamps),
        "panda_link7_wrench_filtered": len(link7_filtered),
        "panda_link7_wrench_calibrated": len(link7_calibrated),
        "panda_link7_force_feedback": len(link7_feedback),
    }
    if use_online_alignment:
        lengths["panda_link7_force_clean"] = len(aligned_clean)
        lengths["panda_link7_force_model"] = len(aligned_model)
    lengths.update({name: len(values) for name, values in streams.items()})
    if len(set(lengths.values())) != 1:
        raise ValueError(f"continuous replay streams are not aligned: {lengths}")

    isolated_highrate = None
    isolated_corrected = None
    isolated_filtered = None
    isolated_filter_alpha = None
    isolated_frame_unscaled = None
    alignment_config = None
    if use_online_alignment:
        if not np.allclose(aligned_clean[:, 3:], 0.0, atol=0.0, rtol=0.0):
            raise ValueError("online-aligned clean torque channels must be exactly zero")
        if not np.allclose(aligned_model[:, 3:], 0.0, atol=0.0, rtol=0.0):
            raise ValueError("online-aligned model torque channels must be exactly zero")
        primary = aligned_model
        primary_source_name = "panda_link7_force_model_base"
    elif use_counterfactual:
        contact_base_k = read_csv(
            force_dir / "panda_link7_wrench_base_at_K_120hz.csv", 7
        ).astype(np.float64)
        no_contact_base_k = read_csv(
            args.no_contact_force_dir / "panda_link7_wrench_base_at_K_120hz.csv",
            7,
        ).astype(np.float64)
        contact_time = contact_base_k[:, 0] - contact_base_k[0, 0]
        no_contact_time = no_contact_base_k[:, 0] - no_contact_base_k[0, 0]
        if len(contact_time) < 2 or len(no_contact_time) < 2:
            raise ValueError("counterfactual high-rate streams are too short")
        if np.any(np.diff(contact_time) <= 0.0) or np.any(np.diff(no_contact_time) <= 0.0):
            raise ValueError("counterfactual high-rate timestamps must increase")
        tolerance = 2.0 * max(
            float(np.median(np.diff(contact_time))),
            float(np.median(np.diff(no_contact_time))),
        )
        if (
            contact_time[0] < no_contact_time[0] - tolerance
            or contact_time[-1] > no_contact_time[-1] + tolerance
        ):
            raise ValueError("no-contact wrench does not cover the contact replay duration")
        no_contact_aligned = np.column_stack(
            [
                np.interp(contact_time, no_contact_time, no_contact_base_k[:, index])
                for index in range(1, 7)
            ]
        )
        isolated_highrate = contact_base_k[:, 1:7] - no_contact_aligned
        baseline_mask = contact_time <= 3.0
        if not np.any(baseline_mask):
            raise ValueError("counterfactual stream has no samples in the first 3 seconds")
        isolated_baseline = np.median(
            isolated_highrate[baseline_mask], axis=0, keepdims=True
        )
        isolated_corrected = isolated_highrate - isolated_baseline
        isolated_despiked = causal_median_filter(isolated_corrected, window=3)
        isolated_filtered, isolated_filter_alpha = causal_lowpass(
            isolated_despiked,
            float(np.median(np.diff(contact_time))),
            float(args.isolated_cutoff_hz),
        )
        frame_time = timestamps.astype(np.float64) - float(timestamps[0])
        isolated_frame_unscaled = interval_mean_to_frames(
            contact_time,
            isolated_filtered,
            frame_time,
        )
        primary = isolated_frame_unscaled * float(args.isolated_force_gain)
        primary_source_name = "panda_link7_wrench_contact_isolated_base_at_K"
    else:
        # Backward-compatible path for callers that do not provide a second
        # no-contact replay. Components remain in the PhysX native frame.
        primary = link7_calibrated
        primary_source_name = "panda_link7_wrench_calibrated"

    if len(primary) != frame_count or primary.shape[1] != 6:
        raise ValueError(f"primary wrench is not aligned Nx6: {primary.shape}")
    if use_counterfactual:
        highrate_table = np.column_stack(
            (
                contact_time,
                isolated_highrate,
                isolated_corrected,
                isolated_filtered,
            )
        )
        highrate_header = ["timestamp"] + [
            f"{prefix}_{name}"
            for prefix in ("raw", "corrected", "filtered")
            for name in WRENCH_HEADER
        ]
        write_csv(
            args.sim_dir
            / "panda_link7_wrench_contact_isolated_base_at_K_120hz.csv",
            highrate_table,
            highrate_header,
        )
        write_csv(
            args.sim_dir
            / "panda_link7_wrench_contact_isolated_base_at_K_unscaled.csv",
            isolated_frame_unscaled,
        )
        write_csv(
            args.sim_dir / "panda_link7_wrench_contact_isolated_base_at_K.csv",
            primary,
        )

    data_path = args.sim_dir / "data.h5"
    copy_real_schema(args.real_h5, data_path, frame_count)
    with h5py.File(data_path, "r+") as handle:
        for suffix in ("", "_base", "_stiffness"):
            replace_dataset(handle, f"obs/state/ee_wrench{suffix}", primary)
            replace_dataset(handle, f"obs/state/ee_force{suffix}", primary[:, :3])
            replace_dataset(handle, f"obs/state/ee_torque{suffix}", primary[:, 3:])
            wrench_dataset = handle[f"obs/state/ee_wrench{suffix}"]
            for key in list(wrench_dataset.attrs):
                del wrench_dataset.attrs[key]
            wrench_dataset.attrs["source"] = primary_source_name
            wrench_dataset.attrs["frame"] = (
                "robot_base_O"
                if use_counterfactual or use_online_alignment
                else "PhysX_native_incoming_joint"
            )
            wrench_dataset.attrs["torque_reference"] = (
                "fingertip_stiffness_frame_K"
                if use_counterfactual
                else "zeroed" if use_online_alignment else "PhysX_native_incoming_joint"
            )
            wrench_dataset.attrs["counterfactual_no_contact_subtracted"] = bool(
                use_counterfactual
            )
            wrench_dataset.attrs["online_causal_alignment"] = bool(use_online_alignment)
            wrench_dataset.attrs["torque_zeroed"] = bool(use_online_alignment)
            for field in ("ee_force", "ee_torque"):
                dataset = handle[f"obs/state/{field}{suffix}"]
                for key in list(dataset.attrs):
                    del dataset.attrs[key]
                dataset.attrs["source"] = primary_source_name
                dataset.attrs["frame"] = wrench_dataset.attrs["frame"]
                dataset.attrs["online_causal_alignment"] = bool(use_online_alignment)
                dataset.attrs["torque_zeroed"] = bool(use_online_alignment)

        state_group = handle["obs/state"]
        for name, values in (
            ("insertion_force_feedback", link7_feedback[:, 0:1]),
            ("insertion_force_substep_rms", link7_feedback[:, 1:2]),
            ("insertion_jam_binary", link7_feedback[:, 2:3]),
        ):
            if name in state_group:
                del state_group[name]
            state_group.create_dataset(name, data=values, compression="gzip")

        sim_force = handle.require_group("sim").require_group("force")
        for name, values in streams.items():
            if name in sim_force:
                del sim_force[name]
            sim_force.create_dataset(name, data=values, compression="gzip")

        if use_online_alignment:
            alignment_header, alignment_values = read_named_csv(
                aligned_force_files["highrate"]
            )
            expected_prefixes = (
                "raw_base",
                "predicted_baseline",
                "residual_tare",
                "baseline_corrected",
                "median_filtered",
                "force_clean",
                "force_model",
            )
            if alignment_header[0] != "timestamp" or alignment_header[-1] != "tare_ready":
                raise ValueError("unexpected online force-alignment CSV contract")
            column_lookup = {name: index for index, name in enumerate(alignment_header)}
            for prefix in expected_prefixes:
                columns = [
                    column_lookup[f"{prefix}_{axis}"] for axis in ("Fx", "Fy", "Fz")
                ]
                dataset_name = f"panda_link7_{prefix}_base_120hz"
                if dataset_name in sim_force:
                    del sim_force[dataset_name]
                sim_force.create_dataset(
                    dataset_name,
                    data=alignment_values[:, columns],
                    compression="gzip",
                )
            for dataset_name, values in (
                (
                    "panda_link7_force_alignment_120hz_timestamps",
                    alignment_values[:, 0],
                ),
                (
                    "panda_link7_force_alignment_tare_ready",
                    alignment_values[:, -1].astype(np.bool_),
                ),
                ("panda_link7_force_clean_base", aligned_clean),
                ("panda_link7_force_model_base", aligned_model),
            ):
                if dataset_name in sim_force:
                    del sim_force[dataset_name]
                sim_force.create_dataset(dataset_name, data=values, compression="gzip")
            alignment_config = json.loads(
                aligned_force_files["config"].read_text(encoding="utf-8")
            )
            if alignment_config.get("version") != "causal_force_alignment_v1":
                raise ValueError("unsupported online force-alignment config version")
            sim_force.attrs["online_force_alignment_config"] = json.dumps(
                alignment_config, sort_keys=True
            )

        if use_counterfactual:
            for name, values in (
                ("panda_link7_wrench_contact_isolated_base_at_K_raw_120hz", isolated_highrate),
                (
                    "panda_link7_wrench_contact_isolated_base_at_K_corrected_120hz",
                    isolated_corrected,
                ),
                (
                    "panda_link7_wrench_contact_isolated_base_at_K_filtered_120hz",
                    isolated_filtered,
                ),
                (
                    "panda_link7_wrench_contact_isolated_base_at_K_unscaled",
                    isolated_frame_unscaled,
                ),
                ("panda_link7_wrench_contact_isolated_base_at_K", primary),
            ):
                if name in sim_force:
                    del sim_force[name]
                sim_force.create_dataset(name, data=values, compression="gzip")
            sim_force.create_dataset(
                "panda_link7_wrench_contact_isolated_120hz_timestamps",
                data=contact_time,
                compression="gzip",
            )
        for name, values in (
            ("panda_link7_wrench_filtered", link7_filtered),
            ("panda_link7_wrench_calibrated", link7_calibrated),
            ("panda_link7_force_norm_filtered", np.linalg.norm(link7_filtered[:, :3], axis=1)),
            ("panda_link7_force_norm_calibrated", np.linalg.norm(link7_calibrated[:, :3], axis=1)),
            ("panda_link7_force_norm_substep_rms", link7_feedback[:, 1]),
            ("panda_link7_jam_binary", link7_feedback[:, 2]),
            ("panda_link7_wrench_120hz_timestamps", link7_120hz[:, 0]),
            ("panda_link7_wrench_120hz_raw", link7_120hz[:, 1:7]),
            ("panda_link7_wrench_120hz_baseline_corrected", link7_120hz[:, 7:13]),
            ("panda_link7_wrench_120hz_filtered", link7_120hz[:, 13:19]),
        ):
            if name in sim_force:
                del sim_force[name]
            sim_force.create_dataset(name, data=values, compression="gzip")

        total_path = force_dir / "contact_total_base.csv"
        if total_path.is_file():
            contact_total = read_csv(total_path, 3)
        else:
            contact_total = np.zeros((frame_count, 3), dtype=np.float32)
        if len(contact_total) != frame_count:
            raise ValueError("contact_total_base length differs from rollout")
        contact_count = (np.linalg.norm(contact_total, axis=1) > 1.0e-6).astype(np.float32)
        contact_wrench = np.column_stack(
            (contact_total, np.zeros((frame_count, 3), dtype=np.float32))
        )
        for name, values in (
            ("contact_total_base", contact_total),
            ("contact_wrench_base", contact_wrench),
            ("contact_count", contact_count),
        ):
            if name in sim_force:
                del sim_force[name]
            sim_force.create_dataset(name, data=values, compression="gzip")

        feedback_path = force_dir / "insertion_feedback.csv"
        if feedback_path.is_file():
            insertion_feedback = read_csv(feedback_path, 3)
            if len(insertion_feedback) != frame_count:
                raise ValueError("insertion_feedback length differs from rollout")
            for name, values in (
                ("insertion_contact_force_norm", insertion_feedback[:, 0]),
                ("insertion_feedback_ema", insertion_feedback[:, 1]),
                ("insertion_contact_binary", insertion_feedback[:, 2]),
            ):
                if name in sim_force:
                    del sim_force[name]
                sim_force.create_dataset(name, data=values, compression="gzip")
            sim_force.attrs["insertion_feedback_role"] = (
                "peg-hole contact feedback for correct-vs-jammed insertion; not a wrist wrench"
            )

        all_body_path = force_dir / "incoming_wrench_all_bodies.csv"
        if all_body_path.is_file():
            header, all_body = read_named_csv(all_body_path)
            if len(all_body) != frame_count or len(header) % 6:
                raise ValueError("incoming_wrench_all_bodies is not aligned Nx(6*body_count)")
            for group_name in (
                "incoming_wrench_by_body_raw",
                "incoming_wrench_by_body_baseline_corrected",
            ):
                if group_name in sim_force:
                    del sim_force[group_name]
            raw_group = sim_force.create_group("incoming_wrench_by_body_raw")
            corrected_group = sim_force.create_group(
                "incoming_wrench_by_body_baseline_corrected"
            )
            baseline_frames = min(30, frame_count)
            for offset in range(0, len(header), 6):
                component_names = header[offset : offset + 6]
                suffixes = tuple(f"_{name}" for name in WRENCH_HEADER)
                if not all(component_names[index].endswith(suffixes[index]) for index in range(6)):
                    raise ValueError(f"unexpected incoming-wrench columns: {component_names}")
                body_name = component_names[0][: -len(suffixes[0])]
                body_values = all_body[:, offset : offset + 6]
                baseline = np.median(body_values[:baseline_frames], axis=0, keepdims=True)
                raw_group.create_dataset(body_name, data=body_values, compression="gzip")
                corrected_group.create_dataset(
                    body_name, data=body_values - baseline, compression="gzip"
                )
            sim_force.attrs["incoming_wrench_body_baseline_frames"] = baseline_frames

        topology_path = force_dir / "articulation_load_path_topology.json"
        if topology_path.is_file():
            topology = json.loads(topology_path.read_text(encoding="utf-8"))
            sim_force.attrs["force_sensor_is_series_ancestor_of_gripper"] = bool(
                topology["force_sensor_is_series_ancestor_of_gripper"]
            )
            recommended_body = topology.get("recommended_topological_wrist_body")
            sim_force.attrs["recommended_topological_wrist_body"] = (
                "none" if recommended_body is None else recommended_body
            )

        sim_force.attrs["training_contract_source"] = primary_source_name
        sim_force.attrs["ppo_hdf5_effort_source"] = primary_source_name
        sim_force.attrs["force_sensor_exported"] = False
        sim_force.attrs["acquisition_source"] = (
            "panda_link7 incoming joint wrench sampled at every 120 Hz PhysX substep, "
            + (
                "directed-load transformed to robot base, q/qd no-contact baseline "
                "predicted, initial residual tared, causal median/low-pass filtered, "
                "interval averaged, and optional AR(1) model residual added"
                if use_online_alignment
                else (
                    "directed-load transformed to robot base/K, same-target no-contact "
                    "wrench subtracted, baseline corrected, causal 3-sample median "
                    "de-spiked, low-pass filtered, then interval averaged"
                    if use_counterfactual
                    else "baseline corrected, causal 3-sample median de-spiked, "
                    "low-pass filtered, then interval averaged"
                )
            )
        )
        sim_force.attrs["panda_link7_component_frame"] = (
            "robot base O, torque at stiffness frame K"
            if use_counterfactual
            else (
                "robot base O; torque zeroed"
                if use_online_alignment
                else "PhysX native incoming-joint frame; force norm only"
            )
        )
        sim_force.attrs["counterfactual_no_contact_subtracted"] = bool(use_counterfactual)
        sim_force.attrs["isolated_force_gain"] = float(args.isolated_force_gain)
        sim_force.attrs["isolated_filter_cutoff_hz"] = float(args.isolated_cutoff_hz)
        sim_force.attrs["online_causal_alignment"] = bool(use_online_alignment)
        sim_force.attrs["torque_zeroed"] = bool(use_online_alignment)
        sim_force.attrs["contact_wrench_role"] = "diagnostic only"
        handle.attrs["sim_force_training_source"] = primary_source_name
        handle.attrs["force_frame"] = (
            "robot_base_O"
            if use_online_alignment or use_counterfactual
            else "PhysX_native_incoming_joint"
        )
        handle.attrs["force_torque_mode"] = (
            "zero" if use_online_alignment else "measured"
        )
        handle.attrs["force_alignment_version"] = (
            "causal_force_alignment_v1" if use_online_alignment else "legacy"
        )
        handle.attrs["continuous_physical_force_replay"] = True
        handle.attrs["visual_replay_source"] = (
            "separate direct joint rollout" if split_visual_force else "continuous physical replay"
        )
        handle.attrs["force_runtime_joint_state_writes"] = False
        handle.attrs["runtime_joint_state_writes"] = bool(split_visual_force)

    # Compatibility names consumed by the existing comparison/validation tools.
    write_csv(args.sim_dir / "O_F_ext_hat_K.csv", primary)
    write_csv(args.sim_dir / "incoming_joint_wrench_base_at_K.csv", primary)
    write_csv(args.sim_dir / "contact_wrench_base_at_K.csv", contact_wrench)
    write_csv(args.sim_dir / "contact_wrench_base.csv", contact_wrench)
    write_csv(args.sim_dir / "contact_count.csv", contact_count[:, None], ("contact_count",))

    metadata_path = args.sim_dir / "replay_metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    force_metadata_path = force_dir / "replay_metadata.json"
    force_metadata = (
        json.loads(force_metadata_path.read_text(encoding="utf-8"))
        if force_metadata_path.is_file()
        else {}
    )
    metadata.update(
        {
            "replay_mode": (
                "direct" if split_visual_force else "servo"
            ),
            "visual_force_decoupled": bool(split_visual_force),
            "runtime_joint_state_writes": bool(split_visual_force),
            "force_runtime_joint_state_writes": False,
            "preparation_joint_state_writes": True,
            "training_wrench_source": primary_source_name,
            "force_replay": {
                "source_directory": str(force_dir.resolve()),
                "controller": force_metadata.get("controller"),
                "physical_pose_error_m": force_metadata.get("physical_pose_error_m"),
                "visual_output_used": False,
            },
            "ppo_force_contract": {
                "hdf5_effort": primary_source_name,
                "sensor": "panda_link7 incoming joint wrench",
                "sample_timing": "every 120 Hz PhysX substep",
                "filtering": (
                    "q/qd motion baseline + residual tare + causal median + "
                    f"{float(alignment_config['filter']['cutoff_hz']):g} Hz causal "
                    "low-pass + interval mean"
                    if use_online_alignment
                    else (
                        "same-target no-contact subtraction + 3 second residual baseline + "
                        "causal 3-sample median de-spike + causal low-pass + interval mean"
                        if use_counterfactual
                        else "3 second baseline + causal 3-sample median de-spike + "
                        "causal low-pass + interval mean"
                    )
                ),
                "component_frame": (
                    "robot base O; torque at stiffness frame K"
                    if use_counterfactual
                    else (
                        "robot base O; force XYZ only; torque zeroed"
                        if use_online_alignment
                        else "PhysX native incoming-joint frame; norm validated"
                    )
                ),
                "alignment_version": (
                    "causal_force_alignment_v1" if use_online_alignment else "legacy"
                ),
                "fixed_delay_frames": 0 if use_online_alignment else None,
                "amplitude_mapping": "identity" if use_online_alignment else None,
                "counterfactual_no_contact_subtracted": bool(use_counterfactual),
                "no_contact_source_directory": (
                    str(args.no_contact_force_dir.resolve())
                    if use_counterfactual
                    else None
                ),
                "contact_proxy_used_for_training": False,
                "load_path_topology": "articulation_load_path_topology.json",
                "all_body_incoming_wrench": "incoming_wrench_all_bodies.csv",
            },
        }
    )
    metadata.setdefault("force_collection", {}).update(
        {
            "primary_file": (
                "panda_link7_force_model.csv"
                if use_online_alignment
                else (
                    "panda_link7_wrench_contact_isolated_base_at_K.csv"
                    if use_counterfactual
                    else "panda_link7_wrench_calibrated.csv"
                )
            ),
            "primary_source": primary_source_name,
            "clean_file": (
                "panda_link7_force_clean.csv" if use_online_alignment else None
            ),
            "torque_zeroed": bool(use_online_alignment),
            "force_sensor_exported": False,
            "contact_proxy_used_for_training": False,
        }
    )
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    summary = {
        "frame_count": frame_count,
        "primary": primary_source_name,
        "visual_replay_mode": "direct" if split_visual_force else "servo",
        "force_replay_mode": "continuous PPO servo",
        "runtime_joint_state_writes": bool(split_visual_force),
        "force_runtime_joint_state_writes": False,
        "online_causal_alignment": bool(use_online_alignment),
        "force_p95_n": float(np.percentile(np.linalg.norm(primary[:, :3], axis=1), 95)),
        "force_max_n": float(np.max(np.linalg.norm(primary[:, :3], axis=1))),
        "joint_tracking_is_expected_to_be_nonzero": True,
    }
    (args.sim_dir / "continuous_ppo_wrench_contract.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(f"[PACKAGE] data={data_path.resolve()}")
    print(f"[PACKAGE] contract={(args.sim_dir / 'continuous_ppo_wrench_contract.json').resolve()}")


if __name__ == "__main__":
    main()
