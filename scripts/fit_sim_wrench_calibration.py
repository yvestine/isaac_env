#!/usr/bin/env python3
"""Fit one global link7 wrench calibration from successful insertions.

All 40 real trajectories are successful insertions. Trajectories 1..39 are
the calibration set and traj_0 is a strict holdout. Each training trajectory
is truncated at its first simulation-success frame so the fit cannot learn
the large force produced by continuing to push after insertion has succeeded.

Only the verified panda_link7/base-K counterfactual stream is accepted. This
script never changes how the physical simulation wrench is acquired.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np
from scipy.optimize import least_squares


NAMES = ("Fx", "Fy", "Fz", "Tx", "Ty", "Tz")
TRAIN_IDS = tuple(range(1, 40))
HOLDOUT_IDS = (0,)
SIM_DATASET = "sim/force/panda_link7_wrench_contact_isolated_base_at_K_unscaled"


def _timestamps(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    if len(values) < 4 or not np.isfinite(values).all() or np.any(np.diff(values) <= 0.0):
        raise ValueError("timestamps must be finite, increasing and contain at least four samples")
    return values - values[0]


def _features(q: np.ndarray, t: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64)
    qd = np.gradient(q, t, axis=0, edge_order=1)
    return np.column_stack((np.ones(len(q)), q, qd))


def _resample(t: np.ndarray, values: np.ndarray, target_t: np.ndarray) -> np.ndarray:
    return np.column_stack(
        [np.interp(target_t, t, values[:, axis]) for axis in range(values.shape[1])]
    )


def _read_real(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with h5py.File(path, "r") as data:
        t = _timestamps(data["timestamps"][:])
        q = np.asarray(data["obs/state/joint_pos"][:], dtype=np.float64)
        wrench = np.asarray(data["obs/state/ee_wrench_base"][:], dtype=np.float64)
    if q.shape != (len(t), 7) or wrench.shape != (len(t), 6):
        raise ValueError(f"{path}: expected joint_pos (T,7) and ee_wrench_base (T,6)")
    if not np.isfinite(q).all() or not np.isfinite(wrench).all():
        raise ValueError(f"{path}: contains NaN/Inf")
    return t, q, wrench


def _first_success_frame(sim_dir: Path, frame_count: int) -> int:
    metadata_path = sim_dir / "replay_metadata.json"
    if not metadata_path.is_file():
        raise FileNotFoundError(f"{metadata_path}: success metadata is required")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("geometric_success") is not True:
        raise RuntimeError(f"{sim_dir}: trajectory is not marked as a successful insertion")
    frame = metadata.get("first_success_frame")
    if frame is None:
        raise RuntimeError(f"{metadata_path}: first_success_frame is missing")
    frame = int(frame)
    if not 0 <= frame < frame_count:
        raise ValueError(f"{metadata_path}: invalid first_success_frame={frame}")
    return frame


def _read_sim(path: Path, target_t: np.ndarray) -> tuple[np.ndarray, float, int]:
    with h5py.File(path, "r") as data:
        if SIM_DATASET not in data:
            raise KeyError(
                f"{path}: missing {SIM_DATASET}; old sim_force/contact-proxy data "
                "cannot be used for this calibration"
            )
        t = _timestamps(data["timestamps"][:])
        wrench = np.asarray(data[SIM_DATASET][:], dtype=np.float64)
        force_group = data["sim/force"]
        if not bool(force_group.attrs.get("counterfactual_no_contact_subtracted", False)):
            raise RuntimeError(f"{path}: same-target no-contact subtraction was not verified")
        source = str(force_group.attrs.get("training_contract_source", ""))
        if source != "panda_link7_wrench_contact_isolated_base_at_K":
            raise RuntimeError(f"{path}: unexpected training force source {source!r}")
    if wrench.shape != (len(t), 6) or not np.isfinite(wrench).all():
        raise ValueError(f"{path}: {SIM_DATASET} must be finite (T,6)")
    success_frame = _first_success_frame(path.parent, len(t))
    return _resample(t, wrench, target_t), float(t[success_frame]), success_frame


def _causal_filter(values: np.ndarray, alpha: float, delay: int) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    ema = np.empty_like(values)
    ema[0] = values[0]
    for index in range(1, len(values)):
        ema[index] = alpha * values[index] + (1.0 - alpha) * ema[index - 1]
    if delay == 0:
        return ema
    output = np.zeros_like(ema)
    output[delay:] = ema[:-delay]
    return output


def _correlation(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 3 or np.std(a) < 1.0e-12 or np.std(b) < 1.0e-12:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def _metrics(target: np.ndarray, estimate: np.ndarray) -> dict[str, float]:
    metrics = {
        f"{name}_pearson": _correlation(target[:, idx], estimate[:, idx])
        for idx, name in enumerate(NAMES)
    }
    metrics["force_norm_pearson"] = _correlation(
        np.linalg.norm(target[:, :3], axis=1), np.linalg.norm(estimate[:, :3], axis=1)
    )
    metrics["rmse"] = float(np.sqrt(np.mean((target - estimate) ** 2)))
    return metrics


def _validate_direction_test(path: Path) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    summary = data.get("summary", {})
    if summary.get("passed") is not True or summary.get("sign_consistent") is not True:
        raise RuntimeError("panda_link7 directed +/-XYZ validation did not pass")
    if data.get("incoming_wrench_read_body") != "panda_link7":
        raise ValueError("directed wrench result was not measured at panda_link7")
    directions = data.get("directions")
    if not isinstance(directions, list) or len(directions) != 6:
        raise ValueError("directed wrench test must contain six +/-XYZ entries")
    leakage_limit = float(summary.get("cross_axis_leakage_limit", 0.15))
    for item in directions:
        if item.get("passed") is not True:
            raise RuntimeError(f"directed wrench entry failed: {item.get('label', '?')}")
        if float(item.get("cross_axis_leakage_ratio", np.inf)) > leakage_limit + 1.0e-9:
            raise RuntimeError(f"directed wrench leakage exceeds its recorded limit: {item}")
    return data


def _apply_model(
    sim: np.ndarray,
    offset: np.ndarray,
    gain: np.ndarray,
    bias: np.ndarray,
    alpha: float,
    delay: int,
) -> np.ndarray:
    force = sim[:, :3]
    torque = sim[:, 3:] + np.cross(np.broadcast_to(offset, force.shape), force)
    filtered = _causal_filter(np.concatenate((force, torque), axis=1), alpha, delay)
    return filtered * gain + bias


def _contact_weights(sim: np.ndarray) -> np.ndarray:
    """Keep long free-space segments from overwhelming the insertion fit."""
    norm = np.linalg.norm(sim[:, :3], axis=1)
    scale = max(float(np.quantile(norm, 0.95)), 1.0e-9)
    return 1.0 + 4.0 * np.clip(norm / scale, 0.0, 1.0)


def _ridge_baseline(features: np.ndarray, target: np.ndarray, ridge: float) -> np.ndarray:
    regularizer = np.eye(features.shape[1]) * ridge
    regularizer[0, 0] = 0.0
    return np.linalg.solve(
        features.T @ features + regularizer,
        features.T @ target,
    )


def _select_baseline_model(
    feature_blocks: list[np.ndarray],
    target_blocks: list[np.ndarray],
    ridge: float,
) -> tuple[np.ndarray, str, dict[str, float]]:
    """Choose static or q/qd baseline by leave-one-trajectory-out RMSE."""
    static_errors = []
    motion_errors = []
    for holdout in range(len(feature_blocks)):
        train_x = np.concatenate(
            [block for index, block in enumerate(feature_blocks) if index != holdout]
        )
        train_y = np.concatenate(
            [block for index, block in enumerate(target_blocks) if index != holdout]
        )
        test_x = feature_blocks[holdout]
        test_y = target_blocks[holdout]
        static_prediction = np.broadcast_to(np.median(train_y, axis=0), test_y.shape)
        motion_prediction = test_x @ _ridge_baseline(train_x, train_y, ridge)
        static_errors.append(float(np.mean((test_y - static_prediction) ** 2)))
        motion_errors.append(float(np.mean((test_y - motion_prediction) ** 2)))

    static_rmse = float(np.sqrt(np.mean(static_errors)))
    motion_rmse = float(np.sqrt(np.mean(motion_errors)))
    all_x = np.concatenate(feature_blocks)
    all_y = np.concatenate(target_blocks)
    if motion_rmse < static_rmse:
        weights = _ridge_baseline(all_x, all_y, ridge)
        model_type = "q_qd_ridge"
    else:
        weights = np.zeros((all_x.shape[1], all_y.shape[1]), dtype=np.float64)
        weights[0] = np.median(all_y, axis=0)
        model_type = "static_median"
    return weights, model_type, {
        "static_median_leave_one_trajectory_out_rmse": static_rmse,
        "q_qd_ridge_leave_one_trajectory_out_rmse": motion_rmse,
    }


def _positive_diagonal_fit(
    pairs: list[tuple[np.ndarray, np.ndarray]],
    alpha: float,
    delay: int,
) -> tuple[np.ndarray, np.ndarray, float]:
    xs, ys, weights = [], [], []
    for sim, target in pairs:
        xs.append(_causal_filter(sim, alpha, delay))
        ys.append(target)
        weights.append(_contact_weights(sim))
    x = np.concatenate(xs)
    y = np.concatenate(ys)
    weight = np.concatenate(weights)
    gain = np.empty(6)
    bias = np.empty(6)
    for axis in range(6):
        design = np.column_stack((x[:, axis], np.ones(len(x))))
        weighted_design = design * np.sqrt(weight)[:, None]
        weighted_target = y[:, axis] * np.sqrt(weight)
        solution, *_ = np.linalg.lstsq(weighted_design, weighted_target, rcond=None)
        gain[axis] = max(float(solution[0]), 1.0e-4)
        bias[axis] = float(solution[1])
    estimate = np.concatenate(
        [_apply_model(sim, np.zeros(3), gain, bias, alpha, delay) for sim, _ in pairs]
    )
    target = np.concatenate([item[1] for item in pairs])
    metric = _metrics(target, estimate)
    score_terms = [metric["Fz_pearson"], metric["force_norm_pearson"]]
    finite = [value for value in score_terms if np.isfinite(value)]
    return gain, bias, float(np.mean(finite)) if finite else -np.inf


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--real-data-dir", type=Path, default=Path("real_data"))
    parser.add_argument("--sim-root", type=Path, required=True)
    parser.add_argument("--direction-test", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--no-contact-seconds", type=float, default=3.0)
    parser.add_argument("--ridge", type=float, default=1.0e-3)
    parser.add_argument("--max-delay-frames", type=int, default=5)
    args = parser.parse_args()
    if args.no_contact_seconds <= 0.0 or args.ridge < 0.0 or args.max_delay_frames < 0:
        raise ValueError("invalid calibration arguments")

    real_ids = sorted(
        int(path.parent.name.removeprefix("traj_"))
        for path in args.real_data_dir.glob("traj_*/data.h5")
    )
    expected_ids = list(range(40))
    if real_ids != expected_ids:
        raise RuntimeError(f"expected exactly real traj_0..traj_39, found {real_ids}")

    direction_path = args.direction_test.expanduser().resolve()
    direction_result = _validate_direction_test(direction_path)
    baseline_x, baseline_y = [], []
    raw_pairs: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    success_frames: dict[str, int] = {}
    for trajectory_id in TRAIN_IDS:
        real_path = args.real_data_dir / f"traj_{trajectory_id}" / "data.h5"
        sim_path = args.sim_root / f"traj_{trajectory_id}" / "data.h5"
        if not sim_path.is_file():
            raise FileNotFoundError(f"missing current link7 paired trajectory: {sim_path}")
        t, q, real = _read_real(real_path)
        features = _features(q, t)
        baseline_mask = t <= args.no_contact_seconds
        if int(baseline_mask.sum()) < 5:
            raise ValueError(f"traj_{trajectory_id}: insufficient initial free-space frames")
        baseline_x.append(features[baseline_mask])
        baseline_y.append(real[baseline_mask])
        sim, success_time, success_frame = _read_sim(sim_path, t)
        success_mask = t <= success_time + 1.0e-9
        if int(success_mask.sum()) < 5:
            raise ValueError(f"traj_{trajectory_id}: invalid success truncation")
        raw_pairs.append((sim[success_mask], features[success_mask], real[success_mask]))
        success_frames[str(trajectory_id)] = success_frame

    baseline, baseline_model_type, baseline_cv = _select_baseline_model(
        baseline_x, baseline_y, args.ridge
    )
    pairs = [(sim, real - features @ baseline) for sim, features, real in raw_pairs]

    candidates = []
    for alpha in (1.0, 0.75, 0.5, 0.35, 0.25, 0.15):
        for delay in range(args.max_delay_frames + 1):
            gain, bias, score = _positive_diagonal_fit(pairs, alpha, delay)
            candidates.append((score, alpha, delay, gain, bias))
    candidates.sort(key=lambda item: item[0], reverse=True)
    score, alpha, delay, start_gain, start_bias = candidates[0]

    def residual(parameters: np.ndarray) -> np.ndarray:
        offset = parameters[:3]
        gain = np.exp(parameters[3:9])
        bias = parameters[9:15]
        rows = []
        for sim, target in pairs:
            estimate = _apply_model(sim, offset, gain, bias, alpha, delay)
            weight = np.sqrt(_contact_weights(sim))[:, None]
            rows.append(((estimate - target) * weight).reshape(-1))
        return np.concatenate(rows)

    lower = np.concatenate(([-0.10] * 3, [-7.0] * 6, [-20.0] * 6))
    upper = np.concatenate(([0.10] * 3, [7.0] * 6, [20.0] * 6))
    initial_unbounded = np.concatenate((np.zeros(3), np.log(start_gain), start_bias))
    # A component whose unconstrained slope is negative is deliberately
    # replaced by a tiny positive gain in _positive_diagonal_fit.  Its log can
    # then fall below the nonlinear optimizer's positive-gain bound.  SciPy
    # requires x0 to be strictly feasible, so project every initial parameter
    # just inside the declared bounds instead of aborting before optimization.
    bound_margin = 1.0e-8
    initial = np.clip(
        initial_unbounded,
        lower + bound_margin,
        upper - bound_margin,
    )
    clipped_initial_indices = np.flatnonzero(
        np.abs(initial - initial_unbounded) > 0.0
    ).astype(int)
    if len(clipped_initial_indices):
        print(
            "[FIT] projected bounded initial parameters at indices "
            f"{clipped_initial_indices.tolist()}",
            flush=True,
        )
    solution = least_squares(
        residual, initial, bounds=(lower, upper), loss="soft_l1", f_scale=1.0
    )
    offset = solution.x[:3]
    gain = np.exp(solution.x[3:9])
    bias = solution.x[9:15]
    train_target = np.concatenate([target for _sim, target in pairs])
    train_estimate = np.concatenate(
        [_apply_model(sim, offset, gain, bias, alpha, delay) for sim, _target in pairs]
    )
    clip_lower = np.quantile(train_target, 0.005, axis=0)
    clip_upper = np.quantile(train_target, 0.995, axis=0)
    result = {
        "version": "link7_success_wrench_calibration_v3",
        "coordinate_contract": "base_frame_force__K_reference_torque",
        "dataset_semantics": {
            "all_40_real_trajectories_are_successful_insertions": True,
            "negative_or_failed_real_examples_available": False,
            "fit_target": "successful insertion wrench trend until first simulation success",
            "not_valid_for": "learning a successful-vs-failed insertion classifier",
        },
        "incoming_wrench_body": "panda_link7",
        "physical_source": (
            "directed-load transformed panda_link7 incoming joint wrench in base/K; "
            "same-target no-contact continuous replay subtracted"
        ),
        "sim_source_dataset": SIM_DATASET,
        "fitted_trajectories": list(TRAIN_IDS),
        "holdout_trajectories": list(HOLDOUT_IDS),
        "training_first_success_frames": success_frames,
        "post_success_samples_used_for_fit": False,
        "no_contact_seconds": float(args.no_contact_seconds),
        "baseline_model_type": baseline_model_type,
        "baseline_model_selection": baseline_cv,
        "baseline_model_features": ["1", *[f"q{i}" for i in range(7)], *[f"qd{i}" for i in range(7)]],
        "baseline_model_weights": baseline.tolist(),
        "axis_permutation": [0, 1, 2, 3, 4, 5],
        "axis_sign": [1.0] * 6,
        "axis_gain": gain.tolist(),
        "constant_bias": bias.tolist(),
        "torque_reference_offset_base_m": offset.tolist(),
        "causal_ema_alpha": float(alpha),
        "causal_delay_frames": int(delay),
        "clip_lower": clip_lower.tolist(),
        "clip_upper": clip_upper.tolist(),
        "clip_quantiles": [0.005, 0.995],
        "direction_test": {
            "path": str(direction_path),
            "sha256": hashlib.sha256(direction_path.read_bytes()).hexdigest(),
            "summary": direction_result.get("summary", {}),
        },
        "candidate_selection_score": float(score),
        "bounded_initialization": {
            "projected": bool(len(clipped_initial_indices)),
            "projected_parameter_indices": clipped_initial_indices.tolist(),
            "initial_axis_gain": start_gain.tolist(),
            "initial_constant_bias": start_bias.tolist(),
        },
        "optimizer_success": bool(solution.success),
        "optimizer_message": str(solution.message),
        "training_metrics": _metrics(train_target, train_estimate),
    }
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
