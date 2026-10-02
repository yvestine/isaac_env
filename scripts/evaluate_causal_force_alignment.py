#!/usr/bin/env python3
"""Evaluate the causal sim/real force contract without fitting on the holdout."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np


PHASES = (
    ("above_10mm", lambda z: z > 0.010),
    ("10_to_6mm", lambda z: (z <= 0.010) & (z > 0.006)),
    ("6_to_3mm", lambda z: (z <= 0.006) & (z > 0.003)),
    ("at_or_below_3mm", lambda z: z <= 0.003),
)


def correlation(first: np.ndarray, second: np.ndarray) -> float:
    if len(first) < 3 or np.std(first) < 1.0e-12 or np.std(second) < 1.0e-12:
        return float("nan")
    return float(np.corrcoef(first, second)[0, 1])


def lagged_correlation(
    real: np.ndarray,
    sim: np.ndarray,
    mask: np.ndarray,
    lag: int,
    component: str,
) -> float:
    indexes = np.flatnonzero(mask)
    real_indexes = indexes + lag
    valid = (real_indexes >= 0) & (real_indexes < len(real))
    real_indexes = real_indexes[valid]
    sim_indexes = indexes[valid]
    if component == "Fz":
        return correlation(real[real_indexes, 2], sim[sim_indexes, 2])
    return correlation(
        np.linalg.norm(real[real_indexes], axis=1),
        np.linalg.norm(sim[sim_indexes], axis=1),
    )


def read_force(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with h5py.File(path, "r") as data:
        if str(data.attrs.get("force_alignment", "")) != "causal_force_alignment_v1":
            raise RuntimeError(f"{path}: not a causal_force_alignment_v1 dataset")
        wrench = np.asarray(data["obs/state/ee_wrench_base"][:], dtype=np.float64)
        time = np.asarray(data["timestamps"][:], dtype=np.float64)
        if not np.allclose(wrench[:, 3:], 0.0, atol=0.0, rtol=0.0):
            raise RuntimeError(f"{path}: torque compatibility channels are not exactly zero")
    return wrench[:, :3], time - time[0]


def ar1(values: np.ndarray) -> np.ndarray:
    if len(values) < 3:
        return np.full(3, np.nan)
    centered = values - np.median(values, axis=0, keepdims=True)
    previous = centered[:-1]
    current = centered[1:]
    denominator = np.sum(previous * previous, axis=0)
    return np.divide(
        np.sum(previous * current, axis=0),
        denominator,
        out=np.zeros(3, dtype=np.float64),
        where=denominator > 1.0e-12,
    )


def evaluate_split(
    ids: tuple[int, ...],
    real_root: Path,
    sim_root: Path,
    geometry_root: Path,
    free_space: dict,
) -> dict:
    correlations = {"Fz": [], "force_norm": []}
    lag_metrics = {lag: {"Fz": [], "force_norm": []} for lag in range(-2, 3)}
    phase_ratios = {name: [] for name, _selector in PHASES}
    distribution_errors = {"rms": [], "std": [], "ar1": []}
    for trajectory in ids:
        real, real_time = read_force(real_root / f"traj_{trajectory}" / "data.h5")
        sim, sim_time = read_force(sim_root / f"traj_{trajectory}" / "data.h5")
        geometry = np.genfromtxt(
            geometry_root / f"traj_{trajectory}" / "gt_replay_validation.csv",
            delimiter=",",
            names=True,
        )
        count = min(len(real), len(sim), len(geometry))
        real = real[:count]
        sim = sim[:count]
        geometry = geometry[:count]
        if not np.allclose(real_time[:count], sim_time[:count], atol=1.0e-5, rtol=1.0e-5):
            raise RuntimeError(f"traj_{trajectory}: real/sim frame timestamps differ")
        insertion = geometry["z_disp_m"] <= 0.010
        correlations["Fz"].append(correlation(real[insertion, 2], sim[insertion, 2]))
        correlations["force_norm"].append(
            correlation(
                np.linalg.norm(real[insertion], axis=1),
                np.linalg.norm(sim[insertion], axis=1),
            )
        )
        for lag in lag_metrics:
            for component in lag_metrics[lag]:
                lag_metrics[lag][component].append(
                    lagged_correlation(real, sim, insertion, lag, component)
                )
        for name, selector in PHASES:
            mask = selector(geometry["z_disp_m"])
            if np.any(mask):
                real_level = float(np.median(np.linalg.norm(real[mask], axis=1)))
                sim_level = float(np.median(np.linalg.norm(sim[mask], axis=1)))
                phase_ratios[name].append(real_level / max(sim_level, 1.0e-9))
        start, stop = free_space[f"traj_{trajectory}"]["free_space_frames"][0]
        real_free = real[start:stop]
        sim_free = sim[start:stop]
        real_rms = float(np.sqrt(np.mean(np.sum(real_free**2, axis=1))))
        sim_rms = float(np.sqrt(np.mean(np.sum(sim_free**2, axis=1))))
        distribution_errors["rms"].append(abs(sim_rms / max(real_rms, 1.0e-9) - 1.0))
        real_std = np.std(real_free, axis=0)
        sim_std = np.std(sim_free, axis=0)
        distribution_errors["std"].extend(
            np.abs(sim_std / np.maximum(real_std, 1.0e-9) - 1.0).tolist()
        )
        distribution_errors["ar1"].extend(np.abs(ar1(sim_free) - ar1(real_free)).tolist())

    median_correlations = {
        name: float(np.nanmedian(values)) for name, values in correlations.items()
    }
    median_lags = {
        str(lag): {
            name: float(np.nanmedian(values)) for name, values in components.items()
        }
        for lag, components in lag_metrics.items()
    }
    best_lag = max(
        lag_metrics,
        key=lambda lag: np.nanmean(
            [median_lags[str(lag)]["Fz"], median_lags[str(lag)]["force_norm"]]
        ),
    )
    ratios = {
        name: float(np.nanmedian(values)) if values else float("nan")
        for name, values in phase_ratios.items()
    }
    distribution = {
        name: float(np.nanmedian(values)) for name, values in distribution_errors.items()
    }
    checks = {
        "Fz_correlation_at_least_0.75": median_correlations["Fz"] >= 0.75,
        "force_norm_correlation_at_least_0.70": median_correlations["force_norm"] >= 0.70,
        "phase_amplitude_ratios_0.7_to_1.4": all(
            np.isfinite(value) and 0.7 <= value <= 1.4 for value in ratios.values()
        ),
        "best_absolute_lag_at_most_one_frame": abs(best_lag) <= 1,
        "free_space_rms_error_at_most_20_percent": distribution["rms"] <= 0.20,
        "free_space_std_error_at_most_20_percent": distribution["std"] <= 0.20,
        "free_space_ar1_absolute_error_at_most_0.20": distribution["ar1"] <= 0.20,
    }
    return {
        "trajectories": list(ids),
        "insertion_correlations": median_correlations,
        "lag_correlations": median_lags,
        "best_lag_frames": int(best_lag),
        "phase_real_over_sim_force_norm": ratios,
        "free_space_distribution_error": distribution,
        "checks": checks,
        "passed": all(checks.values()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--real-root", type=Path, default=Path("real-data-aligned"))
    parser.add_argument("--sim-root", type=Path, default=Path("sim-data-aligned"))
    parser.add_argument("--geometry-root", type=Path, default=Path("sim-data"))
    parser.add_argument(
        "--free-space-intervals",
        type=Path,
        default=Path("real_data/free_space_intervals.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/force_alignment/evaluation.json"),
    )
    args = parser.parse_args()
    free_space = json.loads(args.free_space_intervals.read_text(encoding="utf-8"))[
        "trajectories"
    ]
    result = {
        "success_contract": "z_disp_m < 0.003 and xy_error_m < 0.003",
        "fit_trajectories": list(range(1, 32)),
        "validation": evaluate_split(
            tuple(range(32, 40)),
            args.real_root,
            args.sim_root,
            args.geometry_root,
            free_space,
        ),
        "holdout": evaluate_split(
            (0,),
            args.real_root,
            args.sim_root,
            args.geometry_root,
            free_space,
        ),
    }
    result["passed"] = result["validation"]["passed"] and result["holdout"]["passed"]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
