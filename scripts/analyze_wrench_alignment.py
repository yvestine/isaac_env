#!/usr/bin/env python3
"""Analyze raw/no-contact/residual wrench semantics against real O_F_ext_hat_K."""
from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

FIELDS = ("Fx", "Fy", "Fz", "Tx", "Ty", "Tz")
EMA_ALPHAS = (0.05, 0.10, 0.20, 0.25, 0.30, 0.50)


def read_six_csv(path: Path) -> np.ndarray:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.reader(handle))
    if len(rows) < 2:
        raise ValueError(f"{path}: expected a header and data")
    values = np.asarray([[float(value) for value in row] for row in rows[1:]], dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 6:
        raise ValueError(f"{path}: expected six columns, got {values.shape}")
    if not np.isfinite(values).all():
        raise ValueError(f"{path}: contains NaN or Inf")
    return values


def read_time_csv(path: Path) -> np.ndarray:
    values = np.loadtxt(path, delimiter=",", skiprows=1, ndmin=1).astype(np.float64)
    values = values.reshape(-1)
    if len(values) < 2 or np.any(np.diff(values) <= 0.0):
        raise ValueError(f"{path}: timestamps must be strictly increasing")
    return values


def read_real(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with h5py.File(path, "r") as data:
        for key in ("obs/state/ee_wrench_base", "observations/ee_wrench_base", "state/ee_wrench_base"):
            if key in data:
                wrench = np.asarray(data[key], dtype=np.float64).reshape(-1, 6)
                break
        else:
            raise KeyError(f"{path}: ee_wrench_base not found")
        timestamps = np.asarray(data["timestamps"], dtype=np.float64).reshape(-1)
    if len(wrench) != len(timestamps):
        raise ValueError("real wrench/timestamp lengths differ")
    return timestamps, wrench


def write_six_csv(path: Path, values: np.ndarray) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(FIELDS)
        writer.writerows(np.asarray(values, dtype=np.float64).reshape(-1, 6).tolist())


def correlation(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 2 or np.std(a) < 1.0e-12 or np.std(b) < 1.0e-12:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def ema(values: np.ndarray, alpha: float | None) -> np.ndarray:
    if alpha is None:
        return values.copy()
    if not 0.0 < alpha <= 1.0:
        raise ValueError("EMA alpha must be in (0, 1]")
    result = np.empty_like(values)
    result[0] = values[0]
    for index in range(1, len(values)):
        result[index] = alpha * values[index] + (1.0 - alpha) * result[index - 1]
    return result


def resample(timestamps: np.ndarray, values: np.ndarray, grid: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    return np.column_stack([np.interp(grid, timestamps, values[:, axis]) for axis in range(values.shape[1])])


def common_grid(real_t: np.ndarray, sim_t: np.ndarray, sample_hz: float) -> np.ndarray:
    start = max(float(real_t[0]), float(sim_t[0]))
    stop = min(float(real_t[-1]), float(sim_t[-1]))
    step = 1.0 / float(sample_hz)
    if stop <= start:
        raise ValueError("real/sim timestamp ranges do not overlap")
    grid = np.arange(start, stop + 0.25 * step, step, dtype=np.float64)
    if len(grid) < 2:
        raise ValueError("common resampled grid has fewer than two frames")
    return grid


def metric_values(real: np.ndarray, sim: np.ndarray) -> dict[str, float]:
    real_force_norm = np.linalg.norm(real[:, :3], axis=1)
    sim_force_norm = np.linalg.norm(sim[:, :3], axis=1)
    real_torque_norm = np.linalg.norm(real[:, 3:], axis=1)
    sim_torque_norm = np.linalg.norm(sim[:, 3:], axis=1)
    real_delta = np.diff(real[:, :3], axis=0)
    sim_delta = np.diff(sim[:, :3], axis=0)
    values = {f"{field}_corr": correlation(real[:, index], sim[:, index]) for index, field in enumerate(FIELDS)}
    values.update(
        {
            "F_norm_corr": correlation(real_force_norm, sim_force_norm),
            "T_norm_corr": correlation(real_torque_norm, sim_torque_norm),
            "dFx_corr": correlation(real_delta[:, 0], sim_delta[:, 0]),
            "dFy_corr": correlation(real_delta[:, 1], sim_delta[:, 1]),
            "dFz_corr": correlation(real_delta[:, 2], sim_delta[:, 2]),
            "dF_norm_corr": correlation(
                np.diff(real_force_norm), np.diff(sim_force_norm)
            ),
        }
    )
    return values


def score(values: dict[str, float], trend: bool = False) -> float:
    keys = ("dFx_corr", "dFy_corr", "dFz_corr", "dF_norm_corr") if trend else (
        "Fx_corr", "Fy_corr", "Fz_corr", "F_norm_corr"
    )
    finite = [values[key] for key in keys if np.isfinite(values[key])]
    return float(np.mean(finite)) if finite else float("nan")


def lagged_metrics(real: np.ndarray, sim: np.ndarray, max_lag_frames: int) -> tuple[dict[str, float], int, dict[str, float]]:
    def aligned(lag: int) -> tuple[np.ndarray, np.ndarray]:
        if lag > 0:
            return real[:-lag], sim[lag:]
        if lag < 0:
            return real[-lag:], sim[:lag]
        return real, sim

    zero_real, zero_sim = aligned(0)
    zero = metric_values(zero_real, zero_sim)
    best_lag = 0
    best_score = -np.inf
    best = zero
    for lag in range(-max_lag_frames, max_lag_frames + 1):
        aligned_real, aligned_sim = aligned(lag)
        current = metric_values(aligned_real, aligned_sim)
        current_score = score(current, trend=False)
        if np.isfinite(current_score) and current_score > best_score:
            best_score = current_score
            best_lag = lag
            best = current
    return zero, best_lag, best


def row_for_candidate(
    method: str,
    sign: int,
    filter_name: str,
    alpha: float | None,
    sample_rate: float,
    real_t: np.ndarray,
    real: np.ndarray,
    sim_t: np.ndarray,
    sim: np.ndarray,
    max_lag_frames: int,
) -> dict[str, object]:
    grid = common_grid(real_t, sim_t, 10.0)
    real_10 = resample(real_t, real, grid)
    sim_10 = resample(sim_t, sim, grid) * float(sign)
    sim_10 = ema(sim_10, alpha)
    zero, best_lag, best = lagged_metrics(real_10, sim_10, max_lag_frames)
    row: dict[str, object] = {
        "method": method,
        "sign": "+1" if sign > 0 else "-1",
        "filter": filter_name,
        "alpha": "" if alpha is None else f"{alpha:.2f}",
        "sample_rate": f"{sample_rate:.6f}",
        "zero_lag_seconds": "0.000000",
        "lag_seconds": f"{best_lag / 10.0:.6f}",
        "best_lag_seconds": f"{best_lag / 10.0:.6f}",
        "zero_lag_score": f"{score(zero):.6f}",
        "best_lag_score": f"{score(best):.6f}",
        "zero_lag_trend_score": f"{score(zero, trend=True):.6f}",
        "best_lag_trend_score": f"{score(best, trend=True):.6f}",
    }
    for key in zero:
        row[f"{key}"] = f"{zero[key]:.6f}" if np.isfinite(zero[key]) else "nan"
        row[f"best_{key}"] = f"{best[key]:.6f}" if np.isfinite(best[key]) else "nan"
    return row


def row_float(row: dict[str, object], key: str) -> float:
    try:
        return float(row[key])
    except (TypeError, ValueError):
        return float("nan")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--real-h5", type=Path, required=True)
    parser.add_argument("--contact-dir", type=Path, required=True)
    parser.add_argument("--no-contact-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--sample-hz", type=float, default=10.0)
    parser.add_argument("--max-lag-seconds", type=float, default=1.0)
    args = parser.parse_args()

    real_t, real = read_real(args.real_h5)
    contact_t = read_time_csv(args.contact_dir / "timestamps.csv")
    no_contact_t = read_time_csv(args.no_contact_dir / "timestamps.csv")
    contact_raw = read_six_csv(args.contact_dir / "wrench_raw.csv")
    no_contact_raw = read_six_csv(args.no_contact_dir / "wrench_raw.csv")
    contact_base = read_six_csv(args.contact_dir / "wrench_base.csv")
    no_contact_base = read_six_csv(args.no_contact_dir / "wrench_base.csv")
    if not (len(contact_t) == len(contact_raw) == len(contact_base)):
        raise ValueError("contact signal/timestamp lengths differ")
    if not (len(no_contact_t) == len(no_contact_raw) == len(no_contact_base)):
        raise ValueError("no-contact signal/timestamp lengths differ")
    if not np.allclose(contact_t, no_contact_t, atol=1.0e-5):
        raise ValueError("contact/no-contact timestamps are not aligned")
    if len(real_t) != len(real) or len(real_t) < 2:
        raise ValueError("real data is empty or misaligned")

    residual_raw = contact_raw - no_contact_raw
    residual_base = contact_base - no_contact_base
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    write_six_csv(output / "wrench_raw.csv", contact_raw)
    write_six_csv(output / "wrench_no_contact.csv", no_contact_raw)
    write_six_csv(output / "wrench_residual.csv", residual_raw)
    real_rate = 1.0 / float(np.median(np.diff(real_t)))
    sim_rate = 1.0 / float(np.median(np.diff(contact_t)))
    max_lag_frames = max(1, int(round(args.max_lag_seconds * args.sample_hz)))
    candidates: list[dict[str, object]] = []
    candidates.append(row_for_candidate("raw incoming", 1, "none", None, sim_rate, real_t, real, contact_t, contact_raw, max_lag_frames))
    candidates.append(row_for_candidate("raw + sign", -1, "none", None, sim_rate, real_t, real, contact_t, contact_raw, max_lag_frames))
    candidates.append(row_for_candidate("residual", 1, "none", None, sim_rate, real_t, real, contact_t, residual_raw, max_lag_frames))
    candidates.append(row_for_candidate("residual + base/K transform", 1, "none", None, sim_rate, real_t, real, contact_t, residual_base, max_lag_frames))
    candidates.append(row_for_candidate("residual + base/K transform", -1, "none", None, sim_rate, real_t, real, contact_t, residual_base, max_lag_frames))
    for alpha in (None, *EMA_ALPHAS):
        for sign in (1, -1):
            candidates.append(
                row_for_candidate(
                    "residual + base/K + 10Hz",
                    sign,
                    "none" if alpha is None else "ema",
                    alpha,
                    args.sample_hz,
                    real_t,
                    real,
                    contact_t,
                    residual_base,
                    max_lag_frames,
                )
            )

    fieldnames = list(candidates[0].keys())
    with (output / "wrench_alignment_report.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(candidates)

    # Select sign at zero lag for the unfiltered semantic output, then select
    # the best filtered/lagged variant separately for deployment comparison.
    semantic_rows = [row for row in candidates if row["method"] == "residual + base/K transform" and row["filter"] == "none"]
    semantic_sign_row = max(semantic_rows, key=lambda row: row_float(row, "zero_lag_score"))
    semantic_sign = 1 if semantic_sign_row["sign"] == "+1" else -1
    write_six_csv(output / "wrench_o_f_ext_like_raw.csv", residual_base * semantic_sign)
    processed_rows = [row for row in candidates if row["method"] == "residual + base/K + 10Hz"]
    best_processed_row = max(processed_rows, key=lambda row: row_float(row, "best_lag_score"))
    best_overall_row = max(candidates, key=lambda row: row_float(row, "best_lag_score"))
    best_sign = 1 if best_processed_row["sign"] == "+1" else -1
    best_alpha = None if best_processed_row["alpha"] == "" else float(best_processed_row["alpha"])
    grid = common_grid(real_t, contact_t, args.sample_hz)
    semantic_10 = ema(resample(contact_t, residual_base, grid) * semantic_sign, None)
    best_10 = ema(resample(contact_t, residual_base, grid) * best_sign, best_alpha)
    np.savetxt(output / "timestamps_10hz.csv", grid.reshape(-1, 1), delimiter=",", header="timestamp", comments="")
    write_six_csv(output / "wrench_o_f_ext_like_10hz.csv", semantic_10)

    contact_meta = {}
    for directory in (args.contact_dir, args.no_contact_dir):
        config_path = directory / "replay_config.json"
        if config_path.exists():
            contact_meta[directory.name] = json.loads(config_path.read_text(encoding="utf-8"))
    semantics = {
        "raw_source": "robot.root_physx_view.get_link_incoming_joint_force()",
        "robot_body_names": contact_meta.get(args.contact_dir.name, {}).get("robot_body_names", []),
        "force_sensor_body_idx": contact_meta.get(args.contact_dir.name, {}).get("force_sensor_body_idx"),
        "incoming_wrench_body_idx": contact_meta.get(args.contact_dir.name, {}).get("incoming_wrench_body_idx"),
        "incoming_wrench_frame": contact_meta.get(args.contact_dir.name, {}).get("incoming_wrench_frame", "unknown"),
        "incoming_wrench_torque_reference": contact_meta.get(args.contact_dir.name, {}).get("incoming_wrench_torque_reference", "unknown"),
        "raw_order": list(FIELDS),
        "target_semantics": "robot_base frame, torque about stiffness frame K",
        "transform": "F_base=R_base_world F_world; T_K=R_base_world(T_world+(p_raw-p_K)xF_world)",
        "transform_source": "tacex_tasks.forge.forge_utils.change_FT_frame / ForgeEnv._transform_raw_wrench_to_base",
        "sign_test": "both +1 and -1 evaluated; no hard-coded sign assumed",
        "sample_rate_real_hz": real_rate,
        "sample_rate_sim_hz": sim_rate,
        "contact_config": contact_meta.get(args.contact_dir.name, {}),
        "no_contact_config": contact_meta.get(args.no_contact_dir.name, {}),
    }
    (output / "wrench_semantics.json").write_text(json.dumps(semantics, indent=2), encoding="utf-8")

    with (output / "wrench_alignment_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "best_pipeline": best_overall_row,
                "best_processed_semantic_pipeline": best_processed_row,
                "semantic_unfiltered_sign": semantic_sign_row,
                "real_sample_hz": real_rate,
                "sim_sample_hz": sim_rate,
                "residual_definition": "wrench_raw(contact) - wrench_raw(no_contact)",
                "transformed_definition": "residual base/K = wrench_base(contact) - wrench_base(no_contact)",
                "note": "base/K subtraction is equivalent to transforming the raw residual when the two runs share the same frame geometry; replay metadata records the frame contract.",
            },
            handle,
            indent=2,
        )

    # Compact visualization: real vs raw, residual and best processed force norm;
    # Fz is shown separately because insertion onset is usually axial.
    real_10 = resample(real_t, real, grid)
    raw_10 = resample(contact_t, contact_raw, grid)
    residual_10 = resample(contact_t, residual_raw, grid)
    best_for_plot = best_10
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    axes[0].plot(grid, np.linalg.norm(real_10[:, :3], axis=1), label="real |F|", linewidth=1.5)
    axes[0].plot(grid, np.linalg.norm(raw_10[:, :3], axis=1), label="raw incoming |F|", linewidth=1.0)
    axes[0].plot(grid, np.linalg.norm(residual_10[:, :3], axis=1), label="raw residual |F|", linewidth=1.0)
    axes[0].plot(grid, np.linalg.norm(best_for_plot[:, :3], axis=1), label="best processed |F|", linewidth=1.2)
    axes[0].set_ylabel("force norm (N)")
    axes[0].grid(alpha=0.25)
    axes[0].legend()
    axes[1].plot(grid, real_10[:, 2], label="real Fz", linewidth=1.5)
    axes[1].plot(grid, raw_10[:, 2], label="raw Fz", linewidth=1.0)
    axes[1].plot(grid, residual_10[:, 2], label="residual Fz", linewidth=1.0)
    axes[1].plot(grid, best_for_plot[:, 2], label="best processed Fz", linewidth=1.2)
    axes[1].set_xlabel("timestamp (s)")
    axes[1].set_ylabel("Fz (N)")
    axes[1].grid(alpha=0.25)
    axes[1].legend()
    fig.suptitle("traj0 wrench alignment: raw, residual, base/K and filtered/lagged")
    fig.tight_layout()
    fig.savefig(output / "wrench_alignment.png", dpi=160)
    plt.close(fig)

    print(f"[DONE] final alignment files: {output}")
    print(f"[SEMANTICS] body_names={semantics['robot_body_names']}")
    print(f"[SEMANTICS] force_sensor_body_idx={semantics['force_sensor_body_idx']}")
    print(f"[SEMANTICS] incoming frame={semantics['incoming_wrench_frame']}, torque reference={semantics['incoming_wrench_torque_reference']}")
    print(f"[BEST overall] method={best_overall_row['method']} sign={best_overall_row['sign']} filter={best_overall_row['filter']} alpha={best_overall_row['alpha']} best_lag={best_overall_row['best_lag_seconds']}s score={best_overall_row['best_lag_score']}")
    print(f"[BEST overall] Fz={best_overall_row['best_Fz_corr']} |F|={best_overall_row['best_F_norm_corr']} d|F|={best_overall_row['best_dF_norm_corr']}")
    print(f"[BEST semantic] method={best_processed_row['method']} sign={best_processed_row['sign']} filter={best_processed_row['filter']} alpha={best_processed_row['alpha']} best_lag={best_processed_row['best_lag_seconds']}s score={best_processed_row['best_lag_score']}")
    print(f"[BEST semantic] Fz={best_processed_row['best_Fz_corr']} |F|={best_processed_row['best_F_norm_corr']} d|F|={best_processed_row['best_dF_norm_corr']}")


if __name__ == "__main__":
    main()
