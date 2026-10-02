#!/usr/bin/env python3
"""Summarize the 2-port x 2-action-start x 2-replan TAVLA evaluation grid."""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path


EXPERIMENT_RE = re.compile(r"^port_(8000|8001)_start_(1|5)_steps_(5|10)$")
MODEL_NAMES = {
    8000: "real/sim 50:50 co-train affine force-trend",
    8001: "sim affine force-trend",
}
Z_THRESHOLDS_MM = (3, 5, 8, 10)


def as_float(row: dict[str, str], key: str) -> float | None:
    value = row.get(key, "").strip()
    if not value:
        return None
    try:
        return float(value)
    except ValueError:
        return None


def read_episode(path: Path) -> dict[str, str]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 1:
        raise ValueError(f"expected one episode in {path}, found {len(rows)}")
    return rows[0]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path, nargs="?", default=Path("outputs/tavla_8exp"))
    parser.add_argument("--expected-profiles", type=int, default=40)
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()

    root = args.root.expanduser().resolve()
    summaries = []
    missing = []
    for experiment in sorted(path for path in root.glob("port_*_start_*_steps_*") if path.is_dir()):
        match = EXPERIMENT_RE.match(experiment.name)
        if match is None:
            continue
        port, action_start, replan = map(int, match.groups())
        episode_files = sorted(experiment.glob("profile_*/episodes.csv"))
        rows = [read_episode(path) for path in episode_files]
        conditional_z: list[float] = []
        exact_conditional_count = 0
        xy_count = 0
        for row in rows:
            xy = as_float(row, "best_xy_error_m")
            if xy is None or xy >= 0.003:
                continue
            xy_count += 1
            z = as_float(row, "minimum_z_disp_when_xy_lt_3mm_m")
            if z is not None:
                exact_conditional_count += 1
            else:
                z = as_float(row, "minimum_z_disp_m")
            if z is not None:
                conditional_z.append(z)
        summary = {
            "experiment": experiment.name,
            "port": port,
            "model": MODEL_NAMES[port],
            "action_start_index": action_start,
            "replan_actions": replan,
            "profiles": len(rows),
            "successes": sum(int(row.get("success", "0") or 0) for row in rows),
            "xy_lt_3mm": xy_count,
            "conditional_z_exact_rows": exact_conditional_count,
            "tavla_request_failures": sum(
                int(row.get("tavla_failures_peak", "0") or 0) for row in rows
            ),
            "tavla_timeouts": sum(
                int(row.get("tavla_timeouts_peak", "0") or 0) for row in rows
            ),
        }
        for threshold_mm in Z_THRESHOLDS_MM:
            summary[f"xy_lt_3mm_and_z_lt_{threshold_mm}mm"] = sum(
                z < threshold_mm / 1000.0 for z in conditional_z
            )
        summaries.append(summary)
        if len(rows) != args.expected_profiles:
            missing.append(f"{experiment.name}: {len(rows)}/{args.expected_profiles}")

    expected_experiments = 8
    if len(summaries) != expected_experiments:
        missing.append(f"experiments: {len(summaries)}/{expected_experiments}")

    output_json = root / "metrics_summary.json"
    output_csv = root / "metrics_summary.csv"
    root.mkdir(parents=True, exist_ok=True)
    output_json.write_text(
        json.dumps(
            {
                "models": {str(port): name for port, name in MODEL_NAMES.items()},
                "metric_contract": (
                    "Z is the minimum z displacement observed at a step where XY < 3 mm. "
                    "Rows created before this field existed use the independent minimum-Z fallback."
                ),
                "experiments": summaries,
                "incomplete": missing,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    if summaries:
        with output_csv.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(summaries[0]))
            writer.writeheader()
            writer.writerows(summaries)

    columns = (
        "experiment",
        "profiles",
        "successes",
        "xy_lt_3mm",
        "xy_lt_3mm_and_z_lt_3mm",
        "xy_lt_3mm_and_z_lt_5mm",
        "xy_lt_3mm_and_z_lt_8mm",
        "xy_lt_3mm_and_z_lt_10mm",
    )
    print("\t".join(columns))
    for summary in summaries:
        print("\t".join(str(summary[column]) for column in columns))
    if missing:
        print("[summary] incomplete: " + "; ".join(missing))
    print(f"[summary] wrote {output_csv} and {output_json}")
    return 1 if args.strict and missing else 0


if __name__ == "__main__":
    raise SystemExit(main())
