#!/usr/bin/env python3
"""Subtract a same-trajectory no-contact baseline from a rigid-tool replay."""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path
import numpy as np

def read_csv(path: Path) -> np.ndarray:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.reader(handle))
    if len(rows) < 2:
        raise ValueError(f"{path}: expected a header and at least one frame")
    values = np.asarray([[float(value) for value in row] for row in rows[1:]], dtype=np.float32)
    if values.ndim != 2 or values.shape[1] != 6:
        raise ValueError(f"{path}: expected six wrench columns, got {values.shape}")
    return values

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contact-dir", type=Path, required=True)
    parser.add_argument("--no-contact-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    contact_dir = args.contact_dir.resolve()
    no_contact_dir = args.no_contact_dir.resolve()
    output_dir = args.output_dir.resolve()
    contact = read_csv(contact_dir / "wrench_raw.csv")
    baseline = read_csv(no_contact_dir / "wrench_raw.csv")
    if contact.shape != baseline.shape:
        raise ValueError(f"contact/no-contact shapes differ: {contact.shape} vs {baseline.shape}")
    contact_time = np.loadtxt(contact_dir / "timestamps.csv", delimiter=",", skiprows=1, ndmin=1)
    baseline_time = np.loadtxt(no_contact_dir / "timestamps.csv", delimiter=",", skiprows=1, ndmin=1)
    if contact_time.shape != baseline_time.shape or not np.allclose(contact_time, baseline_time, atol=1.0e-5):
        raise ValueError("contact and no-contact timestamps are not aligned; do not compensate them")
    compensated = contact - baseline
    if not np.isfinite(compensated).all():
        raise ValueError("compensated wrench contains NaN or Inf")
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "wrench_residual.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["Fx", "Fy", "Fz", "Tx", "Ty", "Tz"])
        writer.writerows(compensated.tolist())
    np.savetxt(output_dir / "timestamps.csv", contact_time.reshape(-1, 1), delimiter=",", header="timestamp", comments="")
    metadata = {
        "definition": "wrench_raw(contact) - wrench_raw(no_contact)",
        "input_signal": "wrench_raw.csv",
        "input_semantics": "raw PhysX incoming joint wrench in the configured incoming frame/reference",
        "contact_dir": str(contact_dir),
        "no_contact_dir": str(no_contact_dir),
        "frame_count": int(len(compensated)),
        "component_order": "[Fx,Fy,Fz,Tx,Ty,Tz]",
        "units": "[N,N,N,N*m,N*m]",
    }
    (output_dir / "no_contact_compensation.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"[DONE] wrote {len(compensated)} residual frames to {output_dir / 'wrench_residual.csv'}")

if __name__ == "__main__":
    main()
