#!/usr/bin/env python3
"""Build real-data-aligned/: put the REAL force on the SAME definition as SIM.

The real ee_wrench_base is the raw external-wrench estimate: in free space it still
carries a payload/dynamics baseline (~3.3N) plus sensor noise (~1N), so it looks dirtier
than the sim signal, which is contact-isolated (free space == 0). We remove that baseline
using ONLY the labelled no-contact frames from free_space_intervals.json, so real also
becomes ~contact-isolated. Deployable: at inference the same fit runs on the episode's
initial pre-contact frames (a motion-aware sensor tare) and then applies forward.

MODE = "motion": fit wrench ~ [q, qdot, sin q, cos q, 1] on free-space frames (ridge),
                 subtract the prediction over the whole trajectory (~1.1N floor).
MODE = "tare":   subtract the free-space median (constant per episode, ~1.5N floor).

Torque (Tx,Ty,Tz) is zeroed: sim contact torque is ~0 on every channel (~0.1 vs real
~2 N.m), so the 6-axis effort is aligned as force-only, torque==0 on both real and sim.

Everything else is copied verbatim; videos are symlinked (no duplication). Output mirrors
real_data/ so the SAME convert_user_hdf5_to_lerobot.py works on --raw-dir real-data-aligned.
"""
from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np

ROOT = Path("/home/sutai/workspace/gujiawei/isaac_env")
REAL = ROOT / "real_data"
OUT = ROOT / "real-data-aligned"
FS = json.loads((REAL / "free_space_intervals.json").read_text())["trajectories"]

MODE = "motion"   # "motion" (pose/vel-aware) or "tare" (constant median)
RIDGE = 1.0
BASE_WRENCH = "obs/state/ee_wrench_base"
BASE_FORCE = "obs/state/ee_force_base"
BASE_TORQUE = "obs/state/ee_torque_base"


def features(q, qd):
    return np.concatenate([q, qd, np.sin(q), np.cos(q), np.ones((len(q), 1))], axis=1)


def fit_baseline(wrench, q, qd, fs_slice):
    a, b = fs_slice
    if MODE == "tare":
        base = np.median(wrench[a:b], axis=0)
        return np.broadcast_to(base, wrench.shape).copy()
    X = features(q, qd)
    Xf = X[a:b]
    W = np.linalg.solve(Xf.T @ Xf + RIDGE * np.eye(X.shape[1]), Xf.T @ wrench[a:b])
    return X @ W


def build_one(trajectory: int, out_dir: Path) -> dict:
    src = REAL / f"traj_{trajectory}"
    a, b = FS[f"traj_{trajectory}"]["free_space_frames"][0]
    out_dir.mkdir(parents=True, exist_ok=True)
    dst = out_dir / "data.h5"
    with h5py.File(src / "data.h5", "r") as fs, h5py.File(dst, "w") as fd:
        for name in fs.attrs:
            fd.attrs[name] = fs.attrs[name]
        for key in fs.keys():
            fs.copy(fs[key], fd, name=key)
        wrench = fd[BASE_WRENCH][:].astype(np.float64)
        q = fd["obs/state/joint_pos"][:].astype(np.float64)
        qd = fd["obs/state/joint_vel"][:].astype(np.float64)
        baseline = fit_baseline(wrench, q, qd, (a, b))
        cleaned = wrench - baseline
        cleaned[:, 3:] = 0.0          # drop torque: sim produces ~0 contact torque, so the
                                      # 6-axis effort is force-only (Tx,Ty,Tz==0) on both sides
        fd[BASE_WRENCH][...] = cleaned
        fd[BASE_FORCE][...] = cleaned[:, :3]
        fd[BASE_TORQUE][...] = 0.0
        fd.attrs["force_alignment"] = f"free_space_baseline_removed:{MODE};torque_zeroed"
        fd.attrs["force_baseline_frames"] = [a, b]
        raw_fs = float(np.linalg.norm(wrench[a:b, :3], axis=1).mean())
        res_fs = float(np.linalg.norm(cleaned[a:b, :3], axis=1).mean())
        peak = float(np.linalg.norm(cleaned[:, :3], axis=1).max())
    for name in ("front_camera.mp4", "wrist_camera.mp4"):
        link = out_dir / name
        if link.is_symlink() or link.exists():
            link.unlink()
        link.symlink_to(src / name)
    return {
        "trajectory": trajectory, "frames": int(wrench.shape[0]),
        "free_space_frames": [a, b], "mode": MODE,
        "free_space_F_raw_n": raw_fs, "free_space_F_aligned_n": res_fs,
        "contact_peak_F_n": peak,
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    manifest = []
    for trajectory in range(40):
        info = build_one(trajectory, OUT / f"traj_{trajectory}")
        manifest.append(info)
        print(f"[{trajectory:02d}/39] frames={info['frames']:4d} "
              f"free-space |F| {info['free_space_F_raw_n']:.2f}->{info['free_space_F_aligned_n']:.2f}N "
              f"peak={info['contact_peak_F_n']:.2f}N", flush=True)
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    raw = np.mean([m["free_space_F_raw_n"] for m in manifest])
    ali = np.mean([m["free_space_F_aligned_n"] for m in manifest])
    print(f"[DONE] {len(manifest)} trajectories | mean free-space |F| {raw:.2f}->{ali:.2f}N "
          f"(mode={MODE}) -> {OUT}")


if __name__ == "__main__":
    main()
