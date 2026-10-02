"""Compare one aligned-eval first input against the stored sim-data video."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--sim-data-dir", type=Path, default=Path("sim-data"))
parser.add_argument("--profile-id", type=int, required=True)
parser.add_argument("--eval-dir", type=Path, required=True)
parser.add_argument("--min-psnr-db", type=float, default=35.0)
parser.add_argument("--min-ssim", type=float, default=0.97)
args = parser.parse_args()


def _read_video_first(path: Path) -> np.ndarray:
    capture = cv2.VideoCapture(str(path))
    try:
        ok, frame = capture.read()
    finally:
        capture.release()
    if not ok or frame is None:
        raise RuntimeError(f"Could not decode first frame: {path}")
    return frame


def _read_image(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"Could not read image: {path}")
    return image


def _psnr(reference: np.ndarray, candidate: np.ndarray) -> float:
    error = reference.astype(np.float64) - candidate.astype(np.float64)
    mse = float(np.mean(error * error))
    if mse == 0.0:
        return float("inf")
    return float(20.0 * np.log10(255.0 / np.sqrt(mse)))


def _ssim(reference: np.ndarray, candidate: np.ndarray) -> float:
    """Compute mean local SSIM with the standard 11x11 Gaussian window."""

    ref = reference.astype(np.float64)
    cand = candidate.astype(np.float64)
    c1 = (0.01 * 255.0) ** 2
    c2 = (0.03 * 255.0) ** 2
    mu_ref = cv2.GaussianBlur(ref, (11, 11), 1.5)
    mu_cand = cv2.GaussianBlur(cand, (11, 11), 1.5)
    mu_ref_sq = mu_ref * mu_ref
    mu_cand_sq = mu_cand * mu_cand
    mu_product = mu_ref * mu_cand
    sigma_ref_sq = cv2.GaussianBlur(ref * ref, (11, 11), 1.5) - mu_ref_sq
    sigma_cand_sq = cv2.GaussianBlur(cand * cand, (11, 11), 1.5) - mu_cand_sq
    covariance = cv2.GaussianBlur(ref * cand, (11, 11), 1.5) - mu_product
    numerator = (2.0 * mu_product + c1) * (2.0 * covariance + c2)
    denominator = (mu_ref_sq + mu_cand_sq + c1) * (
        sigma_ref_sq + sigma_cand_sq + c2
    )
    return float(np.mean(numerator / denominator))


def _compare(name: str, reference_path: Path, candidate_path: Path, output: Path):
    reference = _read_video_first(reference_path)
    candidate = _read_image(candidate_path)
    if reference.shape != candidate.shape:
        raise ValueError(
            f"{name} shape mismatch: reference={reference.shape}, candidate={candidate.shape}"
        )
    difference = cv2.absdiff(reference, candidate)
    montage = np.concatenate(
        (
            reference,
            candidate,
            np.clip(difference.astype(np.int16) * 4, 0, 255).astype(np.uint8),
        ),
        axis=1,
    )
    cv2.imwrite(str(output), montage)
    return {
        "reference": str(reference_path),
        "candidate": str(candidate_path),
        "shape": list(reference.shape),
        "psnr_db": _psnr(reference, candidate),
        "ssim": _ssim(reference, candidate),
        "mean_abs_error": float(np.mean(difference)),
        "montage": str(output),
    }


def main() -> None:
    profile_dir = args.sim_data_dir / f"traj_{args.profile_id}"
    eval_dir = args.eval_dir.expanduser().resolve()
    front = _compare(
        "front",
        profile_dir / "front_camera.mp4",
        eval_dir / "policy_input_front_first.png",
        eval_dir / "visual_alignment_front.png",
    )
    wrist = _compare(
        "wrist",
        profile_dir / "wrist_camera.mp4",
        eval_dir / "policy_input_wrist_first.png",
        eval_dir / "visual_alignment_wrist.png",
    )
    passed = all(
        result["psnr_db"] >= args.min_psnr_db
        and result["ssim"] >= args.min_ssim
        for result in (front, wrist)
    )
    report = {
        "profile_id": args.profile_id,
        "thresholds": {
            "min_psnr_db": args.min_psnr_db,
            "min_ssim": args.min_ssim,
        },
        "front": front,
        "wrist": wrist,
        "passed": passed,
    }
    report_path = eval_dir / "visual_alignment.json"
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
