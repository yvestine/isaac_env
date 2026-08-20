#!/usr/bin/env python3
"""Run the four remote TAVLA checkpoints serially and print success rates only."""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from pathlib import Path


MODELS = (
    ("base_50_50", 8000),
    ("realinit_50_50", 8001),
    ("base_70sim_30real", 8002),
    ("realinit_70sim_30real", 8003),
)
SUMMARY_RE = re.compile(
    r"\[TAVLA-SUMMARY\]\s+port=(\d+)\s+successes=(\d+)\s+"
    r"episodes=(\d+)\s+success_rate=([0-9]+(?:\.[0-9]+)?)%"
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Serially evaluate the four TAVLA servers with no trajectory/video output."
    )
    parser.add_argument("--task", default="TacEx-RealSim-PegInsert-TAVLA-Teacher-v0")
    parser.add_argument("--tavla-host", default="10.0.40.113")
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--steps", type=int, default=600)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def _run_one(
    repo_root: Path,
    eval_script: Path,
    task: str,
    host: str,
    port: int,
    episodes: int,
    steps: int,
    seed: int,
    device: str | None,
) -> dict[str, object]:
    command = [
        str(repo_root / "isaaclab.sh"),
        "-p",
        str(eval_script),
        "--task",
        task,
        "--tavla-host",
        host,
        "--tavla-port",
        str(port),
        "--steps",
        str(steps),
        "--episodes",
        str(episodes),
        "--seed",
        str(seed),
        "--summary-only",
        "--headless",
        "--enable_cameras",
    ]
    if device is not None:
        command.extend(["--device", device])

    child_env = os.environ.copy()
    child_env.setdefault("TERM", "xterm")
    completed = subprocess.run(
        command,
        cwd=repo_root,
        env=child_env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    output = f"{completed.stdout}\n{completed.stderr}"
    matches = [match for match in SUMMARY_RE.finditer(output) if int(match.group(1)) == port]
    if completed.returncode != 0 or not matches:
        tail = "\n".join(output.splitlines()[-40:])
        raise RuntimeError(
            f"port {port} evaluation failed (returncode={completed.returncode}).\n{tail}"
        )

    match = matches[-1]
    # Validate completeness before returning the result.
    completed_episodes = int(match.group(3))
    if completed_episodes != episodes:
        raise RuntimeError(
            f"port {port} evaluation incomplete: {completed_episodes}/{episodes} episodes completed"
        )
    return {
        "successes": int(match.group(2)),
        "episodes": int(match.group(3)),
        "success_rate": float(match.group(4)),
    }


def main() -> int:
    args = _parse_args()
    if args.episodes <= 0 or args.steps <= 0:
        raise ValueError("--episodes and --steps must be positive")

    repo_root = Path(__file__).resolve().parents[3]
    eval_script = repo_root / "scripts/reinforcement_learning/rl_games/tavla_eval.py"
    results: list[tuple[str, int, dict[str, object]]] = []

    for model_name, port in MODELS:
        result = _run_one(
            repo_root=repo_root,
            eval_script=eval_script,
            task=args.task,
            host=args.tavla_host,
            port=port,
            episodes=args.episodes,
            steps=args.steps,
            seed=args.seed,
            device=args.device,
        )
        results.append((model_name, port, result))

    print("TAVLA serial evaluation result")
    print(f"episodes per model: {args.episodes}")
    print(f"seed: {args.seed}")
    for model_name, port, result in results:
        print(
            f"{model_name} (port {port}): "
            f"{result['successes']}/{result['episodes']} = {result['success_rate']:.2f}%"
        )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("Interrupted.", file=sys.stderr)
