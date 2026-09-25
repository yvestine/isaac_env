"""Dedicated entry point for continuous EE-pose replay.

This is intentionally a separate command from ``replay_real_joint_ppo.py``.
It delegates to the existing Cartesian/PPO replay implementation, which:

* reads ``obs/state/ee_pose`` from the real H5 trajectory;
* solves the corresponding PPO/DLS arm targets from the recorded EE pose;
* advances the arm and the held peg through PhysX between frames;
* samples the existing PPO wrench streams after the physical substeps; and
* places the fixed hole at the median peg-root pose from the final frame
  window (default: five frames).

The delegate is kept in one place so the known camera/background and Isaac
Sim 6 bootstrap fixes are not duplicated.  The original joint replay remains
untouched and is still the rollback/reference path.
"""

from __future__ import annotations

import runpy
import sys
from pathlib import Path


def _has_option(arguments: list[str], name: str) -> bool:
    """Return whether an argparse option is present in either supported form."""
    prefix = f"{name}="
    return any(argument == name or argument.startswith(prefix) for argument in arguments)


def main() -> None:
    delegate = Path(__file__).with_name("replay_real_ee_cartesian_ppo.py").resolve()
    if not delegate.is_file():
        raise FileNotFoundError(f"Cartesian/PPO replay implementation not found: {delegate}")

    user_arguments = sys.argv[1:]
    default_arguments: list[str] = []
    if not _has_option(user_arguments, "--output-dir"):
        default_arguments.extend(
            ["--output-dir", "outputs/ee_pose_ppo_force/traj_0_aligned_visual"]
        )
    if not _has_option(user_arguments, "--endpoint-window"):
        default_arguments.extend(["--endpoint-window", "5"])
    if not _has_option(user_arguments, "--record-contact-pair"):
        default_arguments.append("--record-contact-pair")

    # The delegate owns AppLauncher argument parsing and must see its own
    # filename in argv[0].  User arguments are appended so explicit values
    # override the dedicated entry-point defaults in argparse.
    sys.argv = [str(delegate), *default_arguments, *user_arguments]
    runpy.run_path(str(delegate), run_name="__main__")


if __name__ == "__main__":
    main()
