"""Capture pre-insertion simulator states for later PPO resets."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="Capture a TA-VLA pre-insertion state database")
parser.add_argument("--episodes", type=int, default=100)
parser.add_argument("--output", type=Path, default=Path("outputs/tavla_baseline/preinsert_states.json"))
parser.add_argument("--adapter-checkpoint", type=str, default="checkpoints/unpaired_sim_to_real_affine.pt")
parser.add_argument("--tavla-host", type=str, default="10.0.40.113")
parser.add_argument("--tavla-port", type=int, default=8000)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
simulation_app = AppLauncher(args).app

import tacex_tasks  # noqa: E402,F401
from tacex_tasks.real2sim.tavla_baseline.isaac_env import TavlaAffineResidualEnv  # noqa: E402
from tacex_tasks.real2sim.tavla_baseline.states import PreInsertState, PreInsertStateDatabase  # noqa: E402
from tacex_tasks.real2sim.tavla_residual_env_cfg import RealSimTavlaResidualPegInsertCfg  # noqa: E402


def main() -> None:
    cfg = RealSimTavlaResidualPegInsertCfg()
    cfg.scene.num_envs = 1
    cfg.teacher_eval_only = True
    cfg.teacher_policy_cfg.host_ip = args.tavla_host
    cfg.teacher_policy_cfg.host_port = args.tavla_port
    cfg.data_collect_cfg["collect_data"] = False
    env = TavlaAffineResidualEnv(cfg, baseline_adapter_path=args.adapter_checkpoint)
    database = PreInsertStateDatabase()
    try:
        for _episode in range(args.episodes):
            env.reset()
            for _step in range(env.max_episode_length):
                action = torch.zeros((1, 8), device=env.device)
                _, _, terminated, truncated, _ = env.step(action)
                if hasattr(env, "held_pos") and hasattr(env, "fixed_pos"):
                    engaged = env._get_curr_successes(env.cfg_task.engage_threshold, check_rot=False)
                    if bool(engaged.reshape(-1)[0]):
                        database.add(PreInsertState.from_arrays(
                            env._current_tavla_state()[0].detach().cpu().numpy(),
                            qvel=env.joint_vel[0].detach().cpu().numpy(),
                            peg_pose=env.held_pos[0].detach().cpu().numpy(),
                            hole_pose=env.fixed_pos[0].detach().cpu().numpy(),
                            insertion_depth=0.0,
                            metadata={"episode": _episode, "step": _step},
                        ))
                        break
                if bool(torch.as_tensor(terminated).reshape(-1)[0]) or bool(torch.as_tensor(truncated).reshape(-1)[0]):
                    break
        database.save(args.output)
        print(f"saved {len(database)} pre-insertion states to {args.output}")
    finally:
        env.close()


if __name__ == "__main__":
    main()

