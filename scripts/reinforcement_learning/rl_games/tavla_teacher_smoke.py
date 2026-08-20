"""Run the frozen TAVLA teacher in the PegInsert simulator without RL."""

import argparse

import torch
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="TAVLA-only RealSim smoke test")
parser.add_argument(
    "--task",
    type=str,
    default="TacEx-RealSim-PegInsert-TAVLA-Teacher-v0",
    help="Teacher-only Gym task id",
)
parser.add_argument("--steps", type=int, default=300, help="Maximum simulation steps")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

simulation_app = AppLauncher(args).app

import gymnasium as gym  # noqa: E402
import tacex_tasks  # noqa: E402,F401
from isaaclab.utils.dict import print_dict  # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402


def main():
    env_cfg = parse_env_cfg(args.task, device=args.device, num_envs=1, use_fabric=not args.disable_fabric)
    env_cfg.teacher_eval_only = True
    env_cfg.data_collect_cfg["collect_data"] = False
    env = gym.make(args.task, cfg=env_cfg)
    reset_result = env.reset()
    obs = reset_result[0] if isinstance(reset_result, tuple) else reset_result
    del obs

    episodes = 0
    for step in range(args.steps):
        action = torch.zeros((1, env.unwrapped.cfg.action_space), device=env.unwrapped.device)
        result = env.step(action)
        if len(result) == 5:
            _, _, terminated, truncated, info = result
            done = bool(torch.as_tensor(terminated).any() or torch.as_tensor(truncated).any())
        else:
            _, _, done, info = result
            done = bool(torch.as_tensor(done).any())
        if done:
            episodes += 1
            if episodes >= 3:
                break

    metrics = env.unwrapped.extras
    scalar_metrics = {}
    for key in (
        "tavla/teacher_joint_error",
        "tavla/gripper_target",
        "tavla/gripper_actual",
        "tavla/wrench_base_mean",
        "tavla/wrench_base_std",
    ):
        value = metrics.get(key)
        if value is not None:
            scalar_metrics[key] = float(value.detach().cpu().item()) if torch.is_tensor(value) else float(value)

    print_dict(
        {
            "steps": step + 1,
            "episodes": episodes,
            "teacher_inference_count": env.unwrapped.teacher_inference_count,
            "teacher_failures": env.unwrapped.teacher_failures,
            "last_teacher_latency_s": env.unwrapped.teacher_inference_latency_s,
            **scalar_metrics,
        },
        nesting=2,
    )
    env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
