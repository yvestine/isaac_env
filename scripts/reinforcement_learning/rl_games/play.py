# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to play a checkpoint if an RL agent from RL-Games."""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher
# import tacex_tasks

import os

# os.environ['http_proxy'] = 'http://127.0.0.1:7890'
# os.environ['https_proxy'] = 'http://127.0.0.1:7890'

    
# def debug():
#     import debugpy
#     debugpy.listen(("0.0.0.0", 5679))
#     print("? Waiting for debugger to attach on port 5678...")
#     debugpy.wait_for_client()

# add argparse arguments
parser = argparse.ArgumentParser(description="Play a checkpoint of an RL agent from RL-Games.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument("--steps", type=int, default=None, help="Stop after this many simulation steps.")
parser.add_argument("--tavla-episodes", type=int, default=None, help="Stop TAVLA play after this many completed episodes.")
parser.add_argument("--seed", type=int, default=None, help="Override the environment reset seed for evaluation.")
parser.add_argument("--episode_length_s", type=float, default=None, help="Maximum play episode length in seconds.")
parser.add_argument("--num_trajectories", type=int, default=None, help="Number of play trajectories to collect before exiting.")
parser.add_argument("--action_scale", type=float, default=1.0, help="Scale policy actions during play to make execution gentler.")
parser.add_argument(
    "--slow_play_controls",
    action="store_true",
    default=False,
    help="Use tighter Cartesian action thresholds during play for gentler data collection.",
)
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--checkpoint", type=str, default=None, help="Path to model checkpoint.")
parser.add_argument("--ppo-hold-steps", type=int, default=1, help="Repeat each PPO action for this many environment steps.")
parser.add_argument(
    "--policy",
    choices=("ppo", "tavla"),
    default="ppo",
    help="Policy backend. Use 'tavla' to deploy the remote TAVLA teacher without a PPO checkpoint.",
)
parser.add_argument(
    "--tavla",
    "--use-tavla",
    "--use_tavla",
    dest="use_tavla",
    action="store_true",
    help="Deploy the remote TAVLA teacher in the existing PegInsert simulation.",
)
parser.add_argument("--tavla-host", type=str, default="10.0.40.113", help="Remote TAVLA server host.")
parser.add_argument("--tavla-port", type=int, default=8000, help="Remote TAVLA server port.")
parser.add_argument(
    "--tavla-control-mode",
    choices=("kinematic_taskspace", "aligned_joint", "ppo_cartesian"),
    default="kinematic_taskspace",
    help="How to execute TAVLA absolute joint targets in simulation.",
)
parser.add_argument(
    "--tavla-hold-steps",
    type=int,
    default=None,
    help="Simulation steps for which each TAVLA action is held (default: 3, matching 30 Hz to 10 Hz).",
)
parser.add_argument(
    "--tavla-replan-actions",
    type=int,
    default=None,
    help="Number of actions consumed from each remote chunk before visual replanning (default: 5).",
)
parser.add_argument(
    "--tavla-visual-profile",
    choices=("raw", "real_aligned"),
    default="raw",
    help="RGB preprocessing applied only to images sent to TAVLA.",
)
parser.add_argument(
    "--tavla-camera-calibration",
    type=str,
    default=None,
    help="JSON calibration used by the real_aligned TAVLA visual profile.",
)
parser.add_argument(
    "--tavla-speed-scale",
    type=float,
    default=None,
    help="Scale the real-data fitted TAVLA joint speed/acceleration limits.",
)
parser.add_argument(
    "--disable-tavla-action-interpolation",
    action="store_true",
    help="Disable the 30 Hz interpolation and real-data motion limits for diagnosis.",
)
parser.add_argument(
    "--disable-tavla-state-alignment",
    action="store_true",
    help="Do not translate the simulator joint state to the real-data policy reference state.",
)
parser.add_argument(
    "--disable-tavla-action-alignment",
    action="store_true",
    help="Do not translate absolute TAVLA joint targets back into simulator coordinates.",
)
failure_group = parser.add_mutually_exclusive_group()
failure_group.add_argument(
    "--save-failed-trajectory",
    "--save_failed_trajectory",
    dest="save_failed_trajectory",
    action="store_true",
    help="Save failed trajectory buffers under the environment output directory.",
)
failure_group.add_argument(
    "--no-save-failed-trajectory",
    "--no_save_failed_trajectory",
    dest="save_failed_trajectory",
    action="store_false",
    help="Discard failed trajectory buffers (the default).",
)
failure_group.set_defaults(save_failed_trajectory=None)
parser.add_argument(
    "--output-dir",
    type=str,
    default=None,
    help="Directory for collected successful/failed trajectories (default: ./data).",
)
parser.add_argument(
    "--use_pretrained_checkpoint",
    action="store_true",
    help="Use the pre-trained checkpoint from Nucleus.",
)
parser.add_argument(
    "--use_last_checkpoint",
    action="store_true",
    help="When no checkpoint provided, use the last saved model. Otherwise use the best saved model.",
)
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time, if possible.")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()
# always enable cameras to record video
if args_cli.video or args_cli.use_tavla or args_cli.policy == "tavla":
    args_cli.enable_cameras = True

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""


import gymnasium as gym
import math
import os
import time
import torch

from rl_games.common import env_configurations, vecenv
from rl_games.common.player import BasePlayer
from rl_games.torch_runner import Runner

from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.dict import print_dict
from isaaclab_rl.utils.pretrained_checkpoint import get_published_pretrained_checkpoint
from isaaclab_rl.rl_games import RlGamesGpuEnv, RlGamesVecEnvWrapper
from isaaclab_tasks.utils import get_checkpoint_path, load_cfg_from_registry, parse_env_cfg

# import isaaclab_tasks  # noqa: F401
import tacex_tasks  # noqa: F401

# PLACEHOLDER: Extension template (do not remove this comment)



def main():
    """Play with PPO or the remote TAVLA teacher."""
    requested_task = args_cli.task
    tavla_mode = (
        args_cli.use_tavla
        or args_cli.policy == "tavla"
        or (requested_task is not None and "TAVLA-Teacher" in requested_task)
    )
    if tavla_mode:
        if requested_task is None or "PegInsert" in requested_task:
            args_cli.task = "TacEx-RealSim-PegInsert-TAVLA-Teacher-v0"
        else:
            raise ValueError("Remote TAVLA deployment currently supports the RealSim PegInsert task only")
    elif requested_task is None:
        raise ValueError("--task is required for PPO play")

    task_name = args_cli.task.split(":")[-1]
    play_num_envs = 1 if tavla_mode else (args_cli.num_envs if args_cli.num_envs is not None else 1)
    env_cfg = parse_env_cfg(
        args_cli.task, device=args_cli.device, num_envs=play_num_envs, use_fabric=not args_cli.disable_fabric
    )
    if args_cli.seed is not None:
        env_cfg.seed = args_cli.seed
    agent_cfg = load_cfg_from_registry(args_cli.task, "rl_games_cfg_entry_point")

    if not tavla_mode:
        log_root_path = os.path.join("logs", "rl_games", agent_cfg["params"]["config"]["name"])
        log_root_path = os.path.abspath(log_root_path)
        print(f"[INFO] Loading experiment from directory: {log_root_path}")
        if args_cli.use_pretrained_checkpoint:
            resume_path = get_published_pretrained_checkpoint("rl_games", task_name)
            if not resume_path:
                print("[INFO] Unfortunately a pre-trained checkpoint is currently unavailable for this task.")
                return
        elif args_cli.checkpoint is None:
            run_dir = agent_cfg["params"]["config"].get("full_experiment_name", ".*")
            checkpoint_file = ".*" if args_cli.use_last_checkpoint else f"{agent_cfg['params']['config']['name']}.pth"
            resume_path = get_checkpoint_path(log_root_path, run_dir, checkpoint_file, other_dirs=["nn"])
        else:
            resume_path = retrieve_file_path(args_cli.checkpoint)
        log_dir = os.path.dirname(os.path.dirname(resume_path))

    # wrap around environment for rl-games
    rl_device = agent_cfg["params"]["config"]["device"]
    clip_obs = agent_cfg["params"]["env"].get("clip_observations", math.inf)
    clip_actions = agent_cfg["params"]["env"].get("clip_actions", math.inf)

    if args_cli.episode_length_s is not None:
        env_cfg.episode_length_s = args_cli.episode_length_s

    if args_cli.slow_play_controls and hasattr(env_cfg, "ctrl"):
        env_cfg.ctrl.default_task_prop_gains = [1400.0, 1400.0, 1400.0, 70.0, 70.0, 70.0]
        env_cfg.ctrl.reset_task_prop_gains = [1600.0, 1600.0, 1600.0, 90.0, 90.0, 90.0]
        env_cfg.ctrl.kp_null = 10
        env_cfg.ctrl.kd_null = 6.3246
        env_cfg.ctrl.pos_action_threshold = [0.020, 0.020, 0.020]
        env_cfg.ctrl.rot_action_threshold = [0.045, 0.045, 0.045]
        env_cfg.ctrl.rot_threshold_noise_level = [0.29, 0.29, 0.29]

    # if hasattr(env_cfg, "robot"):
    #     arm1 = env_cfg.robot.actuators.get("panda_arm1")
    #     arm2 = env_cfg.robot.actuators.get("panda_arm2")
    #     wrist = env_cfg.robot.actuators.get("panda_wrist")
    #     if arm1 is not None:
    #         arm1.damping = 140.0
    #         arm1.friction = 5.0
    #         arm1.armature = 0.2
    #     if arm2 is not None:
    #         arm2.damping = 90.0
    #         arm2.friction = 4.0
    #         arm2.armature = 0.12
    #     if wrist is not None:
    #         wrist.damping = 140.0
    #         wrist.friction = 6.0
    #         wrist.armature = 0.12

    # Play is the real-scene/data-collection path.
    if hasattr(env_cfg, "data_collect_cfg"):
        env_cfg.data_collect_cfg["collect_data"] = True
        env_cfg.data_collect_cfg["immediate_stop"] = True
        if args_cli.save_failed_trajectory is not None:
            env_cfg.data_collect_cfg["save_failed_trajectory"] = args_cli.save_failed_trajectory
        if args_cli.num_trajectories is not None:
            env_cfg.data_collect_cfg["num_trajectories"] = args_cli.num_trajectories


    if tavla_mode:
        if args_cli.tavla_episodes is not None and args_cli.tavla_episodes <= 0:
            raise ValueError("--tavla-episodes must be positive")
        env_cfg.teacher_policy_cfg.host_ip = args_cli.tavla_host
        env_cfg.teacher_policy_cfg.host_port = args_cli.tavla_port
        env_cfg.teacher_eval_only = True
        env_cfg.teacher_control_mode = args_cli.tavla_control_mode
        env_cfg.teacher_visual_profile = args_cli.tavla_visual_profile
        if args_cli.tavla_camera_calibration is not None:
            env_cfg.teacher_camera_calibration = args_cli.tavla_camera_calibration
        env_cfg.teacher_state_alignment = not args_cli.disable_tavla_state_alignment
        env_cfg.teacher_action_state_alignment = not args_cli.disable_tavla_action_alignment
        env_cfg.teacher_action_interpolation = not args_cli.disable_tavla_action_interpolation
        if args_cli.tavla_speed_scale is not None:
            if args_cli.tavla_speed_scale <= 0.0:
                raise ValueError("--tavla-speed-scale must be positive")
            env_cfg.teacher_speed_scale = args_cli.tavla_speed_scale
        if args_cli.tavla_hold_steps is not None:
            if args_cli.tavla_hold_steps <= 0:
                raise ValueError("--tavla-hold-steps must be positive")
            env_cfg.teacher_hold_steps = args_cli.tavla_hold_steps
        if args_cli.tavla_replan_actions is not None:
            if args_cli.tavla_replan_actions <= 0:
                raise ValueError("--tavla-replan-actions must be positive")
            env_cfg.teacher_replan_actions = args_cli.tavla_replan_actions

    # Play uses the full franka_env_backup scene. When cameras are disabled, only
    # skip camera sensors; keep the robot and scene paths from the RealSim config.
    if not tavla_mode and not getattr(args_cli, "enable_cameras", False):
        if hasattr(env_cfg, "wrist_camera"):
            env_cfg.wrist_camera = None
        if hasattr(env_cfg, "tiled_camera"):
            env_cfg.tiled_camera = None

    # create isaac environment
    env_kwargs = {
        "cfg": env_cfg,
        "render_mode": "rgb_array" if args_cli.video else None,
    }
    if args_cli.output_dir is not None:
        env_kwargs["output_dir"] = args_cli.output_dir
    env = gym.make(args_cli.task, **env_kwargs)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # wrap for video recording
    if args_cli.video:
        video_root = os.path.abspath(
            os.path.join("logs", "rl_games", "tavla" if tavla_mode else agent_cfg["params"]["config"]["name"])
        )
        video_kwargs = {
            "video_folder": os.path.join(video_root, "videos", "play"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    if tavla_mode:
        try:
            _run_tavla_play(env)
        finally:
            env.close()
        return

    if args_cli.ppo_hold_steps < 1:
        raise ValueError("--ppo-hold-steps must be positive")
    if args_cli.ppo_hold_steps > 1:
        from tacex_tasks.real2sim.action_repeat import PPOActionRepeatWrapper

        env = PPOActionRepeatWrapper(
            env,
            repeat_steps=args_cli.ppo_hold_steps,
            gamma=float(agent_cfg["params"]["config"].get("gamma", 0.99)),
        )
        print(f"[INFO] PPO action hold enabled: {args_cli.ppo_hold_steps} environment steps/action")

    # wrap around environment for rl-games
    env = RlGamesVecEnvWrapper(env, rl_device, clip_obs, clip_actions)

    # register the environment to rl-games registry
    # note: in agents configuration: environment name must be "rlgpu"
    vecenv.register(
        "IsaacRlgWrapper", lambda config_name, num_actors, **kwargs: RlGamesGpuEnv(config_name, num_actors, **kwargs)
    )
    env_configurations.register("rlgpu", {"vecenv_type": "IsaacRlgWrapper", "env_creator": lambda **kwargs: env})

    # load previously trained model
    agent_cfg["params"]["load_checkpoint"] = True
    agent_cfg["params"]["load_path"] = resume_path
    print(f"[INFO]: Loading model checkpoint from: {agent_cfg['params']['load_path']}")

    # set number of actors into agent config
    agent_cfg["params"]["config"]["num_actors"] = env.unwrapped.num_envs
    # create runner from rl-games
    runner = Runner()
    runner.load(agent_cfg)
    # obtain the agent from the runner
    agent: BasePlayer = runner.create_player()
    agent.restore(resume_path)
    # agent.is_deterministic = True
    agent.reset()

    dt = env.unwrapped.step_dt

    # reset environment
    obs = env.reset()
    if isinstance(obs, dict):
        obs = obs["obs"]
    timestep = 0
    # required: enables the flag for batched observations
    _ = agent.get_batch_size(obs, 1)
    # initialize RNN states if used
    if agent.is_rnn:
        agent.init_rnn()
    # simulate environment
    # note: We simplified the logic in rl-games player.py (:func:`BasePlayer.run()`) function in an
    #   attempt to have complete control over environment stepping. However, this removes other
    #   operations such as masking that is used for multi-agent learning by RL-Games.
    while simulation_app.is_running():
        start_time = time.time()
        # run everything in inference mode
        with torch.inference_mode():
            # convert obs to agent format
            obs = agent.obs_to_torch(obs)
            # agent stepping
            actions = agent.get_action(obs, is_deterministic=agent.is_deterministic)
            actions = actions * args_cli.action_scale
            # env stepping
            obs, _, dones, _ = env.step(actions)
            # perform operations for terminated episodes
            if len(dones) > 0:
                # reset rnn state for terminated episodes
                if agent.is_rnn and agent.states is not None:
                    for s in agent.states:
                        s[:, dones, :] = 0.0
        if args_cli.video:
            timestep += 1
            # exit the play loop after recording one video
            if timestep == args_cli.video_length:
                break
        if args_cli.steps is not None:
            timestep += 1 if not args_cli.video else 0
            if timestep >= args_cli.steps:
                break

        # time delay for real-time evaluation, control the speed of the simulation to match real-time as much as possible
        sleep_time = dt - (time.time() - start_time)
        if args_cli.real_time and sleep_time > 0:
            time.sleep(sleep_time)

    # close the simulator
    env.close()


def _run_tavla_play(env):
    """Run the play loop with zero RL input and TAVLA as the controller."""
    env.reset()
    dt = env.unwrapped.step_dt
    timestep = 0
    completed_episodes = int(getattr(env.unwrapped, "total_times", 0))
    while simulation_app.is_running():
        start_time = time.time()
        with torch.inference_mode():
            zero_action = torch.zeros(
                (env.unwrapped.num_envs, env.unwrapped.cfg.action_space),
                dtype=torch.float32,
                device=env.unwrapped.device,
            )
            env.step(zero_action)
        timestep += 1
        current_completed_episodes = int(getattr(env.unwrapped, "total_times", 0))
        if args_cli.tavla_episodes is not None and current_completed_episodes - completed_episodes >= args_cli.tavla_episodes:
            break
        if args_cli.steps is not None and timestep >= args_cli.steps:
            break
        if args_cli.video and timestep >= args_cli.video_length:
            break
        sleep_time = dt - (time.time() - start_time)
        if args_cli.real_time and sleep_time > 0:
            time.sleep(sleep_time)
    print(
        f"[INFO] TAVLA play finished after {timestep} steps; "
        f"inferences={env.unwrapped.teacher_inference_count}, "
        f"failures={env.unwrapped.teacher_failures}"
    )


if __name__ == "__main__":
    # debug()
    try:
        main()
    finally:
        simulation_app.close()
