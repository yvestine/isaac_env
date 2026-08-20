#!/usr/bin/env python3
import h5py
import numpy as np
import pandas as pd
import cv2
from pathlib import Path
import argparse
from tqdm import tqdm

def debug():
    import debugpy
    debugpy.listen(("0.0.0.0", 5679))
    print("✅ Waiting for debugger to attach on port 5678...")
    debugpy.wait_for_client()

def process_episode(episode_path: Path, h5_file: h5py.File, desired_joints: list = None, action_mode: str = 'end_pose', include_gripper_force: bool = False):
    """
    Processes a single episode's data and writes it to the root of an HDF5 file.
    
    Args:
        episode_path (Path): The path to the episode directory.
        h5_file (h5py.File): The HDF5 file object to write to.
        desired_joints (list, optional): A list of joint names to filter for the qpos dataset.
        action_mode (str): Determines the content of the 'action' dataset. Can be 'end_pose' or 'qpos'.
        include_gripper_force (bool): If True, includes gripper effort in the qpos dataset.
    """
    print(f"\nProcessing {episode_path.name} -> {h5_file.filename}")

    # =============================================================================
    # 1. Process Images: /observations/images/front (WITH BGR -> RGB FIX)
    # =============================================================================
    obs_group = h5_file.create_group('observations')
    images_group = obs_group.create_group('images')
    
    for camera_name in ['front', 'wrist', 'left_gelsight_with_marker']:
        video_path = episode_path / camera_name
        video_files = []
        for ext in ('*.mp4', '*.avi', '*.mov', '*.mkv'):
            video_files.extend(video_path.glob(ext))

        # 读取视频
        for video_file in video_files:
            cap = cv2.VideoCapture(str(video_file))
            if not cap.isOpened():
                print(f"Error opening video file {video_file}")
                continue

            fps = cap.get(cv2.CAP_PROP_FPS)
            print(f"Video FPS: {fps}")

            image_list = []
            print(f"Processing video: {video_file.name}")
            print("    Correcting image color channels (BGR -> RGB)...")
            while True:
                ret, frame = cap.read()
                if not ret:
                    break  # 没有更多帧了
                    
                # Convert from BGR to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image_list.append(frame_rgb)

            images_array = np.array(image_list)
            camera_name = video_file.stem  # 使用视频文件的名字作为camera_name
            if 'front' in camera_name:
                images_group.create_dataset('front', data=images_array, compression='gzip')
                num_timesteps = len(images_array)
            elif 'wrist' in camera_name:
                images_group.create_dataset('left_wrist', data=images_array, compression='gzip')
            elif 'left_gelsight_with_marker' in camera_name:
                images_group.create_dataset('left_gelsight_with_marker', data=images_array, compression='gzip')
            elif 'right_gelsight_with_marker' in camera_name:
                images_group.create_dataset('right_gelsight_with_marker', data=images_array, compression='gzip')
            print(f"    Saved '{camera_name}' video frames with shape: {images_array.shape}")
            cap.release()

    # =============================================================================
    # 2. Process Joint Positions: /observations/qpos
    # =============================================================================
    robot_joints_df = pd.read_csv(episode_path / 'joint_states.csv')
    
    if action_mode == "qpos":
        if desired_joints:
            print(f"    Filtering for specified robot joints: {desired_joints}")
            available_joints = [j for j in desired_joints if j in robot_joints_df.columns]
            if not available_joints:
                print(f"    Warning: None of the specified joints found. Robot qpos will be empty.")
                robot_state = np.empty((num_timesteps, 0))
            else:
                robot_state = robot_joints_df[available_joints].to_numpy()
        else:
            print("    Saving all robot joints for qpos.")
            robot_state = robot_joints_df.drop(columns=['timestamp']).to_numpy()
    elif action_mode == "end_pose":
        print("    Saving end-effector pose for action.")
        gripper = robot_joints_df.to_numpy()[:, -1]  
        robot_state = pd.read_csv(episode_path / 'ee_pose.csv')
        robot_state = np.concatenate((robot_state.to_numpy(), gripper.reshape(-1, 1)), axis=1)
        force_path = episode_path / 'force_local.csv'
        if force_path.exists():
            force_data = pd.read_csv(force_path).to_numpy()

    
    assert robot_state.shape[0] == num_timesteps, "Mismatch in qpos timesteps!"
    obs_group.create_dataset('qpos', data=robot_state)
    obs_group.create_dataset('effort', data=force_data)
    
    print(f"    Saved 'qpos' data with shape: {robot_state.shape}")

    # =============================================================================
    # 3. Process Action Data: /action
    # =============================================================================
    if action_mode == 'end_pose':
        print("    Using end-effector poses for the 'action' dataset.")
        # left_pose_df = pd.read_csv(episode_path / 'ee_pose.csv')
        action_data = pd.read_csv(episode_path / 'actions.csv')
        action_data = action_data.to_numpy()

    elif action_mode == 'qpos':
        print("    Using joint positions (qpos) for the 'action' dataset.")
        # action_data = np.concatenate((qpos_data[1:], qpos_data[-1:])) # 右移一位
        action_data = pd.read_csv(episode_path / 'actions.csv')
        action_data = action_data.to_numpy()
    
    if action_data.shape[0] < num_timesteps:
        last_action = action_data[-1:]
        num_pad = num_timesteps - action_data.shape[0]
        padding = np.repeat(last_action, num_pad, axis=0)
        action_data = np.vstack([action_data, padding])
    elif action_data.shape[0] > num_timesteps:
        raise ValueError("1")
        # action_data = action_data[:num_timesteps]
    assert action_data.shape[0] == num_timesteps, "Mismatch in action timesteps!"
    h5_file.create_dataset('action', data=action_data)
    print(f"    Saved 'action' data with shape: {action_data.shape}")


def main():
    joints = [ # 指定需要保存的关节名称
        "joint_0", "joint_1", "joint_2",
        "joint_3", "joint_4", "joint_5", "joint_6","joint_7"
    ]
    # action_mode = "qpos" # 保存关节角度还是末端位姿
    action_mode = "end_pose" # 保存关节角度还是末端位姿

    parser = argparse.ArgumentParser(description="Convert recorded sensor data into one HDF5 file per episode.")
    parser.add_argument(
        '--source_dir', type=str, default='./data',
        help="The directory containing the 'episode_*' folders."
    )
    parser.add_argument(
        '--output_dir', type=str, default='./dataset',
        help="The directory to save the HDF5 files."
    )
    # Removed the output_file argument as it's now determined automatically
    parser.add_argument(
        '--joints', nargs='+',
        help="A space-separated list of joint names to include in the qpos dataset (e.g., --joints arm_left_1_joint arm_left_2_joint)."
    )
    args = parser.parse_args()

    source_path = Path(args.source_dir)
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    episode_paths = sorted(source_path.glob('episode_*'))

    if not episode_paths:
        print(f"Error: No 'episode_*' directories found in '{source_path}'.")
        return

    print(f"Found {len(episode_paths)} episodes. Starting conversion...")

    for episode_path in episode_paths:
        if episode_path.is_dir():
            # Define output path: save HDF5 files in the specified output directory
            output_h5_path = output_path / f"{episode_path.name}.hdf5"

            with h5py.File(output_h5_path, 'w') as h5_file:
                # process_episode(episode_path, h5_file, args.joints)
                process_episode(episode_path, h5_file, joints,action_mode)

    print(f"\n✅ All episodes processed. HDF5 files have been saved in '{output_path}'.")

if __name__ == '__main__':
    # debug()
    main()