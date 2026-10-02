# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import copy
from pathlib import Path
import numpy as np
import torch
import os
from PIL import Image
import csv
import json
import cv2
import subprocess
import time

from .forge_env import ForgeEnv
from .realsim_env_cfg import RealSimEnvCfg
from isaaclab.sensors import TiledCamera
from isaaclab.utils import math as isaaclab_math
import tacex_tasks.torch_compat as torch_utils
from isaaclab_tasks.direct.factory import factory_control, factory_utils
import isaaclab.sim as sim_utils
import carb

from isaaclab.assets import Articulation
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR


def _sim_flag(sim, name):
    value = getattr(sim, name, False)
    return bool(value() if callable(value) else value)


def _write_h264_mp4(video_path, frames, fps):
    """Write an H.264/yuv420p MP4 that can be previewed by VS Code/Electron."""
    if not frames:
        return

    first_frame = np.ascontiguousarray(frames[0], dtype=np.uint8)
    if first_frame.ndim != 3 or first_frame.shape[2] != 3:
        raise ValueError(
            f"Expected BGR frames with shape (H, W, 3), got {first_frame.shape}"
        )
    height, width = first_frame.shape[:2]

    try:
        import imageio_ffmpeg
    except ImportError as exc:
        raise RuntimeError(
            "H.264 MP4 export requires imageio-ffmpeg in the Isaac Sim Python environment."
        ) from exc

    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    process = subprocess.Popen(
        [
            ffmpeg,
            "-loglevel",
            "error",
            "-y",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "bgr24",
            "-s:v",
            f"{width}x{height}",
            "-r",
            str(fps),
            "-i",
            "-",
            "-an",
            "-vf",
            "pad=ceil(iw/2)*2:ceil(ih/2)*2",
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            "23",
            "-profile:v",
            "baseline",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(video_path),
        ],
        stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )

    try:
        assert process.stdin is not None
        for frame in frames:
            frame = np.ascontiguousarray(frame, dtype=np.uint8)
            if frame.shape != first_frame.shape:
                raise ValueError(
                    f"All video frames must have shape {first_frame.shape}, got {frame.shape}"
                )
            process.stdin.write(frame.tobytes())
        process.stdin.close()
        stderr = process.stderr.read() if process.stderr is not None else b""
        return_code = process.wait()
    except Exception:
        process.kill()
        process.wait()
        if os.path.exists(video_path):
            os.remove(video_path)
        raise

    if return_code != 0:
        if os.path.exists(video_path):
            os.remove(video_path)
        detail = stderr.decode("utf-8", errors="replace").strip()
        raise RuntimeError(f"Could not encode H.264 MP4 {video_path}: {detail}")





class RealSimEnv(ForgeEnv):
    """
    RealSimEnv extension for data collection.
    
    Inherits from ForgeEnv and adds functionality to collect and save:
    - Camera images (front camera, wrist camera)
    - Joint states
    - Gripper states
    - End-effector poses
    - Force/torque sensor data
    - Actions
    """
    
    cfg: RealSimEnvCfg

    def __init__(
        self, 
        cfg: RealSimEnvCfg, 
        render_mode: str | None = None,
        output_dir: str = "./data",
        **kwargs
    ):
        """
        Initialize the data collection environment.
        
        Args:
            cfg: Environment configuration
            render_mode: Rendering mode
            collect_data: Whether to collect data during runtime
            output_dir: Directory to save collected data
            **kwargs: Additional arguments
        """
        super().__init__(cfg, render_mode, **kwargs)
        
        # 创建一个专用的随机数生成器
        # 这里的 seed 可以从 cfg 读取，或者固定
        seed = self.cfg.seed if hasattr(self.cfg, "seed") and self.cfg.seed is not None else 42
        self.rng = torch.Generator(device=self.device)
        self.rng.manual_seed(seed)
        
        
        # ========== 新增: Policy初始化 ==========
        if cfg.policy_cfg:
            from .policy.modeling_pi0remote import PI0RemotePolicy, PI0RemotePolicyTAVLA
            # 根据配置选择policy类型
            if hasattr(cfg.policy_cfg, 'num_history_steps'):
                self.policy = PI0RemotePolicyTAVLA(cfg.policy_cfg)
                print("Using Pi0 Policy")
            else:
                self.policy = PI0RemotePolicy(cfg.policy_cfg)
                print("Using TA-VLA Policy")
        else:
            self.policy = None
        
        self.next_action = []  # 存储policy输出的action
        self.ppo_joint_target = torch.zeros(
            (self.num_envs, 8), dtype=torch.float32, device=self.device
        )
        # The limiter is applied to Cartesian target commands, not to joint
        # torques or the simulator. Keep one target per environment and
        # advance it once per 30 Hz environment action (not per PhysX substep).
        # The parent constructor has not exposed the fingertip state at this
        # point. The actual value is filled in by _reset_idx before stepping.
        self._cartesian_target_pos = torch.zeros(
            (self.num_envs, 3), dtype=torch.float32, device=self.device
        )
        self._cartesian_target_linear_velocity = torch.zeros(
            (self.num_envs, 3), dtype=torch.float32, device=self.device
        )
        self._cartesian_target_euler = torch.zeros(
            (self.num_envs, 3), dtype=torch.float32, device=self.device
        )
        self._cartesian_target_angular_velocity = torch.zeros(
            (self.num_envs, 3), dtype=torch.float32, device=self.device
        )
        self._cartesian_target_update_pending = False
        self._cartesian_orientation_update_pending = False
        self._cartesian_target_initialized = False
        self._cartesian_orientation_initialized = False
        self._joint_target_pos = torch.zeros(
            (self.num_envs, 7), dtype=torch.float32, device=self.device
        )
        self._joint_target_velocity = torch.zeros(
            (self.num_envs, 7), dtype=torch.float32, device=self.device
        )
        self._joint_target_update_pending = False
        self._joint_target_initialized = False
        # ========================================
    
        self.collect_data = cfg.data_collect_cfg["collect_data"]
        # print("Collect Data",self.collect_data)
        self.immediate_stop = cfg.data_collect_cfg["immediate_stop"]
        self.save_failed_trajectory = cfg.data_collect_cfg["save_failed_trajectory"]
        self.num_trajectories = cfg.data_collect_cfg["num_trajectories"]
        self.minimal_output = bool(cfg.data_collect_cfg.get("minimal_output", False))
        self.save_policy_input_video = bool(
            cfg.data_collect_cfg.get("save_policy_input_video", False)
        )
        self.save_raw_camera_video = bool(
            cfg.data_collect_cfg.get("save_raw_camera_video", True)
        )
        self.save_reward_video = bool(
            cfg.data_collect_cfg.get("save_reward_video", True)
        )
        self.save_tavla_hdf5 = bool(cfg.data_collect_cfg.get("save_tavla_hdf5", True))
        self.tavla_hdf5_dir = str(cfg.data_collect_cfg.get("tavla_hdf5_dir", "tavla_raw"))
        self.cur_num_traj = 0

        self.output_dir = output_dir
        self._record_data_mask = None
        self._model_visual_frame_ready = False
        self.last_model_input_front = None
        self.last_model_input_wrist = None

        
        if self.collect_data:
            # Initialize data buffers for each environment
            self.data_buffers = [
                {
                    "camera": {
                        "front": [],
                        "front_transformed": [],
                        "wrist_transformed": [],
                        "wrist": [],
                    },
                    "joints": [],
                    "gripper": [],
                    "ee_pose": [],
                    "ee_pose_xyzw": [],
                    "force": [],  # Backward-compatible model-aligned wrench
                    "force_world": [],  # Backward-compatible parent-frame wrench
                    "force_parent": [],
                    "force_tool": [],
                    "force_model": [],
                    "wrench_raw": [],
                    "wrench_anchor": [],
                    "wrench_base": [],
                    "wrench_corrected": [],
                    "wrench_final": [],
                    "timestamps": [],
                    "actions": [],
                    "ppo_joint_targets": [],
                    "tavla_teacher_actions": [],
                    "tavla_residual_actions": [],
                    "tavla_wrench_base": [],
                    "tavla_wrench_final": [],
                    "tavla_server_effort": [],
                    "tavla_server_effort_matches_final": [],
                    "tavla_policy_wrench": [],
                    "tavla_force_gate_active": [],
                    "tavla_force_gate_contact_norm": [],
                    "tavla_actual_state": [],
                    "tavla_policy_state": [],
                    "tavla_combined_targets": [],
                    "tavla_command_targets": [],
                    "tavla_command_velocity": [],
                    "tavla_action_indices": [],
                    "tavla_executed_targets": [],
                    "tavla_inference_events": [],
                    "tavla_inference_latency_s": [],
                    "tavla_inference_timeouts": [],
                    "tavla_action_nonfinite": [],
                    "tavla_target_out_of_limits": [],
                "tavla_twin_state": [],
                "tavla_taskspace_actions": [],
                "tavla_taskspace_q_deltas": [],
                "tavla_taskspace_targets": [],
                "tavla_twin_ik": [],
                "tavla_force_abort": [],
                    "rewards": [],
                    "reward_terms": []
                }
                for _ in range(self.num_envs)
            ]
            self.reset_data_buffer()
        
        self.success_times = 0
        self.total_times = 0

    @staticmethod
    def _make_background_robot_visual_only(stage, prim_path: str) -> None:
        """Keep a background robot visible without adding physics or control."""

        from pxr import Usd, UsdPhysics

        root = stage.GetPrimAtPath(prim_path)
        if not root.IsValid():
            print(f"[RealSim] Background visual robot not found: {prim_path}")
            return

        rigid_body_count = 0
        collision_count = 0
        joint_paths = []
        for prim in Usd.PrimRange(root):
            if prim.IsA(UsdPhysics.Joint):
                joint_paths.append(str(prim.GetPath()))
            if prim.HasAPI(UsdPhysics.RigidBodyAPI):
                UsdPhysics.RigidBodyAPI(prim).GetRigidBodyEnabledAttr().Set(False)
                rigid_body_count += 1
            if prim.HasAPI(UsdPhysics.CollisionAPI):
                UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Set(False)
                collision_count += 1

        for joint_path in joint_paths:
            stage.OverridePrim(joint_path).SetActive(False)

        print(
            "[RealSim] Kept background robot as visual-only: "
            f"{prim_path} rigid_bodies={rigid_body_count} "
            f"colliders_disabled={collision_count} joints_removed={len(joint_paths)}"
        )



    @staticmethod
    def _spawn_pi0_black_gripper_visual(stage, source_usd: str, hand_path: str) -> None:
        """Mount rollout black finger subtrees on the active Pi0 fingers."""

        from pxr import Gf, Sdf, Usd, UsdGeom, UsdShade

        # Rollout mounts each black long-finger subtree below its corresponding
        # articulated finger, not below panda_hand. Referencing the authored
        # fingertip subtree preserves its exact local transform and material.
        robot_path = hand_path.rsplit("/", 1)[0]
        finger_specs = {
            "left": (
                f"{robot_path}/panda_leftfinger/panda_leftfingertip",
                "/World/Robot/franka/panda_leftfinger/panda_leftfingertip",
            ),
            "right": (
                f"{robot_path}/panda_rightfinger/panda_rightfingertip",
                "/World/Robot/franka/panda_rightfinger/panda_rightfingertip",
            ),
        }
        black_material_path = "/World/Looks/pi0_rollout_plastic_black"
        black_material = UsdShade.Material.Define(stage, black_material_path)
        black_material.GetPrim().GetReferences().AddReference(
            source_usd,
            "/World/Robot/franka/panda_link7/visuals/hand/geometry_01/Looks/PlasticBlack",
        )

        # The generic Franka asset has the same material subset layout as
        # rollout, but its imported PlasticBlack shader is gray. Override
        # that active shader to the rollout value without editing the asset.
        active_black_shader_path = f"{robot_path}/Looks/material_Part__Feature_007/Shader"
        active_black_shader_prim = stage.GetPrimAtPath(active_black_shader_path)
        if active_black_shader_prim.IsValid():
            active_black_shader = UsdShade.Shader(active_black_shader_prim)
            active_black_input = active_black_shader.GetInput("diffuse_color_constant")
            if active_black_input.IsValid():
                active_black_input.Set(Gf.Vec3f(0.0, 0.0, 0.0))

        for name, (target_path, source_path) in finger_specs.items():
            if not stage.GetPrimAtPath(target_path).IsValid():
                fingertip = UsdGeom.Xform.Define(stage, target_path)
                fingertip.GetPrim().GetReferences().AddReference(source_usd, source_path)
            for prim in Usd.PrimRange(stage.GetPrimAtPath(target_path)):
                if prim.IsA(UsdGeom.Mesh):
                    UsdShade.MaterialBindingAPI.Apply(prim).Bind(black_material)
            subset_path = f"{robot_path}/panda_{name}finger/visuals/material_1"
            subset = stage.GetPrimAtPath(subset_path)
            if subset.IsValid():
                UsdShade.MaterialBindingAPI.Apply(subset).Bind(black_material)
        print(f"[RealSim] Restored rollout black gripper under: {robot_path}")

    @staticmethod
    def _spawn_background_robot_visual(
        stage,
        source_usd: str,
        source_prim_path: str,
        prim_path: str,
        offset_pos,
    ) -> None:
        """Reference a robot visual under a translated, non-controlled wrapper."""

        from pxr import Gf, UsdGeom

        if stage.GetPrimAtPath(prim_path).IsValid():
            return

        parent_path = prim_path.rsplit("/", 1)[0]
        parent = UsdGeom.Xform.Define(stage, parent_path)
        parent.AddTranslateOp().Set(Gf.Vec3d(*offset_pos))
        source_root = stage.DefinePrim(prim_path, "Xform")
        source_root.GetReferences().AddReference(source_usd, source_prim_path)

    @staticmethod
    def _hide_active_robot_render_meshes(stage, robot_path: str) -> None:
        """Hide render meshes without touching articulation or collision data."""

        from pxr import Usd, UsdGeom

        root = stage.GetPrimAtPath(robot_path)
        if not root.IsValid():
            print(f"[RealSim] Active robot not found for visual shell: {robot_path}")
            return

        hidden = 0
        for prim in Usd.PrimRange(root):
            prim_path = str(prim.GetPath())
            if prim.IsA(UsdGeom.Mesh) and "/visuals" in prim_path:
                UsdGeom.Imageable(prim).MakeInvisible()
                hidden += 1
        print(
            f"[RealSim] Hid active robot render meshes: {robot_path} meshes={hidden}",
            flush=True,
        )

    @staticmethod
    def _pose_background_robot_visual(
        stage,
        robot_path: str,
        joint_positions,
        gripper_open: float,
        log: bool = True,
    ) -> None:
        """Author a fixed Pi0-only pose for the unactuated right robot.

        The background USD contains an articulated Franka in its all-zero
        pose. Its physics joints are disabled elsewhere, so drive targets
        cannot pose it. This computes the link transforms from the USD joint
        frames and writes visual-only transforms before PhysX starts. The
        left/control articulation is untouched.
        """

        import math

        from pxr import Gf, UsdGeom, UsdPhysics

        def pose_matrix(position, quat):
            if hasattr(quat, "GetReal"):
                quat = Gf.Quatd(
                    float(quat.GetReal()),
                    Gf.Vec3d(*[float(value) for value in quat.GetImaginary()]),
                )
            else:
                quat = Gf.Quatd(
                    float(quat[0]),
                    Gf.Vec3d(float(quat[1]), float(quat[2]), float(quat[3])),
                )
            return Gf.Matrix4d(
                Gf.Matrix3d(quat),
                Gf.Vec3d(*[float(value) for value in position]),
            )

        def axis_rotation(axis, angle_rad):
            rotation = Gf.Rotation(
                Gf.Vec3d(*[float(value) for value in axis]),
                math.degrees(float(angle_rad)),
            )
            return Gf.Matrix4d(Gf.Matrix3d(rotation), Gf.Vec3d(0.0, 0.0, 0.0))

        def axis_translation(axis, distance):
            matrix = Gf.Matrix4d(1.0)
            matrix.SetTranslate(
                Gf.Vec3d(*[float(value) * float(distance) for value in axis])
            )
            return matrix

        def local_matrix(path):
            return UsdGeom.Xformable(stage.GetPrimAtPath(path)).GetLocalTransformation()

        def write_matrix(path, matrix):
            prim = stage.OverridePrim(path)
            xform = UsdGeom.Xformable(prim)
            for op in xform.GetOrderedXformOps():
                if op.GetOpName() == "xformOp:transform:pi0_static_pose":
                    op.Set(matrix)
                    return
            xform.ClearXformOpOrder()
            xform.AddTransformOp(
                precision=UsdGeom.XformOp.PrecisionDouble,
                opSuffix="pi0_static_pose",
            ).Set(matrix)

        q = [float(value) for value in joint_positions]
        if len(q) != 7:
            raise ValueError(
                "background_right_robot_joint_pos must contain exactly 7 values"
            )

        # Gf matrices use USD's row-vector convention. This reproduces every
        # source link transform when q is all zero.
        link7_path = f"{robot_path}/panda_link7"
        link7_source = local_matrix(link7_path)
        link_poses = []
        parent_pose = Gf.Matrix4d(1.0)
        for joint_index, joint_angle in enumerate(q, start=1):
            joint_path = f"{robot_path}/joints/panda_joint{joint_index}"
            if not stage.GetPrimAtPath(joint_path).IsValid():
                # franka_mimic.usd keeps each revolute joint below its
                # preceding link, while the background visual USD collects
                # them below /joints.
                joint_path = (
                    f"{robot_path}/panda_link{joint_index - 1}/"
                    f"panda_joint{joint_index}"
                )
            joint = UsdPhysics.RevoluteJoint(stage.GetPrimAtPath(joint_path))
            if not joint.GetPrim().IsValid():
                raise RuntimeError(f"Missing right-arm joint: {joint_path}")
            frame0 = pose_matrix(
                joint.GetLocalPos0Attr().Get(), joint.GetLocalRot0Attr().Get()
            )
            frame1 = pose_matrix(
                joint.GetLocalPos1Attr().Get(), joint.GetLocalRot1Attr().Get()
            )
            parent_pose = (
                frame1.GetInverse()
                * axis_rotation((0.0, 0.0, 1.0), joint_angle)
                * frame0
                * parent_pose
            )
            link_poses.append(parent_pose)
            write_matrix(f"{robot_path}/panda_link{joint_index}", parent_pose)

        # Preserve the source finger geometry and apply the rollout open
        # displacement to each direct finger root.
        link7_pose = link_poses[-1]
        wrist_delta = link7_source.GetInverse() * link7_pose
        for finger_index in (1, 2):
            joint_path = f"{robot_path}/joints/panda_finger_joint{finger_index}"
            if not stage.GetPrimAtPath(joint_path).IsValid():
                joint_path = (
                    f"{robot_path}/panda_hand/"
                    f"panda_finger_joint{finger_index}"
                )
            joint = UsdPhysics.PrismaticJoint(stage.GetPrimAtPath(joint_path))
            finger_path = (
                f"{robot_path}/panda_leftfinger"
                if finger_index == 1
                else f"{robot_path}/panda_rightfinger"
            )
            frame0 = pose_matrix(
                joint.GetLocalPos0Attr().Get(), joint.GetLocalRot0Attr().Get()
            )
            frame1 = pose_matrix(
                joint.GetLocalPos1Attr().Get(), joint.GetLocalRot1Attr().Get()
            )
            finger_pose = (
                frame1.GetInverse()
                * axis_translation((0.0, 1.0, 0.0), gripper_open)
                * frame0
                * link7_pose
            )
            write_matrix(finger_path, finger_pose)

        fingertip_path = f"{robot_path}/panda_fingertip_centered"
        write_matrix(fingertip_path, local_matrix(fingertip_path) * wrist_delta)
        if log:
            print(
                "[RealSim] Pi0 background visual pose: "
                f"q={q} gripper_open={float(gripper_open):.4f} "
                f"link7_local={tuple(round(float(value), 4) for value in link7_pose.ExtractTranslation())}",
                flush=True,
            )

    @staticmethod
    def _solve_background_robot_endpoint_pose(stage, robot_path: str, cfg) -> None:
        """Solve one fixed endpoint pose for the visual-only right arm."""

        endpoint_pos = np.asarray(
            getattr(cfg, "background_right_robot_endpoint_reference_offset_pos", ()),
            dtype=np.float64,
        )
        endpoint_rot = np.asarray(
            getattr(cfg, "background_right_robot_endpoint_target_rot", ()),
            dtype=np.float64,
        )
        if endpoint_pos.size != 3 or endpoint_rot.size != 9:
            return

        q_seed = np.asarray(
            getattr(cfg, "background_right_robot_joint_pos", ()),
            dtype=np.float64,
        )
        if q_seed.size != 7:
            return

        try:
            from .pi0_static_right_arm import solve_pose_ik

            q_pose, pos_residual, rot_residual = solve_pose_ik(
                stage,
                robot_path,
                endpoint_pos,
                endpoint_rot.reshape(3, 3),
                q_seed.reshape(7),
                max_iterations=500,
                nominal_weight=1.0e-5,
            )
        except Exception as exc:
            print(
                "[RealSim] WARNING: right visual endpoint IK failed; "
                f"using configured joint pose. {type(exc).__name__}: {exc}",
                flush=True,
            )
            return

        cfg.background_right_robot_joint_pos = tuple(float(value) for value in q_pose)
        print(
            "[RealSim] Right visual endpoint IK: "
            f"target={tuple(float(value) for value in endpoint_pos)} "
            f"q={np.array2string(q_pose, precision=7)} "
            f"pos_residual={pos_residual:.6f} m "
            f"rot_residual={rot_residual:.6f} rad",
            flush=True,
        )

    def _setup_scene(self):
        # sensors
        # """Initialize simulation scene."""
        # spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg(), translation=(0.0, 0.0, -1.05))

        # # spawn a usd file of a table into the scene
        # cfg = sim_utils.UsdFileCfg(usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/SeattleLabTable/table_instanceable.usd")
        # cfg.func(
        #     "/World/envs/env_.*/Table", cfg, translation=(0.55, 0.0, 0.0), orientation=(0.70711, 0.0, 0.0, 0.70711)
        # )
        
         # spawn green screen studio
        default_background_usd = Path(__file__).resolve().parents[4] / "franka_env_background_edit" / "franka_env.usd"
        env_cfg = sim_utils.UsdFileCfg(
            usd_path=os.environ.get(
                "TACEX_BACKGROUND_USD",
                str(default_background_usd),
            )
        )

        env_cfg.func(
            "/World/envs/env_.*/franka_env",
            env_cfg,
            translation=tuple(getattr(self.cfg, "robot_base_pos", (0.0, 0.0, 0.0))),
            orientation=tuple(getattr(self.cfg, "robot_base_rot", (1.0, 0.0, 0.0, 0.0))),
            # translation=(1.85718, 0.36375, 0.037),
            # # translation=(0.2005, -1.29732, 0.11821),
            # # orientation=( 0.0,0.0357, -0.9993, 0.0),
            # orientation=( 0.0,0.70711, -0.70711, 0.0),
        )

        if getattr(self.cfg, "pi0_green_table_overlay_enabled", False):
            self._create_pi0_green_table_overlay()
            if getattr(self.cfg, "pi0_green_table_wrist_overlay_enabled", False):
                self._create_pi0_wrist_green_table_overlay()
        if getattr(self.cfg, "pi0_white_table_overlay_enabled", False):
            self._create_pi0_white_table_overlay()
        if getattr(self.cfg, "pi0_reference_cylinder_enabled", False):
            self._create_pi0_reference_cylinder()
        if getattr(self.cfg, "pi0_cable_grommets_enabled", False):
            self._create_pi0_cable_grommets()

        # Some captured backgrounds include a second authored Franka. Paired
        # replay can remove it before PhysX/Fabric scene finalization so only
        # the controlled left articulation remains.
        if getattr(self.cfg, "remove_authored_right_robot", False):
            authored_right_path = "/World/envs/env_0/franka_env/fr3v2_01"
            authored_right = self.sim.stage.GetPrimAtPath(authored_right_path)
            if authored_right.IsValid():
                self.sim.stage.OverridePrim(authored_right_path).SetActive(False)
                print(
                    f"[RealSim] Deactivated authored right robot: {authored_right_path}"
                )

        # The visual background used for the cameras may already contain a
        # static Franka at the same prim path as cfg.robot.  PPO must own the
        # only /Robot/franka articulation; otherwise the static background
        # model hides the movable articulation and PhysX reports a duplicate.
        if getattr(self.cfg, "remove_background_robot", False):
            background_robot_path = "/World/envs/env_0/franka_env/Robot/franka"
            background_robot = self.sim.stage.GetPrimAtPath(background_robot_path)
            if background_robot.IsValid():
                self.sim.stage.RemovePrim(background_robot_path)
                print(f"[RealSim] Removed static background robot: {background_robot_path}")
        elif getattr(self.cfg, "background_robot_visual_only", False):
            self._make_background_robot_visual_only(
                self.sim.stage,
                "/World/envs/env_0/franka_env/Robot/franka",
            )

        if getattr(self.cfg, "replay_left_background_visual_shell", False):
            left_visual_root = "/World/envs/env_0/franka_env/LeftRobotVisual/Robot"
            self._spawn_background_robot_visual(
                self.sim.stage,
                str(env_cfg.usd_path),
                "/World/Robot",
                left_visual_root,
                (0.0, 0.0, 0.0),
            )
            self._make_background_robot_visual_only(
                self.sim.stage,
                left_visual_root + "/franka",
            )
            self._pose_background_robot_visual(
                self.sim.stage,
                left_visual_root + "/franka",
                self.cfg.ctrl.reset_joints,
                0.04,
            )

        if getattr(self.cfg, "background_right_robot_visual_only", False):
            right_robot_root = "/World/envs/env_0/franka_env/RightRobot/Robot"
            copy_active_visual = bool(
                getattr(self.cfg, "background_right_robot_copy_active_visual", False)
            )
            if copy_active_visual:
                right_source_usd = str(
                    Path(__file__).resolve().parents[4]
                    / "assets"
                    / "Factory"
                    / "franka_mimic.usd"
                )
                right_source_prim = "/panda"
            else:
                right_source_usd = str(
                    getattr(
                        self.cfg,
                        "background_right_robot_visual_usd",
                        env_cfg.usd_path,
                    )
                )
                right_source_prim = str(
                    getattr(
                        self.cfg,
                        "background_right_robot_visual_prim",
                        "/World/Robot",
                    )
                )
            self._spawn_background_robot_visual(
                self.sim.stage,
                right_source_usd,
                right_source_prim,
                right_robot_root,
                getattr(self.cfg, "background_right_robot_offset_pos", (0.45, 0.0, 0.0)),
            )
            if copy_active_visual:
                # The factory Franka is referenced after scene creation; load
                # it before traversing joints and authoring the static pose.
                self.sim.stage.Load()
                print(
                    f"[RealSim] Loaded copied right-arm asset: {right_source_usd}",
                    flush=True,
                )
            if copy_active_visual:
                self._spawn_pi0_black_gripper_visual(
                    self.sim.stage,
                    str(default_background_usd),
                    right_robot_root + "/franka/panda_hand",
                )
            # Convert the configured endpoint target into one fixed q before
            # stripping physics from the visual-only right-arm copy. The live
            # PPO left articulation is not involved in this solve.
            self._solve_background_robot_endpoint_pose(
                self.sim.stage,
                right_robot_root + "/franka",
                self.cfg,
            )
            self._make_background_robot_visual_only(
                self.sim.stage,
                right_robot_root + "/franka",
            )
            right_pose = getattr(self.cfg, "background_right_robot_joint_pos", None)
            if right_pose is not None and not bool(
                getattr(self.cfg, "background_right_robot_preposed", False)
            ):
                self._pose_background_robot_visual(
                    self.sim.stage,
                    right_robot_root + "/franka",
                    right_pose,
                    getattr(self.cfg, "background_right_robot_gripper_open", 0.04),
                )

        # The Pi0 scene uses a precomputed, physics-free right-arm asset.
        # Its root transform contains the calibrated base orientation; this
        # wrapper supplies the calibrated base position in franka_env.
        if getattr(self.cfg, "background_fr3v2_right_robot_visual_only", False):
            original_right_root = "/World/envs/env_0/franka_env/fr3v2_01"
            original_right = self.sim.stage.GetPrimAtPath(original_right_root)
            if original_right.IsValid():
                self.sim.stage.OverridePrim(original_right_root).SetActive(False)

            right_root = "/World/envs/env_0/franka_env/RightRobot/Robot"
            right_source = getattr(self, "_pi0_static_right_arm_usd", default_background_usd)
            self._spawn_background_robot_visual(
                self.sim.stage,
                str(right_source),
                "/Root",
                right_root,
                getattr(
                    self.cfg,
                    "background_fr3v2_right_robot_pos",
                    (0.6658084946, -0.08782, 0.0991799997),
                ),
            )

        robot_prim_path = getattr(self.cfg.robot, "prim_path", "")
        if "/ActiveRobot/" in robot_prim_path:
            active_robot_parent = (
                "/World/envs/env_0/franka_env/ActiveRobot"
                if "/franka_env/ActiveRobot/" in robot_prim_path
                else "/World/envs/env_0/ActiveRobot"
            )
            sim_utils.create_prim(
                active_robot_parent,
                "Xform",
                translation=tuple(getattr(self.cfg, "active_robot_base_pos", (0.0, 0.0, 0.0))),
                orientation=tuple(getattr(self.cfg, "active_robot_base_rot", (1.0, 0.0, 0.0, 0.0))),
            )

        # Replay subclasses may author rigid tool geometry under a robot link
        # before the articulation is constructed. This keeps the tool in the
        # same PhysX articulation and lets incoming joint forces carry contact
        # reactions to the force-sensor link.
        prepare_robot_usd = getattr(self, "_prepare_robot_usd_for_replay", None)
        if prepare_robot_usd is not None:
            prepare_robot_usd()

        self._robot = Articulation(self.cfg.robot)
        if getattr(self.cfg, "replay_left_background_visual_shell", False):
            active_robot_path = str(self.cfg.robot.prim_path).replace(
                "env_.*", "env_0"
            )
            self._hide_active_robot_render_meshes(
                self.sim.stage,
                active_robot_path,
            )
        if getattr(self.cfg, "pi0_black_gripper_visual", False):
            robot_instance_path = "/World/envs/env_0/franka_env/Robot/franka"
            if "/ActiveRobot/" in getattr(self.cfg.robot, "prim_path", ""):
                robot_instance_path = getattr(self.cfg.robot, "prim_path", "").replace(
                    "env_.*", "env_0"
                )
            self._spawn_pi0_black_gripper_visual(
                self.sim.stage,
                str(Path(__file__).resolve().parents[4] / "franka_env_background_edit" / "franka_env.usd"),
                f"{robot_instance_path}/panda_hand",
            )
        # Kinematic TAVLA mode uses a second Franka only as a collision-free
        # kinematic model. It is spawned outside the camera view and never
        # receives task commands.
        if getattr(self.cfg, "teacher_control_mode", "") == "kinematic_taskspace":
            twin_cfg = copy.deepcopy(self.cfg.robot)
            sim_utils.create_prim("/World/envs/env_0/tavla_kinematic", "Xform")
            twin_cfg.prim_path = "/World/envs/env_.*/tavla_kinematic/franka"
            twin_cfg.init_state.pos = (10.0, 10.0, 0.0)
            twin_cfg.init_state.rot = (1.0, 0.0, 0.0, 0.0)
            self._tavla_twin_robot = Articulation(twin_cfg)
        self._fixed_asset = Articulation(self.cfg_task.fixed_asset)
        self._held_asset = Articulation(self.cfg_task.held_asset)
        if self.cfg_task.name == "gear_mesh":
            self._small_gear_asset = Articulation(self.cfg_task.small_gear_cfg)
            self._large_gear_asset = Articulation(self.cfg_task.large_gear_cfg)

        self.scene.clone_environments(copy_from_source=False)
        if (
            getattr(self.cfg, "remove_background_robot", False)
            and "/Robot/franka" not in getattr(self.cfg.robot, "prim_path", "")
        ):
            duplicated_background_path = "/World/envs/env_0/franka_env/Robot/franka"
            duplicated_background = self.sim.stage.GetPrimAtPath(duplicated_background_path)
            if duplicated_background.IsValid():
                self.sim.stage.OverridePrim(duplicated_background_path).SetActive(False)
                print(f"[RealSim] Disabled duplicated background robot: {duplicated_background_path}")
        if getattr(self.cfg, "background_robot_visual_only", False):
            self._make_background_robot_visual_only(
                self.sim.stage,
                "/World/envs/env_0/franka_env/Robot/franka",
            )
        if getattr(self.cfg, "background_right_robot_visual_only", False):
            self._make_background_robot_visual_only(
                self.sim.stage,
                "/World/envs/env_0/franka_env/RightRobot/Robot/franka",
            )
        if hasattr(self, "_tavla_twin_robot"):
            # The twin is a purely kinematic coordinate frame. Keep it out of
            # camera images and contact dynamics while retaining its PhysX
            # articulation state for Jacobian/IK queries.
            from pxr import UsdGeom, UsdPhysics

            for prim in self.sim.stage.Traverse():
                prim_path = str(prim.GetPath())
                if "/tavla_kinematic" not in prim_path:
                    continue
                if prim.IsA(UsdGeom.Imageable):
                    UsdGeom.Imageable(prim).MakeInvisible()
                if prim.HasAPI(UsdPhysics.CollisionAPI):
                    UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Set(False)
        if self.device == "cpu":
            # we need to explicitly filter collisions for CPU simulation
            self.scene.filter_collisions()

        self.scene.articulations["robot"] = self._robot
        if hasattr(self, "_tavla_twin_robot"):
            self.scene.articulations["tavla_twin_robot"] = self._tavla_twin_robot
        self.scene.articulations["fixed_asset"] = self._fixed_asset
        self.scene.articulations["held_asset"] = self._held_asset
        if self.cfg_task.name == "gear_mesh":
            self.scene.articulations["small_gear"] = self._small_gear_asset
            self.scene.articulations["large_gear"] = self._large_gear_asset

        if self.cfg.override_held_asset_color:
            self._override_asset_visual_color("HeldAsset", self.cfg.held_asset_visual_color)
        if self.cfg.override_fixed_asset_color:
            self._override_asset_visual_color("FixedAsset", self.cfg.fixed_asset_visual_color)

        # Pi0 visual matching can opt into one explicit light. The imported
        # background USD currently contains two RectLights and one DistantLight
        # in addition to this light, which makes the RGB observation much
        # brighter than the real camera image.
        if getattr(self.cfg, "pi0_disable_background_lights", False):
            self._disable_background_lights_for_pi0()
        light_cfg = sim_utils.DomeLightCfg(
            intensity=float(getattr(self.cfg, "pi0_dome_light_intensity", 2000.0)),
            color=tuple(getattr(self.cfg, "pi0_dome_light_color", (0.75, 0.75, 0.75))),
        )
        light_cfg.func("/World/Light", light_cfg)
        

        # from pxr import UsdGeom

        # prim = self.sim.stage.GetPrimAtPath("/World/envs/env_0/Table")
        # imageable = UsdGeom.Imageable(prim)

        # imageable.MakeInvisible()

        enable_camera_sensors = (
            bool(self.cfg.data_collect_cfg.get("collect_data", False))
            or self.cfg.policy_cfg is not None
            or getattr(self.cfg, "teacher_policy_cfg", None) is not None
            or bool(getattr(self.cfg, "enable_cameras", False))
        )
        if enable_camera_sensors and hasattr(self.cfg, "wrist_camera") and self.cfg.wrist_camera is not None:
            self.wrist_tiled_camera = self.cfg.wrist_camera.class_type(self.cfg.wrist_camera)
            self.scene.sensors["wrist_tiled_camera"] = self.wrist_tiled_camera

        if enable_camera_sensors and hasattr(self.cfg, "tiled_camera") and self.cfg.tiled_camera is not None:
            self.tiled_camera = self.cfg.tiled_camera.class_type(self.cfg.tiled_camera)
            self.scene.sensors["tiled_camera"] = self.tiled_camera

        if getattr(self.cfg, "pi0_disable_background_lights", False):
            self._apply_pi0_camera_exposure()
        
    def _override_asset_visual_color(self, asset_name: str, color: tuple[float, float, float]):
        from pxr import Gf, Sdf, UsdGeom, UsdShade

        stage = self.sim.stage
        material_path = f"/World/Looks/{asset_name.lower()}_color_override"
        material = UsdShade.Material.Define(stage, material_path)
        shader = UsdShade.Shader.Define(stage, f"{material_path}/PreviewSurface")
        shader.CreateIdAttr("UsdPreviewSurface")
        shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
        shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.55)
        shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)
        material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")

        asset_path_fragment = f"/{asset_name}/"
        for prim in stage.Traverse():
            path = str(prim.GetPath())
            if asset_path_fragment not in path or not prim.IsA(UsdGeom.Mesh):
                continue
            UsdShade.MaterialBindingAPI.Apply(prim).Bind(material)

    def _create_pi0_green_table_overlay(self) -> None:
        """Create the green tabletop from its annotated front-view hexagon."""

        from pxr import Gf, Sdf, Usd, UsdGeom, UsdShade

        image_polygon = tuple(
            (float(point[0]), float(point[1]))
            for point in getattr(
                self.cfg,
                "pi0_green_table_overlay_image_polygon",
                (),
            )
        )
        if len(image_polygon) != 6:
            raise ValueError(
                "pi0_green_table_overlay_image_polygon must contain six pixels"
            )

        hole_pos = getattr(
            self.cfg.task.fixed_asset.init_state,
            "pos",
            (0.0, 0.0, 0.0),
        )
        surface_z = float(hole_pos[2]) + float(
            getattr(self.cfg, "pi0_green_table_overlay_z_offset", -0.003)
        )
        stage = self.sim.stage
        camera_prim = next(
            (
                prim
                for prim in stage.Traverse()
                if prim.GetTypeName() == "Camera"
                and str(prim.GetPath()).endswith("/front_camera")
            ),
            None,
        )
        if camera_prim is None:
            raise RuntimeError(
                "Cannot create green tabletop: front_camera prim was not found"
            )

        camera = UsdGeom.Camera(camera_prim)
        width = float(getattr(self.cfg, "pi0_visual_annotation_width", 640))
        height = float(getattr(self.cfg, "pi0_visual_annotation_height", 480))
        focal_length = float(camera.GetFocalLengthAttr().Get())
        horizontal_aperture = float(camera.GetHorizontalApertureAttr().Get())
        vertical_aperture = float(camera.GetVerticalApertureAttr().Get())
        if min(width, height, focal_length, horizontal_aperture, vertical_aperture) <= 0.0:
            raise ValueError("Invalid front-camera projection parameters")

        fx = width * focal_length / horizontal_aperture
        fy = height * focal_length / vertical_aperture
        cx = 0.5 * width
        cy = 0.5 * height
        camera_to_world = UsdGeom.Xformable(camera_prim).ComputeLocalToWorldTransform(
            Usd.TimeCode.Default()
        )
        camera_origin = camera_to_world.Transform(Gf.Vec3d(0.0, 0.0, 0.0))
        def project_pixels(pixel_polygon, target_z):
            world_polygon = []
            for pixel_x, pixel_y in pixel_polygon:
                ray_local = Gf.Vec3d(
                    (pixel_x - cx) / fx,
                    -(pixel_y - cy) / fy,
                    -1.0,
                )
                ray_world = camera_to_world.TransformDir(ray_local)
                ray_z = float(ray_world[2])
                if abs(ray_z) < 1.0e-8:
                    raise ValueError(
                        "Front-camera ray is parallel to the green tabletop"
                    )
                distance = (target_z - float(camera_origin[2])) / ray_z
                if distance <= 0.0:
                    raise ValueError(
                        "Green-table image corner lies behind the front camera"
                    )
                world_polygon.append(camera_origin + ray_world * distance)
            return world_polygon

        parent_path = "/World/envs/env_0/pi0_visual_matching"
        material_path = f"{parent_path}/green_table_material"
        material = UsdShade.Material.Define(stage, material_path)
        shader = UsdShade.Shader.Define(stage, f"{material_path}/PreviewSurface")
        shader.CreateIdAttr("UsdPreviewSurface")
        shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(
            Gf.Vec3f(
                *tuple(
                    float(value)
                    for value in getattr(
                        self.cfg,
                        "pi0_green_table_overlay_color",
                        (0.002, 0.025, 0.018),
                    )
                )
            )
        )
        shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(
            float(getattr(self.cfg, "pi0_green_table_overlay_roughness", 0.85))
        )
        shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)
        material.CreateSurfaceOutput().ConnectToSource(
            shader.ConnectableAPI(), "surface"
        )

        def clip_half_plane(polygon, axis: int, bound: float, keep_greater: bool):
            """Clip a convex image polygon against one axis-aligned boundary."""

            if not polygon:
                return []
            clipped = []
            previous = polygon[-1]
            previous_inside = (
                previous[axis] >= bound if keep_greater else previous[axis] <= bound
            )
            for current in polygon:
                current_inside = (
                    current[axis] >= bound if keep_greater else current[axis] <= bound
                )
                if current_inside != previous_inside:
                    denominator = current[axis] - previous[axis]
                    if abs(denominator) > 1.0e-8:
                        blend = (bound - previous[axis]) / denominator
                        clipped.append(
                            (
                                previous[0] + blend * (current[0] - previous[0]),
                                previous[1] + blend * (current[1] - previous[1]),
                            )
                        )
                if current_inside:
                    clipped.append(current)
                previous = current
                previous_inside = current_inside
            return clipped

        def clip_rect(polygon, xmin, xmax, ymin, ymax):
            result = list(polygon)
            for axis, bound, keep_greater in (
                (0, xmin, True),
                (0, xmax, False),
                (1, ymin, True),
                (1, ymax, False),
            ):
                result = clip_half_plane(result, axis, bound, keep_greater)
            return result

        def create_region(name: str, pixel_region):
            if len(pixel_region) < 3:
                return
            region = UsdGeom.Mesh.Define(stage, f"{parent_path}/{name}")
            # Pixel points are clockwise. Reverse them for an upward-facing mesh.
            region.CreatePointsAttr(list(reversed(project_pixels(pixel_region, surface_z))))
            region.CreateFaceVertexCountsAttr([len(pixel_region)])
            region.CreateFaceVertexIndicesAttr(list(range(len(pixel_region))))
            region.CreateNormalsAttr([Gf.Vec3f(0.0, 0.0, 1.0)])
            region.SetNormalsInterpolation("uniform")
            UsdShade.MaterialBindingAPI(region.GetPrim()).Bind(material)

        cutout = tuple(
            float(value)
            for value in getattr(
                self.cfg,
                "pi0_green_table_hole_cutout_xyxy",
                (),
            )
        )
        if len(cutout) == 4:
            left, top, right, bottom = cutout
            if not left < right or not top < bottom:
                raise ValueError("green-table hole cutout must satisfy x0 < x1 and y0 < y1")
            # Four non-overlapping clipped regions leave the hole's screen
            # region untouched while fully covering the rest of the hexagon.
            infinity = 1.0e9
            regions = (
                ("green_table_top", clip_rect(image_polygon, -infinity, infinity, -infinity, top)),
                ("green_table_bottom", clip_rect(image_polygon, -infinity, infinity, bottom, infinity)),
                ("green_table_left", clip_rect(image_polygon, -infinity, left, top, bottom)),
                ("green_table_right", clip_rect(image_polygon, right, infinity, top, bottom)),
            )
            for name, pixel_region in regions:
                create_region(name, pixel_region)
        elif len(cutout) == 0:
            create_region("green_table", image_polygon)
        else:
            raise ValueError("pi0_green_table_hole_cutout_xyxy must have four values")

        if getattr(self.cfg, "pi0_green_table_right_patch_enabled", False):
            right_patch_polygon = tuple(
                (float(point[0]), float(point[1]))
                for point in getattr(
                    self.cfg,
                    "pi0_green_table_right_patch_image_polygon",
                    (),
                )
            )
            if len(right_patch_polygon) < 3:
                raise ValueError(
                    "pi0_green_table_right_patch_image_polygon needs at least 3 pixels"
                )
            right_patch_z = float(hole_pos[2]) + float(
                getattr(self.cfg, "pi0_green_table_right_patch_z_offset", 0.001)
            )
            right_patch_points = project_pixels(right_patch_polygon, right_patch_z)
            right_patch = UsdGeom.Mesh.Define(
                stage,
                f"{parent_path}/green_table_right_patch",
            )
            right_patch.CreatePointsAttr(list(reversed(right_patch_points)))
            right_patch.CreateFaceVertexCountsAttr([len(right_patch_points)])
            right_patch.CreateFaceVertexIndicesAttr(
                list(range(len(right_patch_points)))
            )
            right_patch.CreateNormalsAttr([Gf.Vec3f(0.0, 0.0, 1.0)])
            right_patch.SetNormalsInterpolation("uniform")
            UsdShade.MaterialBindingAPI(right_patch.GetPrim()).Bind(material)

        print(
            "[Pi0Visual] Created green-table overlay: "
            f"image_polygon={image_polygon} cutout={cutout or None} "
            f"surface_z={surface_z:.6f}",
            flush=True,
        )

    def _create_pi0_wrist_green_table_overlay(self) -> None:
        """Create one static, wrist-only green tabletop coverage mesh.

        The normal green tabletop is sized from the annotated front image and
        must stay tight for the front-camera match.  The wrist camera moves
        with the hand and can see beyond that polygon, so this mesh expands
        the same horizontal tabletop footprint in world XY.  It is authored
        once during scene setup; camera observations only toggle its
        visibility before an already-required camera render.
        """

        from pxr import Gf, Sdf, Usd, UsdGeom, UsdShade

        stage = self.sim.stage
        parent_path = "/World/envs/env_0/pi0_visual_matching"
        material_path = f"{parent_path}/green_table_material"
        material = UsdShade.Material.Get(stage, material_path)
        if not material.GetPrim().IsValid():
            raise RuntimeError(
                "Cannot create wrist green-table overlay before the base green-table material"
            )

        image_polygon = tuple(
            (float(point[0]), float(point[1]))
            for point in getattr(
                self.cfg,
                "pi0_green_table_overlay_image_polygon",
                (),
            )
        )
        if len(image_polygon) < 3:
            raise ValueError(
                "pi0_green_table_overlay_image_polygon needs at least three pixels"
            )

        hole_pos = getattr(
            self.cfg.task.fixed_asset.init_state,
            "pos",
            (0.0, 0.0, 0.0),
        )
        surface_z = float(hole_pos[2]) + float(
            getattr(self.cfg, "pi0_green_table_wrist_overlay_z_offset", -0.006)
        )
        camera_prim = next(
            (
                prim
                for prim in stage.Traverse()
                if prim.GetTypeName() == "Camera"
                and str(prim.GetPath()).endswith("/front_camera")
            ),
            None,
        )
        if camera_prim is None:
            raise RuntimeError(
                "Cannot create wrist green-table overlay: front-camera prim was not found"
            )

        camera = UsdGeom.Camera(camera_prim)
        width = float(getattr(self.cfg, "pi0_visual_annotation_width", 640))
        height = float(getattr(self.cfg, "pi0_visual_annotation_height", 480))
        focal_length = float(camera.GetFocalLengthAttr().Get())
        horizontal_aperture = float(camera.GetHorizontalApertureAttr().Get())
        vertical_aperture = float(camera.GetVerticalApertureAttr().Get())
        if min(width, height, focal_length, horizontal_aperture, vertical_aperture) <= 0.0:
            raise ValueError("Invalid front-camera projection parameters")

        fx = width * focal_length / horizontal_aperture
        fy = height * focal_length / vertical_aperture
        cx = 0.5 * width
        cy = 0.5 * height
        camera_to_world = UsdGeom.Xformable(camera_prim).ComputeLocalToWorldTransform(
            Usd.TimeCode.Default()
        )
        camera_origin = camera_to_world.Transform(Gf.Vec3d(0.0, 0.0, 0.0))
        table_points = []
        for pixel_x, pixel_y in image_polygon:
            ray_local = Gf.Vec3d(
                (pixel_x - cx) / fx,
                -(pixel_y - cy) / fy,
                -1.0,
            )
            ray_world = camera_to_world.TransformDir(ray_local)
            ray_z = float(ray_world[2])
            if abs(ray_z) < 1.0e-8:
                raise ValueError(
                    "Wrist green-table projection ray is parallel to the tabletop"
                )
            distance = (surface_z - float(camera_origin[2])) / ray_z
            if distance <= 0.0:
                raise ValueError(
                    "Wrist green-table projection lies behind the front camera"
                )
            point = camera_origin + ray_world * distance
            table_points.append(point)

        margin = float(
            getattr(self.cfg, "pi0_green_table_wrist_overlay_margin_m", 0.35)
        )
        if margin < 0.0:
            raise ValueError("pi0_green_table_wrist_overlay_margin_m must be non-negative")
        x_values = [float(point[0]) for point in table_points]
        y_values = [float(point[1]) for point in table_points]
        wrist_points = [
            Gf.Vec3d(min(x_values) - margin, min(y_values) - margin, surface_z),
            Gf.Vec3d(max(x_values) + margin, min(y_values) - margin, surface_z),
            Gf.Vec3d(max(x_values) + margin, max(y_values) + margin, surface_z),
            Gf.Vec3d(min(x_values) - margin, max(y_values) + margin, surface_z),
        ]

        overlay_path = f"{parent_path}/green_table_wrist"
        overlay = UsdGeom.Mesh.Define(stage, overlay_path)
        overlay.CreatePointsAttr(list(reversed(wrist_points)))
        overlay.CreateFaceVertexCountsAttr([len(wrist_points)])
        overlay.CreateFaceVertexIndicesAttr(list(range(len(wrist_points))))
        overlay.CreateNormalsAttr([Gf.Vec3f(0.0, 0.0, 1.0)])
        overlay.SetNormalsInterpolation("uniform")
        UsdShade.MaterialBindingAPI(overlay.GetPrim()).Bind(material)

        # Keep the larger mesh out of the front render.  _camera_observation
        # makes it visible only while updating the wrist camera and then
        # leaves the captured wrist tensor intact for data recording.
        UsdGeom.Imageable(overlay.GetPrim()).MakeInvisible()
        self._pi0_wrist_green_table_overlay_path = overlay_path
        print(
            "[Pi0Visual] Created static wrist green-table overlay: "
            f"margin_m={margin:.3f} surface_z={surface_z:.6f}",
            flush=True,
        )

    def _create_pi0_white_table_overlay(self) -> None:
        """Create the Pi0-only white tabletop using the real-view boundary."""

        from pxr import Gf, Sdf, Usd, UsdGeom, UsdShade

        hole_pos = getattr(
            self.cfg.task.fixed_asset.init_state,
            "pos",
            (0.0, 0.0, 0.0),
        )
        hole_z = float(hole_pos[2])
        surface_z = hole_z + float(
            getattr(self.cfg, "pi0_white_table_overlay_z_offset", -0.009)
        )

        stage = self.sim.stage
        parent_path = "/World/envs/env_0/pi0_visual_matching"
        region_path = f"{parent_path}/white_table"

        image_polygon = getattr(self.cfg, "pi0_white_table_overlay_image_polygon", ())
        image_polygon = tuple(
            (float(point[0]), float(point[1])) for point in image_polygon
        )

        # Convert the annotated front-camera pixels to a horizontal world
        # plane.  The background camera is authored in the imported USD, so
        # its actual transform must be queried after the background is loaded
        # rather than duplicated in the Pi0 config.
        camera_prim = next(
            (
                prim
                for prim in stage.Traverse()
                if prim.GetTypeName() == "Camera"
                and str(prim.GetPath()).endswith("/front_camera")
            ),
            None,
        )
        world_points = []
        if len(image_polygon) == 4 and camera_prim is not None:
            camera = UsdGeom.Camera(camera_prim)
            width = float(getattr(self.cfg, "pi0_visual_annotation_width", 640))
            height = float(getattr(self.cfg, "pi0_visual_annotation_height", 480))
            focal_length = float(camera.GetFocalLengthAttr().Get())
            horizontal_aperture = float(camera.GetHorizontalApertureAttr().Get())
            vertical_aperture = float(camera.GetVerticalApertureAttr().Get())
            if min(
                width,
                height,
                focal_length,
                horizontal_aperture,
                vertical_aperture,
            ) <= 0.0:
                raise ValueError("Invalid front-camera projection parameters")

            fx = width * focal_length / horizontal_aperture
            fy = height * focal_length / vertical_aperture
            cx = 0.5 * width
            cy = 0.5 * height
            camera_to_world = UsdGeom.Xformable(camera_prim).ComputeLocalToWorldTransform(
                Usd.TimeCode.Default()
            )
            camera_origin = camera_to_world.Transform(Gf.Vec3d(0.0, 0.0, 0.0))
            for pixel_x, pixel_y in image_polygon:
                ray_local = Gf.Vec3d(
                    (pixel_x - cx) / fx,
                    -(pixel_y - cy) / fy,
                    -1.0,
                )
                ray_world = camera_to_world.TransformDir(ray_local)
                ray_z = float(ray_world[2])
                if abs(ray_z) < 1.0e-8:
                    raise ValueError(
                        "Front-camera ray is parallel to the white tabletop plane"
                    )
                distance = (surface_z - float(camera_origin[2])) / ray_z
                if distance <= 0.0:
                    raise ValueError(
                        "White-table image corner lies behind the front camera"
                    )
                world_points.append(camera_origin + ray_world * distance)
        else:
            x0, x1, y0, y1 = tuple(
                float(value)
                for value in getattr(
                    self.cfg,
                    "pi0_white_table_overlay_bounds",
                    (-0.38, 0.85, -0.335, 0.40),
                )
            )
            if not (x1 > x0 and y1 > y0):
                raise ValueError(
                    "pi0_white_table_overlay_bounds must be (x0, x1, y0, y1)"
                )
            world_points = [
                Gf.Vec3d(x0, y0, surface_z),
                Gf.Vec3d(x1, y0, surface_z),
                Gf.Vec3d(x1, y1, surface_z),
                Gf.Vec3d(x0, y1, surface_z),
            ]

        # Image order is top-left, top-right, bottom-right, bottom-left.  A
        # positive-Z mesh normal needs the reverse order in world XY.
        region = UsdGeom.Mesh.Define(stage, region_path)
        region.CreatePointsAttr(
            [world_points[3], world_points[2], world_points[1], world_points[0]]
        )
        region.CreateFaceVertexCountsAttr([4])
        region.CreateFaceVertexIndicesAttr([0, 1, 2, 3])
        region.CreateNormalsAttr([Gf.Vec3f(0.0, 0.0, 1.0)])
        region.SetNormalsInterpolation("uniform")

        material_path = f"{parent_path}/white_table_material"
        material = UsdShade.Material.Define(stage, material_path)
        shader = UsdShade.Shader.Define(stage, f"{material_path}/PreviewSurface")
        shader.CreateIdAttr("UsdPreviewSurface")
        shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(
            Gf.Vec3f(
                *tuple(
                    float(value)
                    for value in getattr(
                        self.cfg,
                        "pi0_white_table_overlay_color",
                        (0.25, 0.25, 0.30),
                    )
                )
            )
        )
        shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(
            float(getattr(self.cfg, "pi0_white_table_overlay_roughness", 0.90))
        )
        shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)
        material.CreateSurfaceOutput().ConnectToSource(
            shader.ConnectableAPI(), "surface"
        )
        UsdShade.MaterialBindingAPI(region.GetPrim()).Bind(material)

        print(
            "[Pi0Visual] Created white-table overlay: "
            f"image_polygon={image_polygon} "
            f"world_points={[tuple(round(float(v), 6) for v in point) for point in world_points]} "
            f"surface_z={surface_z:.6f} "
            f"color={tuple(getattr(self.cfg, 'pi0_white_table_overlay_color', (0.25, 0.25, 0.30)))}",
            flush=True,
        )

    def _create_pi0_cable_grommets(self) -> None:
        """Create two flush, visual-only flexible cable grommets."""

        from pxr import Gf, Sdf, Usd, UsdGeom, UsdShade

        stage = self.sim.stage
        camera_prim = next(
            (
                prim
                for prim in stage.Traverse()
                if prim.GetTypeName() == "Camera"
                and str(prim.GetPath()).endswith("/front_camera")
            ),
            None,
        )
        if camera_prim is None:
            raise RuntimeError("Cannot place Pi0 cable grommets: front_camera was not found")

        camera = UsdGeom.Camera(camera_prim)
        width = float(getattr(self.cfg, "pi0_visual_annotation_width", 640))
        height = float(getattr(self.cfg, "pi0_visual_annotation_height", 480))
        focal_length = float(camera.GetFocalLengthAttr().Get())
        horizontal_aperture = float(camera.GetHorizontalApertureAttr().Get())
        vertical_aperture = float(camera.GetVerticalApertureAttr().Get())
        fx = width * focal_length / horizontal_aperture
        fy = height * focal_length / vertical_aperture
        camera_to_world = UsdGeom.Xformable(camera_prim).ComputeLocalToWorldTransform(
            Usd.TimeCode.Default()
        )
        camera_origin = camera_to_world.Transform(Gf.Vec3d(0.0, 0.0, 0.0))
        hole_pos = getattr(self.cfg.task.fixed_asset.init_state, "pos", (0.0, 0.0, 0.0))
        surface_z = float(hole_pos[2]) + float(
            getattr(self.cfg, "pi0_cable_grommet_surface_z_offset", -0.0147)
        )

        def material(path: str, color):
            result = UsdShade.Material.Define(stage, path)
            shader = UsdShade.Shader.Define(stage, f"{path}/PreviewSurface")
            shader.CreateIdAttr("UsdPreviewSurface")
            shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(
                Gf.Vec3f(*[float(value) for value in color])
            )
            shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.92)
            shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)
            result.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
            return result

        root_path = "/World/envs/env_0/pi0_visual_matching/cable_grommets"
        outer_material = material(
            f"{root_path}/outer_material", self.cfg.pi0_cable_grommet_outer_color
        )
        inner_material = material(
            f"{root_path}/inner_material", self.cfg.pi0_cable_grommet_inner_color
        )
        slit_material = material(
            f"{root_path}/slit_material", self.cfg.pi0_cable_grommet_slit_color
        )
        outer_radius = float(self.cfg.pi0_cable_grommet_outer_radius_m)
        inner_radius = float(self.cfg.pi0_cable_grommet_inner_radius_m)
        disc_height = 0.00020

        for index, (pixel_x, pixel_y) in enumerate(self.cfg.pi0_cable_grommet_center_pixels):
            ray_local = Gf.Vec3d(
                (float(pixel_x) - 0.5 * width) / fx,
                -(float(pixel_y) - 0.5 * height) / fy,
                -1.0,
            )
            ray_world = camera_to_world.TransformDir(ray_local)
            if abs(float(ray_world[2])) < 1.0e-8:
                raise ValueError("Cable-grommet ray is parallel to the tabletop")
            distance = (surface_z - float(camera_origin[2])) / float(ray_world[2])
            if distance <= 0.0:
                raise ValueError("Cable-grommet pixel lies behind the front camera")
            center = camera_origin + ray_world * distance
            item_path = f"{root_path}/grommet_{index}"
            item = UsdGeom.Xform.Define(stage, item_path)
            item.AddTranslateOp().Set(
                Gf.Vec3d(float(center[0]), float(center[1]), surface_z)
            )

            for name, radius, z_value, bound_material in (
                ("outer_ring", outer_radius, 0.5 * disc_height, outer_material),
                ("soft_center", inner_radius, 1.5 * disc_height, inner_material),
            ):
                cylinder = UsdGeom.Cylinder.Define(stage, f"{item_path}/{name}")
                cylinder.CreateAxisAttr(UsdGeom.Tokens.z)
                cylinder.CreateRadiusAttr(radius)
                cylinder.CreateHeightAttr(disc_height)
                UsdGeom.Xformable(cylinder.GetPrim()).AddTranslateOp().Set(
                    Gf.Vec3d(0.0, 0.0, z_value)
                )
                cylinder.GetPrim().CreateAttribute(
                    "primvars:doNotCastShadows", Sdf.ValueTypeNames.Bool
                ).Set(True)
                UsdShade.MaterialBindingAPI(cylinder.GetPrim()).Bind(bound_material)

            slit_length = 0.030
            slit_width = 0.0011
            slit_height = 0.00008
            for slit_index, angle_deg in enumerate((0.0, 90.0)):
                slit = UsdGeom.Cube.Define(stage, f"{item_path}/slit_{slit_index}")
                slit.CreateSizeAttr(1.0)
                slit_xform = UsdGeom.Xformable(slit.GetPrim())
                slit_xform.AddTranslateOp().Set(
                    Gf.Vec3d(0.0, 0.0, 2.0 * disc_height + 0.5 * slit_height)
                )
                slit_xform.AddRotateZOp().Set(angle_deg)
                slit_xform.AddScaleOp().Set(
                    Gf.Vec3d(slit_length, slit_width, slit_height)
                )
                slit.GetPrim().CreateAttribute(
                    "primvars:doNotCastShadows", Sdf.ValueTypeNames.Bool
                ).Set(True)
                UsdShade.MaterialBindingAPI(slit.GetPrim()).Bind(slit_material)

        print(
            "[RealSim] Added visual-only cable grommets at front pixels: "
            f"{tuple(self.cfg.pi0_cable_grommet_center_pixels)}"
        )

    def _create_pi0_reference_cylinder(self) -> None:
        """Create the real-scene black tube and silver flange as visual geometry."""

        from pxr import Gf, Sdf, Usd, UsdGeom, UsdShade

        stage = self.sim.stage
        camera_prim = next(
            (
                prim
                for prim in stage.Traverse()
                if prim.GetTypeName() == "Camera"
                and str(prim.GetPath()).endswith("/front_camera")
            ),
            None,
        )
        if camera_prim is None:
            raise RuntimeError(
                "Cannot create Pi0 reference cylinder: front_camera was not found"
            )

        camera = UsdGeom.Camera(camera_prim)
        width = float(getattr(self.cfg, "pi0_visual_annotation_width", 640))
        height = float(getattr(self.cfg, "pi0_visual_annotation_height", 480))
        focal_length = float(camera.GetFocalLengthAttr().Get())
        horizontal_aperture = float(camera.GetHorizontalApertureAttr().Get())
        vertical_aperture = float(camera.GetVerticalApertureAttr().Get())
        if min(
            width,
            height,
            focal_length,
            horizontal_aperture,
            vertical_aperture,
        ) <= 0.0:
            raise ValueError("Invalid front-camera projection parameters")

        pixel_x, pixel_y = tuple(
            float(value)
            for value in getattr(
                self.cfg,
                "pi0_reference_cylinder_base_pixel",
                (580.0, 361.0),
            )
        )
        hole_pos = getattr(
            self.cfg.task.fixed_asset.init_state,
            "pos",
            (0.0, 0.0, 0.0),
        )
        surface_z = float(hole_pos[2]) + float(
            getattr(
                self.cfg,
                "pi0_reference_cylinder_surface_z_offset",
                -0.0002,
            )
        )
        fx = width * focal_length / horizontal_aperture
        fy = height * focal_length / vertical_aperture
        camera_to_world = UsdGeom.Xformable(camera_prim).ComputeLocalToWorldTransform(
            Usd.TimeCode.Default()
        )
        camera_origin = camera_to_world.Transform(Gf.Vec3d(0.0, 0.0, 0.0))
        ray_local = Gf.Vec3d(
            (pixel_x - 0.5 * width) / fx,
            -(pixel_y - 0.5 * height) / fy,
            -1.0,
        )
        ray_world = camera_to_world.TransformDir(ray_local)
        if abs(float(ray_world[2])) < 1.0e-8:
            raise ValueError("Reference-cylinder ray is parallel to the tabletop")
        distance = (surface_z - float(camera_origin[2])) / float(ray_world[2])
        if distance <= 0.0:
            raise ValueError("Reference-cylinder pixel lies behind the front camera")
        base_center = camera_origin + ray_world * distance

        parent_path = "/World/envs/env_0/pi0_visual_matching/reference_cylinder"
        root = UsdGeom.Xform.Define(stage, parent_path)
        root.AddTranslateOp().Set(
            Gf.Vec3d(float(base_center[0]), float(base_center[1]), surface_z)
        )

        def create_material(path, color, roughness, metallic):
            material = UsdShade.Material.Define(stage, path)
            shader = UsdShade.Shader.Define(stage, f"{path}/PreviewSurface")
            shader.CreateIdAttr("UsdPreviewSurface")
            shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(
                Gf.Vec3f(*[float(value) for value in color])
            )
            shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(
                float(roughness)
            )
            shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(
                float(metallic)
            )
            material.CreateSurfaceOutput().ConnectToSource(
                shader.ConnectableAPI(), "surface"
            )
            return material

        metal_material = create_material(
            f"{parent_path}/metal_material",
            getattr(
                self.cfg,
                "pi0_reference_cylinder_metal_color",
                (0.18, 0.19, 0.18),
            ),
            roughness=0.85,
            metallic=0.0,
        )
        black_material = create_material(
            f"{parent_path}/black_material",
            getattr(
                self.cfg,
                "pi0_reference_cylinder_black_color",
                (0.006, 0.006, 0.005),
            ),
            roughness=0.58,
            metallic=0.05,
        )
        inner_material = create_material(
            f"{parent_path}/inner_material",
            getattr(
                self.cfg,
                "pi0_reference_cylinder_inner_color",
                (0.0005, 0.0005, 0.0005),
            ),
            roughness=1.0,
            metallic=0.0,
        )

        base_radius = float(
            getattr(self.cfg, "pi0_reference_cylinder_base_radius_m", 0.032)
        )
        base_height = float(
            getattr(self.cfg, "pi0_reference_cylinder_base_height_m", 0.008)
        )
        collar_radius = float(
            getattr(self.cfg, "pi0_reference_cylinder_collar_radius_m", 0.022)
        )
        collar_height = float(
            getattr(self.cfg, "pi0_reference_cylinder_collar_height_m", 0.010)
        )
        tube_radius = float(
            getattr(self.cfg, "pi0_reference_cylinder_tube_radius_m", 0.017)
        )
        tube_height = float(
            getattr(self.cfg, "pi0_reference_cylinder_tube_height_m", 0.052)
        )
        inner_radius = float(
            getattr(self.cfg, "pi0_reference_cylinder_inner_radius_m", 0.0115)
        )
        if not (
            base_radius > collar_radius > tube_radius > inner_radius > 0.0
            and min(base_height, collar_height, tube_height) > 0.0
        ):
            raise ValueError("Invalid Pi0 reference-cylinder dimensions")

        def create_cylinder_mesh(
            name,
            radius,
            cylinder_height,
            center_z,
            material,
            segments=48,
        ):
            """Create a closed polygon mesh without an analytic Cylinder prim."""

            radius = float(radius)
            half_height = 0.5 * float(cylinder_height)
            center_z = float(center_z)
            bottom_z = center_z - half_height
            top_z = center_z + half_height
            points = [
                Gf.Vec3f(0.0, 0.0, bottom_z),
                Gf.Vec3f(0.0, 0.0, top_z),
            ]
            for z_value in (bottom_z, top_z):
                for index in range(segments):
                    angle = 2.0 * np.pi * float(index) / float(segments)
                    points.append(
                        Gf.Vec3f(
                            radius * float(np.cos(angle)),
                            radius * float(np.sin(angle)),
                            z_value,
                        )
                    )

            bottom_start = 2
            top_start = 2 + segments
            face_counts = []
            face_indices = []
            for index in range(segments):
                next_index = (index + 1) % segments
                bottom = bottom_start + index
                bottom_next = bottom_start + next_index
                top = top_start + index
                top_next = top_start + next_index
                # Two side triangles, one bottom cap triangle, and one top
                # cap triangle. Winding points outward on every surface.
                face_counts.extend((3, 3, 3, 3))
                face_indices.extend(
                    (
                        bottom,
                        bottom_next,
                        top_next,
                        bottom,
                        top_next,
                        top,
                        0,
                        bottom_next,
                        bottom,
                        1,
                        top,
                        top_next,
                    )
                )

            mesh = UsdGeom.Mesh.Define(stage, f"{parent_path}/{name}")
            mesh.CreatePointsAttr(points)
            mesh.CreateFaceVertexCountsAttr(face_counts)
            mesh.CreateFaceVertexIndicesAttr(face_indices)
            mesh.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
            mesh.CreateExtentAttr(
                [
                    Gf.Vec3f(-radius, -radius, bottom_z),
                    Gf.Vec3f(radius, radius, top_z),
                ]
            )
            # The imported laboratory background is a Gaussian particle
            # field. Shadows received from ordinary meshes can become large,
            # noisy splat artifacts in the oblique wrist view. This prop is
            # visual context only, so keep it visible but exclude it from the
            # RTX shadow pass.
            mesh.GetPrim().CreateAttribute(
                "primvars:doNotCastShadows",
                Sdf.ValueTypeNames.Bool,
            ).Set(True)
            UsdShade.MaterialBindingAPI(mesh.GetPrim()).Bind(material)
            return mesh

        create_cylinder_mesh(
            "silver_flange",
            base_radius,
            base_height,
            0.5 * base_height,
            metal_material,
        )
        create_cylinder_mesh(
            "silver_collar",
            collar_radius,
            collar_height,
            base_height + 0.5 * collar_height,
            metal_material,
        )
        tube_bottom = base_height + collar_height
        create_cylinder_mesh(
            "black_tube",
            tube_radius,
            tube_height,
            tube_bottom + 0.5 * tube_height,
            black_material,
        )
        # A recessed near-black disk creates the visible hollow opening while
        # keeping the object lightweight and free of boolean geometry.
        create_cylinder_mesh(
            "tube_inner_dark",
            inner_radius,
            0.001,
            tube_bottom + tube_height + 0.00051,
            inner_material,
        )

        print(
            "[Pi0Visual] Created reference cylinder: "
            f"base_pixel=({pixel_x:.1f}, {pixel_y:.1f}) "
            f"base_world=({float(base_center[0]):.6f}, "
            f"{float(base_center[1]):.6f}, {surface_z:.6f})",
            flush=True,
        )

    def _disable_background_lights_for_pi0(self) -> None:
        """Disable lights authored by the imported background for Pi0 only."""

        from pxr import UsdLux

        disabled = []
        for prim in self.sim.stage.Traverse():
            if prim.GetTypeName() not in {
                "DomeLight",
                "DistantLight",
                "RectLight",
                "SphereLight",
                "DiskLight",
                "CylinderLight",
                "PortalLight",
            }:
                continue
            # The explicit /World/Light is created immediately after this
            # method, so every light found here belongs to the background.
            intensity_attr = UsdLux.LightAPI(prim).GetIntensityAttr()
            if intensity_attr.IsValid():
                intensity_attr.Set(0.0)
                disabled.append(str(prim.GetPath()))
        print(
            "[Pi0Visual] Disabled background lights: "
            + (", ".join(disabled) if disabled else "none"),
            flush=True,
        )

    def _apply_pi0_camera_exposure(self) -> None:
        """Apply fixed RTX exposure and disable auto exposure on Pi0 cameras."""

        from pxr import Sdf

        exposure_by_suffix = {
            "/front_camera": float(getattr(self.cfg, "pi0_front_camera_exposure", -1.0)),
            "/wrist_camera": float(getattr(self.cfg, "pi0_wrist_camera_exposure", -2.0)),
        }
        applied = []
        for prim in self.sim.stage.Traverse():
            if prim.GetTypeName() != "Camera":
                continue
            path = str(prim.GetPath())
            exposure = next(
                (value for suffix, value in exposure_by_suffix.items() if path.endswith(suffix)),
                None,
            )
            if exposure is None:
                continue
            prim.AddAppliedSchema("OmniRtxCameraExposureAPI_1")
            prim.AddAppliedSchema("OmniRtxCameraAutoExposureAPI_1")
            prim.CreateAttribute("exposure", Sdf.ValueTypeNames.Float).Set(exposure)
            prim.CreateAttribute(
                "omni:rtx:autoExposure:enabled",
                Sdf.ValueTypeNames.Bool,
            ).Set(False)
            applied.append(f"{path}={exposure:.2f}EV")
        print(
            "[Pi0Visual] Fixed camera exposure: "
            + (", ".join(applied) if applied else "none"),
            flush=True,
        )

    def record_data(self, env_idx=None):
        """
        Record simulation data for one or all environments.
        
        Args:
            env_idx (int, optional): Index of the environment to record data for.
                                    If None, record data for all environments.
        """
        if not self.collect_data:
            return
            
        record_mask = getattr(self, "_record_data_mask", None)
        if env_idx is not None and record_mask is not None and not bool(record_mask[env_idx].item()):
            return

        if env_idx is not None:
            buf = self.data_buffers[env_idx]
            
            # Raw sensor videos are optional diagnostics. Evaluation saves the
            # exact policy-input tensors below instead.
            if self.save_raw_camera_video and hasattr(self, "tiled_camera") and self.tiled_camera is not None:
                buf["camera"]["front"].append(
                    self.tiled_camera.data.output["rgb"][env_idx].to("cpu").clone()
                )
            if self.save_raw_camera_video and hasattr(self, "wrist_tiled_camera") and self.wrist_tiled_camera is not None:
                wrist_frame = self.wrist_tiled_camera.data.output["rgb"][env_idx].to("cpu").clone()
                clean_wrist_frame = getattr(self, "_clean_wrist_camera_frame", None)
                if callable(clean_wrist_frame):
                    wrist_frame = clean_wrist_frame(wrist_frame)
                buf["camera"]["wrist"].append(wrist_frame)
            if self.save_policy_input_video and getattr(self, "_model_visual_frame_ready", False):
                buf["camera"]["front_transformed"].append(
                    self.last_model_input_front.to("cpu").clone()
                )
                buf["camera"]["wrist_transformed"].append(
                    self.last_model_input_wrist.to("cpu").clone()
                )
            
            # Record joint states
            buf["joints"].append(self.joint_pos[env_idx].to("cpu").clone())
            
            # Record gripper state
            if hasattr(self, "_current_tavla_state"):
                gripper = float(self._current_tavla_state()[env_idx, 7].detach().cpu())
            else:
                gripper = float(
                    torch.clamp(self.joint_pos[env_idx, 7:9].mean() / 0.04, 0.0, 1.0).detach().cpu()
                )
            buf["gripper"].append(gripper)
            
            # Record end-effector pose
            ee_pose = torch.cat([
                self.fingertip_midpoint_pos[env_idx],
                self.fingertip_midpoint_quat[env_idx]
            ], dim=0).to("cpu")
            buf["ee_pose"].append(ee_pose)
            # Isaac quaternion order is wxyz; the alignment/HDF5 contract is xyzw.
            buf["ee_pose_xyzw"].append(torch.cat([
                ee_pose[:3], ee_pose[4:7], ee_pose[3:4]
            ]).clone())

            # Preserve legacy model/tool streams and record the explicit wrench contract.
            buf["force"].append(self.wrench_model[env_idx].to("cpu").clone())
            buf["force_world"].append(self.force_sensor_parent_smooth[env_idx].to("cpu").clone())
            buf["force_parent"].append(self.force_sensor_parent_smooth[env_idx].to("cpu").clone())
            buf["force_tool"].append(self.wrench_tool_smooth[env_idx].to("cpu").clone())
            buf["force_model"].append(self.wrench_model[env_idx].to("cpu").clone())
            buf["wrench_raw"].append(self.wrench_raw[env_idx].to("cpu").clone())
            buf["wrench_anchor"].append(self.wrench_anchor[env_idx].to("cpu").clone())
            buf["wrench_base"].append(self.wrench_base[env_idx].to("cpu").clone())
            buf["wrench_corrected"].append(self.wrench_corrected[env_idx].to("cpu").clone())
            buf["wrench_final"].append(self.wrench_final[env_idx].to("cpu").clone())
            timestamp = float(self.episode_length_buf[env_idx].float().cpu() * self.step_dt)
            buf["timestamps"].append(timestamp)
            if hasattr(self, "actions"):
                buf["actions"].append(self.actions[env_idx].to("cpu").clone())
            if hasattr(self, "ppo_joint_target"):
                buf["ppo_joint_targets"].append(self.ppo_joint_target[env_idx].to("cpu").clone())
            if hasattr(self, "reward_buf") and record_mask is not None:
                buf["rewards"].append(float(self.reward_buf[env_idx].detach().cpu()))
                reward_terms = {}
                for name, value in getattr(self, "extras", {}).items():
                    if not name.startswith("logs_rew_"):
                        continue
                    if torch.is_tensor(value):
                        value = value.detach().reshape(-1)
                        if env_idx < value.numel():
                            reward_terms[name[len("logs_rew_"):]] = float(value[env_idx].cpu())
                    else:
                        try:
                            reward_terms[name[len("logs_rew_"):]] = float(value)
                        except (TypeError, ValueError):
                            pass
                buf["reward_terms"].append(reward_terms)
            if hasattr(self, "teacher_target") and hasattr(self, "residual_action"):
                buf["tavla_teacher_actions"].append(self.teacher_target[env_idx].to("cpu").clone())
                buf["tavla_residual_actions"].append(self.residual_action[env_idx].to("cpu").clone())
                buf["tavla_wrench_base"].append(self.wrench_base[env_idx].to("cpu").clone())
                buf["tavla_wrench_final"].append(self.wrench_final[env_idx].to("cpu").clone())
                if hasattr(self, "last_tavla_effort"):
                    buf["tavla_policy_wrench"].append(self.last_tavla_effort[env_idx].to("cpu").clone())
                if hasattr(self, "last_tavla_force_gate_active"):
                    buf["tavla_force_gate_active"].append(
                        float(self.last_tavla_force_gate_active[env_idx].detach().cpu())
                    )
                if hasattr(self, "last_tavla_force_gate_contact_norm"):
                    buf["tavla_force_gate_contact_norm"].append(
                        float(self.last_tavla_force_gate_contact_norm[env_idx].detach().cpu())
                    )
                if hasattr(self, "last_tavla_server_effort"):
                    buf["tavla_server_effort"].append(self.last_tavla_server_effort[env_idx].to("cpu").clone())
                    buf["tavla_server_effort_matches_final"].append(
                        float(self.last_tavla_server_effort_matches_final[env_idx].detach().cpu())
                    )
                if hasattr(self, "last_tavla_actual_state"):
                    buf["tavla_actual_state"].append(self.last_tavla_actual_state[env_idx].to("cpu").clone())
                    buf["tavla_policy_state"].append(self.last_tavla_policy_state[env_idx].to("cpu").clone())
                if hasattr(self, "combined_joint_target"):
                    buf["tavla_combined_targets"].append(self.combined_joint_target[env_idx].to("cpu").clone())
                    buf["tavla_executed_targets"].append(self.combined_joint_target[env_idx].to("cpu").clone())
                if hasattr(self, "_teacher_command_target"):
                    buf["tavla_command_targets"].append(
                        self._teacher_command_target[env_idx].to("cpu").clone()
                    )
                    buf["tavla_command_velocity"].append(
                        self._teacher_command_velocity[env_idx].to("cpu").clone()
                    )
                if hasattr(self, "teacher_action_index"):
                    buf["tavla_action_indices"].append(
                        int(self.teacher_action_index[env_idx].detach().cpu())
                    )
                if hasattr(self, "last_teacher_inference_event"):
                    buf["tavla_inference_events"].append(int(self.last_teacher_inference_event))
                    buf["tavla_inference_latency_s"].append(float(self.teacher_inference_latency_s))
                    buf["tavla_inference_timeouts"].append(int(self.last_teacher_inference_timeout))
                    buf["tavla_action_nonfinite"].append(int(self.last_teacher_action_nonfinite))
                    buf["tavla_target_out_of_limits"].append(int(self.last_teacher_target_out_of_limits))
                if hasattr(self, "_tavla_twin_q"):
                    buf["tavla_twin_state"].append(self._tavla_twin_q[env_idx].to("cpu").clone())
                    buf["tavla_taskspace_actions"].append(self._teacher_taskspace_delta_pose[env_idx].to("cpu").clone())
                    if hasattr(self, "_teacher_taskspace_q_delta"):
                        buf["tavla_taskspace_q_deltas"].append(self._teacher_taskspace_q_delta[env_idx].to("cpu").clone())
                    if hasattr(self, "_teacher_taskspace_target_pos"):
                        buf["tavla_taskspace_targets"].append(torch.cat((
                            self._teacher_taskspace_target_pos[env_idx],
                            self._teacher_taskspace_target_quat[env_idx],
                        )).to("cpu").clone())
                    buf["tavla_twin_ik"].append(torch.tensor([
                        float(self.tavla_twin_ik_position_error[env_idx].detach().cpu()),
                        float(self.tavla_twin_ik_rotation_error[env_idx].detach().cpu()),
                        float(self.tavla_twin_ik_iterations[env_idx].detach().cpu()),
                        float(self.tavla_twin_ik_converged[env_idx].detach().cpu()),
                    ]))
                    buf["tavla_force_abort"].append(int(self.tavla_force_abort[env_idx].detach().cpu()))
            
        else:
            # Record for all environments
            for i in range(self.num_envs):
                self.record_data(i)

    def reset_data_buffer(self, env_idx=None):
        """
        Clear the data buffer for one or all environments.
        
        Args:
            env_idx (int, optional): Index of the environment to reset.
                                    If None, reset all environments.
        """
        if not self.collect_data:
            return

        self._model_visual_frame_ready = False
            
        if env_idx is not None:
            self.data_buffers[env_idx] = {
                "camera": {
                    "front": [],
                    "front_transformed": [],
                    "wrist_transformed": [],
                    "wrist": [],
                },
                "joints": [],
                "gripper": [],
                "ee_pose": [],
                "ee_pose_xyzw": [],
                "force": [],
                "force_world": [],
                "force_parent": [],
                "force_tool": [],
                "force_model": [],
                "wrench_raw": [],
                "wrench_anchor": [],
                "wrench_base": [],
                "wrench_corrected": [],
                "wrench_final": [],
                "timestamps": [],
                "actions": [],
                "ppo_joint_targets": [],
                "tavla_teacher_actions": [],
                "tavla_residual_actions": [],
                "tavla_wrench_base": [],
                "tavla_wrench_final": [],
                "tavla_server_effort": [],
                "tavla_server_effort_matches_final": [],
                "tavla_policy_wrench": [],
                "tavla_force_gate_active": [],
                "tavla_force_gate_contact_norm": [],
                "tavla_actual_state": [],
                "tavla_policy_state": [],
                "tavla_combined_targets": [],
                "tavla_command_targets": [],
                "tavla_command_velocity": [],
                "tavla_action_indices": [],
                "tavla_executed_targets": [],
                "tavla_inference_events": [],
                "tavla_inference_latency_s": [],
                "tavla_inference_timeouts": [],
                "tavla_action_nonfinite": [],
                "tavla_target_out_of_limits": [],
                "tavla_twin_state": [],
                "tavla_taskspace_actions": [],
                "tavla_taskspace_q_deltas": [],
                "tavla_taskspace_targets": [],
                "tavla_twin_ik": [],
                "tavla_force_abort": [],
                "rewards": [],
                "reward_terms": []
            }
        else:
            for i in range(self.num_envs):
                self.reset_data_buffer(i)

    def save_data_to_disk(self, env_idx=None, success=None):
        """
        Save the buffered data to disk for one or more environments.
        
        Args:
            env_idx (int, list, np.ndarray, optional): Index or indices of environments to save.
                                                    If None, save all environments.
            success (bool, optional): Episode outcome metadata for the saved buffer.
        """
        if not self.collect_data:
            return
            
        def preprocess_frame(frame):
            """Convert frame to proper format for saving."""
            # Convert to numpy
            if isinstance(frame, torch.Tensor):
                frame = frame.detach().cpu().numpy()
            elif not isinstance(frame, np.ndarray):
                raise TypeError(
                    f"Unsupported data type: {type(frame)}. "
                    f"Expected torch.Tensor or numpy.ndarray."
                )

            # Normalize [0,1] -> [0,255] if needed
            if frame.dtype != np.uint8:
                if frame.size and float(frame.max()) <= 1.0:
                    frame = frame * 255.0
                frame = np.clip(frame, 0.0, 255.0).astype(np.uint8)
            else:
                frame = np.asarray(frame, dtype=np.uint8)

            # Handle channel order: (C, H, W) -> (H, W, C)
            if frame.ndim != 3:
                raise ValueError(f"Camera frame must be rank-3, got {frame.shape}")
            if frame.shape[0] in (1, 3, 4) and frame.shape[-1] not in (1, 3, 4):
                frame = np.transpose(frame, (1, 2, 0))
            if frame.shape[-1] == 1:
                frame = np.repeat(frame, 3, axis=-1)
            elif frame.shape[-1] == 4:
                frame = frame[..., :3]

            return np.ascontiguousarray(frame)

        base_output_dir = self.output_dir
        os.makedirs(base_output_dir, exist_ok=True)

        # Normalize env indices
        if env_idx is None:
            env_indices = list(range(self.num_envs))
        elif isinstance(env_idx, (list, tuple, np.ndarray)):
            env_indices = [int(i) for i in env_idx]
        else:
            env_indices = [int(env_idx)]

        for idx in env_indices:
            # Find next available episode directory
            episode_idx = 0
            while True:
                episode_dir = os.path.join(base_output_dir, f'episode_{episode_idx}')
                if not os.path.exists(episode_dir):
                    print(f"Saving new data folder for env {idx}: {episode_dir}")
                    os.makedirs(episode_dir, exist_ok=True)
                    break
                episode_idx += 1

            buf = self.data_buffers[idx]

            # Save camera data as video
            for key, camera_list in buf["camera"].items():
                allowed_camera_keys = set()
                if self.save_policy_input_video:
                    allowed_camera_keys.update({"front_transformed", "wrist_transformed"})
                if self.save_raw_camera_video:
                    allowed_camera_keys.update({"front", "wrist"})
                if key not in allowed_camera_keys:
                    continue
                if len(camera_list) <= 1:
                    continue

                save_dir = episode_dir if self.minimal_output else os.path.join(episode_dir, key)
                os.makedirs(save_dir, exist_ok=True)

                # Get frame properties
                first_frame = preprocess_frame(camera_list[0])
                last_frame = preprocess_frame(camera_list[-1])

                # Encode as H.264/yuv420p for VS Code/Electron compatibility.
                fps = int(1 / (self.physics_dt * self.cfg.decimation))
                video_path = os.path.join(save_dir, f'{key}.mp4')
                video_frames = []
                # Keep the existing behavior: the reset frame is not part of the
                # per-step trajectory stream.
                for i in range(1, len(camera_list)):
                    frame = preprocess_frame(camera_list[i])
                    if frame.shape[2] == 1:
                        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
                    else:
                        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                    video_frames.append(frame)

                _write_h264_mp4(video_path, video_frames, fps)
                print(f"Saved {key} video to {video_path}")

                # Keep the original camera video untouched and additionally
                # export a VS Code-friendly H.264 video with the same PPO/
                # Factory reward used by the environment overlaid on every
                # post-reset frame.  The first camera frame is the reset
                # snapshot and has no reward, so reward index i-1 matches
                # camera frame i.
                if self.save_reward_video and key in {"front", "wrist"} and buf.get("rewards"):
                    reward_values = [float(value) for value in buf["rewards"]]
                    reward_terms = buf.get("reward_terms", [])
                    overlay_frames = []
                    cumulative_reward = 0.0
                    for frame_index in range(1, len(camera_list)):
                        frame = preprocess_frame(camera_list[frame_index])
                        if frame.shape[2] == 1:
                            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
                        else:
                            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                        reward_index = frame_index - 1
                        if reward_index < len(reward_values):
                            reward = reward_values[reward_index]
                            cumulative_reward += reward
                            lines = [
                                f"step: {reward_index:04d}",
                                f"PPO reward: {reward:+.4f}",
                                f"cumulative: {cumulative_reward:+.3f}",
                            ]
                            if reward_index < len(reward_terms):
                                terms = reward_terms[reward_index]
                                for name in (
                                    "kp_fine",
                                    "curr_engaged",
                                    "curr_success",
                                    "insertion_progress",
                                    "contact_penalty",
                                ):
                                    if name in terms and terms[name] is not None:
                                        lines.append(f"{name}: {float(terms[name]):+.3f}")
                            for line_index, line in enumerate(lines):
                                cv2.putText(
                                    frame,
                                    line,
                                    (12, 28 + 25 * line_index),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    0.65,
                                    (0, 0, 0),
                                    3,
                                    cv2.LINE_AA,
                                )
                                cv2.putText(
                                    frame,
                                    line,
                                    (12, 28 + 25 * line_index),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    0.65,
                                    (255, 255, 255),
                                    1,
                                    cv2.LINE_AA,
                                )
                        overlay_frames.append(frame)
                    if self.minimal_output:
                        overlay_path = os.path.join(episode_dir, f"{key}_reward.mp4")
                    else:
                        overlay_dir = os.path.join(episode_dir, f"{key}_reward")
                        os.makedirs(overlay_dir, exist_ok=True)
                        overlay_path = os.path.join(overlay_dir, f"{key}_reward.mp4")
                    _write_h264_mp4(overlay_path, overlay_frames, fps)
                    print(f"Saved {key} reward video to {overlay_path}")

                # Save last frame
                if not self.minimal_output:
                    Image.fromarray(last_frame).save(os.path.join(save_dir, 'last_frame.png'))
                if not self.minimal_output:
                    for snapshot_idx in sorted({0, len(camera_list) // 2, len(camera_list) - 1}):
                        snapshot = preprocess_frame(camera_list[snapshot_idx])
                        Image.fromarray(snapshot).save(
                            os.path.join(save_dir, f"frame_{snapshot_idx:06d}.png")
                        )

            # The first observation is captured during reset and has no reward.
            # Drop it from saved per-step streams so all trajectory rows align.
            reward_count = len(buf.get("rewards", []))
            timestamp_count = len(buf.get("timestamps", []))
            trajectory_start = 1 if reward_count and timestamp_count == reward_count + 1 else 0

            def trajectory_series(key):
                values = buf.get(key, [])
                return values[trajectory_start:] if trajectory_start else values

            # Save numerical data to CSV
            self._save_array_to_csv(trajectory_series("joints"), episode_dir, 'joint_states.csv', 'joint')
            self._save_array_to_csv(trajectory_series("force"), episode_dir, 'force_local.csv', 'force')
            self._save_array_to_csv(trajectory_series("force_world"), episode_dir, 'force_world.csv', 'force')
            self._save_array_to_csv(trajectory_series("force_parent"), episode_dir, 'wrench_parent.csv', 'wrench_parent')
            self._save_array_to_csv(trajectory_series("force_tool"), episode_dir, 'wrench_tool.csv', 'wrench_tool')
            self._save_array_to_csv(trajectory_series("force_model"), episode_dir, 'wrench_model.csv', 'wrench_model')
            for key, filename, prefix in (
                ("wrench_raw", "wrench_raw.csv", "wrench_raw"),
                ("wrench_anchor", "wrench_anchor.csv", "wrench_anchor"),
                ("wrench_base", "wrench_base.csv", "wrench_base"),
                ("wrench_corrected", "wrench_corrected.csv", "wrench_corrected"),
                ("wrench_final", "wrench_final.csv", "wrench_final"),
            ):
                self._save_array_to_csv(trajectory_series(key), episode_dir, filename, prefix)
            self._save_array_to_csv(trajectory_series("timestamps"), episode_dir, 'timestamps.csv', 'timestamp')
            self._save_array_to_csv(trajectory_series("gripper"), episode_dir, 'gripper.csv', 'gripper')
            self._save_array_to_csv(buf["actions"], episode_dir, 'actions.csv', 'action')
            self._save_array_to_csv(
                trajectory_series("ppo_joint_targets"), episode_dir, "ppo_joint_targets.csv", "ppo_joint_target"
            )
            self._save_scalar_series_to_csv(buf["rewards"], episode_dir, "reward.csv", "reward")
            self._save_reward_terms_to_csv(buf["reward_terms"], episode_dir)
            if "tavla_teacher_actions" in buf:
                self._save_array_to_csv(
                    trajectory_series("tavla_teacher_actions"), episode_dir, "tavla_teacher_actions.csv", "teacher_action"
                )
            if "tavla_residual_actions" in buf:
                self._save_array_to_csv(
                    trajectory_series("tavla_residual_actions"), episode_dir, "tavla_residual_actions.csv", "residual_action"
                )
            if "tavla_wrench_base" in buf:
                self._save_array_to_csv(
                    trajectory_series("tavla_wrench_base"), episode_dir, "tavla_wrench_base.csv", "wrench_base"
                )
            if "tavla_wrench_final" in buf:
                self._save_array_to_csv(
                    trajectory_series("tavla_wrench_final"), episode_dir, "tavla_wrench_final.csv", "wrench_final"
                )
            for key, filename, prefix in (
                ("tavla_actual_state", "tavla_actual_state.csv", "actual_state"),
                ("tavla_policy_state", "tavla_policy_state.csv", "policy_state"),
                ("tavla_combined_targets", "tavla_combined_targets.csv", "combined_target"),
                ("tavla_executed_targets", "tavla_executed_targets.csv", "executed_target"),
                ("tavla_command_targets", "tavla_command_targets.csv", "command_target"),
                ("tavla_command_velocity", "tavla_command_velocity.csv", "command_velocity"),
                ("tavla_server_effort", "tavla_server_effort.csv", "server_effort"),
                ("tavla_server_effort_matches_final", "tavla_server_effort_matches_final.csv", "matches_final"),
                ("tavla_action_indices", "tavla_action_indices.csv", "action_index"),
                ("tavla_inference_events", "tavla_inference_events.csv", "inference_event"),
                ("tavla_inference_latency_s", "tavla_inference_latency_s.csv", "inference_latency_s"),
                ("tavla_inference_timeouts", "tavla_inference_timeouts.csv", "inference_timeout"),
                ("tavla_action_nonfinite", "tavla_action_nonfinite.csv", "action_nonfinite"),
                ("tavla_target_out_of_limits", "tavla_target_out_of_limits.csv", "target_out_of_limits"),
                ("tavla_policy_wrench", "tavla_policy_wrench.csv", "policy_wrench"),
                ("tavla_force_gate_active", "tavla_force_gate_active.csv", "gate_active"),
                ("tavla_force_gate_contact_norm", "tavla_force_gate_contact_norm.csv", "contact_norm_n"),
                ("tavla_twin_state", "tavla_twin_state.csv", "twin_state"),
                ("tavla_taskspace_actions", "tavla_taskspace_actions.csv", "taskspace_action"),
                ("tavla_taskspace_q_deltas", "tavla_taskspace_q_deltas.csv", "taskspace_q_delta"),
                ("tavla_taskspace_targets", "tavla_taskspace_targets.csv", "taskspace_target"),
                ("tavla_twin_ik", "tavla_twin_ik.csv", "twin_ik"),
                ("tavla_force_abort", "tavla_force_abort.csv", "force_abort"),
            ):
                if key in buf:
                    self._save_array_to_csv(trajectory_series(key), episode_dir, filename, prefix)
            failure_reason = None
            if success is False:
                if bool(torch.any(getattr(self, "_tavla_mapping_failed", torch.zeros(1, dtype=torch.bool, device=self.device)))):
                    failure_reason = "twin_ik_mapping_failure"
                elif int(getattr(self, "teacher_failures", 0)) > 0:
                    failure_reason = "teacher_inference_failure"
                elif int(getattr(self, "tavla_force_abort_count", 0)) > 0:
                    failure_reason = "abnormal_wrench"
                else:
                    failure_reason = "task_timeout_or_termination"
            metadata = {
                "success": None if success is None else bool(success),
                "seed": int(getattr(self.cfg, "seed", 0)),
                "failure_reason": failure_reason,
                "reward_steps": len(buf.get("rewards", [])),
                "episode_duration_s": float(len(buf.get("rewards", [])) * self.step_dt),
                "server_timeout": bool(getattr(self, "teacher_timeouts", 0)),
                "action_nonfinite_count": int(sum(buf.get("tavla_action_nonfinite", []))),
                "target_out_of_limits_count": int(getattr(self, "teacher_target_out_of_limits_count", 0)),
                "teacher_inference_count": int(getattr(self, "teacher_inference_count", 0)),
                "teacher_failures": int(getattr(self, "teacher_failures", 0)),
                "teacher_timeouts": int(getattr(self, "teacher_timeouts", 0)),
                "last_teacher_latency_s": float(getattr(self, "teacher_inference_latency_s", 0.0)),
                "teacher_control_mode": getattr(self, "_teacher_control_mode", None),
                "teacher_hold_steps": int(getattr(self, "teacher_hold_steps", 0)),
                "teacher_replan_actions": int(getattr(self, "teacher_replan_actions", 0)),
                "teacher_visual_profile": getattr(self, "_teacher_visual_profile", "raw"),
                "teacher_camera_calibration": getattr(self, "_teacher_camera_calibration", ""),
                "twin_mapping_failures": int(getattr(self, "tavla_mapping_failures", 0)),
                "force_abort_count": int(getattr(self, "tavla_force_abort_count", 0)),
                "teacher_action_interpolation": bool(getattr(self, "_teacher_action_interpolation", False)),
                "teacher_speed_scale": float(getattr(self, "_teacher_speed_scale", 1.0)),
                "teacher_taskspace_velocity_limits": [
                    float(value) for value in getattr(self.cfg, "teacher_taskspace_velocity_limits", [])
                ],
                "teacher_force_norm_p99": float(getattr(self.cfg, "teacher_force_norm_p99", 0.0)),
                "tavla_force_gate": {
                    "enabled": bool(getattr(self.cfg, "tavla_force_gate_enabled", False)),
                    "threshold_n": float(getattr(self.cfg, "tavla_force_gate_threshold_n", 0.0)),
                    "confirm_steps": int(getattr(self.cfg, "tavla_force_gate_confirm_steps", 0)),
                    "release_steps": int(getattr(self.cfg, "tavla_force_gate_release_steps", 0)),
                    "precontact_wrench": [
                        float(value)
                        for value in getattr(self.cfg, "tavla_force_gate_precontact_wrench", [])
                    ],
                    "contact_view": "HeldAsset -> FixedAsset PhysX contact report",
                },
                "twin_ik_position_error": float(
                    getattr(self, "tavla_twin_ik_position_error", torch.tensor([float("nan")], device=self.device))[0].detach().cpu()
                ),
                "twin_ik_rotation_error": float(
                    getattr(self, "tavla_twin_ik_rotation_error", torch.tensor([float("nan")], device=self.device))[0].detach().cpu()
                ),
                "twin_ik_converged": bool(
                    getattr(self, "tavla_twin_ik_converged", torch.tensor([False], device=self.device))[0].detach().cpu()
                ),
                "teacher_joint_velocity_limits": [float(value) for value in getattr(self.cfg, "teacher_joint_velocity_limits", [])],
                "teacher_joint_acceleration_limits": [float(value) for value in getattr(self.cfg, "teacher_joint_acceleration_limits", [])],
                "teacher_state_alignment": bool(getattr(self, "_teacher_state_alignment", False)),
                "teacher_action_state_alignment": bool(getattr(self, "_teacher_action_state_alignment", False)),
                "teacher_wrench_scale": [float(value) for value in getattr(self.cfg, "teacher_wrench_scale", [])],
                "teacher_wrench_bias": [float(value) for value in getattr(self.cfg, "teacher_wrench_bias", [])],
                "wrench_contract": {
                    "component_order": ["Fx", "Fy", "Fz", "Tx", "Ty", "Tz"],
                    "units": ["N", "N", "N", "N*m", "N*m", "N*m"],
                    "wrench_raw": "PhysX body_incoming_joint_wrench_b source value",
                    "wrench_anchor": "force-sensor joint anchor frame/reference",
                    "wrench_base": "robot base frame, torque about robot-base origin; matches the real training contract",
                    "wrench_corrected": "configured robot-base-frame reference; calibration pending unless ready",
                    "wrench_final": "-wrench_base; final sign-corrected wrench sent to TAVLA",
                    "raw_wrench_frame": getattr(self.cfg, "ft_raw_wrench_frame", "unknown"),
                    "raw_torque_reference": getattr(self.cfg, "ft_raw_torque_reference", "unknown"),
                    "corrected_reference": getattr(self.cfg, "ft_corrected_reference", "unknown"),
                    "corrected_ready": bool(getattr(self.cfg, "ft_corrected_ready", False)),
                    "used_by_tavla": "wrench_final",
                    "torque_translation_formula": "tau_B = tau_A + cross(p_A - p_B, F)",
                    "matrix_convention": "output = M @ input; component order [F,T]",
                    "sign": [float(value) for value in getattr(self.cfg, "ft_corrected_wrench_sign", [1.0] * 6)],
                },
            }
            tavla_hdf5_path = None
            if self.save_tavla_hdf5 and not self.minimal_output:
                tavla_hdf5_path = self._save_tavla_hdf5(
                    buf=buf,
                    trajectory_start=trajectory_start,
                    success=success,
                    episode_idx=episode_idx,
                )
                if tavla_hdf5_path is not None:
                    metadata["tavla_hdf5_path"] = os.path.relpath(tavla_hdf5_path, base_output_dir)

            with open(os.path.join(episode_dir, "episode_metadata.json"), "w", encoding="utf-8") as file:
                json.dump(metadata, file, indent=2, ensure_ascii=False)
            self._save_array_to_csv(trajectory_series("ee_pose"), episode_dir, 'ee_pose.csv', 'ee_pose_wxyz')
            self._save_array_to_csv(
                trajectory_series("ee_pose_xyzw"), episode_dir, 'ee_pose_xyzw.csv', 'ee_pose_xyzw'
            )

    def _save_tavla_hdf5(self, buf, trajectory_start, success, episode_idx):
        """Save one episode in the raw HDF5 layout consumed by TA-VLA converters."""
        try:
            import h5py
        except ImportError:
            print("[TAVLA] h5py is unavailable; skipping raw HDF5 export.")
            return None

        def to_numpy(value):
            if hasattr(value, "detach"):
                value = value.detach().cpu().numpy()
            return np.asarray(value)

        def stack_series(key):
            values = buf.get(key, [])
            values = values[trajectory_start:] if trajectory_start else values
            if not values:
                return None
            return np.stack([to_numpy(value) for value in values])

        def frame_to_rgb(value):
            frame = to_numpy(value)
            if frame.ndim != 3:
                raise ValueError(f"TAVLA camera frame must be rank-3, got {frame.shape}")
            if frame.shape[0] in (1, 3, 4) and frame.shape[-1] not in (1, 3, 4):
                frame = np.transpose(frame, (1, 2, 0))
            if frame.shape[-1] == 1:
                frame = np.repeat(frame, 3, axis=-1)
            elif frame.shape[-1] == 4:
                frame = frame[..., :3]
            if frame.dtype != np.uint8:
                if frame.size and float(frame.max()) <= 1.0:
                    frame = frame * 255.0
                frame = np.clip(frame, 0.0, 255.0).astype(np.uint8)
            if frame.shape[:2] != (480, 640):
                frame = cv2.resize(frame, (640, 480), interpolation=cv2.INTER_LINEAR)
            return np.ascontiguousarray(frame)

        def stack_frames(camera_name):
            values = buf.get("camera", {}).get(camera_name, [])
            values = values[trajectory_start:] if trajectory_start else values
            if not values:
                return None
            return np.stack([frame_to_rgb(value) for value in values])

        joints = stack_series("joints")
        gripper = stack_series("gripper")
        corrected_ready = bool(getattr(self.cfg, "ft_corrected_ready", False))
        effort_key = "wrench_final"
        effort = stack_series(effort_key)
        wrench_streams = {
            key: stack_series(key)
            for key in ("wrench_raw", "wrench_anchor", "wrench_base", "wrench_corrected", "wrench_final")
        }
        ee_pose = stack_series("ee_pose_xyzw")
        action = stack_series("ppo_joint_targets")
        if action is None:
            action = stack_series("tavla_executed_targets")
        timestamps = stack_series("timestamps")
        # New evaluations store the exact policy inputs. Fall back to the
        # legacy raw streams so old buffers remain exportable.
        front = stack_frames("front_transformed")
        if front is None:
            front = stack_frames("front")
        wrist = stack_frames("wrist_transformed")
        if wrist is None:
            wrist = stack_frames("wrist")

        missing = []
        for name, value in (
            ("joints", joints),
            ("gripper", gripper),
            (effort_key, effort),
            ("ppo_joint_targets", action),
            ("front camera", front),
            ("wrist camera", wrist),
        ):
            if value is None:
                missing.append(name)
        if missing:
            print(f"[TAVLA] Cannot export HDF5; missing fields: {', '.join(missing)}")
            return None

        if joints.ndim != 2 or joints.shape[1] < 7:
            raise ValueError(f"TAVLA qpos source must contain at least 7 joints, got {joints.shape}")
        qpos = np.concatenate(
            [joints[:, :7].astype(np.float32), gripper.reshape(-1, 1).astype(np.float32)], axis=1
        )
        effort = effort.reshape(effort.shape[0], -1).astype(np.float32)
        for key, value in wrench_streams.items():
            if value is not None:
                wrench_streams[key] = value.reshape(value.shape[0], -1).astype(np.float32)
        if ee_pose is not None:
            ee_pose = ee_pose.reshape(ee_pose.shape[0], -1).astype(np.float32)
        action = action.reshape(action.shape[0], -1).astype(np.float32)
        timestamps = (
            timestamps.reshape(-1).astype(np.float32)
            if timestamps is not None
            else np.arange(len(qpos), dtype=np.float32) * float(self.step_dt)
        )

        if effort.shape[1] != 6:
            raise ValueError(f"TAVLA effort must have 6 values, got {effort.shape}")
        if action.shape[1] != 8:
            raise ValueError(f"TAVLA action must have 8 values, got {action.shape}")

        lengths = {
            "qpos": len(qpos),
            "effort": len(effort),
            "action": len(action),
            "timestamps": len(timestamps),
            "front": len(front),
            "wrist": len(wrist),
        }
        lengths.update({key: len(value) for key, value in wrench_streams.items() if value is not None})
        if ee_pose is not None:
            lengths["ee_pose"] = len(ee_pose)
        frame_count = min(lengths.values())
        if frame_count <= 0:
            print("[TAVLA] Cannot export HDF5; episode has no aligned frames.")
            return None
        if len(set(lengths.values())) != 1:
            print(f"[TAVLA] Aligning episode streams to {frame_count} frames: {lengths}")

        # Native decimation already makes each recorded environment step one PPO/TAVLA macro-step.
        # Do not downsample it a second time here.
        sample_indices = np.arange(0, frame_count, dtype=np.int64)
        qpos = qpos[:frame_count][sample_indices]
        effort = effort[:frame_count][sample_indices]
        action = action[:frame_count][sample_indices]
        timestamps = timestamps[:frame_count][sample_indices]
        front = front[:frame_count][sample_indices]
        wrist = wrist[:frame_count][sample_indices]
        for key, value in wrench_streams.items():
            if value is not None:
                wrench_streams[key] = value[:frame_count][sample_indices]
        if ee_pose is not None:
            ee_pose = ee_pose[:frame_count][sample_indices]
        frame_count = len(sample_indices)

        raw_dir = os.path.join(self.output_dir, self.tavla_hdf5_dir)
        os.makedirs(raw_dir, exist_ok=True)
        path = os.path.join(raw_dir, f"episode_{episode_idx}.hdf5")
        with h5py.File(path, "w") as data:
            observations = data.create_group("observations")
            images = observations.create_group("images")
            observations.create_dataset("qpos", data=qpos, compression="gzip")
            observations.create_dataset("effort", data=effort, compression="gzip")
            for key, value in wrench_streams.items():
                if value is not None:
                    observations.create_dataset(key, data=value, compression="gzip")
            if ee_pose is not None:
                observations.create_dataset("ee_pose", data=ee_pose, compression="gzip")
            images.create_dataset("cam_high", data=front, compression="gzip")
            images.create_dataset("cam_left_wrist", data=wrist, compression="gzip")
            images["cam_right_wrist"] = images["cam_left_wrist"]
            data.create_dataset("action", data=action, compression="gzip")
            data.create_dataset("timestamp", data=timestamps)
            done = np.zeros((frame_count,), dtype=np.bool_)
            done[-1] = True
            data.create_dataset("done", data=done)
            episode_success = np.zeros((frame_count,), dtype=np.bool_)
            episode_success[-1] = bool(success)
            data.create_dataset("success", data=episode_success)
            data.attrs["format"] = "tavla_raw_v1"
            data.attrs["task"] = str(getattr(self.cfg, "task_prompt", "peg-in-hole"))
            data.attrs["source_fps"] = float(1.0 / self.step_dt)
            data.attrs["fps"] = float(1.0 / self.step_dt)
            data.attrs["state_semantics"] = "7 arm joint positions + normalized gripper"
            data.attrs["effort_semantics"] = f"{effort_key}, six-dimensional TAVLA effort input"
            data.attrs["wrench_component_order"] = "[Fx,Fy,Fz,Tx,Ty,Tz]"
            data.attrs["wrench_units"] = "[N,N,N,N*m,N*m,N*m]"
            data.attrs["wrench_raw_semantics"] = "PhysX body_incoming_joint_wrench_b source value"
            data.attrs["wrench_anchor_semantics"] = "force-sensor joint anchor frame/reference"
            data.attrs["wrench_base_semantics"] = "robot base frame, torque about robot-base origin; real training contract"
            data.attrs["wrench_corrected_semantics"] = "configured robot-base-frame reference"
            data.attrs["wrench_final_semantics"] = "-wrench_base; final sign-corrected wrench sent to TAVLA"
            data.attrs["wrench_final_definition"] = "wrench_final = -wrench_base"
            data.attrs["wrench_corrected_ready"] = corrected_ready
            data.attrs["wrench_raw_frame"] = str(getattr(self.cfg, "ft_raw_wrench_frame", "unknown"))
            data.attrs["wrench_raw_torque_reference"] = str(getattr(self.cfg, "ft_raw_torque_reference", "unknown"))
            data.attrs["wrench_corrected_reference"] = str(getattr(self.cfg, "ft_corrected_reference", "unknown"))
            data.attrs["wrench_torque_translation_formula"] = "tau_B = tau_A + cross(p_A - p_B, F)"
            data.attrs["ee_pose_order"] = "[x,y,z,qx,qy,qz,qw]"
            data.attrs["action_semantics"] = "8-D post-adapter PPO joint target"
            data.attrs["camera_layout"] = "HWC RGB uint8, 480x640"

        print(f"[TAVLA] Saved raw episode to {path} ({frame_count} frames)")
        return path

    def _save_array_to_csv(self, data_list, episode_dir, filename, column_prefix):
        """
        Helper function to save a time-series data list to CSV.
        
        Args:
            data_list (list): List of tensors to save
            episode_dir (str): Directory to save CSV
            filename (str): CSV file name
            column_prefix (str): Prefix for CSV column names
        """
        if self.minimal_output or not data_list:
            print(f"No data ({column_prefix}), skipping save.")
            return

        file_path = os.path.join(episode_dir, filename)

        try:
            # Handle scalar values, including zero-dimensional tensors.
            first_value = data_list[0]
            first_array = first_value.detach().cpu().numpy() if hasattr(first_value, "detach") else np.asarray(first_value)
            if isinstance(first_value, (int, float, np.number)) or first_array.ndim == 0:
                header = [column_prefix]
                with open(file_path, 'w', newline='') as csvfile:
                    csv_writer = csv.writer(csvfile)
                    csv_writer.writerow(header)
                    for i in range(len(data_list)) :
                        csv_writer.writerow([data_list[i]])
            else:
                # Handle tensor/array data
                num_columns = data_list[0].shape[0]
                header = [f'{column_prefix}_{j}' for j in range(num_columns)]

                with open(file_path, 'w', newline='') as csvfile:
                    csv_writer = csv.writer(csvfile)
                    csv_writer.writerow(header)

                    for i in range(len(data_list)) :
                        row_data = data_list[i]
                        if hasattr(row_data, 'cpu'):
                            row_data = row_data.detach().cpu().numpy()
                        csv_writer.writerow(row_data)

            print(f"Data ({column_prefix}) saved to {file_path}")
        except Exception as e:
            print(f"Error saving {filename}: {e}")

    def _save_scalar_series_to_csv(self, values, episode_dir, filename, column_name):
        if self.minimal_output or not values:
            return
        with open(os.path.join(episode_dir, filename), "w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow([column_name])
            writer.writerows([[float(value)] for value in values])

    def _save_reward_terms_to_csv(self, terms, episode_dir):
        if self.minimal_output or not terms:
            return
        names = sorted({name for row in terms for name in row})
        with open(os.path.join(episode_dir, "reward_terms.csv"), "w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(["step", *names])
            for step, row in enumerate(terms):
                writer.writerow([step, *[row.get(name) for name in names]])

    def _get_curr_successes(self, success_threshold, check_rot=False):
        if self.cfg_task.name != "peg_insert":
            return super()._get_curr_successes(success_threshold=success_threshold, check_rot=check_rot)

        held_base_pos, _ = factory_utils.get_held_base_pose(
            self.held_pos, self.held_quat, self.cfg_task.name, self.cfg_task.fixed_asset_cfg, self.num_envs, self.device
        )
        target_held_base_pos, _ = factory_utils.get_target_held_base_pose(
            self.fixed_pos,
            self.fixed_quat,
            self.cfg_task.name,
            self.cfg_task.fixed_asset_cfg,
            self.num_envs,
            self.device,
        )

        xy_dist = torch.linalg.vector_norm(target_held_base_pos[:, 0:2] - held_base_pos[:, 0:2], dim=1)
        z_disp = held_base_pos[:, 2] - target_held_base_pos[:, 2]

        success_xy_threshold = float(getattr(self.cfg_task, "success_xy_threshold", 0.006))
        is_centered = xy_dist < success_xy_threshold
        if bool(getattr(self.cfg_task, "align_only", False)):
            min_height = float(getattr(self.cfg_task, "align_success_min_height", 0.03))
            max_height = float(getattr(self.cfg_task, "align_success_max_height", 0.08))
            is_above_hole = torch.logical_and(z_disp > min_height, z_disp < max_height)
            return torch.logical_and(is_centered, is_above_hole)

        # One explicit geometric insertion contract is shared by replay,
        # force fitting and online evaluation.  The legacy Factory threshold
        # converted success_threshold through the hole height and yielded a
        # stricter 1 mm depth for this task.
        success_z_threshold = float(getattr(self.cfg_task, "success_z_threshold", 0.003))
        is_inserted = z_disp < success_z_threshold
        return torch.logical_and(is_centered, is_inserted)

    def _get_factory_rew_dict(self, curr_successes):
        rew_dict, rew_scales = super()._get_factory_rew_dict(curr_successes)

        if self.cfg_task.name != "peg_insert":
            return rew_dict, rew_scales

        held_base_pos, _ = factory_utils.get_held_base_pose(
            self.held_pos, self.held_quat, self.cfg_task.name, self.cfg_task.fixed_asset_cfg, self.num_envs, self.device
        )
        target_held_base_pos, _ = factory_utils.get_target_held_base_pose(
            self.fixed_pos,
            self.fixed_quat,
            self.cfg_task.name,
            self.cfg_task.fixed_asset_cfg,
            self.num_envs,
            self.device,
        )

        xy_dist = torch.linalg.vector_norm(target_held_base_pos[:, 0:2] - held_base_pos[:, 0:2], dim=1)
        z_disp = held_base_pos[:, 2] - target_held_base_pos[:, 2]

        align_tol = max(float(getattr(self.cfg_task, "alignment_reward_tolerance", 0.006)), 1e-6)
        insert_gate_tol = max(float(getattr(self.cfg_task, "insertion_gate_tolerance", 0.02)), 1e-6)
        aligned = torch.exp(-torch.square(xy_dist / align_tol))
        insertion_gate = torch.exp(-torch.square(xy_dist / insert_gate_tol))
        insert_action = torch.clamp(-self.actions[:, 2], 0.0, 1.0)
        misaligned_down = (1.0 - insertion_gate) * insert_action

        if bool(getattr(self.cfg_task, "align_only", False)):
            z_target = float(getattr(self.cfg_task, "align_above_hole_height", 0.05))
            z_tol = max(float(getattr(self.cfg_task, "align_above_hole_z_tolerance", 0.03)), 1e-6)
            above_hole_z = torch.exp(-torch.square((z_disp - z_target) / z_tol))

            for rew_name in ("kp_baseline", "kp_coarse", "kp_fine", "curr_engaged"):
                if rew_name in rew_scales:
                    rew_scales[rew_name] = 0.0
            if "curr_success" in rew_scales:
                rew_scales["curr_success"] = self.cfg_task.success_bonus_scale

            rew_dict.update(
                {
                    "xy_alignment": aligned,
                    "above_hole_z": above_hole_z,
                    "misaligned_down_action": misaligned_down,
                }
            )
            rew_scales.update(
                {
                    "xy_alignment": self.cfg_task.alignment_reward_scale,
                    "above_hole_z": 1.0,
                    "misaligned_down_action": -self.cfg_task.misaligned_down_penalty_scale,
                }
            )
            return rew_dict, rew_scales

        insert_depth = max(float(getattr(self.cfg_task, "insertion_reward_depth", self.cfg_task.fixed_asset_cfg.height)), 1e-6)
        pre_insert_height = max(float(getattr(self.cfg_task, "pre_insert_reward_height", 0.06)), insert_depth + 1e-6)
        insertion_progress = 1.0 - torch.clamp(z_disp / insert_depth, 0.0, 1.0)
        pre_insert_progress = 1.0 - torch.clamp(
            (z_disp - insert_depth) / (pre_insert_height - insert_depth), 0.0, 1.0
        )

        rew_dict.update(
            {
                "xy_alignment": aligned,
                "pre_insert_progress": insertion_gate * pre_insert_progress,
                "insertion_progress": insertion_gate * insertion_progress,
                "aligned_down_action": insertion_gate * insert_action,
                "misaligned_down_action": misaligned_down,
            }
        )
        rew_scales.update(
            {
                "xy_alignment": self.cfg_task.alignment_reward_scale,
                "pre_insert_progress": self.cfg_task.pre_insert_reward_scale,
                "insertion_progress": self.cfg_task.insertion_reward_scale,
                "aligned_down_action": self.cfg_task.down_action_reward_scale,
                "misaligned_down_action": -self.cfg_task.misaligned_down_penalty_scale,
            }
        )
        return rew_dict, rew_scales

    def _get_observations(self):
        """Override to record data after computing observations."""
        obs = super()._get_observations()
        
        # ========== 新增: Policy推理 ==========
        if self.policy:
            next_action = self.select_action(self.policy)
            self.next_action = torch.stack(next_action).to(obs["policy"].device)

        # ======================================
        
        # Record data if collection is enabled
        if self.collect_data:
            self.record_data()
        
        return obs

    def _pre_physics_step(self, action):
        super()._pre_physics_step(action)
        self._cartesian_target_update_pending = True
        self._cartesian_orientation_update_pending = True
        self._joint_target_update_pending = True

    def _limit_cartesian_target_speed(self, target_pos):
        """Limit the virtual Cartesian target without feeding measured velocity into PPO."""
        if not getattr(self.cfg.ctrl, "cartesian_target_speed_limit_enabled", True):
            return target_pos
        if not self._cartesian_target_initialized:
            self._cartesian_target_pos = self.fingertip_midpoint_pos.detach().clone()
            self._cartesian_target_initialized = True
        if not self._cartesian_target_update_pending:
            return self._cartesian_target_pos

        self._cartesian_target_update_pending = False
        max_speed = max(float(getattr(self.cfg.ctrl, "cartesian_target_speed_limit_mps", 0.08)), 0.0)
        speed_cap = max(float(getattr(self.cfg.ctrl, "cartesian_target_speed_cap_mps", 0.144)), 0.0)
        if speed_cap > 0.0:
            max_speed = min(max_speed, speed_cap)
        max_delta = max_speed * float(self.step_dt)
        delta = target_pos - self._cartesian_target_pos
        limited_target = self._cartesian_target_pos + torch.clamp(delta, -max_delta, max_delta)
        self._cartesian_target_pos = limited_target.detach().clone()
        return self._cartesian_target_pos

    def _limit_cartesian_target_orientation(self, target_quat):
        """Apply the real-data angular speed and acceleration limits."""
        if not getattr(self.cfg.ctrl, "cartesian_target_speed_limit_enabled", True):
            return target_quat
        if not self._cartesian_orientation_initialized:
            current_euler = torch.stack(torch_utils.get_euler_xyz(self.fingertip_midpoint_quat), dim=1)
            self._cartesian_target_euler = current_euler.detach().clone()
            self._cartesian_orientation_initialized = True
        if not self._cartesian_orientation_update_pending:
            return torch_utils.quat_from_euler_xyz(
                self._cartesian_target_euler[:, 0],
                self._cartesian_target_euler[:, 1],
                self._cartesian_target_euler[:, 2],
            )

        self._cartesian_orientation_update_pending = False
        dt = float(self.step_dt)
        target_euler = torch.stack(torch_utils.get_euler_xyz(target_quat), dim=1)
        delta = target_euler - self._cartesian_target_euler
        delta = (delta + torch.pi) % (2.0 * torch.pi) - torch.pi
        desired_velocity = delta / max(dt, 1.0e-6)
        max_speed = max(
            float(getattr(self.cfg.ctrl, "cartesian_target_angular_speed_limit_radps", 0.40)), 0.0
        )
        desired_velocity = torch.clamp(desired_velocity, -max_speed, max_speed)
        max_velocity_change = max(
            float(getattr(self.cfg.ctrl, "cartesian_target_angular_acceleration_limit_radps2", 1.0)), 0.0
        ) * dt
        limited_velocity = self._cartesian_target_angular_velocity + torch.clamp(
            desired_velocity - self._cartesian_target_angular_velocity,
            -max_velocity_change,
            max_velocity_change,
        )
        self._cartesian_target_angular_velocity = limited_velocity.detach().clone()
        self._cartesian_target_euler = (
            self._cartesian_target_euler + limited_velocity * dt
        ).detach().clone()
        return torch_utils.quat_from_euler_xyz(
            self._cartesian_target_euler[:, 0],
            self._cartesian_target_euler[:, 1],
            self._cartesian_target_euler[:, 2],
        )

    def _is_peg_insert_yaw_locked(self):
        return (
            getattr(self.cfg_task, "name", "") == "peg_insert"
            and bool(getattr(self.cfg.ctrl, "lock_fingertip_yaw", False))
        )

    def _constrain_fingertip_orientation(self, target_quat):
        """Keep peg-in-hole yaw fixed while allowing a small roll/pitch tilt."""
        target_roll, target_pitch, target_yaw = torch_utils.get_euler_xyz(target_quat)
        if not self._is_peg_insert_yaw_locked():
            return target_quat

        if getattr(self.cfg.ctrl, "allow_fingertip_tilt", False):
            tilt_limit = max(float(getattr(self.cfg.ctrl, "fingertip_tilt_limit_rad", 0.0873)), 0.0)
            roll_ref = torch.full_like(target_roll, torch.pi)
            roll_delta = (target_roll - roll_ref + torch.pi) % (2.0 * torch.pi) - torch.pi
            pitch_delta = (target_pitch + torch.pi) % (2.0 * torch.pi) - torch.pi
            target_roll = roll_ref + torch.clamp(roll_delta, -tilt_limit, tilt_limit)
            target_pitch = torch.clamp(pitch_delta, -tilt_limit, tilt_limit)
        elif getattr(self.cfg.ctrl, "lock_fingertip_downward", False):
            target_roll = torch.full_like(target_roll, torch.pi)
            target_pitch = torch.zeros_like(target_pitch)

        if self._is_peg_insert_yaw_locked():
            target_yaw = torch.full_like(
                target_yaw, float(getattr(self.cfg.ctrl, "locked_fingertip_yaw", -1.5708))
            )

        return torch_utils.quat_from_euler_xyz(target_roll, target_pitch, target_yaw)

    def _get_dones(self):
        """Check which environments are terminated.

        For Factory reset logic, it is important that all environments
        stay in sync (i.e., _get_dones should return all true or all false).
        """
        self._compute_intermediate_values(dt=self.physics_dt)
        time_out = self.episode_length_buf >= self.max_episode_length - 1

        if self.immediate_stop :
            curr_successes = self._get_curr_successes(
                success_threshold=self.cfg_task.success_threshold, check_rot=self.cfg_task.name == "nut_thread"
            )
            terminated = time_out | curr_successes
        else:
            terminated = time_out

        return terminated, time_out
    
    def step(self, action: torch.Tensor):
        """Execute one time-step of the environment's dynamics.

        The environment steps forward at a fixed time-step, while the physics simulation is decimated at a
        lower time-step. This is to ensure that the simulation is stable. These two time-steps can be configured
        independently using the :attr:`DirectRLEnvCfg.decimation` (number of simulation steps per environment step)
        and the :attr:`DirectRLEnvCfg.sim.physics_dt` (physics time-step). Based on these parameters, the environment
        time-step is computed as the product of the two.

        This function performs the following steps:

        1. Pre-process the actions before stepping through the physics.
        2. Apply the actions to the simulator and step through the physics in a decimated manner.
        3. Compute the reward and done signals.
        4. Reset environments that have terminated or reached the maximum episode length.
        5. Apply interval events if they are enabled.
        6. Compute observations.

        Args:
            action: The actions to apply on the environment. Shape is (num_envs, action_dim).

        Returns:
            A tuple containing the observations, rewards, resets (terminated and truncated) and extras.
        """
        action = action.to(self.device)
        # Use pi0 (zero actions) for inference instead of RL actions
        if hasattr(self, 'policy') and self.policy is not None:
            action = torch.zeros_like(action)
        # add action noise
        if self.cfg.action_noise_model:
            action = self._action_noise_model(action)

        # process actions
        self._pre_physics_step(action)

        # check if we need to do rendering within the physics loop
        # note: checked here once to avoid multiple checks within the loop
        is_rendering = _sim_flag(self.sim, "has_gui") or _sim_flag(self.sim, "has_rtx_sensors")

        start = time.time()
        if getattr(self, "_sim_data_force_zero_gravity", False):
            sim_utils.SimulationContext.instance().physics_sim_view.set_gravity(
                carb.Float3(0.0, 0.0, 0.0)
            )
        # perform physics stepping
        for _ in range(self.cfg.decimation):
            self._sim_step_counter += 1
            # set actions into buffers
            self._apply_action()
            # set actions into simulator
            self.scene.write_data_to_sim()
            # simulate
            self.sim.step(render=False)
            # render between steps only if the GUI or an RTX sensor needs it
            # note: we assume the render interval to be the shortest accepted rendering interval.
            #    If a camera needs rendering at a faster frequency, this will lead to unexpected behavior.
            if self._sim_step_counter % self.cfg.sim.render_interval == 0 and is_rendering:
                self.sim.render()
            # update buffers at sim dt
            self.scene.update(dt=self.physics_dt)
            if getattr(self, "_sim_data_trace_physics", False):
                trace_step = int(getattr(self, "_sim_data_trace_step", 0))
                if trace_step < 12:
                    q_trace = self.joint_pos[0, :7].detach().cpu().numpy()
                    v_trace = self.joint_vel[0, :7].detach().cpu().numpy()
                    q_target_trace = self._robot.data.joint_pos_target[0, :7].detach().cpu().numpy()
                    v_target_trace = self._robot.data.joint_vel_target[0, :7].detach().cpu().numpy()
                    applied_trace = self._robot.data.applied_torque[0, :7].detach().cpu().numpy()
                    computed_trace = self._robot.data.computed_torque[0, :7].detach().cpu().numpy()
                    stiffness_trace = self._robot.data.joint_stiffness[0, :7].detach().cpu().numpy()
                    damping_trace = self._robot.data.joint_damping[0, :7].detach().cpu().numpy()
                    print(
                        f"[SimDataPhysicsTrace] substep={trace_step} "
                        f"q={np.array2string(q_trace, precision=6)} "
                        f"vel={np.array2string(v_trace, precision=6)} "
                        f"q_target={np.array2string(q_target_trace, precision=6)} "
                        f"vel_target={np.array2string(v_target_trace, precision=6)} "
                        f"applied={np.array2string(applied_trace, precision=6)} "
                        f"computed={np.array2string(computed_trace, precision=6)} "
                        f"stiffness={np.array2string(stiffness_trace, precision=3)} "
                        f"damping={np.array2string(damping_trace, precision=3)}",
                        flush=True,
                    )
                self._sim_data_trace_step = trace_step + 1

        end = time.time()
        # print(f"Cost time:{end-start}")
        # print(self.episode_length_buf)

        # -- update env counters (used for curriculum generation)
        self.episode_length_buf += 1  # step in current episode (per env)
        self.common_step_counter += 1  # total step (common for all envs)

        self.reset_terminated[:], self.reset_time_outs[:] = self._get_dones()
        self.reset_buf = self.reset_terminated | self.reset_time_outs
        self.reward_buf = self._get_rewards()

        # -- reset envs that terminated/timed-out and log the episode information
        reset_env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        reset_env_mask = torch.zeros((self.num_envs,), dtype=torch.bool, device=self.device)
        if len(reset_env_ids) > 0:
            reset_env_mask[reset_env_ids] = True
        if len(reset_env_ids) > 0:
            if hasattr(self, "_tavla_mapping_failed"):
                print(
                    "[TAVLA] episode termination: "
                    f"mapping_failed={bool(torch.any(self._tavla_mapping_failed[reset_env_ids]))}, "
                    f"teacher_failures={int(getattr(self, 'teacher_failures', 0))}, "
                    f"force_aborts={int(getattr(self, 'tavla_force_abort_count', 0))}"
                )
            success = self._get_curr_successes(
                success_threshold=self.cfg_task.success_threshold, check_rot=False
            )
            # ep_succeeded is latched during _get_rewards, so preserve
            # success when the final pose has drifted out of tolerance.
            episode_success = torch.logical_or(self.ep_succeeded.bool(), success)
            current_total = len(reset_env_ids)
            current_successes = int(episode_success[reset_env_ids].sum().item())
            current_success_rate = (current_successes / current_total) * 100 if current_total > 0 else 0.0

            self.success_times += current_successes
            self.total_times += current_total
            cumulative_success_rate = (self.success_times / self.total_times) * 100 if self.total_times > 0 else 0.0
            # 保存数据。Any reset closes the current episode buffer. Saving is optional; clearing is not.
            for env_ids in reset_env_ids.to("cpu").numpy().tolist():
                if self.collect_data:
                    if bool(episode_success[env_ids]):
                        print("Task success!")
                        self.cur_num_traj += 1
                        self.save_data_to_disk(env_ids, success=True)
                    elif self.save_failed_trajectory:
                        self.cur_num_traj += 1
                        print("Task Failed!")
                        self.save_data_to_disk(env_ids, success=False)
                    else:
                        print("Task Failed! Discarding trajectory buffer.")
                    self.reset_data_buffer(env_ids)

            self._reset_idx(reset_env_ids)
            avg_reward = self.reward_buf.mean()
            print(
                f"Current Success rate: {current_successes} / {current_total} = "
                f"{current_success_rate:.2f}%"
            )
            print(
                f"Cumulative Success rate: {self.success_times} / {self.total_times} = "
                f"{cumulative_success_rate:.2f}%"
            )
            print(f"Average Reward: {avg_reward.item():.6f}")

            # update articulation kinematics
            self.scene.write_data_to_sim()
            self.sim.forward()
            # if sensors are added to the scene, make sure we render to reflect changes in reset
            if _sim_flag(self.sim, "has_rtx_sensors") and self.cfg.rerender_on_reset:
                self.sim.render()

        if self.collect_data and self.cur_num_traj >= self.num_trajectories and not getattr(self.cfg, "teacher_eval_only", False):
            exit(0)
            
        # Save only environments that did not terminate in this step. This
        # keeps the reset state out of the previous episode's reward buffer.
        self._record_data_mask = ~reset_env_mask

        # post-step: step interval event
        if self.cfg.events:
            if "interval" in self.event_manager.available_modes:
                self.event_manager.apply(mode="interval", dt=self.step_dt)

        # 保存数据
        # post-step:
         # 保存 action 的数据
        if self.collect_data:
            env_num = self.fingertip_midpoint_pos.shape[0]
            ctrl_target_gripper_dof_pos = 0.0
            gripper = torch.tensor(ctrl_target_gripper_dof_pos, device=self.device)
            gripper = gripper.expand(env_num, 1)
            for i in range(self.num_envs):
                if not bool(reset_env_mask[i].item()):
                    self.data_buffers[i]["actions"].append(self.next_action[i])

        # update observations
        self.obs_buf = self._get_observations()
        self._record_data_mask = None

        # add observation noise
        # note: we apply no noise to the state space (since it is used for critic networks)
        if self.cfg.observation_noise_model:
            self.obs_buf["policy"] = self._observation_noise_model(self.obs_buf["policy"])

        # return observations, rewards, resets and extras
        return self.obs_buf, self.reward_buf, self.reset_terminated, self.reset_time_outs, self.extras
    
    def _apply_action(self):
        """FORGE actions are defined as targets relative to the fixed asset."""
        if self.last_update_timestamp < self._robot._data._sim_timestamp:
            self._compute_intermediate_values(dt=self.physics_dt)

        # Step (0): Scale actions to allowed range.
        pos_actions = self.actions[:, 0:3]
        pos_actions = pos_actions @ torch.diag(torch.tensor(self.cfg.ctrl.pos_action_bounds, device=self.device))

        rot_actions = self.actions[:, 3:6]
        rot_actions = rot_actions @ torch.diag(torch.tensor(self.cfg.ctrl.rot_action_bounds, device=self.device))

        # Step (1): Compute desired pose targets in EE frame.
        # (1.a) Position. Action frame is assumed to be the top of the bolt (noisy estimate).
        fixed_pos_action_frame = self.fixed_pos_obs_frame + self.init_fixed_pos_obs_noise
        ctrl_target_fingertip_preclipped_pos = fixed_pos_action_frame + pos_actions
        # (1.b) Enforce rotation action constraints.

        if self.cfg.disable_xy_rot:
            rot_actions[:, 0:2] = 0.0

        # Assumes joint limit is in (+x, -y)-quadrant of world frame.
        rot_actions[:, 2] = np.deg2rad(-180.0) + np.deg2rad(270.0) * (rot_actions[:, 2] + 1.0) / 2.0  # Joint limit.
        # (1.c) Get desired orientation target.
        bolt_frame_quat = torch_utils.quat_from_euler_xyz(
            roll=rot_actions[:, 0], pitch=rot_actions[:, 1], yaw=rot_actions[:, 2]
        )

        rot_180_euler = torch.tensor([np.pi, 0.0, 0.0], device=self.device).repeat(self.num_envs, 1)
        quat_bolt_to_ee = torch_utils.quat_from_euler_xyz(
            roll=rot_180_euler[:, 0], pitch=rot_180_euler[:, 1], yaw=rot_180_euler[:, 2]
        )

        ctrl_target_fingertip_preclipped_quat = torch_utils.quat_mul(quat_bolt_to_ee, bolt_frame_quat)

        # Step (2): Clip targets if they are too far from current EE pose.
        # (2.a): Clip position targets.
        self.delta_pos = ctrl_target_fingertip_preclipped_pos - self.fingertip_midpoint_pos  # Used for action_penalty.
        pos_error_clipped = torch.clip(self.delta_pos, -self.pos_threshold, self.pos_threshold)
        ctrl_target_fingertip_midpoint_pos = self.fingertip_midpoint_pos + pos_error_clipped

        # (2.b) Clip orientation targets. Use Euler angles. We assume we are near upright, so
        # clipping yaw will effectively cause slow motions. When we clip, we also need to make
        # sure we avoid the joint limit.

        # (2.b.i) Get current and desired Euler angles.
        curr_roll, curr_pitch, curr_yaw = torch_utils.get_euler_xyz(self.fingertip_midpoint_quat)
        desired_roll, desired_pitch, desired_yaw = torch_utils.get_euler_xyz(ctrl_target_fingertip_preclipped_quat)
        desired_xyz = torch.stack([desired_roll, desired_pitch, desired_yaw], dim=1)

        # (2.b.ii) Correct the direction of motion to avoid joint limit.
        # Map yaws between [-125, 235] degrees (so that angles appear on a continuous span uninterrupted by the joint limit).
        curr_yaw = factory_utils.wrap_yaw(curr_yaw)
        desired_yaw = factory_utils.wrap_yaw(desired_yaw)

        # (2.b.iii) Clip motion in the correct direction.
        self.delta_yaw = desired_yaw - curr_yaw  # Used later for action_penalty.
        clipped_yaw = torch.clip(self.delta_yaw, -self.rot_threshold[:, 2], self.rot_threshold[:, 2])
        desired_xyz[:, 2] = curr_yaw + clipped_yaw

        # (2.b.iv) Clip roll and pitch.
        desired_roll = torch.where(desired_roll < 0.0, desired_roll + 2 * torch.pi, desired_roll)
        desired_pitch = torch.where(desired_pitch < 0.0, desired_pitch + 2 * torch.pi, desired_pitch)

        delta_roll = desired_roll - curr_roll
        clipped_roll = torch.clip(delta_roll, -self.rot_threshold[:, 0], self.rot_threshold[:, 0])
        desired_xyz[:, 0] = curr_roll + clipped_roll

        curr_pitch = torch.where(curr_pitch > torch.pi, curr_pitch - 2 * torch.pi, curr_pitch)
        desired_pitch = torch.where(desired_pitch > torch.pi, desired_pitch - 2 * torch.pi, desired_pitch)

        delta_pitch = desired_pitch - curr_pitch
        clipped_pitch = torch.clip(delta_pitch, -self.rot_threshold[:, 1], self.rot_threshold[:, 1])
        desired_xyz[:, 1] = curr_pitch + clipped_pitch

        ctrl_target_fingertip_midpoint_quat = self._constrain_fingertip_orientation(
            torch_utils.quat_from_euler_xyz(
                roll=desired_xyz[:, 0], pitch=desired_xyz[:, 1], yaw=desired_xyz[:, 2]
            )
        )
        if self._is_peg_insert_yaw_locked():
            self.delta_yaw = torch.zeros_like(self.delta_yaw)

        # ========== 新增: 判断使用RL action还是Policy action ==========
        if hasattr(self, 'policy') and self.policy is not None:
            # 使用外部policy的action (绝对位置和四元数)
            policy_target_pos = self._limit_cartesian_target_speed(self.next_action[:, :3])
            policy_target_quat = self._constrain_fingertip_orientation(self.next_action[:, 3:7])
            policy_target_quat = self._limit_cartesian_target_orientation(policy_target_quat)
            self.ctrl_target_fingertip_midpoint_pos = policy_target_pos
            self.ctrl_target_fingertip_midpoint_quat = policy_target_quat
            self.ctrl_target_gripper_dof_pos = 0.0
            self.generate_ctrl_signals(
                ctrl_target_fingertip_midpoint_pos=policy_target_pos,
                ctrl_target_fingertip_midpoint_quat=policy_target_quat,
                ctrl_target_gripper_dof_pos=0.0,
            )
        else:
            ctrl_target_gripper_dof_pos = 0.0
            gripper = torch.tensor(ctrl_target_gripper_dof_pos, device="cuda")
            gripper = gripper.expand(self.fingertip_midpoint_pos.shape[0], 1)

            ctrl_target_fingertip_midpoint_pos = self._limit_cartesian_target_speed(
                ctrl_target_fingertip_midpoint_pos
            )
            # Keep the original PPO orientation response. The peg-in-hole-only
            # yaw constraint is applied by _constrain_fingertip_orientation above.
            self.ppo_joint_target = self._taskspace_target_to_joint_target(
                ctrl_target_fingertip_midpoint_pos,
                ctrl_target_fingertip_midpoint_quat,
                gripper,
            )
            # Keep the final PPO command in task space. The joint target is
            # retained for diagnostics/data export, but converting it back via
            # the Jacobian can invalidate the Cartesian speed bound.
            self.next_action = torch.cat(
                [
                    ctrl_target_fingertip_midpoint_pos,
                    ctrl_target_fingertip_midpoint_quat,
                    gripper,
                ],
                dim=1,
            )

            # 使用RL计算的action (相对增量)
            self.ctrl_target_fingertip_midpoint_pos = ctrl_target_fingertip_midpoint_pos
            self.ctrl_target_fingertip_midpoint_quat = ctrl_target_fingertip_midpoint_quat
            self.ctrl_target_gripper_dof_pos = ctrl_target_gripper_dof_pos
            self.generate_ctrl_signals(
                ctrl_target_fingertip_midpoint_pos=ctrl_target_fingertip_midpoint_pos,
                ctrl_target_fingertip_midpoint_quat=ctrl_target_fingertip_midpoint_quat,
                ctrl_target_gripper_dof_pos=ctrl_target_gripper_dof_pos,
            )
        # ===========================================================
        
            
    def select_action(self, policy):
        """
        使用policy为每个环境生成action。
        
        Args:
            policy: Policy模型实例
            
        Returns:
            action_list: 每个环境的action列表
        """
        action_list = []
        for env_idx in range(self.num_envs):
            # 准备当前末端执行器位姿
            cur_ee_pose = torch.cat([
                self.fingertip_midpoint_pos[env_idx], 
                self.fingertip_midpoint_quat[env_idx]
            ], dim=0)
            
            # 准备相机图像
            head_img_tensor = self.tiled_camera.data.output["rgb"][env_idx].to("cpu").clone().unsqueeze(0)
            wrist_img_tensor = self.wrist_tiled_camera.data.output["rgb"][env_idx].to("cpu").clone().unsqueeze(0)
            
            # 准备状态信息 (末端位姿: xyz + quat，不包含 gripper)
            _state = cur_ee_pose.to("cpu").clone()
            _state = _state.view(1, -1)
            
            # 准备任务提示
            prompt_data = self.cfg.task_prompt
            
            # 准备力/力矩传感器数据
            effort = self.wrench_final[env_idx].unsqueeze(0).to("cpu")
            # effort = torch.nn.functional.pad(effort, (0, 2, 0, 0), mode='constant', value=0)
            # print("effort type", type(effort))
            # print("effort ",effort)
            
            # 打包输入字典 (键名必须与模型内部映射一致)
            batch_input = {
                "observation.images.front": head_img_tensor,
                "observation.images.left_wrist": wrist_img_tensor,
                "observation.state": _state,
                "observation.effort": effort,
                "task": prompt_data,
            }
            
            # 调用policy推理
            next_action = policy.select_action(batch_input)
            action_list.append(next_action)
        
        return action_list
    
    def _detach_held_asset(self, env_ids):
        """Remove the peg attachment before resetting its pose."""
        if getattr(self.cfg_task, "name", "") != "peg_insert":
            return
        from pxr import Sdf

        stage = self.sim.stage
        for env_id in env_ids.detach().cpu().tolist():
            joint_path = Sdf.Path(f"/World/envs/env_{int(env_id)}/PegFixedJoint")
            if stage.GetPrimAtPath(joint_path).IsValid():
                stage.RemovePrim(joint_path)

    def _attach_held_asset(self, env_ids):
        """Leave peg as an independent dynamic body held by finger contacts."""

        del env_ids
        return

    def _reset_idx(self, env_ids):
        """Perform additional randomizations."""
        self._detach_held_asset(env_ids)
        super()._reset_idx(env_ids)
        self._cartesian_target_initialized = True
        self._cartesian_target_pos[env_ids] = self.fingertip_midpoint_pos[env_ids].detach()
        self._cartesian_target_linear_velocity[env_ids] = 0.0
        self._cartesian_orientation_initialized = True
        self._cartesian_target_euler[env_ids] = torch.stack(
            torch_utils.get_euler_xyz(self.fingertip_midpoint_quat[env_ids]), dim=1
        ).detach()
        self._cartesian_target_angular_velocity[env_ids] = 0.0
        self._cartesian_target_update_pending = False
        self._cartesian_orientation_update_pending = False
        self._joint_target_initialized = True
        self._joint_target_pos[env_ids] = self.joint_pos[env_ids, :7].detach()
        self._joint_target_velocity[env_ids] = 0.0
        self._joint_target_update_pending = False
        # ========== 新增: 重置policy状态 ==========
        if hasattr(self, 'policy') and self.policy is not None:
            self.policy.reset()
        # =========================================
        
        
    def randomize_initial_state(self, env_ids):
        """Randomize initial state and perform any episode-level randomization."""
        # Disable gravity.
        physics_sim_view = sim_utils.SimulationContext.instance().physics_sim_view
        physics_sim_view.set_gravity(carb.Float3(0.0, 0.0, 0.0))

        # (1.) Randomize fixed asset pose.
        fixed_state = self._fixed_asset.data.default_root_state.clone()[env_ids]
        # (1.a.) Position
        # Evaluators may provide the same per-episode offsets used by the Pi0
        # randomized evaluator.  With no schedule, preserve the original
        # seeded randomization exactly.
        reset_schedule = getattr(self, "tavla_reset_schedule", None)
        reset_schedule_index = int(getattr(self, "tavla_reset_schedule_index", 0))
        schedule_row = None
        if reset_schedule:
            schedule_row = reset_schedule[min(reset_schedule_index, len(reset_schedule) - 1)]
            fixed_pos_init_rand = torch.as_tensor(
                schedule_row["hole_offset_m"], dtype=torch.float32, device=self.device
            ).view(1, 3).repeat(len(env_ids), 1)
        else:
            rand_sample = torch.rand(
                (len(env_ids), 3), generator=self.rng, dtype=torch.float32, device=self.device
            )
            fixed_pos_init_rand = 2 * (rand_sample - 0.5)  # [-1, 1]
            fixed_asset_init_pos_rand = torch.tensor(
                self.cfg_task.fixed_asset_init_pos_noise, dtype=torch.float32, device=self.device
            )
            fixed_pos_init_rand = fixed_pos_init_rand @ torch.diag(fixed_asset_init_pos_rand)
        if not hasattr(self, "tavla_last_hole_position_offset_m"):
            self.tavla_last_hole_position_offset_m = torch.zeros(
                (self.num_envs, 3), dtype=torch.float32, device=self.device
            )
        self.tavla_last_hole_position_offset_m[env_ids] = fixed_pos_init_rand
        fixed_state[:, 0:3] += fixed_pos_init_rand + self.scene.env_origins[env_ids]
        # (1.b.) Orientation
        fixed_orn_init_yaw = np.deg2rad(self.cfg_task.fixed_asset_init_orn_deg)
        fixed_orn_yaw_range = np.deg2rad(self.cfg_task.fixed_asset_init_orn_range_deg)
        
        # [MODIFIED] 使用局部生成器 self.rng
        rand_sample = torch.rand((len(env_ids), 3), generator=self.rng, dtype=torch.float32, device=self.device)
        
        fixed_orn_euler = fixed_orn_init_yaw + fixed_orn_yaw_range * rand_sample
        fixed_orn_euler[:, 0:2] = 0.0  # Only change yaw.
        fixed_orn_quat = torch_utils.quat_from_euler_xyz(
            fixed_orn_euler[:, 0], fixed_orn_euler[:, 1], fixed_orn_euler[:, 2]
        )
        fixed_state[:, 3:7] = fixed_orn_quat
        use_pi0_reset_pose = bool(getattr(self.cfg_task, "use_pi0_reset_pose", False))
        if use_pi0_reset_pose and hasattr(self.cfg, "pi0_hole_init_rot"):
            fixed_state[:, 3:7] = torch.as_tensor(
                self.cfg.pi0_hole_init_rot, dtype=torch.float32, device=self.device
            ).view(1, 4)
        # (1.c.) Velocity
        fixed_state[:, 7:] = 0.0  # vel
        # (1.d.) Update values.
        self._fixed_asset.write_root_pose_to_sim(fixed_state[:, 0:7], env_ids=env_ids)
        self._fixed_asset.write_root_velocity_to_sim(fixed_state[:, 7:], env_ids=env_ids)
        self._fixed_asset.reset()

        # (1.e.) Noisy position observation.
        # [MODIFIED] 使用局部生成器 self.rng
        fixed_asset_pos_noise = torch.randn((len(env_ids), 3), generator=self.rng, dtype=torch.float32, device=self.device)
        
        fixed_asset_pos_rand = torch.tensor(self.cfg.obs_rand.fixed_asset_pos, dtype=torch.float32, device=self.device)
        fixed_asset_pos_noise = fixed_asset_pos_noise @ torch.diag(fixed_asset_pos_rand)
        self.init_fixed_pos_obs_noise[env_ids] = fixed_asset_pos_noise

        self.step_sim_no_action()

        # Compute the frame on the bolt that would be used as observation: fixed_pos_obs_frame
        # For example, the tip of the bolt can be used as the observation frame
        fixed_tip_pos_local = torch.zeros((self.num_envs, 3), device=self.device)
        fixed_tip_pos_local[:, 2] += self.cfg_task.fixed_asset_cfg.height
        fixed_tip_pos_local[:, 2] += self.cfg_task.fixed_asset_cfg.base_height
        if self.cfg_task.name == "gear_mesh":
            fixed_tip_pos_local[:, 0] = self.cfg_task.fixed_asset_cfg.medium_gear_base_offset[0]

        _, fixed_tip_pos = torch_utils.tf_combine(
            self.fixed_quat,
            self.fixed_pos,
            torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device).unsqueeze(0).repeat(self.num_envs, 1),
            fixed_tip_pos_local,
        )
        self.fixed_pos_obs_frame[:] = fixed_tip_pos
        # (2) Move gripper to randomizes location above fixed asset. Keep trying until IK succeeds.
        # (a) get position vector to target
        hand_down_quat = torch.zeros((self.num_envs, 4), dtype=torch.float32, device=self.device)
        bad_envs = env_ids.clone()
        ik_attempt = 0
        MAX_ATTEMPTS = 10
        reset_ik_debug = bool(getattr(self.cfg_task, "reset_ik_debug", False))
        debug_print = print if reset_ik_debug else (lambda *args, **kwargs: None)
        skip_reset_ik = bool(getattr(self.cfg_task, "skip_reset_ik", False))
        if skip_reset_ik:
            direct_reset_schedule = getattr(self, "tavla_direct_reset_joints_schedule", None)
            direct_reset_index = int(getattr(self, "tavla_reset_schedule_index", 0))
            if direct_reset_schedule:
                direct_reset_joints = direct_reset_schedule[
                    min(direct_reset_index, len(direct_reset_schedule) - 1)
                ]
                self._set_franka_to_default_pose(
                    joints=direct_reset_joints,
                    env_ids=env_ids,
                )
                print(
                    "[RealSim] reset IK skipped; using PI0 direct joints "
                    f"for episode index {direct_reset_index}."
                )
            else:
                print("[RealSim] reset IK skipped; using reset_joints directly.")

        while not skip_reset_ik:
            n_bad = bad_envs.shape[0]

            # ✅ 打印fingertip当前位姿
            fingertip_pos_cur = self._robot.data.body_pos_w[:, self.fingertip_body_idx]
            fingertip_quat_cur = self._robot.data.body_quat_w[:, self.fingertip_body_idx]
            debug_print(f"\n[DEBUG] ========== IK attempt {ik_attempt} | bad_envs: {n_bad} ==========")
            debug_print(f"[DEBUG] fingertip_pos  (current)[0]: {fingertip_pos_cur[0].cpu().numpy()}")
            debug_print(f"[DEBUG] fingertip_quat (current)[0]: {fingertip_quat_cur[0].cpu().numpy()}")

            # ✅ 计算目标位置
            above_fixed_pos = fixed_tip_pos.clone()
            above_fixed_pos[:, 2] += self.cfg_task.hand_init_pos[2]
            debug_print(f"[DEBUG] fixed_tip_pos          [0]: {fixed_tip_pos[0].cpu().numpy()}")
            debug_print(f"[DEBUG] hand_init_pos z offset    : {self.cfg_task.hand_init_pos[2]}")
            debug_print(f"[DEBUG] above_fixed_pos (no noise)[0]: {above_fixed_pos[0].cpu().numpy()}")

            # ✅ 加噪声
            if schedule_row is not None:
                above_fixed_pos_rand = torch.as_tensor(
                    schedule_row["hand_offset_m"], dtype=torch.float32, device=self.device
                ).view(1, 3).repeat(n_bad, 1)
            else:
                rand_sample = torch.rand(
                    (n_bad, 3), generator=self.rng, dtype=torch.float32, device=self.device
                )
                above_fixed_pos_rand = 2 * (rand_sample - 0.5)
                hand_init_pos_rand = torch.tensor(self.cfg_task.hand_init_pos_noise, device=self.device)
                above_fixed_pos_rand = above_fixed_pos_rand @ torch.diag(hand_init_pos_rand)
            if not hasattr(self, "tavla_last_hand_position_offset_m"):
                self.tavla_last_hand_position_offset_m = torch.zeros(
                    (self.num_envs, 3), dtype=torch.float32, device=self.device
                )
            self.tavla_last_hand_position_offset_m[bad_envs] = above_fixed_pos_rand
            above_fixed_pos[bad_envs] += above_fixed_pos_rand
            debug_print(f"[DEBUG] above_fixed_pos (w/ noise)[0]: {above_fixed_pos[0].cpu().numpy()}")

            # ✅ 打印目标和当前fingertip的差值
            delta = above_fixed_pos - fingertip_pos_cur
            debug_print(f"[DEBUG] pos delta (target - current)[0]: {delta[0].cpu().numpy()}")
            debug_print(f"[DEBUG] pos delta norm[0]: {torch.linalg.norm(delta[0]):.4f} m")

            # ✅ 计算目标姿态
            hand_down_euler = (
                torch.tensor(self.cfg_task.hand_init_orn, device=self.device).unsqueeze(0).repeat(n_bad, 1)
            )
            rand_sample = torch.rand((n_bad, 3), generator=self.rng, dtype=torch.float32, device=self.device)
            above_fixed_orn_noise = 2 * (rand_sample - 0.5)
            hand_init_orn_rand = torch.tensor(self.cfg_task.hand_init_orn_noise, device=self.device)
            above_fixed_orn_noise = above_fixed_orn_noise @ torch.diag(hand_init_orn_rand)
            hand_down_euler += above_fixed_orn_noise
            debug_print(f"[DEBUG] hand_down_euler (w/ noise)[0]: {hand_down_euler[0].cpu().numpy()}")

            hand_down_quat[bad_envs, :] = torch_utils.quat_from_euler_xyz(
                roll=hand_down_euler[:, 0], pitch=hand_down_euler[:, 1], yaw=hand_down_euler[:, 2]
            )
            debug_print(f"[DEBUG] hand_down_quat (target)  [0]: {hand_down_quat[0].cpu().numpy()}")

            # ✅ 执行IK
            pos_error, aa_error = self.set_pos_inverse_kinematics(
                ctrl_target_fingertip_midpoint_pos=above_fixed_pos,
                ctrl_target_fingertip_midpoint_quat=hand_down_quat,
                env_ids=bad_envs,
            )

            pos_err_val = torch.linalg.norm(pos_error, dim=1)
            angle_err_val = torch.norm(aa_error, dim=1)
            debug_print(f"[DEBUG] pos_err   — max: {pos_err_val.max():.4f} | mean: {pos_err_val.mean():.4f}")
            debug_print(f"[DEBUG] angle_err — max: {angle_err_val.max():.4f} | mean: {angle_err_val.mean():.4f}")

            pos_error_mask = pos_err_val > 1e-3
            angle_error_mask = angle_err_val > 1e-3
            any_error = torch.logical_or(pos_error_mask, angle_error_mask)
            bad_envs = bad_envs[any_error.nonzero(as_tuple=False).squeeze(-1)]
            debug_print(f"[DEBUG] 收敛后仍失败的env数: {bad_envs.shape[0]}")

            if bad_envs.shape[0] == 0:
                debug_print(f"[DEBUG] ✅ IK 全部收敛，共尝试 {ik_attempt + 1} 次")
                break

            # ✅ 超过最大次数强制退出
            if ik_attempt >= MAX_ATTEMPTS:
                print(f"[WARNING] ❌ IK 在 {MAX_ATTEMPTS} 次后仍未收敛!")
                print(f"[WARNING] 仍失败的env索引: {bad_envs.cpu().numpy()}")
                print(f"[WARNING] 最后目标位置:  {above_fixed_pos[bad_envs[0]].cpu().numpy()}")
                print(f"[WARNING] 最后fingertip: {fingertip_pos_cur[bad_envs[0]].cpu().numpy()}")
                print(f"[WARNING] pos_err:   {pos_err_val[any_error].cpu().numpy()}")
                print(f"[WARNING] angle_err: {angle_err_val[any_error].cpu().numpy()}")
                break

            self._set_franka_to_default_pose(
                joints=self.cfg.ctrl.reset_joints,
                env_ids=bad_envs
            )
            ik_attempt += 1

        self.step_sim_no_action()
        if schedule_row is not None:
            # DirectRLEnv performs an internal reset after every completed
            # episode; consume exactly one schedule row per reset and reuse the
            # final row only for any cleanup reset after the requested run.
            self.tavla_reset_schedule_index = min(
                reset_schedule_index + 1, len(reset_schedule) - 1
            )

        # Add flanking gears after servo (so arm doesn't move them).
        if self.cfg_task.name == "gear_mesh" and self.cfg_task.add_flanking_gears:
            small_gear_state = self._small_gear_asset.data.default_root_state.clone()[env_ids]
            small_gear_state[:, 0:7] = fixed_state[:, 0:7]
            small_gear_state[:, 7:] = 0.0  # vel
            self._small_gear_asset.write_root_pose_to_sim(small_gear_state[:, 0:7], env_ids=env_ids)
            self._small_gear_asset.write_root_velocity_to_sim(small_gear_state[:, 7:], env_ids=env_ids)
            self._small_gear_asset.reset()

            large_gear_state = self._large_gear_asset.data.default_root_state.clone()[env_ids]
            large_gear_state[:, 0:7] = fixed_state[:, 0:7]
            large_gear_state[:, 7:] = 0.0  # vel
            self._large_gear_asset.write_root_pose_to_sim(large_gear_state[:, 0:7], env_ids=env_ids)
            self._large_gear_asset.write_root_velocity_to_sim(large_gear_state[:, 7:], env_ids=env_ids)
            self._large_gear_asset.reset()

        # (3) Randomize asset-in-gripper location.
        # flip gripper z orientation
        flip_z_quat = torch.tensor([0.0, 0.0, 1.0, 0.0], device=self.device).unsqueeze(0).repeat(self.num_envs, 1)
        fingertip_flipped_quat, fingertip_flipped_pos = torch_utils.tf_combine(
            q1=self.fingertip_midpoint_quat,
            t1=self.fingertip_midpoint_pos,
            q2=flip_z_quat,
            t2=torch.zeros((self.num_envs, 3), device=self.device),
        )

        # get default gripper in asset transform
        held_asset_relative_pos, held_asset_relative_quat = self.get_handheld_asset_relative_pose()
        asset_in_hand_quat, asset_in_hand_pos = torch_utils.tf_inverse(
            held_asset_relative_quat, held_asset_relative_pos
        )

        translated_held_asset_quat, translated_held_asset_pos = torch_utils.tf_combine(
            q1=fingertip_flipped_quat, t1=fingertip_flipped_pos, q2=asset_in_hand_quat, t2=asset_in_hand_pos
        )

        # Add asset in hand randomization
        # [MODIFIED] 使用局部生成器 self.rng
        # 注意：原文这里是对 self.num_envs 进行随机，建议保持原逻辑以防维度不匹配
        # 如果只想对 env_ids 随机，需要修改这部分逻辑，但为了稳妥起见，我们只替换生成器
        rand_sample = torch.rand((self.num_envs, 3), generator=self.rng, dtype=torch.float32, device=self.device)
        
        held_asset_pos_noise = 2 * (rand_sample - 0.5)  # [-1, 1]
        if self.cfg_task.name == "gear_mesh":
            held_asset_pos_noise[:, 2] = -rand_sample[:, 2]  # [-1, 0]

        held_asset_pos_noise_level = torch.tensor(self.cfg_task.held_asset_pos_noise, device=self.device)
        held_asset_pos_noise = held_asset_pos_noise @ torch.diag(held_asset_pos_noise_level)
        # translated_held_asset_quat, translated_held_asset_pos = torch_utils.tf_combine(
        #     q1=translated_held_asset_quat,
        #     t1=translated_held_asset_pos,
        #     q2=torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device).unsqueeze(0).repeat(self.num_envs, 1),
        #     t2=held_asset_pos_noise,
        # )
        
        # ================= NEW: 添加旋转随机化 =================
        if self.cfg_task.name == "peg_insert" and not use_pi0_reset_pose:
            rot_noise_deg = float(getattr(self.cfg_task, "held_asset_rot_noise_deg", 0.0))
            rot_noise_rad = np.deg2rad(rot_noise_deg)
            
            # [MODIFIED] 使用局部生成器 self.rng
            rand_rot_sample = torch.rand((self.num_envs, 3), generator=self.rng, dtype=torch.float32, device=self.device)
            
            held_asset_rpy_noise = 2 * (rand_rot_sample - 0.5) * rot_noise_rad 

            # 转换为四元数
            held_asset_rot_noise_quat = torch_utils.quat_from_euler_xyz(
                held_asset_rpy_noise[:, 0], 
                held_asset_rpy_noise[:, 1], 
                held_asset_rpy_noise[:, 2]
            )

            # 3. 第一步结合：将噪声施加到夹爪坐标系上
            # 这一步得到的是一个“带有误差的夹爪中心位姿”
            noisy_gripper_quat, noisy_gripper_pos = torch_utils.tf_combine(
                q1=fingertip_flipped_quat, 
                t1=fingertip_flipped_pos, 
                q2=held_asset_rot_noise_quat,   # 在夹爪中心施加旋转
                t2=held_asset_pos_noise         # 在夹爪中心施加位移
            )

            # 4. 第二步结合：加上物体相对于夹爪的固定偏移
            # 因为前一步旋转了坐标系，这一步的偏移向量会跟着旋转，从而实现“绕点旋转”
            translated_held_asset_quat, translated_held_asset_pos = torch_utils.tf_combine(
                q1=noisy_gripper_quat, 
                t1=noisy_gripper_pos, 
                q2=asset_in_hand_quat, 
                t2=asset_in_hand_pos
            )
            debug_print("随机位置和旋转")
        elif not use_pi0_reset_pose:
            translated_held_asset_quat, translated_held_asset_pos = torch_utils.tf_combine(
                q1=translated_held_asset_quat,
                t1=translated_held_asset_pos,
                q2=torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device).unsqueeze(0).repeat(self.num_envs, 1),
                t2=held_asset_pos_noise,
            )

        if use_pi0_reset_pose:
            # Use Pi0's IsaacLab-6 XYZW frame chain for the held peg. The
            # legacy RealSim chain flips the peg into a different position.
            identity_quat = torch.zeros((self.num_envs, 4), device=self.device)
            identity_quat[:, 3] = 1.0
            zero_pos = torch.zeros((self.num_envs, 3), device=self.device)
            flip_y_quat = torch.zeros((self.num_envs, 4), device=self.device)
            flip_y_quat[:, 1] = 1.0
            flipped_pos, flipped_quat = isaaclab_math.combine_frame_transforms(
                self.fingertip_midpoint_pos,
                self.fingertip_midpoint_quat,
                zero_pos,
                flip_y_quat,
            )
            relative_pos = torch.zeros((self.num_envs, 3), device=self.device)
            relative_pos[:, 2] = (0.050 - 0.017608) + float(
                getattr(self.cfg, "pi0_peg_mount_depth_adjust_m", 0.0)
            )
            translated_held_asset_pos, translated_held_asset_quat = (
                isaaclab_math.combine_frame_transforms(
                    flipped_pos,
                    flipped_quat,
                    -relative_pos,
                    identity_quat,
                )
            )

        held_state = self._held_asset.data.default_root_state.clone()
        held_state[:, 0:3] = translated_held_asset_pos + self.scene.env_origins
        held_state[:, 3:7] = translated_held_asset_quat
        held_state[:, 7:] = 0.0
        self._held_asset.write_root_pose_to_sim(held_state[:, 0:7])
        self._held_asset.write_root_velocity_to_sim(held_state[:, 7:])
        self._held_asset.reset()

        #  Close hand
        # Set gains to use for quick resets.
        reset_task_prop_gains = torch.tensor(self.cfg.ctrl.reset_task_prop_gains, device=self.device).repeat(
            (self.num_envs, 1)
        )
        self.task_prop_gains = reset_task_prop_gains
        self.task_deriv_gains = factory_utils.get_deriv_gains(
            reset_task_prop_gains, self.cfg.ctrl.reset_rot_deriv_scale
        )

        self.step_sim_no_action()

        grasp_time = 0.0
        grasp_close_time_s = float(getattr(self.cfg_task, "grasp_close_time_s", 0.25))
        while grasp_time < grasp_close_time_s:
            self.ctrl_target_joint_pos[env_ids, 7:] = 0.0  # Close gripper.
            self.close_gripper_in_place()
            self.step_sim_no_action()
            grasp_time += self.sim.get_physics_dt()

        if getattr(self.cfg_task, "snap_held_asset_after_grasp", False):
            flip_z_quat = torch.tensor([0.0, 0.0, 1.0, 0.0], dtype=torch.float32, device=self.device).unsqueeze(0).repeat(
                self.num_envs, 1
            )
            fingertip_flipped_quat, fingertip_flipped_pos = torch_utils.tf_combine(
                q1=self.fingertip_midpoint_quat,
                t1=self.fingertip_midpoint_pos,
                q2=flip_z_quat,
                t2=torch.zeros((self.num_envs, 3), device=self.device),
            )
            held_asset_relative_pos, held_asset_relative_quat = self.get_handheld_asset_relative_pose()
            asset_in_hand_quat, asset_in_hand_pos = torch_utils.tf_inverse(
                held_asset_relative_quat, held_asset_relative_pos
            )
            snapped_held_asset_quat, snapped_held_asset_pos = torch_utils.tf_combine(
                q1=fingertip_flipped_quat,
                t1=fingertip_flipped_pos,
                q2=asset_in_hand_quat,
                t2=asset_in_hand_pos,
            )
            held_state = self._held_asset.data.root_state_w.clone()
            held_state[:, 0:3] = snapped_held_asset_pos + self.scene.env_origins
            held_state[:, 3:7] = snapped_held_asset_quat
            held_state[:, 7:] = 0.0
            self._held_asset.write_root_pose_to_sim(held_state[env_ids, 0:7], env_ids=env_ids)
            self._held_asset.write_root_velocity_to_sim(held_state[env_ids, 7:], env_ids=env_ids)
            self._held_asset.reset()
            if use_pi0_reset_pose:
                identity_quat = torch.zeros((self.num_envs, 4), device=self.device)
                identity_quat[:, 3] = 1.0
                zero_pos = torch.zeros((self.num_envs, 3), device=self.device)
                flip_y_quat = torch.zeros((self.num_envs, 4), device=self.device)
                flip_y_quat[:, 1] = 1.0
                flipped_pos, flipped_quat = isaaclab_math.combine_frame_transforms(
                    self.fingertip_midpoint_pos,
                    self.fingertip_midpoint_quat,
                    zero_pos,
                    flip_y_quat,
                )
                relative_pos = torch.zeros((self.num_envs, 3), device=self.device)
                relative_pos[:, 2] = (0.050 - 0.017608) + float(
                    getattr(self.cfg, "pi0_peg_mount_depth_adjust_m", 0.0)
                )
                snapped_held_asset_pos, snapped_held_asset_quat = (
                    isaaclab_math.combine_frame_transforms(
                        flipped_pos,
                        flipped_quat,
                        -relative_pos,
                        identity_quat,
                    )
                )
                held_state[:, 0:3] = snapped_held_asset_pos + self.scene.env_origins
                held_state[:, 3:7] = snapped_held_asset_quat
                self._held_asset.write_root_pose_to_sim(
                    held_state[env_ids, 0:7], env_ids=env_ids
                )
                self._held_asset.write_root_velocity_to_sim(
                    held_state[env_ids, 7:], env_ids=env_ids
                )
                self._held_asset.reset()
            self.step_sim_no_action()

        grasp_settle_time_s = float(getattr(self.cfg_task, "grasp_settle_time_s", 0.0))
        grasp_settle_time = 0.0
        while grasp_settle_time < grasp_settle_time_s:
            self.close_gripper_in_place()
            self.step_sim_no_action()
            grasp_settle_time += self.sim.get_physics_dt()

        self._attach_held_asset(env_ids)

        self.prev_joint_pos = self.joint_pos[:, 0:7].clone()
        self.prev_fingertip_pos = self.fingertip_midpoint_pos.clone()
        self.prev_fingertip_quat = self.fingertip_midpoint_quat.clone()

        # Set initial actions to involve no-movement. Needed for EMA/correct penalties.
        self.actions = torch.zeros_like(self.actions)
        self.prev_actions = torch.zeros_like(self.actions)

        # Zero initial velocity.
        self.ee_angvel_fd[:, :] = 0.0
        self.ee_linvel_fd[:, :] = 0.0

        # Set initial gains for the episode.
        self.task_prop_gains = self.default_gains
        self.task_deriv_gains = factory_utils.get_deriv_gains(self.default_gains)

        physics_sim_view.set_gravity(carb.Float3(*self.cfg.sim.gravity))
    def _joint_position_limits(self):
        """Return arm joint limits in the batched shape used by the controller."""
        limits = getattr(self._robot.data, "soft_joint_pos_limits", None)
        if limits is None:
            limits = getattr(self._robot.data, "joint_pos_limits", None)
        if limits is None:
            raise RuntimeError("Isaac articulation does not expose joint position limits")
        if limits.ndim == 2:
            limits = limits.unsqueeze(0)
        return limits[:, :7, 0], limits[:, :7, 1]

    def _taskspace_target_to_joint_target(self, target_pos, target_quat, gripper_fraction):
        """Convert the executed PPO task-space target to an 8D TAVLA target."""
        position_error, rotation_error = factory_control.get_pose_error(
            fingertip_midpoint_pos=self.fingertip_midpoint_pos,
            fingertip_midpoint_quat=self.fingertip_midpoint_quat,
            ctrl_target_fingertip_midpoint_pos=target_pos,
            ctrl_target_fingertip_midpoint_quat=target_quat,
            jacobian_type="geometric",
            rot_error_type="axis_angle",
        )
        delta_pose = torch.cat((position_error, rotation_error), dim=-1)
        delta_q = factory_control.get_delta_dof_pos(
            delta_pose,
            "dls",
            self.fingertip_midpoint_jacobian[:, :6, :7],
            self.device,
        )
        q_target = self.joint_pos[:, :7] + delta_q
        lower, upper = self._joint_position_limits()
        q_target = torch.minimum(torch.maximum(q_target, lower), upper)
        if getattr(self.cfg.ctrl, "joint_target_dynamics_limit_enabled", True):
            if not self._joint_target_initialized:
                self._joint_target_pos = self.joint_pos[:, :7].detach().clone()
                self._joint_target_velocity.zero_()
                self._joint_target_initialized = True
            if not self._joint_target_update_pending:
                q_target = self._joint_target_pos
            else:
                dt = float(self.step_dt)
                desired_velocity = (q_target - self._joint_target_pos) / max(dt, 1.0e-6)
                max_speed = max(
                    float(getattr(self.cfg.ctrl, "joint_target_velocity_limit_radps", 0.5)), 0.0
                )
                desired_velocity = torch.clamp(desired_velocity, -max_speed, max_speed)
                max_velocity_change = max(
                    float(getattr(self.cfg.ctrl, "joint_target_acceleration_limit_radps2", 0.7)), 0.0
                ) * dt
                limited_velocity = self._joint_target_velocity + torch.clamp(
                    desired_velocity - self._joint_target_velocity,
                    -max_velocity_change,
                    max_velocity_change,
                )
                q_target = self._joint_target_pos + limited_velocity * dt
                q_target = torch.minimum(torch.maximum(q_target, lower), upper)
                self._joint_target_velocity = limited_velocity.detach().clone()
                self._joint_target_pos = q_target.detach().clone()
                self._joint_target_update_pending = False
        if gripper_fraction.ndim == 1:
            gripper_fraction = gripper_fraction.unsqueeze(-1)
        gripper_fraction = torch.clamp(gripper_fraction, 0.0, 1.0)
        return torch.cat((q_target, gripper_fraction), dim=-1)

    def _joint_target_to_taskspace_target(self, q_target):
        """Map the limited joint target back to a task-space target for the existing controller."""
        q_delta = q_target[:, :7] - self.joint_pos[:, :7]
        delta_twist = (
            self.fingertip_midpoint_jacobian[:, :6, :7] @ q_delta.unsqueeze(-1)
        ).squeeze(-1)
        target_pos = self.fingertip_midpoint_pos + delta_twist[:, :3]
        rotation_delta = delta_twist[:, 3:]
        angle = torch.linalg.vector_norm(rotation_delta, dim=-1)
        axis = rotation_delta / torch.clamp(angle.unsqueeze(-1), min=1.0e-6)
        delta_quat = torch_utils.quat_from_angle_axis(angle, axis)
        identity = torch.zeros_like(delta_quat)
        identity[:, 0] = 1.0
        delta_quat = torch.where((angle > 1.0e-6).unsqueeze(-1), delta_quat, identity)
        target_quat = torch_utils.quat_mul(delta_quat, self.fingertip_midpoint_quat)
        return target_pos, target_quat
