"""Isaac Sim 6.0 standalone RealSim peg-in-hole PPO smoke trainer.

This file intentionally does not modify or import the legacy RealSimEnv step
and reset implementations.  It reuses the existing RealSim Franka/peg/hole
configuration, but follows the Isaac Lab 3.x FactoryEnv scene, reset, IK, and
rendering path.  The play path loads one visual copy of the user's background USD outside the
vectorized environment. The reconstructed Gaussian tabletop is replaced by
calibrated geometric tabletop regions, with separate static PhysX collision
geometry in the PPO/TAVLA frame.

The default run is deliberately small:
    num_envs=4, horizon_length=64, minibatch_size=256, max_iterations=1

Override --max_iterations only after the smoke run is clean.
"""

from __future__ import annotations

import argparse
import json
import csv
import tempfile
import subprocess
import numpy as np
import os
import sys
import time
from pathlib import Path

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--max_iterations", type=int, default=1)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--play", action="store_true")
parser.add_argument("--checkpoint", type=str, default=None)
parser.add_argument("--output_dir", type=str, default="data/realsim_isaac6_play")
parser.add_argument("--num_episodes", type=int, default=1)
parser.add_argument("--save_failed_episodes", action="store_true")
parser.add_argument("--play_background", action="store_true")
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0], *hydra_args]

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


# Isaac Sim 6 keeps the legacy Semantics Python extension in an extension
# cache.  TacEx's sensor package still imports the legacy module while its
# task package is scanned.  Add the cache before importing tacex_tasks.
import isaacsim  # noqa: E402

_isaacsim_root = Path(isaacsim.__file__).resolve().parent
_isaacsim_exts = _isaacsim_root / "exts"
_isaacsim_extscache = _isaacsim_root / "extscache"
for _cache_dir in _isaacsim_extscache.glob("omni.usd.schema.semantics-*"):
    if _cache_dir.is_dir():
        sys.path.append(str(_cache_dir))
for _extension_name in ("isaacsim.core.prims", "isaacsim.core.utils", "isaacsim.core.api"):
    _extension_pkg = _isaacsim_exts / _extension_name / "isaacsim"
    if _extension_pkg.is_dir() and str(_extension_pkg) not in isaacsim.__path__:
        isaacsim.__path__.append(str(_extension_pkg))


import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
import yaml  # noqa: E402
import carb  # noqa: E402
from isaaclab.assets import Articulation  # noqa: E402
from isaaclab.sensors import Camera, CameraCfg  # noqa: E402
import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.envs import DirectMARLEnv  # noqa: E402
from isaaclab_rl.rl_games import RlGamesGpuEnv, RlGamesVecEnvWrapper  # noqa: E402
from isaaclab.utils.assets import retrieve_file_path  # noqa: E402
from isaaclab.utils import math as isaaclab_math  # noqa: E402
from rl_games.common import env_configurations, vecenv  # noqa: E402
from rl_games.common.algo_observer import IsaacAlgoObserver  # noqa: E402
from rl_games.torch_runner import Runner  # noqa: E402
from rl_games.common.player import BasePlayer  # noqa: E402
from pxr import Gf, Sdf, Usd, UsdGeom, UsdShade  # noqa: E402


from isaaclab_tasks.direct.factory.factory_env import FactoryEnv  # noqa: E402
from isaaclab_tasks.direct.factory.factory_env_cfg import FactoryTaskPegInsertCfg  # noqa: E402
from isaaclab_tasks.direct.factory import factory_utils  # noqa: E402
from isaaclab.envs import multi_agent_to_single_agent  # noqa: E402
_LOCAL_ASSET_DIR = Path(__file__).resolve().parents[3] / "assets" / "Factory"


_PLAY_BACKGROUND_SOURCE = Path(__file__).resolve().parents[3] / "franka_env_background_edit" / "franka_env.usd"
_PLAY_BACKGROUND_USD = None
_PLAY_BACKGROUND_WRIST_CAMERA_PRIM = (
    "/World/envs/env_.*/franka_env/Robot/franka/"
    "panda_link7/panda_link8/panda_hand/wrist_camera"
)

# Pose of the active PPO Franka in the authored RealSim background.
# The parent /World/Robot is Rz(-90 deg) and the local franka offset is
# (-0.16491293, 0, 0), giving this world-space root pose.
_PLAY_VISUAL_TRANSLATION = (-0.23579, 0.07709, 0.07101863)
_PLAY_VISUAL_ROTATION_XYZW = (0.0, 0.0, -0.70710678, 0.70710678)

# The calibrated peg root offset from the fingertip center is 50 - 17.608 =
# 32.392 mm. Keep the same geometry used by the original replay script.
_PEG_FRANKA_FINGERPAD_LENGTH_M = 0.017608
_PEG_ROOT_OFFSET_M = 0.050 - _PEG_FRANKA_FINGERPAD_LENGTH_M


def _make_rollout_franka_usd():
    """Expose rollout's complete black-gripper Franka as one active asset."""
    generated = Path("/tmp/tacex_isaac6_rollout_franka_full_black_v1.usd")
    if not generated.is_file():
        if not _PLAY_BACKGROUND_SOURCE.is_file():
            raise FileNotFoundError(_PLAY_BACKGROUND_SOURCE)
        stage = Usd.Stage.CreateNew(str(generated))
        root = stage.DefinePrim("/franka", "Xform")
        root.GetReferences().AddReference(
            str(_PLAY_BACKGROUND_SOURCE), "/World/Robot/franka"
        )
        stage.SetDefaultPrim(root)
        stage.GetRootLayer().Save()
    return generated

# Wrist-only visual workspace from the working 5.1/Isaac6 replay. The
# replacement cubes are authored in the background branch and are visible only
# during the deferred wrist pass; the front Gaussian stream is untouched.
_PLAY_TABLE_SURFACE_Z_VISUAL = _PLAY_VISUAL_TRANSLATION[2]
_PLAY_TABLE_HALF_THICKNESS = 0.01
_PLAY_TABLE_REGIONS = (
    # name, x_min, x_max, y_min, y_max, diffuse RGB, roughness
    # The wrist frustum reaches the old white-table region after the Isaac6
    # visual-frame transform. It is the same green work surface seen by the
    # front camera in the real scene, so keep the whole wrist workspace green
    # except for the silver/brown seam. These meshes are hidden in front pass.
    ("white_table", -5.00, 5.00, -5.00, -0.580, (0.003, 0.115, 0.103), 0.90),
    ("silver_table", -5.00, -0.350, -0.580, 5.00, (0.087, 0.05, 0.028), 0.72),
    ("green_mat", -0.350, 5.00, -0.580, 5.00, (0.003, 0.115, 0.103), 0.94),
)


def _spawn_play_table_geometry(env):
    """Create the working rollout-compatible visual tabletop meshes.

    The legacy rollout uses visual-only cubes in the background branch. Keep
    PPO physics unchanged and avoid adding a second collision scene during
    deferred camera rendering.
    """
    stage = env.sim.stage
    visual_z = _PLAY_TABLE_SURFACE_Z_VISUAL + _PLAY_TABLE_HALF_THICKNESS
    def define_cube(prim_path, size, center):
        # Use IsaacLab's official spawn_cuboid hierarchy. Isaac6 RTX/Fabric
        # registers the geometry/mesh child reliably; a Cube at the root prim
        # can remain USD-visible but be omitted from the render scene.
        root = UsdGeom.Xform.Define(stage, prim_path)
        root.AddTranslateOp().Set(Gf.Vec3d(*center))
        mesh = UsdGeom.Cube.Define(stage, f"{prim_path}/geometry/mesh")
        base_size = min(size)
        mesh.CreateSizeAttr(base_size)
        mesh.AddScaleOp().Set(
            Gf.Vec3f(size[0] / base_size, size[1] / base_size, size[2] / base_size)
        )
        return mesh.GetPrim()

    for env_id in range(env.num_envs):
        env_root = f"/World/envs/env_{env_id}"
        for name, x_min, x_max, y_min, y_max, color, roughness in _PLAY_TABLE_REGIONS:
            size_visual = (x_max - x_min, y_max - y_min, 2.0 * _PLAY_TABLE_HALF_THICKNESS)
            center_visual = (
                0.5 * (x_min + x_max),
                0.5 * (y_min + y_max),
                visual_z,
            )
            visual_prim = define_cube(
                f"{env_root}/franka_env/ReplayTableVisual/{name}",
                size_visual,
                center_visual,
            )
            UsdGeom.Cube(visual_prim).CreateDisplayColorPrimvar(
                UsdGeom.Tokens.constant
            ).Set([Gf.Vec3f(*color)])
            material_path = f"{env_root}/ReplayTableMaterials/{name}"
            material = UsdShade.Material.Define(stage, material_path)
            shader = UsdShade.Shader.Define(stage, f"{material_path}/PreviewSurface")
            shader.CreateIdAttr("UsdPreviewSurface")
            shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
            shader.CreateInput("emissiveColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
            shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(roughness)
            material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
            UsdShade.MaterialBindingAPI(visual_prim).Bind(material)
            UsdGeom.Imageable(visual_prim).MakeVisible()

    print(
        f"[Isaac6] spawned {3 * env.num_envs} wrist-only visual tabletop meshes",
        flush=True,
    )

def _init_play_render_layers(env):
    """Cache background Gaussian prims and replacement tabletop prims."""
    env.play_gaussian_backgrounds = []
    env.play_table_visuals = []
    for env_id in range(env.num_envs):
        gaussian_path = f"/World/envs/env_{env_id}/franka_env/lab_0/gauss/gauss"
        gaussian = env.sim.stage.GetPrimAtPath(gaussian_path)
        if gaussian.IsValid() and gaussian.IsA(UsdGeom.Imageable):
            env.play_gaussian_backgrounds.append(UsdGeom.Imageable(gaussian))
        for name, *_ in _PLAY_TABLE_REGIONS:
            table_path = f"/World/envs/env_{env_id}/franka_env/ReplayTableVisual/{name}/geometry/mesh"
            table = env.sim.stage.GetPrimAtPath(table_path)
            if table.IsValid() and table.IsA(UsdGeom.Imageable):
                env.play_table_visuals.append(UsdGeom.Imageable(table))
    if not env.play_gaussian_backgrounds:
        raise RuntimeError("Isaac6 play background Gaussian prim was not found")
    if len(env.play_table_visuals) != 3 * env.num_envs:
        raise RuntimeError(
            f"Isaac6 play tabletop clone mismatch: got {len(env.play_table_visuals)} "
            f"visual cubes for {env.num_envs} environments"
        )


def _set_play_render_layer(env, wrist_pass: bool):
    """Use geometric tables as an overlay with Gaussian as the fallback."""
    # Keep Gaussian visible for both streams. The thin replacement tabletop
    # meshes cover the calibrated work surface during the wrist pass, while
    # Gaussian fills the rest of the camera frustum instead of rendering black
    # below the tabletop edge.
    for gaussian in env.play_gaussian_backgrounds:
        gaussian.MakeVisible()
    for table in env.play_table_visuals:
        table.MakeVisible() if wrist_pass else table.MakeInvisible()


def _make_play_background_usd():
    # Use franka_env.usd as the only scene source.  Isaac Sim 6 does not
    # expand its legacy nested ./fr3v2.usd reference, so repair that one
    # reference in memory and export a physics-free visual compatibility copy.
    # The copy contains the original non-robot background only.
    global _PLAY_BACKGROUND_USD
    if _PLAY_BACKGROUND_USD is not None:
        return _PLAY_BACKGROUND_USD
    generated = Path("/tmp/tacex_isaac6_source_background_compat_v3.usd")
    if not generated.is_file():
        if not _PLAY_BACKGROUND_SOURCE.is_file():
            raise FileNotFoundError(_PLAY_BACKGROUND_SOURCE)
        source_stage = Usd.Stage.Open(str(_PLAY_BACKGROUND_SOURCE))
        visual_stage = Usd.Stage.Open(source_stage.Flatten())
        visual_stage.OverridePrim("/World/PhysicsScene").SetActive(False)
        # The camera sees /World/Robot/franka on the left and the legacy
        # /World/fr3v2_01 on the right. The background robot visuals and
        # physics are removed because the PPO articulation is still authored
        # separately by FactoryEnv. Do not recreate any /World/Robot parent
        # chain here: the active full-rollout Franka is spawned separately
        # under /franka_env/Robot/franka.
        visual_stage.RemovePrim("/World/Robot/franka")
        visual_stage.RemovePrim("/World/fr3v2_01")
        visual_stage.RemovePrim("/World/Robot")
        # Keep only the parent scope required by the nested RealSim path.
        # The active full-rollout Franka creates the /franka child itself.
        UsdGeom.Xform.Define(visual_stage, "/World/Robot")

        remove_paths = []
        for prim in list(visual_stage.Traverse()):
            prim_path = str(prim.GetPath()).lower()
            type_name = prim.GetTypeName()
            if ("/collisions" in prim_path or "/joints" in prim_path
                    or type_name.startswith("Physics")
                    or type_name.startswith("Physx")):
                remove_paths.append(prim.GetPath())
                continue
            for schema in list(prim.GetAppliedSchemas()):
                if schema.startswith("Physics") or schema.startswith("Physx"):
                    prim.RemoveAppliedSchema(schema)
        for prim_path in sorted(remove_paths, key=lambda path: len(str(path)), reverse=True):
            if visual_stage.GetPrimAtPath(prim_path).IsValid():
                visual_stage.RemovePrim(prim_path)

        temp_path = generated.with_name(f"{generated.name}.{os.getpid()}.tmp.usd")
        visual_stage.GetRootLayer().Export(str(temp_path))
        os.replace(temp_path, generated)
        print(
            "[Isaac6] generated source-compatible background: no robot physics, "
            "background robot removed; active Franka owns the wrist-camera path",
            flush=True,
        )
    _PLAY_BACKGROUND_USD = generated
    return _PLAY_BACKGROUND_USD


def _write_camera_video(path, frames, fps=15):
    if len(frames) < 2:
        return
    first = np.asarray(frames[0])
    if first.ndim == 4:
        first = first[0]
    if first.shape[-1] == 4:
        first = first[..., :3]
    height, width = first.shape[:2]
    command = [
        "ffmpeg",
        "-y",
        "-loglevel",
        "error",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s",
        f"{width}x{height}",
        "-r",
        str(fps),
        "-i",
        "-",
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        "medium",
        "-crf",
        "18",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(path),
    ]
    process = subprocess.Popen(
        command,
        stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )
    try:
        for frame in frames:
            frame = np.asarray(frame)
            if frame.ndim == 4:
                frame = frame[0]
            if frame.shape[-1] == 4:
                frame = frame[..., :3]
            if frame.dtype != np.uint8:
                frame = np.clip(frame * 255.0 if frame.max() <= 1.0 else frame, 0, 255).astype(np.uint8)
            process.stdin.write(np.ascontiguousarray(frame).tobytes())
    except BrokenPipeError:
        pass
    finally:
        process.stdin.close()
    stderr = process.stderr.read().decode("utf-8", "replace")
    return_code = process.wait()
    if return_code != 0:
        raise RuntimeError(f"H264 video encoding failed: {stderr[-1000:]}")

class Isaac6PegFactoryEnv(FactoryEnv):
    """Official FactoryEnv mechanics with the existing RealSim peg assets."""

    def __init__(self, cfg, render_mode=None, **kwargs):
        # The legacy train.py passes collect_data=False to gym.make().  The
        # official FactoryEnv does not need that argument.
        kwargs.pop("collect_data", None)
        kwargs.pop("output_dir", None)
        self.last_ik_pos_error = None
        self.last_ik_rot_error = None
        self.success_times = 0
        self.total_times = 0
        super().__init__(cfg, render_mode, **kwargs)

    def _setup_scene(self):
        """Create the Factory scene and the play-only hybrid background.

        The background is composed before articulation views are created. Its
        nested PhysicsScene and duplicate robot are removed before PhysX starts.
        The Gaussian tabletop is replaced by visual cubes plus local-frame
        static colliders when play cameras are enabled.
        """
        if args_cli.play and args_cli.play_background:
            # Match the legacy RealSim scene: the same visual-only USD is
            # mounted at franka_env, so its authored /World/front_camera is
            # resolved as /World/envs/env_0/franka_env/front_camera.
            background_cfg = sim_utils.UsdFileCfg(usd_path=str(_make_play_background_usd()))
            background_cfg.func(
                "/World/envs/env_.*/franka_env",
                background_cfg,
                translation=(0.0, 0.0, 0.0),
                orientation=(0.0, 0.0, 0.0, 1.0),
            )
            print("[Isaac6] mounted one visual-only background with legacy franka_env camera", flush=True)
        self._robot = Articulation(self.cfg.robot)
        self._fixed_asset = Articulation(self.cfg_task.fixed_asset)
        self._held_asset = Articulation(self.cfg_task.held_asset)

        if args_cli.play and args_cli.play_background and getattr(args_cli, "enable_cameras", False):
            _init_play_cameras(self)

        self.scene.clone_environments(copy_from_source=False)
        if args_cli.play and args_cli.play_background:
            _spawn_play_table_geometry(self)
        if args_cli.play and args_cli.play_background and getattr(args_cli, "enable_cameras", False):
            _init_play_render_layers(self)
        if self.device == "cpu":
            self.scene.filter_collisions()

        self.scene.articulations["robot"] = self._robot
        self.scene.articulations["fixed_asset"] = self._fixed_asset
        self.scene.articulations["held_asset"] = self._held_asset

    def set_pos_inverse_kinematics(self, *args, **kwargs):
        pos_error, rot_error = super().set_pos_inverse_kinematics(*args, **kwargs)
        self.last_ik_pos_error = torch.linalg.norm(pos_error.detach(), dim=-1)
        self.last_ik_rot_error = torch.linalg.norm(rot_error.detach(), dim=-1)
        return pos_error, rot_error

    def randomize_initial_state(self, env_ids):
        """Reset the peg with IsaacLab 6's XYZW convention.

        The local legacy FactoryEnv still calls ``tacex_tasks.torch_compat``
        for reset transforms.  That compatibility module uses WXYZ, while
        IsaacLab 6 articulation tensors and ``isaaclab.utils.math`` use XYZW.
        Calling the inherited method therefore rotates the peg before the
        later post-grasp snap.  This standalone play environment owns the
        reset so there is only one convention and one grasp placement.
        """
        physics_sim_view = sim_utils.SimulationContext.instance().physics_sim_view
        physics_sim_view.set_gravity(carb.Float3(0.0, 0.0, 0.0))

        # The play config has all reset noise disabled.  Keep the fixed asset
        # reset explicit so the code remains correct if a future play config
        # changes its default pose or adds noise.
        fixed_pose = self._fixed_asset.data.default_root_pose.torch.clone()[env_ids]
        fixed_vel = self._fixed_asset.data.default_root_vel.torch.clone()[env_ids]
        fixed_pose[:, 0:3] += self.scene.env_origins[env_ids]
        fixed_vel.zero_()
        self._fixed_asset.write_root_pose_to_sim_index(root_pose=fixed_pose, env_ids=env_ids)
        self._fixed_asset.write_root_velocity_to_sim_index(root_velocity=fixed_vel, env_ids=env_ids)
        self._fixed_asset.reset()
        self.init_fixed_pos_obs_noise[env_ids].zero_()
        self.step_sim_no_action()

        identity_quat = torch.zeros((self.num_envs, 4), device=self.device)
        identity_quat[:, 3] = 1.0
        fixed_tip_pos_local = torch.zeros((self.num_envs, 3), device=self.device)
        fixed_tip_pos_local[:, 2] = (
            self.cfg_task.fixed_asset_cfg.height + self.cfg_task.fixed_asset_cfg.base_height
        )
        fixed_tip_pos, _ = isaaclab_math.combine_frame_transforms(
            self.fixed_pos,
            self.fixed_quat,
            fixed_tip_pos_local,
            identity_quat,
        )
        self.fixed_pos_obs_frame[:] = fixed_tip_pos

        # Solve the same downward IK target as FactoryEnv, but produce its
        # quaternion directly in XYZW.  This is deterministic in play,
        # matching the old RealSim replay reset.
        bad_envs = env_ids.clone()
        hand_down_euler = torch.tensor(
            self.cfg_task.hand_init_orn, dtype=torch.float32, device=self.device
        ).view(1, 3).repeat(self.num_envs, 1)
        hand_down_quat = isaaclab_math.quat_from_euler_xyz(
            hand_down_euler[:, 0], hand_down_euler[:, 1], hand_down_euler[:, 2]
        )
        above_fixed_pos = fixed_tip_pos.clone()
        above_fixed_pos[:, 2] += self.cfg_task.hand_init_pos[2]

        while bad_envs.numel() > 0:
            pos_error, aa_error = self.set_pos_inverse_kinematics(
                ctrl_target_fingertip_midpoint_pos=above_fixed_pos,
                ctrl_target_fingertip_midpoint_quat=hand_down_quat,
                env_ids=bad_envs,
            )
            bad_mask = torch.logical_or(
                torch.linalg.norm(pos_error, dim=1) > 1e-3,
                torch.linalg.norm(aa_error, dim=1) > 1e-3,
            )
            bad_envs = bad_envs[bad_mask]
            if bad_envs.numel() > 0:
                self._set_franka_to_default_pose(
                    joints=[0.00871, -0.10368, -0.00794, -1.49139, -0.00083, 1.38774, 0.0],
                    env_ids=bad_envs,
                )

        self.step_sim_no_action()

        # Place the 50 mm peg midpoint at the fingertip midpoint.  The peg
        # Put the peg root 57.392 mm from the active fingertip: the original
        # black-gripper calibration is 32.392 mm, plus 25 mm of hole-facing
        # exposure. Rotate the peg 180 degrees around Y so that exposed half
        # points into the hole. Both transforms below are XYZW.
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
        relative_pos[:, 2] = _PEG_ROOT_OFFSET_M
        held_pos, held_quat = isaaclab_math.combine_frame_transforms(
            flipped_pos,
            flipped_quat,
            -relative_pos,
            identity_quat,
        )
        held_pose = self._held_asset.data.default_root_pose.torch.clone()[env_ids]
        held_vel = self._held_asset.data.default_root_vel.torch.clone()[env_ids]
        held_pose[:, 0:3] = held_pos[env_ids] + self.scene.env_origins[env_ids]
        held_pose[:, 3:7] = held_quat[env_ids]
        held_vel.zero_()
        self._held_asset.write_root_pose_to_sim_index(root_pose=held_pose, env_ids=env_ids)
        self._held_asset.write_root_velocity_to_sim_index(root_velocity=held_vel, env_ids=env_ids)
        self._held_asset.reset()

        # Match the legacy 0.6 s close + 0.2 s settle timing.  The inherited
        # FactoryEnv reset's hard-coded 0.25 s close is intentionally bypassed.
        reset_task_prop_gains = torch.tensor(
            self.cfg.ctrl.reset_task_prop_gains, device=self.device
        ).repeat((self.num_envs, 1))
        self.task_prop_gains = reset_task_prop_gains
        self.task_deriv_gains = factory_utils.get_deriv_gains(
            reset_task_prop_gains, self.cfg.ctrl.reset_rot_deriv_scale
        )
        self.step_sim_no_action()
        grasp_time = 0.0
        while grasp_time < 0.6:
            self.ctrl_target_joint_pos[env_ids, 7:] = 0.0
            self.close_gripper_in_place()
            self.step_sim_no_action()
            grasp_time += self.physics_dt

        settle_time = 0.0
        while settle_time < 0.2:
            self.ctrl_target_joint_pos[env_ids, 7:] = 0.0
            self.close_gripper_in_place()
            self.step_sim_no_action()
            settle_time += self.physics_dt

        self.prev_joint_pos = self.joint_pos[:, 0:7].clone()
        self.prev_fingertip_pos = self.fingertip_midpoint_pos.clone()
        self.prev_fingertip_quat = self.fingertip_midpoint_quat.clone()
        # Match the official IsaacLab reset implementation.  Reallocate these
        # buffers instead of mutating the action tensor in-place; the local
        # rl_games bridge may retain a view of the old tensor during reset.
        self.actions = torch.zeros_like(self.actions)
        self.prev_actions = torch.zeros_like(self.actions)
        self.ee_angvel_fd.zero_()
        self.ee_linvel_fd.zero_()
        self.task_prop_gains = self.default_gains
        self.task_deriv_gains = factory_utils.get_deriv_gains(self.default_gains)
        physics_sim_view.set_gravity(carb.Float3(*self.cfg.sim.gravity))

    def _reset_idx(self, env_ids):
        # Keep FactoryEnv's reset bookkeeping and default-pose preparation.
        # Its call to ``self.randomize_initial_state`` dispatches to the XYZW
        # implementation above, so the legacy WXYZ implementation is never
        # entered while all IsaacLab reset hooks remain intact.
        super()._reset_idx(env_ids)

    def step(self, action):
        result = super().step(action)
        reset_mask = result[2] | result[3]
        if bool(torch.any(reset_mask)):
            success_rate = result[4].get("log", {}).get("Metrics/success_rate")
            if success_rate is not None:
                current_total = int(reset_mask.sum().item())
                current_successes = int(round(float(success_rate) * self.num_envs))
                current_successes = min(current_successes, current_total)
                current_rate = (current_successes / current_total) * 100 if current_total else 0.0
                self.success_times += current_successes
                self.total_times += current_total
                cumulative_rate = (self.success_times / self.total_times) * 100 if self.total_times else 0.0
                print(f"Current Success rate: {current_successes} / {current_total} = {current_rate:.2f}%", flush=True)
                print(f"Cumulative Success rate: {self.success_times} / {self.total_times} = {cumulative_rate:.2f}%", flush=True)
        return result


def _make_cfg():
    cfg = FactoryTaskPegInsertCfg()
    # Use the same active articulation path as replay_real_ee_cartesian_ppo.py.
    # The background robot is removed from the visual compatibility layer, so
    # this Franka is the only robot and its authored gripper stays connected to
    # the arm articulation.
    cfg.robot.prim_path = "/World/envs/env_.*/franka_env/Robot/franka"
    cfg.robot.spawn.usd_path = str(_make_rollout_franka_usd())
    cfg.task.fixed_asset_cfg.usd_path = str(_LOCAL_ASSET_DIR / "factory_hole_8mm.usd")
    cfg.task.fixed_asset.spawn.usd_path = cfg.task.fixed_asset_cfg.usd_path
    cfg.task.held_asset_cfg.usd_path = str(_LOCAL_ASSET_DIR / "factory_peg_8mm.usd")
    cfg.task.held_asset.spawn.usd_path = cfg.task.held_asset_cfg.usd_path
    cfg.scene.num_envs = int(args_cli.num_envs)
    cfg.scene.env_spacing = 2.0
    cfg.scene.replicate_physics = True
    cfg.scene.clone_in_fabric = False
    cfg.seed = int(args_cli.seed)
    cfg.sim.device = args_cli.device or cfg.sim.device

    # FactoryEnv computes the held-asset pose as
    # Keep train/play geometry identical to the calibrated black-gripper pose.
    cfg.task.robot_cfg.franka_fingerpad_length = _PEG_FRANKA_FINGERPAD_LENGTH_M

    if args_cli.play:
        # Training keeps these perturbations.  A recorded play trajectory
        # must start from the nominal grasp so the physical peg is centered
        # between the two black fingertips, matching the legacy replay path.
        cfg.task.fixed_asset_init_pos_noise = [0.0, 0.0, 0.0]
        cfg.task.fixed_asset_init_orn_range_deg = 0.0
        cfg.task.hand_init_pos_noise = [0.0, 0.0, 0.0]
        cfg.task.hand_init_orn_noise = [0.0, 0.0, 0.0]
        cfg.task.held_asset_pos_noise = [0.0, 0.0, 0.0]

    # Keep the simulation buffers bounded while retaining the Factory contact
    # solver settings needed by the peg/hole geometry.
    cfg.sim.render_interval = cfg.decimation
    cfg.sim.physics.gpu_max_rigid_contact_count = 2**20
    cfg.sim.physics.gpu_max_rigid_patch_count = 2**20
    cfg.sim.physics.gpu_collision_stack_size = 2**26
    cfg.sim.physics.gpu_max_num_partitions = 1
    return cfg


def _load_agent_cfg():
    cfg_path = Path(__file__).resolve().parents[3] / "source/tacex_tasks/tacex_tasks/real2sim/agents/rl_games_ppo_cfg.yaml"
    with cfg_path.open("r", encoding="utf-8") as stream:
        agent_cfg = yaml.safe_load(stream)

    params = agent_cfg["params"]
    params["seed"] = int(args_cli.seed)
    network = params["network"]
    network["mlp"]["units"] = [128, 64]
    network["rnn"]["units"] = 128
    network["rnn"]["layers"] = 1
    params["config"].update(
        {
            "name": "RealSimIsaac6Peg4HalfGrasp",
            "full_experiment_name": "smoke_half_grasp",
            "device": args_cli.device or "cuda:0",
            "num_actors": int(args_cli.num_envs),
            "horizon_length": 64,
            "minibatch_size": 256,
            "seq_length": 32,
            "mini_epochs": 2,
            "max_epochs": int(args_cli.max_iterations),
            "save_frequency": 1000,
            "save_best_after": 1000,
        }
    )
    central = params["config"].get("central_value_config", {})
    if central:
        central["minibatch_size"] = 256
        central["mini_epochs"] = 2
        central.setdefault("rnn", {})["units"] = 128
        central.setdefault("rnn", {})["layers"] = 1
        central.setdefault("mlp", {})["units"] = [128, 64]
    return agent_cfg


def _print_memory(tag):
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**2
        reserved = torch.cuda.memory_reserved() / 1024**2
        peak = torch.cuda.max_memory_allocated() / 1024**2
        print(f"[Isaac6][memory] {tag}: allocated={allocated:.1f}MiB reserved={reserved:.1f}MiB peak={peak:.1f}MiB", flush=True)



def _make_play_camera_cfgs():
    # Legacy RealSim front camera: this prim is authored inside
    # franka_env.usd. Do not spawn a second camera or replace its pose.
    front_cfg = CameraCfg(
        prim_path="/World/envs/env_.*/franka_env/front_camera",
        update_period=0,
        height=480,
        width=640,
        data_types=["rgb"],
        spawn=None,
    )
    # Legacy RealSim wrist camera: attach to the hand. The 5.1 replay path
    # overwrote this camera with the calibrated end-effector-relative pose;
    # keep the same calibration in IsaacLab 6 (XYZW, OpenGL convention).
    wrist_cfg = CameraCfg(
        prim_path=_PLAY_BACKGROUND_WRIST_CAMERA_PRIM,
        update_period=0,
        height=480,
        width=640,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0,
            focus_distance=400.0,
            horizontal_aperture=20.955,
            clipping_range=(0.1, 1.0e5),
        ),
        offset=CameraCfg.OffsetCfg(
            pos=(0.07813, -0.00845, -0.0073),
            rot=(0.71266, 0.68644, 0.07985, 0.12057),
            convention="opengl",
        ),
    )
    return front_cfg, wrist_cfg

def _init_play_cameras(env):
    front_cfg, wrist_cfg = _make_play_camera_cfgs()
    env.play_front_camera = Camera(front_cfg)
    env.play_wrist_camera = Camera(wrist_cfg)
    env.scene.sensors["play_front_camera"] = env.play_front_camera
    env.scene.sensors["play_wrist_camera"] = env.play_wrist_camera

def _get_ik_summary(env):
    if env.last_ik_pos_error is None or env.last_ik_rot_error is None:
        return {
            "ik_success": False,
            "ik_max_position_error": None,
            "ik_max_rotation_error": None,
        }
    max_pos = float(env.last_ik_pos_error.max().item())
    max_rot = float(env.last_ik_rot_error.max().item())
    return {
        "ik_success": bool(max_pos <= 2.0e-3 and max_rot <= 2.0e-3),
        "ik_max_position_error": max_pos,
        "ik_max_rotation_error": max_rot,
    }


def _save_play_episode(rows, episode, success, ik_summary):
    output = Path(args_cli.output_dir) / f"episode_{episode:04d}"
    output.mkdir(parents=True, exist_ok=True)
    with (output / "trajectory.jsonl").open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row) + "\n")
    with (output / "metadata.json").open("w", encoding="utf-8") as stream:
        json.dump(
            {
                "success": bool(success),
                "steps": len(rows),
                "checkpoint": str(Path(args_cli.checkpoint).resolve()),
                "robot_asset": str(_PLAY_BACKGROUND_SOURCE) + "#/World/Robot/franka",
                "gripper_visual": "full_rollout_franka:panda_long_finger:PlasticBlack",
                "background": str(_PLAY_BACKGROUND_SOURCE) if args_cli.play_background else None,
                "camera": bool(args_cli.enable_cameras and args_cli.play_background),
                "front_camera_prim": "/World/envs/env_.*/franka_env/front_camera",
                "wrist_camera_prim": _PLAY_BACKGROUND_WRIST_CAMERA_PRIM,
                "camera_capture": "post_step",
                "quaternion_convention": "xyzw",
                **ik_summary,
            },
            stream,
            indent=2,
        )
    print(f"Saved trajectory: {output} success={success}", flush=True)


def _visual_scene_quat(env):
    return torch.as_tensor(
        _PLAY_VISUAL_ROTATION_XYZW,
        dtype=torch.float32,
        device=env.device,
    ).view(1, 4).expand(env.num_envs, -1)


def _visual_scene_translation(env):
    return torch.as_tensor(
        _PLAY_VISUAL_TRANSLATION,
        dtype=torch.float32,
        device=env.device,
    ).view(1, 3).expand(env.num_envs, -1)


def _to_visual_pose(env, position, quaternion):
    """Map PPO-local poses to the authored left-Franka visual frame."""
    origins = env.scene.env_origins
    scene_quat = _visual_scene_quat(env)
    scene_translation = _visual_scene_translation(env)
    visual_position = (
        isaaclab_math.quat_apply(scene_quat, position - origins)
        + scene_translation
        + origins
    )
    visual_quaternion = isaaclab_math.quat_mul(scene_quat, quaternion)
    return visual_position, visual_quaternion


def _set_visual_scene_pose(env):
    """Temporarily move all PPO physics objects to the RealSim visual frame."""
    original_poses = []
    origins = env.scene.env_origins
    scene_quat = _visual_scene_quat(env)
    scene_translation = _visual_scene_translation(env)
    for asset in (env._robot, env._fixed_asset, env._held_asset):
        position = asset.data.root_pos_w.torch.clone()
        quaternion = asset.data.root_quat_w.torch.clone()
        original_poses.append(torch.cat((position, quaternion), dim=-1))
        visual_position = (
            isaaclab_math.quat_apply(scene_quat, position - origins)
            + scene_translation
            + origins
        )
        visual_quaternion = isaaclab_math.quat_mul(scene_quat, quaternion)
        asset.write_root_pose_to_sim_index(
            root_pose=torch.cat((visual_position, visual_quaternion), dim=-1)
        )
    env.sim.forward()
    return original_poses


def _restore_scene_pose(env, original_poses):
    for asset, pose in zip((env._robot, env._fixed_asset, env._held_asset), original_poses):
        asset.write_root_pose_to_sim_index(root_pose=pose)
    env.sim.forward()


def _capture_replay_state(env, env_id):
    """Copy the physical state needed for the deferred wrist-camera pass."""
    def root_pose(asset):
        return torch.cat(
            (
                asset.data.root_pos_w.torch[env_id],
                asset.data.root_quat_w.torch[env_id],
            )
        ).detach().cpu().numpy().copy()

    return {
        "robot_joint_pos": env._robot.data.joint_pos.torch[env_id].detach().cpu().numpy().copy(),
        "robot_root_pose": root_pose(env._robot),
        "fixed_root_pose": root_pose(env._fixed_asset),
        "held_root_pose": root_pose(env._held_asset),
    }


def _write_replay_state(env, state):
    """Restore one cached physical state without advancing the simulation."""
    if env.num_envs != 1:
        raise RuntimeError("The rollout-compatible deferred camera pass requires --num_envs 1")

    joint_pos = torch.as_tensor(
        state["robot_joint_pos"], dtype=torch.float32, device=env.device
    ).view(1, -1)
    joint_ids = list(range(joint_pos.shape[-1]))
    env._robot.write_joint_position_to_sim_index(position=joint_pos, joint_ids=joint_ids)
    env._robot.write_joint_velocity_to_sim_index(
        velocity=torch.zeros_like(joint_pos), joint_ids=joint_ids
    )
    for asset, key in (
        (env._robot, "robot_root_pose"),
        (env._fixed_asset, "fixed_root_pose"),
        (env._held_asset, "held_root_pose"),
    ):
        pose = torch.as_tensor(state[key], dtype=torch.float32, device=env.device).view(1, 7)
        asset.write_root_pose_to_sim_index(root_pose=pose)
    env.sim.forward()
    env.scene.update(dt=env.physics_dt)


def _set_wrist_camera_pose_from_ee(env):
    """Apply the legacy wrist-camera calibration to the hand link.

    The Camera prim is mounted below ``panda_hand`` in the background Franka
    hierarchy. Use the live hand body's world pose here because FactoryEnv's
    ``fingertip_midpoint_pos`` is expressed relative to the environment origin.
    """
    if not hasattr(env, "legacy_wrist_hand_body_idx"):
        try:
            env.legacy_wrist_hand_body_idx = env._robot.body_names.index("panda_hand")
        except ValueError as exc:
            raise RuntimeError(
                f"The PPO Franka has no panda_hand body; bodies={env._robot.body_names}"
            ) from exc

    # FactoryEnv/IsaacLab stores body quaternions in XYZW order. The caller
    # has already moved the PPO robot into the authored visual frame, so do
    # not apply the visual-frame transform a second time.
    hand_idx = env.legacy_wrist_hand_body_idx
    visual_hand_pos = env._robot.data.body_pos_w.torch[:, hand_idx]
    visual_hand_quat = env._robot.data.body_quat_w.torch[:, hand_idx]
    offset_pos = torch.as_tensor(
        (0.07813, -0.00845, -0.0073),
        dtype=torch.float32,
        device=env.device,
    ).view(1, 3).expand(env.num_envs, -1)
    offset_quat = torch.as_tensor(
        (0.71266, 0.68644, 0.07985, 0.12057),
        dtype=torch.float32,
        device=env.device,
    ).view(1, 4).expand(env.num_envs, -1)
    camera_pos, camera_quat = isaaclab_math.combine_frame_transforms(
        visual_hand_pos, visual_hand_quat, offset_pos, offset_quat
    )
    env.play_wrist_camera.set_world_poses(
        camera_pos, camera_quat, convention="opengl"
    )


def _render_wrist_pass(env, replay_states):
    """Render the wrist stream after the front Gaussian stream is complete."""
    if not replay_states:
        return []
    if env.num_envs != 1:
        raise RuntimeError("The rollout-compatible deferred camera pass requires --num_envs 1")

    _set_play_render_layer(env, wrist_pass=True)
    wrist_frames = []
    try:
        for index, state in enumerate(replay_states):
            _write_replay_state(env, state)
            visual_poses = _set_visual_scene_pose(env)
            try:
                env.scene.update(dt=env.physics_dt)
                # The camera is a child of the live rollout Franka hand. Its
                # CameraCfg offset is therefore evaluated by Isaac6 from the
                # same hand link; do not overwrite that attached prim with a
                # world pose during the deferred pass.
                # The extra first renders flush temporal history after replacing
                # the Gaussian background by the geometric tabletop.
                render_count = 3 if index == 0 else 1
                for _ in range(render_count):
                    env.sim.render()
                env.play_wrist_camera.update(env.physics_dt, force_recompute=True)
                wrist_frames.append(
                    env.play_wrist_camera.data.output["rgb"].detach().cpu().numpy()[0]
                )
                if index == 0:
                    print("[Isaac6] wrist pass first frame captured", flush=True)
            finally:
                _restore_scene_pose(env, visual_poses)
    finally:
        _set_play_render_layer(env, wrist_pass=False)
    return wrist_frames


def _transform_action_for_visual_frame(env, action):
    scene_quat = _visual_scene_quat(env)
    transformed = action.clone()
    transformed[:, :3] = isaaclab_math.quat_apply(scene_quat, action[:, :3])
    transformed[:, 3:6] = isaaclab_math.quat_apply(scene_quat, action[:, 3:6])
    return transformed


def _run_play(env, agent_cfg):
    if not args_cli.checkpoint:
        raise ValueError("--checkpoint is required with --play")
    rl_device = agent_cfg["params"]["config"]["device"]
    wrapped = RlGamesVecEnvWrapper(
        env,
        rl_device,
        agent_cfg["params"]["env"].get("clip_observations", float("inf")),
        agent_cfg["params"]["env"].get("clip_actions", float("inf")),
    )
    vecenv.register("Isaac6PlayVecEnv", lambda config_name, num_actors, **kwargs: RlGamesGpuEnv(config_name, num_actors, **kwargs))
    env_configurations.register("rlgpu_isaac6_play", {"vecenv_type": "Isaac6PlayVecEnv", "env_creator": lambda **kwargs: wrapped})
    agent_cfg["params"]["config"]["env_name"] = "rlgpu_isaac6_play"
    agent_cfg["params"]["config"]["num_actors"] = env.unwrapped.num_envs
    agent_cfg["params"]["load_checkpoint"] = True
    agent_cfg["params"]["load_path"] = str(Path(args_cli.checkpoint).resolve())

    runner = Runner()
    runner.load(agent_cfg)
    agent: BasePlayer = runner.create_player()
    agent.restore(str(Path(args_cli.checkpoint).resolve()))
    agent.is_deterministic = True
    agent.reset()
    obs = wrapped.reset()
    ik_summary = _get_ik_summary(env.unwrapped)
    ik_success = ik_summary["ik_success"]
    ik_max_position_error = ik_summary["ik_max_position_error"]
    ik_max_rotation_error = ik_summary["ik_max_rotation_error"]
    print(
        f"[Isaac6][IK] success={ik_success} "
        f"max_position_error={ik_max_position_error} "
        f"max_rotation_error={ik_max_rotation_error}",
        flush=True,
    )
    if isinstance(obs, dict):
        obs = obs["obs"]
    _ = agent.get_batch_size(obs, 1)
    if agent.is_rnn:
        agent.init_rnn()

    num_envs = int(env.unwrapped.num_envs)
    rows = [[] for _ in range(num_envs)]
    front_frames = [[] for _ in range(num_envs)]
    wrist_frames = [[] for _ in range(num_envs)]
    replay_states = [[] for _ in range(num_envs)]
    completed = 0
    while completed < args_cli.num_episodes:
        with torch.inference_mode():
            action = agent.get_action(agent.obs_to_torch(obs), is_deterministic=True)
            obs, reward, dones, _ = wrapped.step(action)

        # Keep PPO physics in its trained local frame, but render and save the
        # scene in the authored RealSim left-Franka frame.
        unwrapped = env.unwrapped
        for env_id in range(num_envs):
            replay_states[env_id].append(_capture_replay_state(unwrapped, env_id))
        visual_action = _transform_action_for_visual_frame(unwrapped, action)
        visual_ee_pos, visual_ee_quat = _to_visual_pose(
            unwrapped,
            unwrapped.fingertip_midpoint_pos.detach(),
            unwrapped.fingertip_midpoint_quat.detach(),
        )
        visual_poses = None
        try:
            if hasattr(unwrapped, "play_front_camera"):
                visual_poses = _set_visual_scene_pose(unwrapped)

                # Match the legacy rollout renderer: capture the complete
                # front Gaussian stream first.  The wrist stream is rendered
                # later from cached physical states, after one visibility switch.
                _set_play_render_layer(unwrapped, wrist_pass=False)
                unwrapped.sim.render()
                unwrapped.play_front_camera.update(unwrapped.physics_dt, force_recompute=True)
                front_rgb = unwrapped.play_front_camera.data.output["rgb"].detach().cpu().numpy()

                if front_rgb.shape[0] != num_envs:
                    raise RuntimeError(
                        f"Front camera batch mismatch: front={front_rgb.shape}, envs={num_envs}"
                    )
                for env_id in range(num_envs):
                    front_frames[env_id].append(front_rgb[env_id])
        finally:
            if hasattr(unwrapped, "play_gaussian_backgrounds"):
                _set_play_render_layer(unwrapped, wrist_pass=False)
            if visual_poses is not None:
                _restore_scene_pose(unwrapped, visual_poses)

        reward_cpu = reward.detach().cpu().reshape(-1).tolist()
        action_cpu = visual_action.detach().cpu().tolist()
        joint_cpu = env.unwrapped._robot.data.joint_pos.torch.detach().cpu().tolist()
        ee_pos_cpu = visual_ee_pos.detach().cpu().tolist()
        ee_quat_cpu = visual_ee_quat.detach().cpu().tolist()
        ee_quat_wxyz_cpu = [
            [quat[3], quat[0], quat[1], quat[2]] for quat in ee_quat_cpu
        ]
        done_ids = torch.nonzero(dones.detach().cpu().reshape(-1), as_tuple=False).flatten().tolist()
        for env_id in range(num_envs):
            rows[env_id].append({
                "step": len(rows[env_id]),
                "reward": float(reward_cpu[env_id]),
                "action": action_cpu[env_id],
                "joint_pos": joint_cpu[env_id],
                "ee_pos": ee_pos_cpu[env_id],
                "ee_quat_xyzw": ee_quat_cpu[env_id],
                "ee_quat_wxyz": ee_quat_wxyz_cpu[env_id],
            })
        # Do not index recurrent states here: rl_games keeps a single player
        # state tensor even when the Isaac Lab vectorized environment has many actors.
        # This branch is only intended to capture the first successful rollout.
        for env_id in done_ids:
            success = bool(env.unwrapped.ep_succeeded[env_id].item())
            print(f"Play rollout: env={env_id} success={success} steps={len(rows[env_id])}", flush=True)
            if success:
                completed += 1
                if hasattr(unwrapped, "play_wrist_camera"):
                    wrist_frames[env_id] = _render_wrist_pass(
                        unwrapped, replay_states[env_id]
                    )
                _save_play_episode(rows[env_id], completed, success, ik_summary)
                episode_dir = Path(args_cli.output_dir) / f"episode_{completed:04d}"
                _write_camera_video(episode_dir / "front_camera.mp4", front_frames[env_id])
                _write_camera_video(episode_dir / "wrist_camera.mp4", wrist_frames[env_id])
                with (episode_dir / "joint_states.csv").open("w", newline="") as stream:
                    writer = csv.writer(stream)
                    writer.writerow([f"joint_{i}" for i in range(len(rows[env_id][0]["joint_pos"]))])
                    writer.writerows(row["joint_pos"] for row in rows[env_id])
                with (episode_dir / "actions.csv").open("w", newline="") as stream:
                    writer = csv.writer(stream)
                    writer.writerow([f"action_{i}" for i in range(len(rows[env_id][0]["action"]))])
                    writer.writerows(row["action"] for row in rows[env_id])
            elif args_cli.save_failed_episodes:
                failed_episode = completed + 1
                if hasattr(unwrapped, "play_wrist_camera"):
                    wrist_frames[env_id] = _render_wrist_pass(
                        unwrapped, replay_states[env_id]
                    )
                _save_play_episode(rows[env_id], failed_episode, False, ik_summary)
                episode_dir = Path(args_cli.output_dir) / f"episode_{failed_episode:04d}"
                _write_camera_video(episode_dir / "front_camera.mp4", front_frames[env_id])
                _write_camera_video(episode_dir / "wrist_camera.mp4", wrist_frames[env_id])
                with (episode_dir / "joint_states.csv").open("w", newline="") as stream:
                    writer = csv.writer(stream)
                    writer.writerow([f"joint_{i}" for i in range(len(rows[env_id][0]["joint_pos"]))])
                    writer.writerows(row["joint_pos"] for row in rows[env_id])
                with (episode_dir / "actions.csv").open("w", newline="") as stream:
                    writer = csv.writer(stream)
                    writer.writerow([f"action_{i}" for i in range(len(rows[env_id][0]["action"]))])
                    writer.writerows(row["action"] for row in rows[env_id])
                completed += 1
                print(f"Saved failed diagnostic episode: {episode_dir}", flush=True)
            front_frames[env_id] = []
            wrist_frames[env_id] = []
            replay_states[env_id] = []
            if completed >= args_cli.num_episodes:
                break
    wrapped.close()

def main():
    cfg = _make_cfg()
    agent_cfg = _load_agent_cfg()
    print(f"[Isaac6] task=TacEx-RealSim-PegInsert-Direct-v0 num_envs={cfg.scene.num_envs}", flush=True)
    print(f"[Isaac6] background={bool(args_cli.play_background)} cameras={bool(args_cli.enable_cameras and args_cli.play_background)} official Factory reset/IK enabled", flush=True)

    env = Isaac6PegFactoryEnv(cfg, render_mode=None)
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    _print_memory("after_env_create")
    if args_cli.play:
        print("[Isaac6] entering play runner", flush=True)
        _run_play(env, agent_cfg)
        return
    env.reset(seed=int(args_cli.seed))
    unwrapped = env.unwrapped
    if unwrapped.last_ik_pos_error is None or unwrapped.last_ik_rot_error is None:
        raise RuntimeError("IK did not execute during reset")
    max_pos = float(unwrapped.last_ik_pos_error.max().item())
    max_rot = float(unwrapped.last_ik_rot_error.max().item())
    ik_success = max_pos <= 2.0e-3 and max_rot <= 2.0e-3
    print(f"[Isaac6][IK] success={ik_success} max_position_error={max_pos:.6e} max_rotation_error={max_rot:.6e}", flush=True)
    if max_pos > 2.0e-3 or max_rot > 2.0e-3:
        raise RuntimeError(f"Isaac6 peg reset IK did not converge: pos={max_pos} rot={max_rot}")

    rl_device = agent_cfg["params"]["config"]["device"]
    env = RlGamesVecEnvWrapper(
        env,
        rl_device,
        agent_cfg["params"]["env"].get("clip_observations", float("inf")),
        agent_cfg["params"]["env"].get("clip_actions", float("inf")),
    )
    vecenv.register(
        "Isaac6Peg4VecEnv",
        lambda config_name, num_actors, **kwargs: RlGamesGpuEnv(config_name, num_actors, **kwargs),
    )
    env_configurations.register(
        "rlgpu_isaac6_peg4",
        {"vecenv_type": "Isaac6Peg4VecEnv", "env_creator": lambda **kwargs: env},
    )
    agent_cfg["params"]["config"]["env_name"] = "rlgpu_isaac6_peg4"
    agent_cfg["params"]["config"]["num_actors"] = int(env.unwrapped.num_envs)

    runner = Runner(IsaacAlgoObserver())
    runner.load(agent_cfg)
    runner.reset()
    _print_memory("before_ppo")
    runner.run({"train": True, "play": False})
    _print_memory("after_ppo")
    env.close()


gym.register(
    id="TacEx-RealSim-PegInsert-Isaac6-Peg4-v0",
    entry_point=f"{__name__}:Isaac6PegFactoryEnv",
    disable_env_checker=True,
)


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
