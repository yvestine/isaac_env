"""Shared visual contract for paired sim-data generation and evaluation."""

from __future__ import annotations

import json
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import h5py
import numpy as np

from .pi0_env import (
    _composite_pi0_cable_grommets,
    _composite_pi0_green_table_front,
    _composite_pi0_reference_cylinder,
    _match_pi0_wrist_appearance,
    _pi0_wrist_camera_rotation,
    _resize_pi0_rgb,
)
from .pi0_env_cfg import RealSimPi0PegInsertCfg


SIM_DATA_GRIPPER_STATE = 0.0865
SIM_DATA_IMAGE_SEED_XOR = 0x51AACC

_DOMAIN_RANDOMIZATION_KEYS = (
    "light_intensity_scale",
    "light_red_blue_tint",
    "front_exposure_delta_ev",
    "wrist_exposure_delta_ev",
    "front_focal_length_scale",
    "wrist_focal_length_scale",
    "table_color_scale",
    "object_color_scale",
    "material_roughness_delta",
    "image_brightness",
    "image_contrast",
    "image_saturation",
    "image_gamma",
    "image_noise_sigma_255",
    "image_blur_sigma_px",
)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[4]


def _resolve_repo_path(path: str | Path) -> Path:
    resolved = Path(path).expanduser()
    if not resolved.is_absolute():
        resolved = _repo_root() / resolved
    return resolved.resolve()


@dataclass(frozen=True)
class SimDataProfile:
    """Validated reset and visual parameters for one generated trajectory."""

    profile_id: int
    directory: Path
    h5_path: Path
    metadata_path: Path
    initial_joint_pos: np.ndarray
    hole_position_m: tuple[float, float, float]
    hole_reference: str
    domain_randomization: dict[str, Any]
    frame_count: int
    gripper_state: float = SIM_DATA_GRIPPER_STATE

    def as_dict(self) -> dict[str, Any]:
        return {
            "profile_id": self.profile_id,
            "directory": str(self.directory),
            "h5_path": str(self.h5_path),
            "metadata_path": str(self.metadata_path),
            "initial_joint_pos": self.initial_joint_pos.tolist(),
            "hole_position_m": list(self.hole_position_m),
            "hole_reference": self.hole_reference,
            "domain_randomization": dict(self.domain_randomization),
            "frame_count": self.frame_count,
            "gripper_state": self.gripper_state,
        }


def load_sim_data_profile(
    sim_data_dir: str | Path,
    profile_id: int,
) -> SimDataProfile:
    """Load one ``sim-data/traj_N`` profile and reject incomplete contracts."""

    if int(profile_id) < 0:
        raise ValueError(f"profile_id must be non-negative, got {profile_id}")
    root = _resolve_repo_path(sim_data_dir)
    directory = root / f"traj_{int(profile_id)}"
    h5_path = directory / "data.h5"
    metadata_path = directory / "replay_metadata.json"
    if not h5_path.is_file():
        raise FileNotFoundError(f"sim-data H5 not found: {h5_path}")
    if not metadata_path.is_file():
        raise FileNotFoundError(f"sim-data metadata not found: {metadata_path}")

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    hole = np.asarray(metadata.get("hole_position_sim_m"), dtype=np.float64)
    if hole.shape != (3,) or not np.isfinite(hole).all():
        raise ValueError(f"Invalid hole_position_sim_m in {metadata_path}: {hole}")
    hole_reference = str(metadata.get("hole_reference", ""))
    if hole_reference != "gt-final-xy-fixed-z":
        raise ValueError(
            f"Expected gt-final-xy-fixed-z in {metadata_path}, got {hole_reference!r}"
        )
    frame_count = int(metadata.get("frame_count", 0))
    if frame_count <= 0:
        raise ValueError(f"Invalid frame_count in {metadata_path}: {frame_count}")

    randomization = metadata.get("domain_randomization")
    if not isinstance(randomization, dict) or not bool(randomization.get("enabled")):
        raise ValueError(f"Enabled domain_randomization is missing in {metadata_path}")
    missing = [key for key in _DOMAIN_RANDOMIZATION_KEYS if key not in randomization]
    if missing:
        raise KeyError(f"Missing domain-randomization fields in {metadata_path}: {missing}")
    numeric_values = np.asarray(
        [float(randomization[key]) for key in _DOMAIN_RANDOMIZATION_KEYS],
        dtype=np.float64,
    )
    if not np.isfinite(numeric_values).all():
        raise ValueError(f"Domain randomization contains NaN/Inf in {metadata_path}")
    seed = int(randomization.get("seed", -1))
    if seed < 0:
        raise ValueError(f"Invalid domain-randomization seed in {metadata_path}: {seed}")
    if seed != int(profile_id):
        raise ValueError(
            f"Expected domain-randomization seed {profile_id}, got {seed} in {metadata_path}"
        )

    with h5py.File(h5_path, "r") as h5:
        if "obs/state/joint_pos" not in h5:
            raise KeyError(f"Missing obs/state/joint_pos in {h5_path}")
        joint_dataset = h5["obs/state/joint_pos"]
        if int(joint_dataset.shape[0]) != frame_count:
            raise ValueError(
                f"H5/metadata frame count mismatch in {directory}: "
                f"{joint_dataset.shape[0]} != {frame_count}"
            )
        q0 = np.asarray(joint_dataset[0], dtype=np.float32)
        if "obs/state/gripper_pos" not in h5:
            raise KeyError(f"Missing obs/state/gripper_pos in {h5_path}")
        gripper = float(np.asarray(h5["obs/state/gripper_pos"][0]).reshape(-1)[0])
    if q0.shape != (7,) or not np.isfinite(q0).all():
        raise ValueError(f"Initial joint state must be finite (7,), got {q0}")
    if not np.isfinite(gripper) or abs(gripper - SIM_DATA_GRIPPER_STATE) > 1.0e-6:
        raise ValueError(
            f"Expected unified gripper {SIM_DATA_GRIPPER_STATE}, got {gripper} in {h5_path}"
        )

    return SimDataProfile(
        profile_id=int(profile_id),
        directory=directory,
        h5_path=h5_path,
        metadata_path=metadata_path,
        initial_joint_pos=q0,
        hole_position_m=tuple(float(value) for value in hole),
        hole_reference=hole_reference,
        domain_randomization=dict(randomization),
        frame_count=frame_count,
    )


def _scaled_color(
    color,
    scale: float,
    red_blue_tint: float = 0.0,
) -> tuple[float, float, float]:
    value = np.asarray(color, dtype=np.float64) * float(scale)
    value *= np.asarray((1.0 + red_blue_tint, 1.0, 1.0 - red_blue_tint))
    return tuple(float(channel) for channel in np.clip(value, 0.0, 1.0))


def apply_visual_domain_randomization(cfg, params: dict[str, Any]) -> None:
    """Apply a previously sampled rollout-level visual domain."""

    missing = [key for key in _DOMAIN_RANDOMIZATION_KEYS if key not in params]
    if missing:
        raise KeyError(f"Missing domain-randomization fields: {missing}")
    cfg.pi0_dome_light_intensity *= float(params["light_intensity_scale"])
    cfg.pi0_dome_light_color = _scaled_color(
        cfg.pi0_dome_light_color,
        1.0,
        float(params["light_red_blue_tint"]),
    )
    cfg.pi0_front_camera_exposure += float(params["front_exposure_delta_ev"])
    cfg.pi0_wrist_camera_exposure += float(params["wrist_exposure_delta_ev"])
    wrist_focal_scale = float(params["wrist_focal_length_scale"])
    front_focal_scale = float(params["front_focal_length_scale"])
    cfg.pi0_wrist_camera_focal_length *= wrist_focal_scale
    if cfg.wrist_camera.spawn is not None:
        cfg.wrist_camera.spawn.focal_length *= wrist_focal_scale
    if cfg.tiled_camera.spawn is not None:
        cfg.tiled_camera.spawn.focal_length *= front_focal_scale
    cfg.pi0_green_table_overlay_color = _scaled_color(
        cfg.pi0_green_table_overlay_color,
        float(params["table_color_scale"]),
    )
    cfg.pi0_green_table_overlay_roughness = float(
        np.clip(
            cfg.pi0_green_table_overlay_roughness
            + float(params["material_roughness_delta"]),
            0.0,
            1.0,
        )
    )
    object_scale = float(params["object_color_scale"])
    cfg.held_asset_visual_color = _scaled_color(
        cfg.held_asset_visual_color,
        object_scale,
    )
    cfg.fixed_asset_visual_color = _scaled_color(
        cfg.fixed_asset_visual_color,
        object_scale,
    )
    cfg.pi0_front_hole_visual_color = _scaled_color(
        cfg.pi0_front_hole_visual_color,
        object_scale,
    )
    cfg.pi0_wrist_hole_visual_color = _scaled_color(
        cfg.pi0_wrist_hole_visual_color,
        object_scale,
    )


def sample_visual_domain_randomization(
    cfg,
    h5_path: Path,
    seed: int | None,
) -> dict[str, Any]:
    """Sample and apply the exact domain used by paired rollout generation."""

    if seed is None:
        seed = zlib.crc32(str(h5_path.resolve()).encode("utf-8")) & 0x7FFFFFFF
    rng = np.random.default_rng(seed)
    light_intensity_scale = float(rng.uniform(0.75, 1.25))
    light_red_blue_tint = float(rng.uniform(-0.06, 0.06))
    exposure_shared = float(rng.uniform(-0.40, 0.40))
    front_exposure_delta = exposure_shared + float(rng.uniform(-0.08, 0.08))
    wrist_exposure_delta = exposure_shared + float(rng.uniform(-0.08, 0.08))
    wrist_focal_scale = float(rng.uniform(0.98, 1.02))
    front_focal_scale = float(rng.uniform(0.98, 1.02))
    params: dict[str, Any] = {
        "enabled": True,
        "seed": int(seed),
        "sampling_scope": "one sample per rollout",
        "light_intensity_scale": light_intensity_scale,
        "light_red_blue_tint": light_red_blue_tint,
        "front_exposure_delta_ev": front_exposure_delta,
        "wrist_exposure_delta_ev": wrist_exposure_delta,
        "front_focal_length_scale": front_focal_scale,
        "wrist_focal_length_scale": wrist_focal_scale,
        "table_color_scale": float(rng.uniform(0.85, 1.15)),
        "object_color_scale": float(rng.uniform(0.90, 1.10)),
        "material_roughness_delta": float(rng.uniform(-0.08, 0.08)),
        "image_brightness": float(rng.uniform(0.92, 1.08)),
        "image_contrast": float(rng.uniform(0.92, 1.08)),
        "image_saturation": float(rng.uniform(0.90, 1.10)),
        "image_gamma": float(rng.uniform(0.94, 1.06)),
        "image_noise_sigma_255": float(rng.uniform(0.0, 2.5)),
        "image_blur_sigma_px": float(rng.uniform(0.0, 0.55)),
    }
    apply_visual_domain_randomization(cfg, params)
    return params


def apply_rgb_domain_randomization(
    image: np.ndarray,
    params: dict[str, Any],
    rng: np.random.Generator,
) -> np.ndarray:
    """Apply rollout-level color response plus deterministic per-frame noise."""

    rgb = image[..., :3].astype(np.float32) / 255.0
    gray = np.mean(rgb, axis=-1, keepdims=True)
    rgb = gray + (rgb - gray) * float(params["image_saturation"])
    rgb = (rgb - 0.5) * float(params["image_contrast"]) + 0.5
    rgb *= float(params["image_brightness"])
    rgb = np.power(np.clip(rgb, 0.0, 1.0), float(params["image_gamma"]))
    noise_sigma = float(params["image_noise_sigma_255"]) / 255.0
    if noise_sigma > 0.0:
        rgb += rng.normal(0.0, noise_sigma, size=rgb.shape).astype(np.float32)
    rgb = np.clip(rgb, 0.0, 1.0)
    blur_sigma = float(params["image_blur_sigma_px"])
    if blur_sigma >= 0.12:
        rgb = cv2.GaussianBlur(rgb, (3, 3), blur_sigma)
    output = image.copy()
    output[..., :3] = np.rint(rgb * 255.0).astype(np.uint8)
    return output


def configure_pi0_cameras(cfg) -> None:
    """Apply the camera contract used by paired rollout generation."""

    render_width = int(getattr(cfg, "pi0_camera_render_width", 1280))
    render_height = int(getattr(cfg, "pi0_camera_render_height", 960))
    cfg.tiled_camera.width = render_width
    cfg.tiled_camera.height = render_height
    cfg.wrist_camera.width = render_width
    cfg.wrist_camera.height = render_height
    if cfg.wrist_camera.spawn is not None:
        cfg.wrist_camera.spawn.focal_length = float(
            getattr(cfg, "pi0_wrist_camera_focal_length", 24.0)
        )
        cfg.wrist_camera.spawn.vertical_aperture_offset = float(
            getattr(cfg, "pi0_wrist_camera_vertical_aperture_offset", 0.65)
        )
        cfg.wrist_camera.spawn.clipping_range = tuple(
            getattr(cfg, "pi0_wrist_camera_clipping_range", (0.01, 1.0e5))
        )
    cfg.wrist_camera.prim_path = (
        "/World/envs/env_.*/franka_env/Robot/franka/"
        "panda_link7/panda_link8/panda_hand/wrist_camera"
    )
    cfg.wrist_camera.offset.pos = tuple(cfg.pi0_wrist_camera_offset_pos)
    cfg.wrist_camera.offset.rot = _pi0_wrist_camera_rotation(cfg)
    cfg.wrist_camera.offset.convention = "opengl"


def configure_sim_data_visual_scene(cfg) -> None:
    """Apply the exact scene settings used to produce the current sim-data."""

    pi0_visual_cfg = RealSimPi0PegInsertCfg()
    for name in dir(pi0_visual_cfg):
        if name.startswith("pi0_"):
            setattr(cfg, name, getattr(pi0_visual_cfg, name))
    cfg.override_held_asset_color = pi0_visual_cfg.override_held_asset_color
    cfg.held_asset_visual_color = pi0_visual_cfg.held_asset_visual_color
    cfg.override_fixed_asset_color = pi0_visual_cfg.override_fixed_asset_color
    cfg.fixed_asset_visual_color = pi0_visual_cfg.fixed_asset_visual_color
    cfg.background_right_robot_offset_pos = tuple(
        pi0_visual_cfg.background_right_robot_offset_pos
    )
    cfg.background_right_robot_joint_pos = tuple(
        pi0_visual_cfg.background_right_robot_joint_pos
    )
    cfg.background_right_robot_gripper_open = float(
        pi0_visual_cfg.background_right_robot_gripper_open
    )
    cfg.background_right_robot_endpoint_reference_offset_pos = tuple(
        pi0_visual_cfg.background_right_robot_endpoint_reference_offset_pos
    )
    cfg.background_right_robot_endpoint_target_rot = tuple(
        pi0_visual_cfg.background_right_robot_endpoint_target_rot
    )

    cfg.remove_background_robot = True
    cfg.background_robot_visual_only = False
    cfg.replay_left_background_visual_shell = True
    cfg.background_right_robot_visual_only = True
    cfg.background_fr3v2_right_robot_visual_only = False
    cfg.background_right_robot_copy_active_visual = False
    cfg.remove_authored_right_robot = True
    cfg.pi0_green_table_overlay_enabled = True
    cfg.pi0_green_table_overlay_z_offset = -0.005
    cfg.pi0_green_table_wrist_overlay_enabled = False
    cfg.pi0_green_table_wrist_edge_cleanup_enabled = False
    cfg.pi0_green_table_front_composite_enabled = False
    cfg.pi0_green_table_right_patch_enabled = False
    cfg.pi0_green_table_right_edge_front_repair_enabled = True
    cfg.pi0_green_table_right_edge_front_repair_alpha = 1.0
    cfg.pi0_green_table_right_edge_front_repair_soft_mask_enabled = True
    cfg.pi0_green_table_right_edge_front_repair_soft_hue_margin = 12.0
    cfg.pi0_green_table_right_edge_front_repair_soft_saturation_floor = 2.0
    cfg.pi0_green_table_right_edge_front_repair_soft_saturation_full = 24.0
    cfg.pi0_green_table_right_edge_front_repair_soft_blur_sigma = 8.0
    cfg.pi0_green_table_right_patch_image_polygon = (
        (540.0, 237.0),
        (625.0, 237.0),
        (680.0, 342.0),
        (680.0, 500.0),
        (540.0, 500.0),
    )
    cfg.pi0_green_table_right_patch_z_offset = 0.0018
    cfg.pi0_green_table_hole_cutout_xyxy = ()
    cfg.pi0_green_table_border_front_repair_enabled = True
    cfg.pi0_green_table_border_front_repair_alpha = 1.0
    cfg.pi0_green_table_border_front_repair_soft_mask_enabled = True
    cfg.pi0_green_table_border_front_repair_soft_hue_margin = 12.0
    cfg.pi0_green_table_border_front_repair_soft_saturation_floor = 2.0
    cfg.pi0_green_table_border_front_repair_soft_saturation_full = 24.0
    cfg.pi0_green_table_border_front_repair_soft_blur_sigma = 0.8
    cfg.pi0_green_table_border_front_repair_soft_close_kernel_size = 3
    cfg.pi0_green_table_border_image_polygon = (
        (70.0, 231.0),
        (613.0, 229.0),
        (619.0, 242.0),
        (72.0, 244.0),
    )
    cfg.pi0_white_table_overlay_enabled = True
    cfg.pi0_white_table_overlay_image_polygon = (
        (171.0, 105.0),
        (472.0, 103.0),
        (566.0, 250.0),
        (84.0, 250.0),
    )
    cfg.pi0_white_table_overlay_z_offset = -0.032
    cfg.pi0_black_gripper_visual = False
    cfg.robot_base_rot = (0.0, 0.0, 0.0, 1.0)
    cfg.pi0_hole_init_rot = (0.0, 0.0, -0.258819, 0.965926)
    cfg.task.fixed_asset.init_state.pos = tuple(cfg.pi0_hole_init_pos)
    cfg.task.fixed_asset.init_state.rot = tuple(cfg.pi0_hole_init_rot)
    cfg.task.fixed_asset_init_pos_noise = [0.0, 0.0, 0.0]
    cfg.task.fixed_asset_init_orn_range_deg = 0.0
    cfg.task.fixed_asset_init_orn_deg = 0.0
    cfg.task.hand_init_pos_noise = [0.0, 0.0, 0.0]
    cfg.task.hand_init_orn_noise = [0.0, 0.0, 0.0]
    cfg.task.held_asset_pos_noise = [0.0, 0.0, 0.0]
    configure_pi0_cameras(cfg)


def make_image_rng(params: dict[str, Any]) -> np.random.Generator:
    return np.random.default_rng(int(params.get("seed", 0)) ^ SIM_DATA_IMAGE_SEED_XOR)


def capture_sim_data_rgb(
    env,
    domain_randomization: dict[str, Any] | None = None,
    image_rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Capture both cameras from one render and return policy-ready RGB."""

    env.sim.render()
    env.tiled_camera.update(env.physics_dt, force_recompute=True)
    env.wrist_tiled_camera.update(env.physics_dt, force_recompute=True)
    front = env.tiled_camera.data.output["rgb"][0].detach().cpu().numpy().copy()
    wrist = env.wrist_tiled_camera.data.output["rgb"][0].detach().cpu().numpy().copy()

    front = _resize_pi0_rgb(front, env.cfg)
    front = _composite_pi0_green_table_front(front, env.cfg)
    front = _composite_pi0_green_table_front(
        front,
        env.cfg,
        polygon_attr="pi0_green_table_right_patch_image_polygon",
        enabled_attr="pi0_green_table_right_edge_front_repair_enabled",
        alpha_attr="pi0_green_table_right_edge_front_repair_alpha",
        apply_hole_cutout=False,
    )
    front = _composite_pi0_green_table_front(
        front,
        env.cfg,
        polygon_attr="pi0_green_table_border_image_polygon",
        enabled_attr="pi0_green_table_border_front_repair_enabled",
        alpha_attr="pi0_green_table_border_front_repair_alpha",
        apply_hole_cutout=False,
    )
    front = _composite_pi0_reference_cylinder(front, env.cfg, "rgb")
    front = _composite_pi0_cable_grommets(front, env.cfg, "rgb")
    wrist = _resize_pi0_rgb(wrist, env.cfg)
    wrist = _match_pi0_wrist_appearance(wrist, env.cfg, "rgb")
    if (
        domain_randomization
        and bool(domain_randomization.get("enabled"))
        and image_rng is not None
    ):
        front = apply_rgb_domain_randomization(front, domain_randomization, image_rng)
        wrist = apply_rgb_domain_randomization(wrist, domain_randomization, image_rng)
    return np.ascontiguousarray(front[..., :3]), np.ascontiguousarray(wrist[..., :3])


def capture_sim_data_bgr(
    env,
    domain_randomization: dict[str, Any] | None = None,
    image_rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Compatibility wrapper for the rollout's OpenCV video writer."""

    front, wrist = capture_sim_data_rgb(env, domain_randomization, image_rng)
    return (
        cv2.cvtColor(front, cv2.COLOR_RGB2BGR),
        cv2.cvtColor(wrist, cv2.COLOR_RGB2BGR),
    )
