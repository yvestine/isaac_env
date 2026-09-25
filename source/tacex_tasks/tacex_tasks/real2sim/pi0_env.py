from __future__ import annotations

from pathlib import Path
import time

import carb
import cv2
import numpy as np
import torch
import isaaclab.sim as sim_utils
from isaaclab.utils import math as isaaclab_math
from isaaclab_tasks.direct.factory import factory_utils

from .policy.modeling_pi0 import PI0RemoteJointPolicy
from .realsim_env import RealSimEnv


_PI0_PEG_ROOT_OFFSET_M = 0.050 - 0.017608
_PI0_REFERENCE_SPRITE_CACHE: dict[tuple[str, tuple[int, ...]], np.ndarray] = {}


def _pi0_wrist_camera_rotation(cfg) -> tuple[float, float, float, float]:
    """Return the Pi0 wrist-camera quaternion after local XYZ correction."""

    base_rot = torch.as_tensor(
        cfg.pi0_wrist_camera_offset_rot,
        dtype=torch.float32,
    ).view(1, 4)
    correction_deg = torch.as_tensor(
        getattr(cfg, "pi0_wrist_camera_rotation_correction_deg", (0.0, 0.0, 0.0)),
        dtype=torch.float32,
    )
    correction_rad = torch.deg2rad(correction_deg)
    correction_rot = isaaclab_math.quat_from_euler_xyz(
        correction_rad[0:1], correction_rad[1:2], correction_rad[2:3]
    )
    corrected_rot = isaaclab_math.quat_mul(base_rot, correction_rot)[0]
    return tuple(float(value) for value in corrected_rot)


def _resize_pi0_rgb(frame: np.ndarray, cfg) -> np.ndarray:
    """Lanczos-downsample a supersampled Pi0 RGB frame to policy resolution."""

    image = np.asarray(frame)
    output_width = int(getattr(cfg, "pi0_camera_output_width", 640))
    output_height = int(getattr(cfg, "pi0_camera_output_height", 480))
    if image.shape[:2] == (output_height, output_width):
        return np.ascontiguousarray(image)
    return np.ascontiguousarray(
        cv2.resize(
            image,
            (output_width, output_height),
            interpolation=cv2.INTER_LANCZOS4,
        )
    )


def _sanitize_pi0_wrist_green_overlay(frame: np.ndarray, cfg) -> np.ndarray:
    """Replace only the bright cyan fringe of the wrist tabletop overlay."""

    image = np.ascontiguousarray(np.asarray(frame)).copy()
    if not bool(getattr(cfg, "pi0_green_table_wrist_edge_cleanup_enabled", False)):
        return image
    if image.ndim != 3 or image.shape[2] < 3:
        raise ValueError(f"Wrist RGB image must be HxWx3/4, got {image.shape}")

    rgb = np.ascontiguousarray(image[:, :, :3])
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    hue_low, hue_high = tuple(
        int(value)
        for value in getattr(
            cfg, "pi0_green_table_wrist_edge_cleanup_hue_range", (75, 105)
        )
    )
    saturation = hsv[:, :, 1]
    value = hsv[:, :, 2]
    red = rgb[:, :, 0].astype(np.int16)
    green = rgb[:, :, 1].astype(np.int16)
    blue = rgb[:, :, 2].astype(np.int16)
    mask = (
        (hsv[:, :, 0] >= hue_low)
        & (hsv[:, :, 0] <= hue_high)
        & (saturation >= int(getattr(cfg, "pi0_green_table_wrist_edge_cleanup_min_saturation", 80)))
        & (value >= int(getattr(cfg, "pi0_green_table_wrist_edge_cleanup_min_value", 90)))
        & (green >= int(getattr(cfg, "pi0_green_table_wrist_edge_cleanup_min_green", 100)))
        & (blue >= int(getattr(cfg, "pi0_green_table_wrist_edge_cleanup_min_blue", 90)))
        & (red <= int(getattr(cfg, "pi0_green_table_wrist_edge_cleanup_max_red", 70)))
        & ((green - red) >= 45)
        & ((blue - red) >= 35)
    )
    if np.any(mask):
        target = np.asarray(
            getattr(cfg, "pi0_green_table_wrist_edge_cleanup_rgb", (24, 75, 65)),
            dtype=image.dtype,
        )
        image[mask, :3] = target
    return image


def _composite_pi0_green_table_front(
    frame: np.ndarray,
    cfg,
    *,
    polygon_attr: str = "pi0_green_table_overlay_image_polygon",
    enabled_attr: str = "pi0_green_table_front_composite_enabled",
    alpha_attr: str = "pi0_green_table_front_composite_alpha",
    apply_hole_cutout: bool = True,
) -> np.ndarray:
    """Match the real green tabletop in the front RGB image only."""

    image = np.ascontiguousarray(frame).copy()
    if not bool(getattr(cfg, enabled_attr, False)):
        return image
    if image.ndim != 3 or image.shape[2] < 3:
        raise ValueError(f"Front RGB image must be HxWx3/4, got {image.shape}")

    height, width = image.shape[:2]
    polygon = np.asarray(getattr(cfg, polygon_attr, ()), dtype=np.float32)
    if polygon.ndim != 2 or polygon.shape[0] < 3 or polygon.shape[1] != 2:
        raise ValueError(f"{polygon_attr} must be an (N, 2) polygon with N >= 3")
    annotation_width = float(getattr(cfg, "pi0_visual_annotation_width", 640))
    annotation_height = float(getattr(cfg, "pi0_visual_annotation_height", 480))
    polygon[:, 0] *= float(width) / annotation_width
    polygon[:, 1] *= float(height) / annotation_height

    table_mask = np.zeros((height, width), dtype=np.uint8)
    cv2.fillPoly(table_mask, [np.rint(polygon).astype(np.int32)], 255)
    cutout = (
        tuple(
            float(value)
            for value in getattr(cfg, "pi0_green_table_hole_cutout_xyxy", ())
        )
        if apply_hole_cutout
        else ()
    )
    if len(cutout) == 4:
        x0, y0, x1, y1 = cutout
        x0 = int(np.floor(x0 * width / annotation_width))
        x1 = int(np.ceil(x1 * width / annotation_width))
        y0 = int(np.floor(y0 * height / annotation_height))
        y1 = int(np.ceil(y1 * height / annotation_height))
        cv2.rectangle(table_mask, (x0, y0), (x1, y1), 0, thickness=-1)
    elif cutout:
        raise ValueError("pi0_green_table_hole_cutout_xyxy must have four values")

    rgb = np.ascontiguousarray(image[:, :, :3])
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    hue_low, hue_high = tuple(
        int(value)
        for value in getattr(cfg, "pi0_green_table_front_composite_hue_range", (40, 110))
    )
    alpha = float(getattr(cfg, alpha_attr, 0.92))
    if not 0.0 <= alpha <= 1.0:
        raise ValueError(f"{alpha_attr} must be in [0, 1]")
    target = np.asarray(
        getattr(cfg, "pi0_green_table_front_composite_rgb", (24, 75, 65)),
        dtype=np.float32,
    ).reshape(3)

    repair_prefix = (
        enabled_attr[: -len("_enabled")]
        if enabled_attr.endswith("_enabled")
        else enabled_attr
    )
    if bool(getattr(cfg, f"{repair_prefix}_soft_mask_enabled", False)):
        hue_margin = float(getattr(cfg, f"{repair_prefix}_soft_hue_margin", 10.0))
        saturation_floor = float(
            getattr(cfg, f"{repair_prefix}_soft_saturation_floor", 2.0)
        )
        saturation_full = float(
            getattr(cfg, f"{repair_prefix}_soft_saturation_full", 25.0)
        )
        blur_sigma = float(getattr(cfg, f"{repair_prefix}_soft_blur_sigma", 0.8))
        close_kernel_size = int(
            getattr(cfg, f"{repair_prefix}_soft_close_kernel_size", 0)
        )
        if hue_margin <= 0.0:
            raise ValueError(f"{repair_prefix}_soft_hue_margin must be positive")
        if saturation_full <= saturation_floor:
            raise ValueError(
                f"{repair_prefix}_soft_saturation_full must exceed its floor"
            )
        if blur_sigma < 0.0:
            raise ValueError(f"{repair_prefix}_soft_blur_sigma must be non-negative")
        if close_kernel_size < 0 or (
            close_kernel_size > 1 and close_kernel_size % 2 == 0
        ):
            raise ValueError(
                f"{repair_prefix}_soft_close_kernel_size must be zero or an odd integer"
            )

        hue = hsv[:, :, 0].astype(np.float32)
        saturation = hsv[:, :, 1].astype(np.float32)
        hue_weight = np.minimum(
            np.clip((hue - (hue_low - hue_margin)) / hue_margin, 0.0, 1.0),
            np.clip(((hue_high + hue_margin) - hue) / hue_margin, 0.0, 1.0),
        )
        saturation_weight = np.clip(
            (saturation - saturation_floor) / (saturation_full - saturation_floor),
            0.0,
            1.0,
        )
        chroma_weight = hue_weight * saturation_weight
        if close_kernel_size > 1:
            close_kernel = np.ones(
                (close_kernel_size, close_kernel_size),
                dtype=np.uint8,
            )
            chroma_weight = cv2.morphologyEx(
                chroma_weight,
                cv2.MORPH_CLOSE,
                close_kernel,
            )
            # Do not let mask closing paint over the orange peg or other
            # strongly colored foreground objects crossing the table edge.
            non_green_foreground = (
                (saturation >= 64.0)
                & (
                    (hue < hue_low - hue_margin)
                    | (hue > hue_high + hue_margin)
                )
            )
            chroma_weight[non_green_foreground] = 0.0
        spatial_weight = table_mask.astype(np.float32) / 255.0
        if blur_sigma > 0.0:
            spatial_weight = cv2.GaussianBlur(
                spatial_weight,
                (0, 0),
                sigmaX=blur_sigma,
                sigmaY=blur_sigma,
            )
        blend_weight = np.clip(
            alpha * spatial_weight * chroma_weight,
            0.0,
            1.0,
        )[:, :, None]
        source = image[:, :, :3].astype(np.float32)
        image[:, :, :3] = np.rint(
            source * (1.0 - blend_weight) + target * blend_weight
        ).astype(np.uint8)
        return image

    green_background = (
        (hsv[:, :, 0] >= hue_low)
        & (hsv[:, :, 0] <= hue_high)
        & (
            hsv[:, :, 1]
            >= int(getattr(cfg, "pi0_green_table_front_composite_min_saturation", 25))
        )
    )
    mask = (table_mask > 0) & green_background
    if not np.any(mask):
        return image

    image[:, :, :3][mask] = np.rint(
        (1.0 - alpha) * image[:, :, :3][mask].astype(np.float32)
        + alpha * target
    ).astype(np.uint8)
    return image


def _composite_pi0_reference_cylinder(
    frame: np.ndarray,
    cfg,
    channel_order: str = "rgb",
) -> np.ndarray:
    """Draw the static real-scene reference prop into the front image only."""

    image = np.ascontiguousarray(frame).copy()
    if not bool(
        getattr(cfg, "pi0_reference_cylinder_front_composite_enabled", False)
    ):
        return image
    if image.ndim != 3 or image.shape[2] < 3:
        raise ValueError(f"Front RGB image must be HxWx3/4, got {image.shape}")
    if channel_order not in {"rgb", "bgr"}:
        raise ValueError(f"Unsupported channel order: {channel_order}")

    repo_root = Path(__file__).resolve().parents[4]
    source_video = Path(
        getattr(
            cfg,
            "pi0_reference_cylinder_source_video",
            "real_data/traj_0/front_camera.mp4",
        )
    )
    if not source_video.is_absolute():
        source_video = repo_root / source_video
    crop_xyxy = tuple(
        int(value)
        for value in getattr(
            cfg,
            "pi0_reference_cylinder_source_crop_xyxy",
            (548, 278, 610, 338),
        )
    )
    if len(crop_xyxy) != 4:
        raise ValueError("Reference-cylinder source crop must be x0,y0,x1,y1")
    cache_key = (str(source_video), crop_xyxy)
    source_crop_bgr = _PI0_REFERENCE_SPRITE_CACHE.get(cache_key)
    if source_crop_bgr is None:
        capture = cv2.VideoCapture(str(source_video))
        if not capture.isOpened():
            raise RuntimeError(f"Cannot open reference-cylinder video: {source_video}")
        try:
            ok, source_frame_bgr = capture.read()
        finally:
            capture.release()
        if not ok or source_frame_bgr is None:
            raise RuntimeError(
                f"Cannot read reference-cylinder frame from: {source_video}"
            )
        x0, y0, x1, y1 = crop_xyxy
        frame_height, frame_width = source_frame_bgr.shape[:2]
        if not (0 <= x0 < x1 <= frame_width and 0 <= y0 < y1 <= frame_height):
            raise ValueError(
                f"Reference-cylinder crop {crop_xyxy} is outside "
                f"{frame_width}x{frame_height}"
            )
        source_crop_bgr = np.ascontiguousarray(
            source_frame_bgr[y0:y1, x0:x1, :3]
        )
        _PI0_REFERENCE_SPRITE_CACHE[cache_key] = source_crop_bgr

    source_crop = (
        cv2.cvtColor(source_crop_bgr, cv2.COLOR_BGR2RGB)
        if channel_order == "rgb"
        else source_crop_bgr.copy()
    )
    image_height, image_width = image.shape[:2]
    scale_x = float(image_width) / 640.0
    scale_y = float(image_height) / 480.0
    target_width = max(1, int(round(source_crop.shape[1] * scale_x)))
    target_height = max(1, int(round(source_crop.shape[0] * scale_y)))
    if source_crop.shape[:2] != (target_height, target_width):
        source_crop = cv2.resize(
            source_crop,
            (target_width, target_height),
            interpolation=cv2.INTER_LANCZOS4,
        )
    center_x, center_y = tuple(
        float(value)
        for value in getattr(
            cfg,
            "pi0_reference_cylinder_front_center_pixel",
            (579.0, 308.0),
        )
    )
    center_x = int(round(center_x * scale_x))
    center_y = int(round(center_y * scale_y))
    target_x0 = center_x - target_width // 2
    target_y0 = center_y - target_height // 2
    target_x1 = target_x0 + target_width
    target_y1 = target_y0 + target_height
    if not (
        0 <= target_x0 < target_x1 <= image_width
        and 0 <= target_y0 < target_y1 <= image_height
    ):
        raise ValueError("Reference-cylinder destination crop is outside front image")

    destination = image[target_y0:target_y1, target_x0:target_x1, :3]
    feather = max(
        1,
        int(
            round(
                float(
                    getattr(cfg, "pi0_reference_cylinder_composite_feather_px", 7)
                )
                * min(scale_x, scale_y)
            )
        ),
    )
    # Match the source table border to the simulated table before blending;
    # this removes the rectangular crop boundary without changing the real
    # object's internal texture and perspective.
    border_mask = np.zeros((target_height, target_width), dtype=bool)
    border_width = min(max(2, feather), target_height // 3, target_width // 3)
    border_mask[:border_width, :] = True
    border_mask[-border_width:, :] = True
    border_mask[:, :border_width] = True
    border_mask[:, -border_width:] = True
    source_float = source_crop[:, :, :3].astype(np.float32)
    destination_float = destination.astype(np.float32)
    color_shift = np.median(destination_float[border_mask], axis=0) - np.median(
        source_float[border_mask], axis=0
    )
    source_float = np.clip(source_float + color_shift.reshape(1, 1, 3), 0.0, 255.0)

    y_indices, x_indices = np.indices((target_height, target_width))
    edge_distance = np.minimum.reduce(
        (
            x_indices,
            target_width - 1 - x_indices,
            y_indices,
            target_height - 1 - y_indices,
        )
    ).astype(np.float32)
    alpha = np.clip(edge_distance / float(feather), 0.0, 1.0)[..., None]
    blended = source_float * alpha + destination_float * (1.0 - alpha)
    image[target_y0:target_y1, target_x0:target_x1, :3] = np.clip(
        blended,
        0.0,
        255.0,
    ).astype(np.uint8)
    return np.ascontiguousarray(image)


def _composite_pi0_cable_grommets(
    frame: np.ndarray,
    cfg,
    channel_order: str = "rgb",
) -> np.ndarray:
    """Composite the real flexible cable grommet into the fixed front view."""

    image = np.ascontiguousarray(frame).copy()
    if not bool(
        getattr(cfg, "pi0_cable_grommet_front_composite_enabled", False)
    ):
        return image
    if image.ndim != 3 or image.shape[2] < 3:
        raise ValueError(f"Front RGB image must be HxWx3/4, got {image.shape}")
    if channel_order not in {"rgb", "bgr"}:
        raise ValueError(f"Unsupported channel order: {channel_order}")

    repo_root = Path(__file__).resolve().parents[4]
    source_video = Path(
        getattr(
            cfg,
            "pi0_cable_grommet_source_video",
            "real_data/traj_0/front_camera.mp4",
        )
    )
    if not source_video.is_absolute():
        source_video = repo_root / source_video
    crop_xyxy = tuple(
        int(value)
        for value in getattr(
            cfg,
            "pi0_cable_grommet_source_crop_xyxy",
            (493, 194, 543, 244),
        )
    )
    cache_key = (str(source_video), crop_xyxy)
    source_crop_bgr = _PI0_REFERENCE_SPRITE_CACHE.get(cache_key)
    if source_crop_bgr is None:
        capture = cv2.VideoCapture(str(source_video))
        if not capture.isOpened():
            raise RuntimeError(f"Cannot open cable-grommet video: {source_video}")
        try:
            ok, source_frame_bgr = capture.read()
        finally:
            capture.release()
        if not ok or source_frame_bgr is None:
            raise RuntimeError(f"Cannot read cable-grommet frame: {source_video}")
        x0, y0, x1, y1 = crop_xyxy
        source_crop_bgr = np.ascontiguousarray(source_frame_bgr[y0:y1, x0:x1, :3])
        _PI0_REFERENCE_SPRITE_CACHE[cache_key] = source_crop_bgr

    source_crop = (
        cv2.cvtColor(source_crop_bgr, cv2.COLOR_BGR2RGB)
        if channel_order == "rgb"
        else source_crop_bgr.copy()
    )
    image_height, image_width = image.shape[:2]
    scale_x = float(image_width) / 640.0
    scale_y = float(image_height) / 480.0
    target_width = max(1, int(round(source_crop.shape[1] * scale_x)))
    target_height = max(1, int(round(source_crop.shape[0] * scale_y)))
    if source_crop.shape[:2] != (target_height, target_width):
        source_crop = cv2.resize(
            source_crop,
            (target_width, target_height),
            interpolation=cv2.INTER_LANCZOS4,
        )

    feather = max(
        1,
        int(
            round(
                float(getattr(cfg, "pi0_cable_grommet_composite_feather_px", 7))
                * min(scale_x, scale_y)
            )
        ),
    )
    y_indices, x_indices = np.indices((target_height, target_width))
    center_crop_x = 0.5 * float(target_width - 1)
    center_crop_y = 0.5 * float(target_height - 1)
    radial_distance = np.sqrt(
        (x_indices.astype(np.float32) - center_crop_x) ** 2
        + (y_indices.astype(np.float32) - center_crop_y) ** 2
    )
    radius = float(
        getattr(cfg, "pi0_cable_grommet_composite_radius_px", 23.0)
    ) * min(scale_x, scale_y)
    alpha = np.clip(
        (radius - radial_distance) / float(feather),
        0.0,
        1.0,
    )[..., None]

    for center_pixel in getattr(cfg, "pi0_cable_grommet_center_pixels", ()):
        center_x = int(round(float(center_pixel[0]) * scale_x))
        center_y = int(round(float(center_pixel[1]) * scale_y))
        target_x0 = center_x - target_width // 2
        target_y0 = center_y - target_height // 2
        target_x1 = target_x0 + target_width
        target_y1 = target_y0 + target_height
        if not (
            0 <= target_x0 < target_x1 <= image_width
            and 0 <= target_y0 < target_y1 <= image_height
        ):
            raise ValueError("Cable-grommet destination crop is outside front image")

        destination = image[target_y0:target_y1, target_x0:target_x1, :3]
        source_float = source_crop[:, :, :3].astype(np.float32)
        destination_float = destination.astype(np.float32)
        border_mask = radial_distance > max(0.0, radius - float(feather))
        destination_table_color = np.median(destination_float[border_mask], axis=0)
        color_shift = destination_table_color - np.median(
            source_float[border_mask], axis=0
        )
        source_float = np.clip(
            source_float + color_shift.reshape(1, 1, 3),
            0.0,
            255.0,
        )
        table_color_distance = np.linalg.norm(
            destination_float - destination_table_color.reshape(1, 1, 3),
            axis=2,
        )
        table_threshold = float(
            getattr(cfg, "pi0_cable_grommet_table_color_threshold", 42.0)
        )
        table_only_alpha = alpha * (
            table_color_distance <= table_threshold
        )[..., None].astype(np.float32)
        blended = (
            source_float * table_only_alpha
            + destination_float * (1.0 - table_only_alpha)
        )
        image[target_y0:target_y1, target_x0:target_x1, :3] = np.clip(
            blended,
            0.0,
            255.0,
        ).astype(np.uint8)

    return np.ascontiguousarray(image)


def _match_pi0_wrist_appearance(
    frame: np.ndarray,
    cfg,
    channel_order: str = "rgb",
) -> np.ndarray:
    """Match the smooth left-to-right response of the real wrist camera."""

    image = np.asarray(frame)
    if not bool(getattr(cfg, "pi0_wrist_visual_match_enabled", False)):
        return np.ascontiguousarray(image)
    if image.ndim != 3 or image.shape[2] < 3:
        raise ValueError(f"Wrist RGB image must be HxWx3/4, got {image.shape}")
    if channel_order not in {"rgb", "bgr"}:
        raise ValueError(f"Unsupported channel order: {channel_order}")

    output = image.copy()
    rgb = output[:, :, :3].astype(np.float32)
    if channel_order == "bgr":
        rgb = rgb[:, :, ::-1]
    width = rgb.shape[1]
    horizontal = np.linspace(0.0, 1.0, width, dtype=np.float32).reshape(1, width, 1)
    gain_left = float(getattr(cfg, "pi0_wrist_visual_gain_left", 0.72))
    gain_right = float(getattr(cfg, "pi0_wrist_visual_gain_right", 0.48))
    vertical = np.linspace(
        0.0,
        1.0,
        rgb.shape[0],
        dtype=np.float32,
    ).reshape(rgb.shape[0], 1, 1)
    bottom_left_gain = float(
        getattr(cfg, "pi0_wrist_visual_gain_bottom_left", 0.0)
    )
    gain = (
        gain_left
        + (gain_right - gain_left) * horizontal
        + bottom_left_gain * (1.0 - horizontal) * vertical
    )

    gray = (
        0.299 * rgb[:, :, 0:1]
        + 0.587 * rgb[:, :, 1:2]
        + 0.114 * rgb[:, :, 2:3]
    )
    left_desaturation = float(
        getattr(cfg, "pi0_wrist_visual_left_desaturation", 0.35)
    )
    desaturation = np.clip(
        left_desaturation * (1.0 - horizontal),
        0.0,
        1.0,
    )
    rgb = rgb * (1.0 - desaturation) + gray * desaturation
    rgb = np.clip(rgb * gain, 0.0, 255.0).astype(np.uint8)
    if channel_order == "bgr":
        rgb = rgb[:, :, ::-1]
    output[:, :, :3] = rgb
    return np.ascontiguousarray(output)


class Pi0RealSimEnv(RealSimEnv):
    """RealSim environment controlled by image/state-only Pi0 joint targets."""

    def __init__(self, cfg, render_mode=None, **kwargs):
        self._pi0_ready = False
        self._pi0_policy = None
        self._pi0_chunk = np.empty((0, 8), dtype=np.float32)
        self._pi0_chunk_index = 0
        self._pi0_chunk_end = 0
        self._pi0_skip_advance_once = False
        self._pi0_target = None
        self._pi0_desired_q = None
        self._pi0_command_q = None
        self._pi0_command_gripper = None
        self._pi0_last_chunk = np.empty((0, 8), dtype=np.float32)
        self._pi0_chunk_id = -1
        self._pi0_last_chunk_id = -1
        self._pi0_last_chunk_action_index = -1
        self._pi0_finger_hold_target = None
        self._pi0_wrist_hold_target = None
        self.pi0_inference_count = 0
        self.pi0_failures = 0
        self.pi0_last_latency_s = 0.0
        self.pi0_last_error = ""
        self.pi0_target_clipped = 0
        self._pi0_retry_after = 0.0

        # Prevent RealSimEnv from instantiating its legacy Cartesian policy
        # adapter. This class owns the Pi0 client and action path.
        cfg.policy_cfg = None
        # Pi0 evaluation does not consume the generic TA-VLA raw HDF5 export.
        # Keep the per-episode MP4/CSV/metadata collection, but avoid writing
        # a second full copy of every camera frame into HDF5.
        if hasattr(cfg, "data_collect_cfg"):
            cfg.data_collect_cfg["save_tavla_hdf5"] = False
        cfg.enable_cameras = True
        render_width = int(getattr(cfg, "pi0_camera_render_width", 1280))
        render_height = int(getattr(cfg, "pi0_camera_render_height", 960))
        if hasattr(cfg, "tiled_camera") and cfg.tiled_camera is not None:
            cfg.tiled_camera.width = render_width
            cfg.tiled_camera.height = render_height
        if hasattr(cfg, "wrist_camera") and cfg.wrist_camera is not None:
            cfg.wrist_camera.width = render_width
            cfg.wrist_camera.height = render_height
            if cfg.wrist_camera.spawn is not None:
                cfg.wrist_camera.spawn.focal_length = float(
                    cfg.pi0_wrist_camera_focal_length
                )
                cfg.wrist_camera.spawn.vertical_aperture_offset = float(
                    cfg.pi0_wrist_camera_vertical_aperture_offset
                )
                cfg.wrist_camera.spawn.clipping_range = tuple(
                    cfg.pi0_wrist_camera_clipping_range
                )
        # Match the Isaac6 PPO/replay wrist-camera mount exactly. The shared
        # RealSim config still contains the legacy WXYZ value because it is
        # also used by non-Pi0 tasks.
        if hasattr(cfg, "wrist_camera") and cfg.wrist_camera is not None:
            cfg.wrist_camera.prim_path = (
                "/World/envs/env_.*/franka_env/Robot/franka/"
                "panda_link7/panda_link8/panda_hand/wrist_camera"
            )
            cfg.wrist_camera.offset.pos = tuple(cfg.pi0_wrist_camera_offset_pos)
            cfg.wrist_camera.offset.rot = _pi0_wrist_camera_rotation(cfg)
            cfg.wrist_camera.offset.convention = "opengl"
        # Pi0 keeps the active rollout Franka unchanged on the left. The
        # right visual is an exact copy of that asset, translated by the
        # configured base offset;
        # it is never actuated and has no task objects attached.
        cfg.remove_background_robot = True
        cfg.background_robot_visual_only = False
        cfg.background_right_robot_visual_only = True
        cfg.background_fr3v2_right_robot_visual_only = False
        cfg.background_right_robot_copy_active_visual = False
        cfg.robot.prim_path = "/World/envs/env_.*/franka_env/Robot/franka"
        cfg.active_robot_base_pos = tuple(cfg.active_robot_base_pos)
        cfg.active_robot_base_rot = tuple(cfg.active_robot_base_rot)

        # Override the shared legacy RealSim pose only for Pi0. The shared
        # replay/TAVLA configs intentionally remain in their authored frame.
        cfg.task.fixed_asset.init_state.pos = tuple(cfg.pi0_hole_init_pos)
        cfg.task.fixed_asset.init_state.rot = tuple(cfg.pi0_hole_init_rot)
        cfg.robot.init_state.rot = tuple(cfg.pi0_robot_init_rot)
        # The shared RealSim robot intentionally has zero arm actuator gains
        # for its task-space torque controller. Pi0 uses absolute joint
        # targets, so give this Pi0-only articulation a stable implicit PD
        # actuator instead of adding an independent hand-written torque loop.
        if hasattr(cfg.robot, "actuators"):
            for actuator_name in ("panda_arm1", "panda_arm2"):
                actuator = cfg.robot.actuators.get(actuator_name)
                if actuator is not None:
                    actuator.stiffness = float(cfg.pi0_policy_cfg.implicit_arm_stiffness)
                    actuator.damping = float(cfg.pi0_policy_cfg.implicit_arm_damping)
        # Keep long randomized evaluations from spending unbounded CPU time
        # on a high-force peg/finger/hole contact manifold.  The authored
        # RealSim assets use 192 position iterations and a 5 mm robot contact
        # shell; those settings are unnecessary for this 8 mm peg and can
        # make a bad edge contact dominate the whole Isaac process.
        for asset_cfg in (cfg.robot, cfg.task.fixed_asset, cfg.task.held_asset):
            rigid_props = getattr(asset_cfg.spawn, "rigid_props", None)
            if rigid_props is not None:
                rigid_props.solver_position_iteration_count = 64
                rigid_props.max_depenetration_velocity = 1.0
        robot_collision_props = getattr(cfg.robot.spawn, "collision_props", None)
        if robot_collision_props is not None:
            robot_collision_props.contact_offset = 0.001
        robot_articulation_props = getattr(cfg.robot.spawn, "articulation_props", None)
        if robot_articulation_props is not None:
            robot_articulation_props.solver_position_iteration_count = 64
        # IsaacLab 6 spawners consume XYZW. Keep the background camera's
        # authored local pose, but do not apply the legacy WXYZ identity to
        # its parent transform.
        cfg.robot_base_rot = (0.0, 0.0, 0.0, 1.0)
        cfg.task.fixed_asset_init_pos_noise = [0.0, 0.0, 0.0]
        cfg.task.fixed_asset_init_orn_range_deg = 0.0
        cfg.task.fixed_asset_init_orn_deg = 0.0
        cfg.task.hand_init_pos_noise = [0.0, 0.0, 0.0]
        cfg.task.hand_init_orn_noise = [0.0, 0.0, 0.0]

        # Legacy IK-based right-arm construction is disabled for Pi0. The
        # right visual is spawned from the same asset as the active left arm
        # and synchronized to the left reset pose after reset.
        self._pi0_static_right_arm_usd = None
        self._pi0_right_arm_ik_q = None
        self._pi0_right_arm_ik_residual = None
        if getattr(cfg, "background_fr3v2_right_robot_visual_only", False):
            from .pi0_static_right_arm import build_static_right_arm_usd

            source_usd = (
                Path(__file__).resolve().parents[4]
                / "franka_env_background_edit"
                / "fr3v2.usd"
            )
            print("[Pi0RightIK] building", source_usd, flush=True)
            try:
                (
                    self._pi0_static_right_arm_usd,
                    self._pi0_right_arm_ik_q,
                    self._pi0_right_arm_ik_residual,
                ) = build_static_right_arm_usd(
                source_usd,
                target_pos=getattr(cfg, "background_right_robot_ee_target_pos", ()),
                target_frame=getattr(
                    cfg, "background_right_robot_ee_target_frame", "franka_env"
                ),
                nominal_q=getattr(cfg, "background_right_robot_joint_pos", ()),
                base_pos=getattr(
                    cfg,
                    "background_fr3v2_right_robot_pos",
                    (0.6658084946, -0.08782, 0.0991799997),
                ),
                base_rot=getattr(
                    cfg,
                    "background_fr3v2_right_robot_rot",
                    (0.70710677, 0.0, 0.0, -0.70710677),
                ),
                    gripper_open=getattr(cfg, "background_right_robot_gripper_open", 0.04),
                )
            except BaseException as exc:
                import traceback
                print(f"[Pi0RightIK] build failed: {type(exc).__name__}: {exc}", flush=True)
                traceback.print_exc()
                raise
            print(f"[Pi0RightIK] build complete q={self._pi0_right_arm_ik_q} residual={self._pi0_right_arm_ik_residual}", flush=True)

        # The endpoint position was tuned from the camera image with a
        # position-only IK seed. Re-solve it against the active left-arm
        # endpoint orientation before scene creation so the static right
        # gripper does not acquire a wrist rotation. This uses the source USD
        # and therefore does not depend on already-authored scene transforms.
        # The right base offset is not part of this solve.
        endpoint_rot = np.asarray(
            getattr(cfg, "background_right_robot_endpoint_target_rot", ()),
            dtype=np.float64,
        )
        endpoint_pos = np.asarray(
            getattr(cfg, "background_right_robot_endpoint_reference_offset_pos", ()),
            dtype=np.float64,
        )
        if (
            getattr(cfg, "background_right_robot_visual_only", False)
            and endpoint_pos.size == 3
            and endpoint_rot.size == 9
        ):
            from pxr import Usd
            from .pi0_static_right_arm import solve_pose_ik

            source_usd = (
                Path(__file__).resolve().parents[4]
                / "franka_env_background_edit"
                / "franka_env.usd"
            )
            source_stage = Usd.Stage.Open(str(source_usd))
            if source_stage is None:
                raise RuntimeError(
                    f"Could not open right-arm pose-IK source: {source_usd}"
                )
            q_seed = np.asarray(
                getattr(cfg, "background_right_robot_joint_pos", ()),
                dtype=np.float64,
            ).reshape(7)
            q_pose, pos_residual, rot_residual = solve_pose_ik(
                source_stage,
                "/World/Robot/franka",
                endpoint_pos,
                endpoint_rot.reshape(3, 3),
                q_seed,
                max_iterations=500,
                nominal_weight=1.0e-5,
            )
            cfg.background_right_robot_joint_pos = tuple(
                float(value) for value in q_pose
            )
            base_offset = tuple(
                getattr(cfg, "background_right_robot_offset_pos", (0.86, 0.0, 0.0))
            )
            print(
                "[Pi0RightPoseIK] endpoint position/orientation constrained: "
                + "q="
                + np.array2string(q_pose, precision=8)
                + " pos_residual_m="
                + f"{pos_residual:.6f}"
                + " rot_residual_rad="
                + f"{rot_residual:.6f}"
                + " base_offset="
                + str(base_offset),
                flush=True,
            )

        print("[Pi0RightIK] entering RealSimEnv", flush=True)
        super().__init__(cfg, render_mode, **kwargs)
        self._override_pi0_active_robot_base_gray_to_white()

        if self.num_envs != 1:
            raise ValueError("Pi0 remote evaluation currently supports num_envs=1")
        self._pi0_policy = PI0RemoteJointPolicy(cfg.pi0_policy_cfg)
        self._pi0_ready = True

    @property
    def _pi0_cfg(self):
        return self.cfg.pi0_policy_cfg

    def _override_pi0_active_robot_base_gray_to_white(self):
        """Match the active and static-right Franka shells to the real image."""
        from pxr import Gf, Sdf, UsdShade

        material_path = "/World/Looks/pi0_active_robot_base_white"
        material = UsdShade.Material.Define(self.sim.stage, material_path)
        shader = UsdShade.Shader.Define(
            self.sim.stage, f"{material_path}/PreviewSurface"
        )
        shader.CreateIdAttr("UsdPreviewSurface")
        shader.CreateInput(
            "diffuseColor", Sdf.ValueTypeNames.Color3f
        ).Set(Gf.Vec3f(*self.cfg.pi0_robot_visual_color))
        shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(
            float(self.cfg.pi0_robot_visual_roughness)
        )
        shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)
        material.CreateSurfaceOutput().ConnectToSource(
            shader.ConnectableAPI(), "surface"
        )
        robot_roots = (
            "/World/envs/env_0/franka_env/Robot/franka",
            "/World/envs/env_0/franka_env/ActiveRobot/franka",
            "/World/envs/env_0/ActiveRobot/franka",
            "/World/envs/env_0/franka_env/RightRobot/Robot/franka",
        )
        rebound_paths = []
        for robot_root in robot_roots:
            base_mesh_path = f"{robot_root}/base_link/visuals/link0/mesh"
            base_mesh = self.sim.stage.GetPrimAtPath(base_mesh_path)
            if base_mesh.IsValid():
                UsdShade.MaterialBindingAPI.Apply(base_mesh).Bind(
                    material,
                    bindingStrength=UsdShade.Tokens.strongerThanDescendants,
                )
                rebound_paths.append(base_mesh_path)

            for prim in self.sim.stage.Traverse():
                prim_path = str(prim.GetPath())
                if not prim_path.startswith(robot_root + "/"):
                    continue
                binding = prim.GetRelationship("material:binding")
                if not binding or not binding.IsValid():
                    continue
                targets = binding.GetTargets()
                is_plastic_white = any(
                    str(target).endswith("/PlasticWhite") for target in targets
                )
                is_base_connector = prim_path.endswith(
                    (
                        "/base_link/visuals/link0/mesh/Connectors_Metal",
                        "/base_link/visuals/link0/mesh/Connectors_Black",
                    )
                )
                if not is_plastic_white and not is_base_connector:
                    continue
                UsdShade.MaterialBindingAPI.Apply(prim).Bind(material)
                rebound_paths.append(prim_path)
        if not rebound_paths:
            print(
                "[Pi0] PlasticWhite bindings not found under Pi0 robots: "
                f"{robot_roots}",
                flush=True,
            )
            return
        print(
            "[Pi0] Rebound Pi0 robot light/gray geometry to real gray-white: "
            + ", ".join(rebound_paths),
            flush=True,
        )

    def close(self):
        static_usd = getattr(self, "_pi0_static_right_arm_usd", None)
        try:
            return super().close()
        finally:
            if static_usd is not None:
                Path(static_usd).unlink(missing_ok=True)
                self._pi0_static_right_arm_usd = None

    def _reset_pi0_joint_state_direct(self, env_ids) -> bool:
        """Optionally replace the Pi0 reset-time IK with a direct q reset.

        The default Pi0 environment keeps the authored IK reset. A diagnostic
        subclass can override this hook, write the robot joint state, and
        return True. The rest of the reset (held peg, gripper grasp, and
        observation bookkeeping) is shared by both paths.
        """
        del env_ids
        return False

    def _sync_pi0_static_right_arm_to_left(self):
        """Apply the fixed right-arm pose while keeping its configured base."""
        if not getattr(self.cfg, "background_right_robot_visual_only", False):
            return
        right_path = "/World/envs/env_0/franka_env/RightRobot/Robot/franka"
        if not self.sim.stage.GetPrimAtPath(right_path).IsValid():
            return
        q_right = np.asarray(
            getattr(self.cfg, "background_right_robot_joint_pos", ()),
            dtype=np.float64,
        ).reshape(7)
        self._pose_background_robot_visual(
            self.sim.stage,
            right_path,
            q_right,
            float(getattr(self.cfg, "background_right_robot_gripper_open", 0.04)),
        )
        self._pi0_static_right_arm_q = q_right.astype(np.float32)
        print(
            "[Pi0] Static right arm: base_offset="
            f"{tuple(getattr(self.cfg, 'background_right_robot_offset_pos', (0.86, 0.0, 0.0)))} "
            f"q={np.array2string(q_right, precision=6)} "
            "endpoint_target_local="
            f"{tuple(getattr(self.cfg, "background_right_robot_endpoint_reference_offset_pos", (0.40, -0.18, 0.14)))} "
            "gripper_open=0.04",
            flush=True,
        )

    def randomize_initial_state(self, env_ids):
        """Reset Pi0 with the IsaacLab 6 XYZW hand-down pose."""
        physics_sim_view = sim_utils.SimulationContext.instance().physics_sim_view
        physics_sim_view.set_gravity(carb.Float3(0.0, 0.0, 0.0))

        fixed_pose = self._fixed_asset.data.default_root_pose.torch.clone()[env_ids]
        fixed_vel = self._fixed_asset.data.default_root_vel.torch.clone()[env_ids]
        hole_noise_range = torch.as_tensor(
            getattr(self.cfg, "pi0_hole_position_noise_m", (0.0, 0.0, 0.0)),
            dtype=torch.float32,
            device=self.device,
        ).view(1, 3)
        reset_schedule = getattr(self, "pi0_reset_schedule", None)
        reset_schedule_index = int(getattr(self, "pi0_reset_schedule_index", 0))
        if reset_schedule is not None:
            # DirectRLEnv performs one internal reset immediately after the
            # final done step. Reuse the last scheduled pose for that cleanup
            # reset; the first N episodes still consume rows 0..N-1 exactly.
            schedule_row = reset_schedule[min(reset_schedule_index, len(reset_schedule) - 1)]
            hole_position_offset = torch.as_tensor(
                schedule_row["hole_offset_m"],
                dtype=torch.float32,
                device=self.device,
            ).view(1, 3).repeat(len(env_ids), 1)
        else:
            hole_position_offset = (
                2.0
                * torch.rand(
                    (len(env_ids), 3),
                    generator=self.rng,
                    dtype=torch.float32,
                    device=self.device,
                )
                - 1.0
            ) * hole_noise_range
        fixed_pose[:, 0:3] += self.scene.env_origins[env_ids] + hole_position_offset
        if not hasattr(self, "pi0_last_hole_position_offset_m"):
            self.pi0_last_hole_position_offset_m = torch.zeros(
                (self.num_envs, 3), dtype=torch.float32, device=self.device
            )
        self.pi0_last_hole_position_offset_m[env_ids] = hole_position_offset
        # Keep the Pi0 hole orientation explicit.  Do not rely only on the
        # inherited default pose: the legacy RealSim reset path rewrites the
        # quaternion through its old WXYZ yaw-only helper.
        fixed_pose[:, 3:7] = torch.as_tensor(
            self.cfg.pi0_hole_init_rot,
            dtype=torch.float32,
            device=self.device,
        ).view(1, 4)
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

        above_fixed_pos = fixed_tip_pos.clone()
        above_fixed_pos[:, 2] += self.cfg_task.hand_init_pos[2]
        hand_noise_range = torch.as_tensor(
            getattr(self.cfg, "pi0_hand_position_noise_m", (0.0, 0.0, 0.0)),
            dtype=torch.float32,
            device=self.device,
        ).view(1, 3)
        if reset_schedule is not None:
            hand_position_offset = torch.as_tensor(
                schedule_row["hand_offset_m"],
                dtype=torch.float32,
                device=self.device,
            ).view(1, 3).repeat(len(env_ids), 1)
            self.pi0_reset_schedule_index = min(
                reset_schedule_index + 1, len(reset_schedule) - 1
            )
        else:
            hand_position_offset = (
                2.0
                * torch.rand(
                    (len(env_ids), 3),
                    generator=self.rng,
                    dtype=torch.float32,
                    device=self.device,
                )
                - 1.0
            ) * hand_noise_range
        above_fixed_pos[env_ids] += hand_position_offset
        if not hasattr(self, "pi0_last_hand_position_offset_m"):
            self.pi0_last_hand_position_offset_m = torch.zeros(
                (self.num_envs, 3), dtype=torch.float32, device=self.device
            )
        self.pi0_last_hand_position_offset_m[env_ids] = hand_position_offset

        if not self._reset_pi0_joint_state_direct(env_ids):
            # hand_init_orn is [pi, 0, -pi/2], matching PPO/replay. Build the
            # quaternion directly with IsaacLab 6 XYZW convention.
            bad_envs = env_ids.clone()
            ik_attempt = 0
            max_ik_attempts = 10
            hand_down_euler = torch.tensor(
                self.cfg_task.hand_init_orn, dtype=torch.float32, device=self.device
            ).view(1, 3).repeat(self.num_envs, 1)
            hand_down_quat = isaaclab_math.quat_from_euler_xyz(
                hand_down_euler[:, 0], hand_down_euler[:, 1], hand_down_euler[:, 2]
            )
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
                    ik_attempt += 1
                    if ik_attempt >= max_ik_attempts:
                        pos_norm = torch.linalg.norm(pos_error, dim=1)
                        aa_norm = torch.linalg.norm(aa_error, dim=1)
                        print(
                            f"[Pi0] IK did not converge after {max_ik_attempts} attempts: "
                            f"pos={pos_norm.detach().cpu().numpy()}, "
                            f"angle={aa_norm.detach().cpu().numpy()}"
                        )
                        break
                    self._set_franka_to_default_pose(
                        joints=[0.00871, -0.10368, -0.00794, -1.49139, -0.00083, 1.38774, 0.0],
                        env_ids=bad_envs,
                    )

        self.step_sim_no_action()

        # Place the held peg using the same XYZW frame chain as PPO/replay.
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
        relative_pos[:, 2] = _PI0_PEG_ROOT_OFFSET_M + float(
            getattr(self.cfg, "pi0_peg_mount_depth_adjust_m", 0.0)
        )
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

        # Match rollout's final post-grasp snap. Without this write, the
        # independent HeldAsset can settle away from the fingertip during the
        # close/settle phase, so the first Pi0 image shows a misplaced peg.
        if bool(getattr(self.cfg_task, "snap_held_asset_after_grasp", False)):
            snapped_zero = torch.zeros((self.num_envs, 3), device=self.device)
            snapped_flip = torch.zeros((self.num_envs, 4), device=self.device)
            snapped_flip[:, 1] = 1.0
            snapped_pos, snapped_quat = isaaclab_math.combine_frame_transforms(
                self.fingertip_midpoint_pos,
                self.fingertip_midpoint_quat,
                snapped_zero,
                snapped_flip,
            )
            snapped_relative = torch.zeros((self.num_envs, 3), device=self.device)
            snapped_relative[:, 2] = (
                self.cfg_task.held_asset_cfg.height
                - self.cfg_task.robot_cfg.franka_fingerpad_length
                + float(getattr(self.cfg, "pi0_peg_mount_depth_adjust_m", 0.0))
            )
            snapped_pos, snapped_quat = isaaclab_math.combine_frame_transforms(
                snapped_pos,
                snapped_quat,
                -snapped_relative,
                identity_quat,
            )
            snapped_pose = self._held_asset.data.default_root_pose.torch.clone()[env_ids]
            snapped_vel = self._held_asset.data.default_root_vel.torch.clone()[env_ids]
            snapped_pose[:, :3] = snapped_pos[env_ids] + self.scene.env_origins[env_ids]
            snapped_pose[:, 3:7] = snapped_quat[env_ids]
            snapped_vel.zero_()
            self._held_asset.write_root_pose_to_sim_index(
                root_pose=snapped_pose, env_ids=env_ids
            )
            self._held_asset.write_root_velocity_to_sim_index(
                root_velocity=snapped_vel, env_ids=env_ids
            )
            self._held_asset.reset()
            self.step_sim_no_action()

        self._attach_held_asset(env_ids)

        self.prev_joint_pos = self.joint_pos[:, 0:7].clone()
        self.prev_fingertip_pos = self.fingertip_midpoint_pos.clone()
        self.prev_fingertip_quat = self.fingertip_midpoint_quat.clone()
        self.actions = torch.zeros_like(self.actions)
        self.prev_actions = torch.zeros_like(self.actions)
        self.ee_angvel_fd.zero_()
        self.ee_linvel_fd.zero_()
        self.task_prop_gains = self.default_gains
        self.task_deriv_gains = factory_utils.get_deriv_gains(self.default_gains)
        # Copy only after the active arm has completed its reset IK/grasp
        # sequence; the right arm remains visual-only afterwards.
        self._sync_pi0_static_right_arm_to_left()
        physics_sim_view.set_gravity(carb.Float3(*self.cfg.sim.gravity))

    def _joint_limits(self):
        limits = getattr(self._robot.data, "soft_joint_pos_limits", None)
        if limits is None:
            limits = getattr(self._robot.data, "joint_pos_limits", None)
        if limits is None:
            raise RuntimeError("Isaac articulation does not expose joint position limits")
        if limits.ndim == 2:
            limits = limits.unsqueeze(0)
        return limits[:, :9, 0], limits[:, :9, 1]

    def _current_pi0_state(self) -> torch.Tensor:
        if self.joint_pos.shape[1] < 9:
            raise RuntimeError("Pi0 joint control requires 7 arm joints and 2 finger joints")
        state_override = getattr(self._pi0_cfg, "gripper_state_override", None)
        if state_override is None:
            gripper = self.joint_pos[:, 7:9].sum(dim=1) * float(self._pi0_cfg.gripper_state_scale)
        else:
            gripper = torch.full(
                (self.num_envs,), float(state_override), dtype=torch.float32, device=self.device
            )
        gripper = torch.clamp(
            gripper,
            float(self._pi0_cfg.gripper_policy_min),
            float(self._pi0_cfg.gripper_policy_max),
        ).unsqueeze(-1)
        return torch.cat((self.joint_pos[:, :7], gripper), dim=-1)

    def _set_pi0_hole_visual_color(self, color) -> None:
        """Set the bound hole material color for one camera render."""
        from pxr import Gf, UsdShade

        shader = UsdShade.Shader.Get(
            self.sim.stage,
            "/World/Looks/fixedasset_color_override/PreviewSurface",
        )
        if not shader.GetPrim().IsValid():
            raise RuntimeError("Pi0 hole color-override shader was not created")
        diffuse = shader.GetInput("diffuseColor")
        if not diffuse:
            raise RuntimeError("Pi0 hole color-override shader has no diffuseColor input")
        diffuse.Set(Gf.Vec3f(*[float(value) for value in color]))

    def _clean_wrist_camera_frame(self, frame):
        """Apply the same wrist-edge cleanup to the continuous video frame."""

        if torch.is_tensor(frame):
            image = frame.detach().cpu().numpy()
        else:
            image = np.asarray(frame)
        cleaned = _sanitize_pi0_wrist_green_overlay(image, self.cfg)
        return torch.from_numpy(cleaned)

    def _camera_observation(self):
        if not hasattr(self, "tiled_camera") or self.tiled_camera is None:
            raise RuntimeError("Pi0 requires the front camera")
        if not hasattr(self, "wrist_tiled_camera") or self.wrist_tiled_camera is None:
            raise RuntimeError("Pi0 requires the wrist camera")

        # The rear white tabletop is a front-camera visual match.  The real
        # wrist camera sees only the green mat in this region, so hide the
        # visual-only white overlay while rendering the wrist frame.
        from pxr import UsdGeom

        white_overlay = self.sim.stage.GetPrimAtPath(
            "/World/envs/env_0/pi0_visual_matching/white_table"
        )
        white_overlay_imageable = (
            UsdGeom.Imageable(white_overlay)
            if white_overlay.IsValid() and white_overlay.IsA(UsdGeom.Imageable)
            else None
        )
        white_visibility = (
            white_overlay_imageable.GetVisibilityAttr().Get()
            if white_overlay_imageable is not None
            else None
        )
        wrist_green_overlay = self.sim.stage.GetPrimAtPath(
            getattr(self, "_pi0_wrist_green_table_overlay_path", "")
        )
        wrist_green_overlay_imageable = (
            UsdGeom.Imageable(wrist_green_overlay)
            if wrist_green_overlay.IsValid() and wrist_green_overlay.IsA(UsdGeom.Imageable)
            else None
        )
        if wrist_green_overlay_imageable is not None:
            # The enlarged mesh is wrist-only. Keep it out of the front
            # render, where the tight annotated polygon is still used.
            wrist_green_overlay_imageable.MakeInvisible()
        self._set_pi0_hole_visual_color(self.cfg.pi0_front_hole_visual_color)
        self.sim.render()
        self.tiled_camera.update(self.physics_dt, force_recompute=True)
        front = (
            self.tiled_camera.data.output["rgb"][0]
            .detach()
            .cpu()
            .numpy()
            .copy()
        )

        self._set_pi0_hole_visual_color(self.cfg.pi0_wrist_hole_visual_color)
        if white_overlay_imageable is not None:
            white_overlay_imageable.MakeInvisible()
        if wrist_green_overlay_imageable is not None:
            wrist_green_overlay_imageable.MakeVisible()
        # RTX material edits become visible one render later. Prime the
        # material change before rendering the frame consumed by the sensor.
        self.sim.render()
        self.sim.render()
        self.wrist_tiled_camera.update(self.physics_dt, force_recompute=True)

        if white_overlay_imageable is not None:
            if white_visibility == UsdGeom.Tokens.invisible:
                white_overlay_imageable.MakeInvisible()
            else:
                white_overlay_imageable.MakeVisible()
        self._set_pi0_hole_visual_color(self.cfg.pi0_front_hole_visual_color)
        wrist = self.wrist_tiled_camera.data.output["rgb"][0].detach().cpu().numpy()
        if wrist_green_overlay_imageable is not None:
            # Do not render again here. The wrist tensor already contains the
            # static overlay and is what the recorder/policy consumes.
            wrist_green_overlay_imageable.MakeInvisible()
        front = _resize_pi0_rgb(front, self.cfg)
        front = _composite_pi0_green_table_front(front, self.cfg)
        front = _composite_pi0_green_table_front(
            front,
            self.cfg,
            polygon_attr="pi0_green_table_right_patch_image_polygon",
            enabled_attr="pi0_green_table_right_edge_front_repair_enabled",
            alpha_attr="pi0_green_table_right_edge_front_repair_alpha",
            apply_hole_cutout=False,
        )
        front = _composite_pi0_reference_cylinder(front, self.cfg, "rgb")
        front = _composite_pi0_cable_grommets(front, self.cfg, "rgb")
        wrist = _resize_pi0_rgb(wrist, self.cfg)
        wrist = _sanitize_pi0_wrist_green_overlay(wrist, self.cfg)
        wrist = _match_pi0_wrist_appearance(wrist, self.cfg, "rgb")
        return front, wrist

    def _continuous_policy_video_observation(self):
        """Build a continuous video frame from already-rendered sensor data.

        The server input path above intentionally performs the camera-specific
        material switches and RTX renders.  Video recording must not repeat
        those renders at every control step, so this path only reads the latest
        sensor buffers and applies the CPU-side image transforms.
        """

        if not hasattr(self, "tiled_camera") or self.tiled_camera is None:
            raise RuntimeError("Pi0 requires the front camera")
        if not hasattr(self, "wrist_tiled_camera") or self.wrist_tiled_camera is None:
            raise RuntimeError("Pi0 requires the wrist camera")

        front = (
            self.tiled_camera.data.output["rgb"][0]
            .detach()
            .cpu()
            .numpy()
            .copy()
        )
        wrist = (
            self.wrist_tiled_camera.data.output["rgb"][0]
            .detach()
            .cpu()
            .numpy()
            .copy()
        )

        front = _resize_pi0_rgb(front, self.cfg)
        front = _composite_pi0_green_table_front(front, self.cfg)
        front = _composite_pi0_green_table_front(
            front,
            self.cfg,
            polygon_attr="pi0_green_table_right_patch_image_polygon",
            enabled_attr="pi0_green_table_right_edge_front_repair_enabled",
            alpha_attr="pi0_green_table_right_edge_front_repair_alpha",
            apply_hole_cutout=False,
        )
        front = _composite_pi0_reference_cylinder(front, self.cfg, "rgb")
        front = _composite_pi0_cable_grommets(front, self.cfg, "rgb")
        wrist = _resize_pi0_rgb(wrist, self.cfg)
        wrist = _sanitize_pi0_wrist_green_overlay(wrist, self.cfg)
        wrist = _match_pi0_wrist_appearance(wrist, self.cfg, "rgb")
        return np.ascontiguousarray(front), np.ascontiguousarray(wrist)

    def _refresh_model_input_video_frame(self):
        """Cache a continuous transformed frame without triggering RTX render."""

        front, wrist = self._continuous_policy_video_observation()
        self.last_model_input_front = torch.from_numpy(
            np.ascontiguousarray(front)
        ).clone()
        self.last_model_input_wrist = torch.from_numpy(
            np.ascontiguousarray(wrist)
        ).clone()
        self._model_visual_frame_ready = True
        return True

    def _pi0_observation(self):
        front, wrist = self._camera_observation()
        # Cache the exact tensors passed to the Pi0 policy. The evaluator can
        # record these without falling back to the later raw camera buffer.
        self.last_model_input_front = torch.from_numpy(
            np.ascontiguousarray(front)
        ).clone()
        self.last_model_input_wrist = torch.from_numpy(
            np.ascontiguousarray(wrist)
        ).clone()
        self._model_visual_frame_ready = True
        state = self._current_pi0_state()[0].detach().cpu().numpy()
        # The 8001 Pi0 checkpoint was trained with the current six-dimensional
        # real-data-compatible wrench frame. ``wrench_final`` is the same
        # sign-corrected quantity used by the existing TAVLA rollout path.
        effort = self.wrench_final[0].detach().cpu().numpy().astype(np.float32)
        effort = effort.reshape(1, 6)
        if not np.isfinite(effort).all():
            raise FloatingPointError("Pi0 effort contains NaN or Inf")
        return {
            "front_rgb": front,
            "wrist_rgb": wrist,
            "state8": state,
            "effort": effort,
            "prompt": "peg-in-hole",
        }

    def _fetch_pi0_chunk(self) -> bool:
        now = time.perf_counter()
        if now < self._pi0_retry_after:
            return False
        try:
            started = now
            chunk = self._pi0_policy.predict_action_chunk(self._pi0_observation())
            elapsed = time.perf_counter() - started
            self._pi0_chunk = np.asarray(chunk, dtype=np.float32)
            if self._pi0_chunk.ndim != 2 or self._pi0_chunk.shape[1] != 8:
                raise ValueError(
                    f"Pi0 action chunk must have shape (H, 8), got {self._pi0_chunk.shape}"
                )
            self._pi0_last_chunk = self._pi0_chunk.copy()
            self._pi0_chunk_id += 1
            self._pi0_chunk_index = max(1, int(self.cfg.pi0_action_start_index))
            self._pi0_chunk_end = min(
                self._pi0_chunk.shape[0],
                self._pi0_chunk_index + max(1, int(self.cfg.pi0_replan_actions)),
            )
            self.pi0_inference_count += 1
            self.pi0_last_latency_s = float(elapsed)
            self.pi0_last_error = ""
            self._pi0_retry_after = 0.0
            return True
        except Exception as exc:
            self.pi0_failures += 1
            self.pi0_last_error = str(exc)
            retry_base = max(
                float(getattr(self._pi0_cfg, "inference_retry_backoff_s", 1.0)),
                0.1,
            )
            retry_delay = min(
                5.0,
                retry_base * (2.0 ** min(self.pi0_failures - 1, 3)),
            )
            self._pi0_retry_after = time.perf_counter() + retry_delay
            self._pi0_chunk = np.empty((0, 8), dtype=np.float32)
            self._pi0_chunk_index = 0
            self._pi0_chunk_end = 0
            return False

    def _set_pi0_target(self, action) -> None:
        action = torch.as_tensor(action, dtype=torch.float32, device=self.device).reshape(-1)
        if action.shape != (8,) or not torch.isfinite(action).all():
            raise ValueError(f"Pi0 target must be finite with shape (8,), got {tuple(action.shape)}")

        lower, upper = self._joint_limits()
        self._pi0_desired_q = action[:7].detach().clone().view(1, 7)
        desired_q = torch.minimum(torch.maximum(action[:7].unsqueeze(0), lower[:, :7]), upper[:, :7])
        if not torch.equal(desired_q, action[:7].unsqueeze(0)):
            self.pi0_target_clipped += 1

        if self._pi0_command_q is None:
            self._pi0_command_q = self.joint_pos[:, :7].detach().clone()
        step_dt = max(float(self.step_dt), 1.0e-6)
        velocity_limits = torch.as_tensor(
            self._pi0_cfg.joint_velocity_limits, dtype=torch.float32, device=self.device
        ).view(1, 7)
        max_delta = velocity_limits * step_dt
        self._pi0_command_q = self._pi0_command_q + torch.clamp(
            desired_q - self._pi0_command_q, -max_delta, max_delta
        )
        self._pi0_command_q = torch.minimum(torch.maximum(self._pi0_command_q, lower[:, :7]), upper[:, :7])
        if getattr(self._pi0_cfg, "hold_wrist_joints", False):
            if self._pi0_wrist_hold_target is None:
                self._pi0_wrist_hold_target = self.joint_pos[:, 4:7].detach().clone()
            self._pi0_command_q[:, 4:7] = self._pi0_wrist_hold_target

        if getattr(self._pi0_cfg, "hold_gripper", True):
            # The real trajectory keeps the gripper fixed. Do not execute the
            # model's eighth action dimension in the simulator, but keep the
            # state sent to Pi0 in the raw training domain.
            self._pi0_command_gripper = torch.full(
                (self.num_envs,),
                float(getattr(self._pi0_cfg, "gripper_state_override", 0.08652404)),
                dtype=torch.float32,
                device=self.device,
            )
        else:
            desired_gripper = torch.clamp(
                action[7],
                float(self._pi0_cfg.gripper_policy_min),
                float(self._pi0_cfg.gripper_policy_max),
            )
            if float(desired_gripper) != float(action[7]):
                self.pi0_target_clipped += 1
            if self._pi0_command_gripper is None:
                self._pi0_command_gripper = self._current_pi0_state()[:, 7].detach().clone()
            gripper_delta = float(self._pi0_cfg.gripper_policy_velocity_limit) * step_dt
            self._pi0_command_gripper = self._pi0_command_gripper + torch.clamp(
                desired_gripper - self._pi0_command_gripper, -gripper_delta, gripper_delta
            )
            self._pi0_command_gripper = torch.clamp(
                self._pi0_command_gripper,
                float(self._pi0_cfg.gripper_policy_min),
                float(self._pi0_cfg.gripper_policy_max),
            )
        self._pi0_target = torch.cat((self._pi0_command_q, self._pi0_command_gripper.unsqueeze(-1)), dim=-1)

    def _advance_pi0_target(self) -> None:
        if self._pi0_chunk_index >= self._pi0_chunk_end:
            if not self._fetch_pi0_chunk():
                return
        if self._pi0_chunk_index < self._pi0_chunk_end:
            action_index = self._pi0_chunk_index
            self._set_pi0_target(self._pi0_chunk[action_index])
            self._pi0_last_chunk_id = self._pi0_chunk_id
            self._pi0_last_chunk_action_index = int(action_index)
            self._pi0_chunk_index += 1

    def _reset_pi0_runtime(self) -> None:
        self._pi0_policy.reset()
        current = self._current_pi0_state()
        self._pi0_command_q = current[:, :7].detach().clone()
        self._pi0_desired_q = current[:, :7].detach().clone()
        self._pi0_command_gripper = current[:, 7].detach().clone()
        # Keep a finite closed preload while Pi0 controls the arm.  Holding
        # the measured post-grasp opening removes the preload, while a zero
        # target can drive both fingers into the 40 N/side actuator limit when
        # the peg contacts the hole.  A 2.5 mm target gives roughly 25 N per
        # finger at the measured 5.8 mm contact opening.
        self._pi0_finger_hold_target = torch.full_like(self.joint_pos[:, 7:9], 0.0025)
        self._pi0_wrist_hold_target = self.joint_pos[:, 4:7].detach().clone()
        self._pi0_target = current.detach().clone()
        self._pi0_chunk = np.empty((0, 8), dtype=np.float32)
        self._pi0_last_chunk = np.empty((0, 8), dtype=np.float32)
        self._pi0_chunk_id = -1
        self._pi0_last_chunk_id = -1
        self._pi0_last_chunk_action_index = -1
        self._pi0_chunk_index = 0
        self._pi0_chunk_end = 0
        self._pi0_skip_advance_once = True
        self.pi0_inference_count = 0
        self.pi0_failures = 0
        self.pi0_last_error = ""
        self._pi0_retry_after = 0.0
        self._advance_pi0_target()

    def _get_observations(self):
        if self._pi0_ready:
            if self._pi0_skip_advance_once:
                self._pi0_skip_advance_once = False
            else:
                self._advance_pi0_target()
            if self._pi0_target is not None:
                self.next_action = self._pi0_target.detach().clone()
                self.ppo_joint_target = self._pi0_target.detach().clone()
        return super()._get_observations()

    def _apply_action(self):
        if not self._pi0_ready or self._pi0_target is None:
            return super()._apply_action()
        if self.last_update_timestamp < self._robot._data._sim_timestamp:
            self._compute_intermediate_values(dt=self.physics_dt)

        lower, upper = self._joint_limits()
        q_target = torch.minimum(torch.maximum(self._pi0_target[:, :7], lower[:, :7]), upper[:, :7])

        if getattr(self._pi0_cfg, "hold_gripper", True):
            if self._pi0_finger_hold_target is None:
                self._pi0_finger_hold_target = self.joint_pos[:, 7:9].detach().clone()
            finger_target = self._pi0_finger_hold_target.detach().clone()
        else:
            finger_target = self._pi0_target[:, 7:8] * float(self._pi0_cfg.gripper_action_to_finger_scale)
            finger_target = torch.cat((finger_target, finger_target), dim=-1)
        finger_target = torch.minimum(torch.maximum(finger_target, lower[:, 7:9]), upper[:, 7:9])

        self.ctrl_target_joint_pos[:, :7] = q_target
        self.ctrl_target_joint_pos[:, 7:9] = finger_target

        if getattr(self._pi0_cfg, "use_taskspace_controller", True):
            # Task-space mode is optional. Only this mode may alter the
            # orientation represented by Pi0's joint target.
            target_pos, target_quat = self._joint_target_to_taskspace_target(q_target)
            target_pos = self._limit_cartesian_target_speed(target_pos)
            if getattr(self._pi0_cfg, "enforce_fingertip_orientation", False):
                target_quat = self._constrain_fingertip_orientation(target_quat)
            target_quat = self._limit_cartesian_target_orientation(target_quat)
            self.ctrl_target_fingertip_midpoint_pos = target_pos
            self.ctrl_target_fingertip_midpoint_quat = target_quat
            self.ctrl_target_gripper_dof_pos = finger_target
            self.generate_ctrl_signals(
                ctrl_target_fingertip_midpoint_pos=target_pos,
                ctrl_target_fingertip_midpoint_quat=target_quat,
                ctrl_target_gripper_dof_pos=finger_target,
            )
            return

        if getattr(self._pi0_cfg, "use_implicit_position_controller", True):
            # Match rollout's joint-space handling: preserve the absolute
            # seven-joint target without an extra FK/IK orientation projection.
            self._robot.set_joint_position_target(self.ctrl_target_joint_pos)
            self._robot.set_joint_effort_target(torch.zeros_like(self.joint_pos))
            return

        # Fallback direct joint torque path, retained for explicit diagnostics.
        kp = torch.as_tensor(self._pi0_cfg.joint_kp, dtype=torch.float32, device=self.device).view(1, 7)
        kd = torch.as_tensor(self._pi0_cfg.joint_kd, dtype=torch.float32, device=self.device).view(1, 7)
        effort_limits = torch.as_tensor(
            self._pi0_cfg.joint_effort_limits, dtype=torch.float32, device=self.device
        ).view(1, 7)
        q_error = (q_target - self.joint_pos[:, :7] + torch.pi) % (2.0 * torch.pi) - torch.pi
        torque = torch.clamp(kp * q_error - kd * self.joint_vel[:, :7], -effort_limits, effort_limits)
        full_torque = torch.zeros_like(self.joint_pos)
        full_torque[:, :7] = torque
        self._robot.set_joint_position_target(self.ctrl_target_joint_pos)
        self._robot.set_joint_effort_target(full_torque)

    def _get_rewards(self):
        # The inherited Forge reward contains a Cartesian action penalty. Pi0
        # supplies joint targets, so there is no Cartesian delta to penalize.
        self.delta_pos = torch.zeros((self.num_envs, 3), dtype=torch.float32, device=self.device)
        self.delta_yaw = torch.zeros((self.num_envs,), dtype=torch.float32, device=self.device)
        return super()._get_rewards()

    def _reset_idx(self, env_ids):
        super()._reset_idx(env_ids)
        # The inherited reset finishes the active left-arm IK/grasp sequence
        # first. Copy that final q once; the right arm remains visual-only.
        self._sync_pi0_static_right_arm_to_left()
        if self._pi0_ready:
            self._reset_pi0_runtime()

    def close(self):
        if self._pi0_policy is not None:
            self._pi0_policy.close()
        super().close()
