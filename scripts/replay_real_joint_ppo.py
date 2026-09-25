"""Replay traj data by writing the measured Franka joint state directly.

This is the joint-space path used by RealSim/PPO.  It deliberately does not
convert the measured end-effector pose and does not call the DLS IK solver.
The existing Cartesian replay remains untouched.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import zlib
from pathlib import Path

import cv2
import h5py
import numpy as np
import torch
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--h5", type=Path, default=Path("real_data/traj_0/data.h5"))
parser.add_argument(
    "--data-dir",
    type=Path,
    default=None,
    help="Replay a contiguous batch from data-dir/traj_<i>/data.h5 in one Isaac Sim process.",
)
parser.add_argument(
    "--trajectory-start",
    type=int,
    default=0,
    help="First trajectory index for --data-dir batch replay.",
)
parser.add_argument(
    "--trajectory-count",
    type=int,
    default=None,
    help="Number of trajectories for --data-dir batch replay.",
)
parser.add_argument(
    "--output-root",
    type=Path,
    default=Path("outputs/sim_force"),
    help="Root directory for --data-dir batch outputs; each trajectory uses traj_<i>.",
)
parser.add_argument("--task", type=str, default="TacEx-RealSim-PegInsert-Direct-v0")
parser.add_argument("--output-dir", type=Path, default=Path("outputs/traj0_joint_ppo_rollout"))
parser.add_argument("--fps", type=float, default=10.0)
parser.add_argument(
    "--domain-randomization",
    action="store_true",
    help="Sample one coherent visual domain for the complete rollout.",
)
parser.add_argument(
    "--dr-seed",
    type=int,
    default=None,
    help="Domain-randomization seed; defaults to a stable seed derived from --h5.",
)
parser.add_argument(
    "--max-frames",
    type=int,
    default=None,
    help="Replay only the first N frames; omit to replay the complete trajectory.",
)
parser.add_argument(
    "--gripper-constant",
    type=float,
    default=None,
    help=(
        "Override the per-frame gripper opening with a single constant for the "
        "whole rollout (normalized units, same scale as action/actual/gripper). "
        "Use 0.0865 to force the unified peg-grasp pose so every rollout renders "
        "the fingers identically regardless of the recorded gripper channel."
    ),
)
parser.add_argument(
    "--save-sim-force",
    action="store_true",
    help=(
        "Record the PPO/PhysX wrench streams and replace the training-contract "
        "base wrench in the paired H5 with simulated values."
    ),
)
parser.add_argument(
    "--physical-wrist-force",
    action="store_true",
    help=(
        "Author a non-rendered peg collider under the robot fingertip before "
        "articulation creation and read the calibrated force_sensor incoming wrench. "
        "Measured joints are still written directly every recorded frame."
    ),
)
parser.add_argument(
    "--wrench-baseline-seconds",
    type=float,
    default=3.0,
    help="Initial no-contact interval used for causal wrist-wrench zeroing.",
)
parser.add_argument(
    "--wrench-ema-alpha",
    type=float,
    default=0.25,
    help="Causal EMA alpha applied after wrist-wrench baseline subtraction.",
)
parser.add_argument(
    "--training-wrench-source",
    choices=("incoming-joint", "contact-equivalent"),
    default="incoming-joint",
    help=(
        "Signal written to obs/state/ee_wrench_base. 'incoming-joint' keeps the "
        "articulation reaction; 'contact-equivalent' uses the peg/hole contact "
        "wrench transported to the Franka K point."
    ),
)
parser.add_argument(
    "--counterfactual-wrench-subtraction",
    action="store_true",
    help=(
        "Run the extra no-contact pass before subtracting direct-write articulation "
        "reaction. Diagnostic only; unnecessary for contact-equivalent training."
    ),
)
parser.add_argument(
    "--contact-compliant-stiffness-n-m",
    type=float,
    default=1000.0,
    help="Compliant-contact stiffness assigned to the integrated replay peg.",
)
parser.add_argument(
    "--enable-compliant-contact",
    action="store_true",
    help=(
        "Opt in to PhysX compliant contact on the integrated peg. Disabled by "
        "default because this Isaac Sim contact view reports zero force for the "
        "tested compliant material path."
    ),
)
parser.add_argument(
    "--contact-compliant-damping-n-s-m",
    type=float,
    default=5.0,
    help="Compliant-contact damping assigned to the integrated replay peg.",
)
parser.add_argument(
    "--hole-reference",
    choices=("gt-final-xy-fixed-z", "gt-final", "configured"),
    default="gt-final-xy-fixed-z",
    help=(
        "Choose the fixed-hole pose. 'gt-final-xy-fixed-z' infers only XY from "
        "the final GT peg pose and keeps the calibrated hole Z; 'gt-final' "
        "retains the legacy behavior that copies final GT XYZ; 'configured' "
        "keeps the complete fixed pose used by Pi0 evaluation."
    ),
)
parser.add_argument(
    "--resolve-asset-collisions",
    action="store_true",
    help=(
        "Run PhysX contact resolution for the peg/hole after each recorded pose. "
        "The measured robot joints remain authoritative."
    ),
)
parser.add_argument(
    "--collision-substeps",
    type=int,
    default=4,
    help="Physics substeps used by --resolve-asset-collisions (default: 4).",
)
parser.add_argument(
    "--asset-contact-offset-m",
    type=float,
    default=None,
    help=(
        "Optional peg/hole PhysX contact offset in metres. The configured "
        "asset value is preserved when omitted."
    ),
)
parser.add_argument(
    "--asset-max-depenetration-velocity-m-s",
    type=float,
    default=None,
    help=(
        "Optional peg/hole maximum depenetration velocity in m/s. The "
        "configured asset value is preserved when omitted."
    ),
)
parser.add_argument(
    "--incoming-wrench-body",
    choices=(
        "force_sensor",
        "panda_hand",
        "panda_fingertip_centered",
        "panda_link7",
        "panda_link8",
    ),
    default=None,
    help=(
        "Override the robot body used for PhysX incoming-joint-wrench reads. "
        "Default keeps the existing force_sensor behavior."
    ),
)
parser.add_argument(
    "--record-contact-pair",
    action="store_true",
    help=(
        "Record direct PhysX peg-hole normal/friction/total contact forces. "
        "Diagnostic only; does not change robot or peg commands."
    ),
)
parser.add_argument(
    "--contact-force-smoothing-alpha",
    type=float,
    default=0.25,
    help=(
        "EMA alpha for the recorded contact-force trend, in (0,1]. "
        "The raw substep-averaged signal is also saved."
    ),
)
parser.add_argument(
    "--tavla-only-output",
    action="store_true",
    help=(
        "Keep only TAVLA handoff files and videos in --output-dir after the "
        "replay. Existing control and force generation are unchanged."
    ),
)
parser.add_argument(
    "--probe-pi0",
    action="store_true",
    help=(
        "At each GT pose, send the rendered sim images plus the GT state to Pi0 "
        "without executing the returned actions."
    ),
)
parser.add_argument("--pi0-host", type=str, default="114.214.164.36")
parser.add_argument("--pi0-port", type=int, default=8000)
parser.add_argument(
    "--pi0-action-index",
    type=int,
    default=1,
    help="Pi0 chunk index to compare with the next GT joint state (default: 1).",
)
parser.add_argument(
    "--real-probe-summary",
    type=Path,
    default=None,
    help="Optional summary.json from diagnose_pi0_traj0_offline.py for direct comparison.",
)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if args.collision_substeps <= 0:
    raise ValueError("--collision-substeps must be greater than zero")
if args.asset_contact_offset_m is not None and args.asset_contact_offset_m <= 0.0:
    raise ValueError("--asset-contact-offset-m must be greater than zero")
if (
    args.asset_max_depenetration_velocity_m_s is not None
    and args.asset_max_depenetration_velocity_m_s <= 0.0
):
    raise ValueError(
        "--asset-max-depenetration-velocity-m-s must be greater than zero"
    )
if args.trajectory_start < 0:
    raise ValueError("--trajectory-start must be non-negative")
if args.trajectory_count is not None and args.trajectory_count <= 0:
    raise ValueError("--trajectory-count must be greater than zero")
if args.data_dir is not None and args.trajectory_count is None:
    raise ValueError("--data-dir requires --trajectory-count")
if not 0.0 < args.contact_force_smoothing_alpha <= 1.0:
    raise ValueError("--contact-force-smoothing-alpha must be in (0, 1]")
if not 0.0 < args.wrench_ema_alpha <= 1.0:
    raise ValueError("--wrench-ema-alpha must be in (0, 1]")
if args.wrench_baseline_seconds <= 0.0:
    raise ValueError("--wrench-baseline-seconds must be greater than zero")
if args.contact_compliant_stiffness_n_m <= 0.0:
    raise ValueError("--contact-compliant-stiffness-n-m must be greater than zero")
if args.contact_compliant_damping_n_s_m < 0.0:
    raise ValueError("--contact-compliant-damping-n-s-m must be non-negative")
if args.training_wrench_source == "contact-equivalent" and not args.record_contact_pair:
    raise ValueError("contact-equivalent training requires --record-contact-pair")
if args.save_sim_force and not args.record_contact_pair and not args.physical_wrist_force:
    raise ValueError("--save-sim-force requires --record-contact-pair")
if args.save_sim_force and not args.resolve_asset_collisions:
    raise ValueError("--save-sim-force requires --resolve-asset-collisions")
if args.physical_wrist_force and not args.save_sim_force:
    raise ValueError("--physical-wrist-force requires --save-sim-force")
if args.pi0_action_index < 0:
    raise ValueError("--pi0-action-index must be >= 0")
# Isaac Sim/RTX needs the host GPU.  Some launch shells export an empty
# CUDA_VISIBLE_DEVICES, which makes PyTorch report zero devices even when the
# NVIDIA driver is available; treat an empty value as unset.
if os.environ.get("CUDA_VISIBLE_DEVICES", "").strip() == "":
    os.environ.pop("CUDA_VISIBLE_DEVICES", None)
if hasattr(args, "enable_cameras"):
    args.enable_cameras = True

# Fabric transform reads and RTX geometry streaming publish dynamic Franka
# links out of sync in Isaac Sim 6.  Keep Fabric for PPO articulation replay,
# but disable geometry streaming before Kit starts.
_kit_args = (
    "--/log/level=warning --/log/outputStreamLevel=warning "
    "--/rtx-transient/resourcemanager/enableGeometryStreaming=false"
)
args.kit_args = f"{getattr(args, 'kit_args', '')} {_kit_args}".strip()

simulation_app = AppLauncher(args).app

# Isaac Sim 6 keeps this legacy Python module in an extension directory that
# is not on isaacsim.__path__ by default. TacEx still imports XFormPrim from it.
import isaacsim  # noqa: E402
_ISAACSIM_EXTS = Path(isaacsim.__file__).resolve().parent / "exts"
for _extension_name in ("isaacsim.core.prims", "isaacsim.core.utils"):
    _extension_pkg = _ISAACSIM_EXTS / _extension_name / "isaacsim"
    if _extension_pkg.is_dir() and str(_extension_pkg) not in isaacsim.__path__:
        isaacsim.__path__.append(str(_extension_pkg))

import tacex_tasks  # noqa: E402,F401
from tacex_tasks import torch_compat as torch_utils  # noqa: E402
from isaaclab.utils import math as math_utils  # noqa: E402
from isaaclab_tasks.direct.factory import factory_utils  # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402
from tacex_tasks.real2sim.pi0_env import (  # noqa: E402
    Pi0RealSimEnv,
    _composite_pi0_green_table_front,
    _composite_pi0_cable_grommets,
    _composite_pi0_reference_cylinder,
    _match_pi0_wrist_appearance,
    _pi0_wrist_camera_rotation,
    _resize_pi0_rgb,
)
from tacex_tasks.real2sim.pi0_env_cfg import RealSimPi0PegInsertCfg  # noqa: E402
from tacex_tasks.real2sim.realsim_env import RealSimEnv, _write_h264_mp4  # noqa: E402

if args.probe_pi0:
    from openpi_client.websocket_client_policy import (  # noqa: E402
        WebsocketClientPolicy,
    )


def _scaled_color(color, scale: float, red_blue_tint: float = 0.0) -> tuple[float, float, float]:
    value = np.asarray(color, dtype=np.float64) * float(scale)
    value *= np.asarray((1.0 + red_blue_tint, 1.0, 1.0 - red_blue_tint))
    return tuple(float(channel) for channel in np.clip(value, 0.0, 1.0))


def sample_visual_domain_randomization(cfg, h5_path: Path, seed: int | None) -> dict:
    """Sample conservative rollout-level visual randomization parameters."""

    if seed is None:
        seed = zlib.crc32(str(h5_path.resolve()).encode("utf-8")) & 0x7FFFFFFF
    rng = np.random.default_rng(seed)

    light_scale = float(rng.uniform(0.75, 1.25))
    light_tint = float(rng.uniform(-0.06, 0.06))
    exposure_shared = float(rng.uniform(-0.40, 0.40))
    front_exposure_delta = exposure_shared + float(rng.uniform(-0.08, 0.08))
    wrist_exposure_delta = exposure_shared + float(rng.uniform(-0.08, 0.08))
    wrist_focal_scale = float(rng.uniform(0.98, 1.02))
    front_focal_scale = float(rng.uniform(0.98, 1.02))
    table_color_scale = float(rng.uniform(0.85, 1.15))
    object_color_scale = float(rng.uniform(0.90, 1.10))
    material_roughness_delta = float(rng.uniform(-0.08, 0.08))

    cfg.pi0_dome_light_intensity *= light_scale
    cfg.pi0_dome_light_color = _scaled_color(
        cfg.pi0_dome_light_color, 1.0, light_tint
    )
    cfg.pi0_front_camera_exposure += front_exposure_delta
    cfg.pi0_wrist_camera_exposure += wrist_exposure_delta
    cfg.pi0_wrist_camera_focal_length *= wrist_focal_scale
    if cfg.wrist_camera.spawn is not None:
        cfg.wrist_camera.spawn.focal_length *= wrist_focal_scale
    if cfg.tiled_camera.spawn is not None:
        cfg.tiled_camera.spawn.focal_length *= front_focal_scale
    cfg.pi0_green_table_overlay_color = _scaled_color(
        cfg.pi0_green_table_overlay_color, table_color_scale
    )
    cfg.pi0_green_table_overlay_roughness = float(
        np.clip(cfg.pi0_green_table_overlay_roughness + material_roughness_delta, 0.0, 1.0)
    )
    cfg.held_asset_visual_color = _scaled_color(
        cfg.held_asset_visual_color, object_color_scale
    )
    cfg.fixed_asset_visual_color = _scaled_color(
        cfg.fixed_asset_visual_color, object_color_scale
    )
    cfg.pi0_front_hole_visual_color = _scaled_color(
        cfg.pi0_front_hole_visual_color, object_color_scale
    )
    cfg.pi0_wrist_hole_visual_color = _scaled_color(
        cfg.pi0_wrist_hole_visual_color, object_color_scale
    )

    return {
        "enabled": True,
        "seed": int(seed),
        "sampling_scope": "one sample per rollout",
        "light_intensity_scale": light_scale,
        "light_red_blue_tint": light_tint,
        "front_exposure_delta_ev": front_exposure_delta,
        "wrist_exposure_delta_ev": wrist_exposure_delta,
        "front_focal_length_scale": front_focal_scale,
        "wrist_focal_length_scale": wrist_focal_scale,
        "table_color_scale": table_color_scale,
        "object_color_scale": object_color_scale,
        "material_roughness_delta": material_roughness_delta,
        "image_brightness": float(rng.uniform(0.92, 1.08)),
        "image_contrast": float(rng.uniform(0.92, 1.08)),
        "image_saturation": float(rng.uniform(0.90, 1.10)),
        "image_gamma": float(rng.uniform(0.94, 1.06)),
        "image_noise_sigma_255": float(rng.uniform(0.0, 2.5)),
        "image_blur_sigma_px": float(rng.uniform(0.0, 0.55)),
    }


def apply_rgb_domain_randomization(
    image: np.ndarray,
    params: dict,
    rng: np.random.Generator,
) -> np.ndarray:
    """Apply temporally coherent color response plus per-frame sensor noise."""

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


def write_paired_cotraining_h5(
    source_path: Path,
    output_path: Path,
    frame_count: int,
    domain_randomization: dict,
    sim_force_streams: dict[str, np.ndarray] | None = None,
    contact_force_base: np.ndarray | None = None,
    contact_wrench_base: np.ndarray | None = None,
    contact_wrench_base_raw: np.ndarray | None = None,
    contact_count: np.ndarray | None = None,
) -> None:
    """Mirror the real H5 schema and aligned labels for paired visual co-training."""

    with h5py.File(source_path, "r") as source, h5py.File(output_path, "w") as output:
        source_frames = int(source["timestamps"].shape[0])

        def copy_group(source_group, output_group) -> None:
            for key, value in source_group.attrs.items():
                output_group.attrs[key] = value
            for name, node in source_group.items():
                if isinstance(node, h5py.Group):
                    copy_group(node, output_group.create_group(name))
                    continue
                if node.ndim > 0 and node.shape[0] == source_frames:
                    data = node[:frame_count]
                else:
                    data = node[()]
                kwargs = {}
                if np.asarray(data).ndim > 0 and node.compression is not None:
                    kwargs["compression"] = node.compression
                dataset = output_group.create_dataset(
                    name, data=data, dtype=node.dtype, **kwargs
                )
                for attr_name, attr_value in node.attrs.items():
                    dataset.attrs[attr_name] = attr_value

        copy_group(source, output)
        if sim_force_streams is not None:
            def replace_dataset(path: str, values: np.ndarray) -> None:
                if path not in output:
                    raise KeyError(f"Simulation-force target dataset is missing: {path}")
                dataset = output[path]
                values = np.asarray(values, dtype=np.float32)
                if dataset.shape != values.shape:
                    raise ValueError(
                        f"Simulation-force shape mismatch for {path}: "
                        f"H5={dataset.shape}, simulated={values.shape}"
                    )
                dataset[...] = values.astype(dataset.dtype, copy=False)

            training_source = (
                "contact_equivalent_wrench_base_at_K"
                if "contact_equivalent_wrench_base_at_K" in sim_force_streams
                else (
                    "wrench_calibrated"
                    if "wrench_calibrated" in sim_force_streams
                    else "wrench_final"
                )
            )
            training_wrench = np.asarray(
                sim_force_streams[training_source], dtype=np.float32
            )

            for suffix in ("", "_base", "_stiffness"):
                replace_dataset(f"obs/state/ee_wrench{suffix}", training_wrench)
                replace_dataset(
                    f"obs/state/ee_force{suffix}", training_wrench[:, :3]
                )
                replace_dataset(
                    f"obs/state/ee_torque{suffix}", training_wrench[:, 3:]
                )

            sim_force_group = output.require_group("sim").require_group("force")
            for name, values in sim_force_streams.items():
                if name in sim_force_group:
                    del sim_force_group[name]
                sim_force_group.create_dataset(
                    name,
                    data=np.asarray(values, dtype=np.float32),
                    compression="gzip",
                )
            if contact_force_base is not None:
                sim_force_group.create_dataset(
                    "contact_total_base",
                    data=contact_force_base,
                    compression="gzip",
                )
            for name, values, width in (
                ("contact_wrench_base", contact_wrench_base, 6),
                ("contact_wrench_base_raw", contact_wrench_base_raw, 6),
                ("contact_count", contact_count, 1),
            ):
                if values is None:
                    continue
                values = np.asarray(values, dtype=np.float32)
                expected_shape = (
                    (frame_count,)
                    if width == 1
                    else (frame_count, width)
                )
                if values.shape != expected_shape:
                    raise ValueError(
                        f"{name} shape mismatch: expected {expected_shape}, "
                        f"got {values.shape}"
                    )
                sim_force_group.create_dataset(
                    name,
                    data=values,
                    compression="gzip",
                )
            sim_force_group.attrs["component_order"] = "[Fx,Fy,Fz,Tx,Ty,Tz]"
            sim_force_group.attrs["units"] = "[N,N,N,N*m,N*m,N*m]"
            sim_force_group.attrs["training_contract_source"] = training_source
            if contact_force_base is not None:
                sim_force_group.attrs["contact_total_base_role"] = (
                    "diagnostic only; never written to obs/state training fields"
                )
            if contact_wrench_base is not None:
                sim_force_group.attrs["contact_wrench_base_definition"] = (
                    "[contact_total_base, torque_about_K_base] after causal EMA"
                )
                sim_force_group.attrs["contact_wrench_base_raw_definition"] = (
                    "diagnostic substep-averaged [contact_total_base, torque_about_K_base]"
                )
            sim_force_group.attrs["wrench_final_definition"] = (
                "directed-load-calibrated child-joint wrench transformed to base/K; "
                "diagnostic under direct state replay"
            )

        output.attrs["synthetic_visual_rollout"] = True
        output.attrs["paired_real_source"] = str(source_path.resolve())
        output.attrs["domain_randomization"] = json.dumps(domain_randomization)
        output.attrs["sim_force_saved"] = sim_force_streams is not None
        if sim_force_streams is not None:
            output.attrs["sim_force_training_field"] = "obs/state/ee_wrench_base"
            output.attrs["sim_force_training_source"] = training_source
            output.attrs["sim_force_proxy_fields_synchronized"] = True


def _wrench_row(env: RealSimEnv, attribute: str) -> np.ndarray:
    """Read one finite six-dimensional simulated wrench after a physics step."""

    value = getattr(env, attribute, None)
    if value is None:
        raise RuntimeError(f"Simulation wrench attribute is unavailable: {attribute}")
    row = value[0].detach().cpu().numpy().astype(np.float32, copy=False).reshape(-1)
    if row.shape != (6,) or not np.isfinite(row).all():
        raise RuntimeError(
            f"Simulation wrench {attribute} must be finite with shape (6,), got {row.shape}"
        )
    return row.copy()


_COUNTERFACTUAL_WRENCH_ATTRIBUTES = (
    "wrench_raw",
    "wrench_child_joint_parent",
    "wrench_anchor",
    "wrench_base",
    "wrench_corrected",
    "wrench_final",
)


def _write_wrench_csv(path: Path, values: np.ndarray, prefix: str) -> None:
    values = np.asarray(values, dtype=np.float32)
    if values.ndim != 2 or values.shape[1] != 6:
        raise ValueError(f"Simulation wrench CSV must have shape (T,6), got {values.shape}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow([f"{prefix}_{index}" for index in range(6)])
        writer.writerows(values.tolist())


def _write_vector_csv(path: Path, values: np.ndarray, prefix: str, dimension: int) -> None:
    values = np.asarray(values, dtype=np.float32)
    if values.ndim != 2 or values.shape[1] != dimension:
        raise ValueError(
            f"Simulation vector CSV must have shape (T,{dimension}), got {values.shape}"
        )
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow([f"{prefix}_{index}" for index in range(dimension)])
        writer.writerows(values.tolist())


def _write_scalar_csv(path: Path, values: list[float], name: str) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow([name])
        writer.writerows([[float(value)] for value in values])


_TAVLA_ONLY_REMOVE_FILES = (
    "contact_count.csv",
    "contact_wrench_base.csv",
    "contact_wrench_base_raw.csv",
    "contact_friction_base.csv",
    "contact_friction_base_raw.csv",
    "contact_friction_world.csv",
    "contact_friction_world_raw.csv",
    "contact_normal_base.csv",
    "contact_normal_base_raw.csv",
    "contact_normal_world.csv",
    "contact_normal_world_raw.csv",
    "contact_total_base.csv",
    "contact_total_base_raw.csv",
    "contact_total_world.csv",
    "contact_total_world_raw.csv",
    "contact_torque_k_base.csv",
    "contact_torque_k_base_raw.csv",
    "contact_torque_k_world.csv",
    "contact_torque_k_world_raw.csv",
    "ee_pose_sim.csv",
    "gt_replay_validation.csv",
    "joint_pos_sim.csv",
    "sim_front_pi0_probe.mp4",
    "sim_wrist_pi0_probe.mp4",
    "sim_pi0_probe_summary.json",
    "sim_pi0_probe_steps.jsonl",
    "sim_pi0_probe_error.csv",
    "timestamps.csv",
    "wrench_anchor.csv",
    "wrench_base.csv",
    "wrench_contact.csv",
    "wrench_contact_held.csv",
    "wrench_corrected.csv",
    "wrench_final.csv",
    "wrench_model.csv",
    "wrench_model_clean.csv",
    "wrench_raw.csv",
    "wrench_tool_smooth.csv",
    "wrist_force_overlay.mp4",
    "wrist_force_overlay.mp4v.backup.mp4",
    "replay_metadata.json",
    "run.log",
)


def _cleanup_tavla_only_output(output_dir: Path) -> None:
    """Remove replay diagnostics and CSVs; preserve TAVLA handoff artifacts."""
    removed = []
    for filename in _TAVLA_ONLY_REMOVE_FILES:
        path = output_dir / filename
        if path.is_file():
            path.unlink()
            removed.append(filename)
    for path in sorted(output_dir.rglob("*.csv")):
        if path.is_file():
            path.unlink()
            removed.append(str(path.relative_to(output_dir)))
    visualization_dir = output_dir / "wrench_visualization"
    if visualization_dir.is_dir():
        shutil.rmtree(visualization_dir)
        removed.append("wrench_visualization/")
    print(
        "[TAVLAOutput] kept=data.h5, front_camera.mp4, wrist_camera.mp4; "
        f"removed={removed}",
        flush=True,
    )


def _contact_array(value) -> np.ndarray:
    """Convert IsaacLab PhysX tensor-view output to NumPy."""
    if hasattr(value, "numpy"):
        value = value.numpy()
    elif hasattr(value, "torch"):
        value = value.torch
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=np.float64)


def _active_contact_indices(counts, starts, capacity: int) -> np.ndarray:
    """Expand PhysX contact spans without assuming that every span starts at zero."""
    counts_array = _contact_array(counts).reshape(-1).astype(np.int64)
    starts_array = _contact_array(starts).reshape(-1).astype(np.int64)
    indices: list[int] = []
    for count, start in zip(counts_array, starts_array, strict=False):
        if count <= 0 or start < 0 or start >= capacity:
            continue
        indices.extend(range(int(start), min(int(start + count), capacity)))
    return np.asarray(indices, dtype=np.int64)


class PhysxPegHoleContactProbe:
    """Read direct PhysX contact forces for the independent peg and hole."""

    def __init__(self, env: RealSimEnv) -> None:
        from pxr import Usd

        def contact_report_paths(root_path: str) -> list[str]:
            root = env.sim.stage.GetPrimAtPath(root_path)
            return [
                str(prim.GetPath())
                for prim in Usd.PrimRange(root)
                if "PhysxContactReportAPI" in prim.GetAppliedSchemas()
            ]

        held_root = str(
            getattr(
                env.cfg,
                "replay_peg_contact_body_path",
                "/World/envs/env_0/HeldAsset",
            )
        )
        held_paths = contact_report_paths(held_root)
        fixed_paths = contact_report_paths("/World/envs/env_0/FixedAsset")
        if not held_paths or not fixed_paths:
            raise RuntimeError(
                "Could not find PhysxContactReportAPI bodies for peg/hole: "
                f"held={held_paths}, fixed={fixed_paths}"
            )
        physics_view = env.sim.physics_manager.get_physics_sim_view()
        if physics_view is None:
            raise RuntimeError("IsaacLab PhysX simulation view is unavailable")
        self._contact_view = physics_view.create_rigid_contact_view(
            held_paths,
            filter_patterns=[fixed_paths for _ in held_paths],
            max_contact_data_count=256,
        )
        self._debug_reported = False
        print(
            "[ContactProbe] direct PhysX peg-hole contact view ready: "
            f"held_root={held_root} held={held_paths}, fixed={fixed_paths}",
            flush=True,
        )

    def read(self, env: RealSimEnv, dt: float) -> dict[str, np.ndarray | float]:
        force_matrix = _contact_array(self._contact_view.get_contact_force_matrix(dt=dt))
        (
            normal_magnitudes,
            normal_points,
            normal_directions,
            _normal_distances,
            normal_counts,
            normal_starts,
        ) = self._contact_view.get_contact_data(dt=dt)
        friction_forces, friction_points, friction_counts, friction_starts = (
            self._contact_view.get_friction_data(dt=dt)
        )
        normal_magnitudes = _contact_array(normal_magnitudes).reshape(-1)
        normal_points = _contact_array(normal_points).reshape(-1, 3)
        normal_directions = _contact_array(normal_directions).reshape(-1, 3)
        friction_forces = _contact_array(friction_forces)
        friction_points = _contact_array(friction_points).reshape(-1, 3)
        friction_counts = _contact_array(friction_counts).reshape(-1)

        normal_entries = force_matrix[0] if force_matrix.ndim == 3 else force_matrix
        normal_entries = np.asarray(normal_entries, dtype=np.float64).reshape(-1, 3)
        normal_world = normal_entries.sum(axis=0) if len(normal_entries) else np.zeros(3)
        friction_count = int(np.maximum(friction_counts, 0).sum())
        friction_world = (
            friction_forces[:friction_count].sum(axis=0)
            if friction_count
            else np.zeros(3)
        )
        total_world = normal_world + friction_world

        # Diagnostic only: synthesize the contact moment about the Franka K
        # origin.  The existing contact_total_base training force is unchanged.
        k_position_world = (
            env.fingertip_midpoint_pos[0].detach().cpu().numpy().astype(np.float64)
        )
        normal_indices = _active_contact_indices(
            normal_counts,
            normal_starts,
            min(len(normal_magnitudes), len(normal_points), len(normal_directions)),
        )
        friction_indices = _active_contact_indices(
            friction_counts,
            friction_starts,
            min(len(friction_forces), len(friction_points)),
        )
        torque_k_world = np.zeros(3, dtype=np.float64)
        if len(normal_indices):
            point_normal_forces = (
                normal_magnitudes[normal_indices, None]
                * normal_directions[normal_indices]
            )
            if np.dot(point_normal_forces.sum(axis=0), normal_world) < 0.0:
                point_normal_forces *= -1.0
            torque_k_world += np.cross(
                normal_points[normal_indices] - k_position_world,
                point_normal_forces,
            ).sum(axis=0)
        if len(friction_indices):
            point_friction_forces = friction_forces.reshape(-1, 3)[friction_indices]
            if np.dot(point_friction_forces.sum(axis=0), friction_world) < 0.0:
                point_friction_forces *= -1.0
            torque_k_world += np.cross(
                friction_points[friction_indices] - k_position_world,
                point_friction_forces,
            ).sum(axis=0)

        root_quat_w = env._robot.data.root_quat_w[0]
        root_quat_w = root_quat_w.torch if hasattr(root_quat_w, "torch") else root_quat_w
        root_quat_w = torch.as_tensor(root_quat_w, dtype=torch.float32, device=env.device).view(1, 4)
        world_to_base = torch_utils.quat_conjugate(root_quat_w)

        def to_base(vector_world: np.ndarray) -> np.ndarray:
            vector = torch.as_tensor(vector_world, dtype=torch.float32, device=env.device).view(1, 3)
            return torch_utils.quat_apply(world_to_base, vector)[0].detach().cpu().numpy()

        contact_count = int(np.count_nonzero(np.linalg.norm(normal_entries, axis=1) > 1.0e-8))
        if not self._debug_reported:
            print(
                "[ContactProbe] first sample: "
                f"normal={normal_world.tolist()} friction={friction_world.tolist()} "
                f"total={total_world.tolist()} count={contact_count}",
                flush=True,
            )
            self._debug_reported = True
        return {
            "normal_world": normal_world.astype(np.float32),
            "friction_world": friction_world.astype(np.float32),
            "total_world": total_world.astype(np.float32),
            "normal_base": to_base(normal_world).astype(np.float32),
            "friction_base": to_base(friction_world).astype(np.float32),
            "total_base": to_base(total_world).astype(np.float32),
            "torque_k_world": torque_k_world.astype(np.float32),
            "torque_k_base": to_base(torque_k_world).astype(np.float32),
            "contact_count": float(contact_count),
        }


_CONTACT_VECTOR_KEYS = (
    "normal_world",
    "friction_world",
    "total_world",
    "normal_base",
    "friction_base",
    "total_base",
    "torque_k_world",
    "torque_k_base",
)


def _average_contact_rows(rows: list[dict[str, np.ndarray | float]]) -> dict[str, np.ndarray | float]:
    """Average substep forces into one frame-level contact signal."""
    if not rows:
        raise ValueError("Cannot average an empty contact-row list")
    averaged: dict[str, np.ndarray | float] = {}
    for key in _CONTACT_VECTOR_KEYS:
        averaged[key] = np.mean(
            np.stack([np.asarray(row[key], dtype=np.float32) for row in rows]),
            axis=0,
        )
    # A contact in any substep means the frame had contact.
    averaged["contact_count"] = float(max(float(row["contact_count"]) for row in rows))
    return averaged


def _smooth_contact_row(
    raw: dict[str, np.ndarray | float],
    previous: dict[str, np.ndarray | float] | None,
    alpha: float,
) -> dict[str, np.ndarray | float]:
    """Apply one EMA to contact vectors; preserve the binary contact count."""
    if previous is None:
        smoothed = {
            key: np.asarray(raw[key], dtype=np.float32).copy()
            for key in _CONTACT_VECTOR_KEYS
        }
    else:
        smoothed = {
            key: alpha * np.asarray(raw[key], dtype=np.float32)
            + (1.0 - alpha) * np.asarray(previous[key], dtype=np.float32)
            for key in _CONTACT_VECTOR_KEYS
        }
    smoothed["contact_count"] = float(raw["contact_count"])
    return smoothed


def _causal_ema(values: np.ndarray, alpha: float) -> np.ndarray:
    """Filter a wrench without using future frames."""
    values = np.asarray(values, dtype=np.float32)
    filtered = np.empty_like(values)
    filtered[0] = values[0]
    for index in range(1, len(values)):
        filtered[index] = alpha * values[index] + (1.0 - alpha) * filtered[index - 1]
    return filtered


class Pi0VisualJointReplayEnv(RealSimEnv):
    """Pi0 background with one independently simulated PPO left arm."""

    def __init__(self, cfg, **kwargs):
        # Keep RealSimEnv's episode/data-collection lifecycle inactive while
        # DirectRLEnv constructs the replay-only articulation.
        self.collect_data = False
        self.immediate_stop = False
        self.save_failed_trajectory = False
        self.num_trajectories = 1_000_000
        self.cur_num_traj = 0
        super().__init__(cfg, **kwargs)

    def _setup_scene(self) -> None:
        super()._setup_scene()
        # The imported visual background has an authored PhysicsScene.  The
        # active PPO Franka must belong solely to IsaacLab's PhysicsScene.
        from pxr import UsdPhysics

        nested_scenes = [
            prim.GetPath()
            for prim in self.sim.stage.Traverse()
            if prim.IsA(UsdPhysics.Scene)
            and str(prim.GetPath()).startswith("/World/envs/")
        ]
        for scene_path in nested_scenes:
            self.sim.stage.RemovePrim(scene_path)

    def _prepare_robot_usd_for_replay(self) -> None:
        """Add a fixed peg collider to the existing fingertip rigid body.

        This runs before IsaacLab constructs the articulation.  The peg is
        therefore part of the robot load path without a runtime fixed joint,
        and no per-frame collider teleport is required.
        """
        if not bool(getattr(self.cfg, "replay_peg_in_articulation", False)):
            return

        from omni.physx.scripts import physicsUtils
        from pxr import Gf, PhysxSchema, UsdGeom, UsdPhysics, UsdShade

        robot_root = str(self.cfg.robot.prim_path).replace("env_.*", "env_0")
        parent_path = f"{robot_root}/panda_fingertip_centered"
        parent = self.sim.stage.GetPrimAtPath(parent_path)
        if not parent.IsValid() or not parent.HasAPI(UsdPhysics.RigidBodyAPI):
            raise RuntimeError(f"Missing fingertip rigid body for physical replay: {parent_path}")

        peg_root_path = f"{parent_path}/ReplayPeg"
        peg_mesh_path = f"{peg_root_path}/mesh"
        if self.sim.stage.GetPrimAtPath(peg_root_path).IsValid():
            self.sim.stage.RemovePrim(peg_root_path)
        peg_root = UsdGeom.Xform.Define(self.sim.stage, peg_root_path)
        mount_depth = (
            float(self.cfg.task.held_asset_cfg.height)
            - float(self.cfg.task.robot_cfg.franka_fingerpad_length)
            + float(getattr(self.cfg, "pi0_peg_mount_depth_adjust_m", 0.0))
        )
        # Match held_peg_pose() exactly. That helper first applies the 180 deg
        # Y rotation and then translates by -mount_depth in the rotated frame,
        # which is +mount_depth on the fingertip parent's local Z axis. Using
        # -mount_depth here displaced the hidden collision peg by 2*mount_depth
        # (about 64.8 mm) from the orange visual peg.
        peg_root.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, mount_depth))
        peg_root.AddOrientOp().Set(Gf.Quatf(0.0, Gf.Vec3f(0.0, 1.0, 0.0)))
        peg_mesh = UsdGeom.Cylinder.Define(self.sim.stage, peg_mesh_path)
        peg_mesh.GetRadiusAttr().Set(0.5 * float(self.cfg.task.held_asset_cfg.diameter))
        peg_mesh.GetHeightAttr().Set(float(self.cfg.task.held_asset_cfg.height))
        peg_mesh.GetAxisAttr().Set("Z")
        peg_mesh.GetPurposeAttr().Set(UsdGeom.Tokens.proxy)
        peg_mesh.AddTranslateOp().Set(
            Gf.Vec3d(0.0, 0.0, 0.5 * float(self.cfg.task.held_asset_cfg.height))
        )
        UsdPhysics.CollisionAPI.Apply(peg_mesh.GetPrim())
        stiffness = float(self.cfg.replay_contact_compliant_stiffness_n_m)
        damping = float(self.cfg.replay_contact_compliant_damping_n_s_m)
        compliant_enabled = bool(self.cfg.replay_enable_compliant_contact)
        if compliant_enabled:
            material_path = f"{peg_root_path}/compliant_contact_material"
            UsdShade.Material.Define(self.sim.stage, material_path)
            material_prim = self.sim.stage.GetPrimAtPath(material_path)
            usd_material = UsdPhysics.MaterialAPI.Apply(material_prim)
            usd_material.CreateStaticFrictionAttr().Set(0.75)
            usd_material.CreateDynamicFrictionAttr().Set(0.75)
            usd_material.CreateRestitutionAttr().Set(0.0)
            physx_material = PhysxSchema.PhysxMaterialAPI.Apply(material_prim)
            physx_material.CreateCompliantContactStiffnessAttr().Set(stiffness)
            physx_material.CreateCompliantContactDampingAttr().Set(damping)
            physicsUtils.add_physics_material_to_prim(
                self.sim.stage, peg_mesh.GetPrim(), material_path
            )
        PhysxSchema.PhysxContactReportAPI.Apply(parent)
        self.cfg.replay_peg_contact_body_path = parent_path
        if compliant_enabled:
            print(
                "[PhysicalWristForce] integrated compliant peg collider authored under "
                f"{parent_path}: stiffness={stiffness:.3f}N/m "
                f"damping={damping:.3f}Ns/m",
                flush=True,
            )
        else:
            print(
                "[PhysicalWristForce] integrated rigid peg collider authored under "
                f"{parent_path}: local_z={mount_depth:.6f}m",
                flush=True,
            )

    def prepare_replay_state(self) -> None:
        """Commit the complete initial Franka state before first capture."""

        env_ids = torch.arange(self.num_envs, dtype=torch.long, device=self.device)
        self.scene.reset(env_ids)
        self._set_assets_to_default_pose(env_ids)
        joint_pos = self._robot.data.default_joint_pos[env_ids].clone()
        joint_pos[:, :7] = torch.as_tensor(
            self.cfg.ctrl.reset_joints, dtype=torch.float32, device=self.device
        ).view(1, 7)
        joint_vel = torch.zeros_like(joint_pos)
        self.ctrl_target_joint_pos[env_ids] = joint_pos
        self._robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)
        self._robot.set_joint_position_target(joint_pos, env_ids=env_ids)
        self._robot.set_joint_effort_target(
            torch.zeros_like(joint_pos), env_ids=env_ids
        )
        self.scene.write_data_to_sim()
        self.sim.step(render=False)
        self.sim.forward()
        self.scene.update(dt=self.physics_dt)
        self._compute_intermediate_values(dt=self.physics_dt)


def load_real_data(path: Path):
    with h5py.File(path, "r") as h5:
        q = np.asarray(h5["obs/state/joint_pos"][:], dtype=np.float32)
        ee_pose = np.asarray(h5["obs/state/ee_pose"][:], dtype=np.float32)
        timestamps = np.asarray(h5["timestamps"][:], dtype=np.float64)
        if "action/actual/gripper" in h5:
            gripper = np.asarray(h5["action/actual/gripper"][:], dtype=np.float32).reshape(-1)
        else:
            gripper = np.full(len(q), 0.15, dtype=np.float32)
    if q.ndim != 2 or q.shape[1] != 7:
        raise ValueError(f"joint_pos must be (N,7), got {q.shape}")
    if ee_pose.shape != (len(q), 7) or timestamps.shape != (len(q),):
        raise ValueError("q, ee_pose, and timestamps have different lengths")
    if gripper.shape != (len(q),):
        raise ValueError("gripper and q have different lengths")
    if not all(np.isfinite(value).all() for value in (q, ee_pose, timestamps, gripper)):
        raise ValueError("trajectory contains NaN or Inf")
    return q, np.clip(gripper, 0.0, 1.0), ee_pose, timestamps


def set_robot_q(
    env: RealSimEnv,
    q: np.ndarray,
    gripper: float,
    *,
    write_scene: bool = True,
) -> None:
    """Write the measured robot state without overwriting collision-resolved assets."""

    q_tensor = torch.as_tensor(q, dtype=torch.float32, device=env.device).view(1, 7)
    full_q = env.joint_pos.clone()
    full_q[:, :7] = q_tensor
    full_q[:, 7:9] = float(gripper) * 0.05
    zero_velocity = torch.zeros_like(full_q)
    env._robot.write_joint_state_to_sim(full_q, zero_velocity)
    env.ctrl_target_joint_pos[:] = full_q
    env._robot.set_joint_position_target(full_q)
    env._robot.set_joint_effort_target(torch.zeros_like(full_q))
    if write_scene:
        env.scene.write_data_to_sim()
    else:
        # During collision-resolved replay, writing the whole scene here would
        # re-apply the pre-contact HeldAsset root pose.  Only flush the robot
        # actuator command after the PhysX contact solve.
        env._robot.write_data_to_sim()
    # Keep the original forward-only path by default.  Collision-resolved
    # replay calls this helper after PhysX has stepped and only reasserts the
    # robot state, leaving the contact-corrected HeldAsset untouched.
    env.sim.forward()
    env.scene.update(dt=env.physics_dt)
    env._compute_intermediate_values(dt=env.physics_dt)
    if getattr(env.cfg, "replay_left_background_visual_shell", False):
        env._pose_background_robot_visual(
            env.sim.stage,
            "/World/envs/env_0/franka_env/LeftRobotVisual/Robot/franka",
            q,
            float(gripper) * 0.04,
            log=False,
        )


def held_peg_pose(env: RealSimEnv) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the Peg8mm root pose rigidly mounted in the left gripper."""

    # Isaac Lab 6 stores frame quaternions as XYZW.  The old WXYZ literal
    # [0, 0, 1, 0] is therefore a Z flip here; the required 180-degree Y flip
    # is [0, 1, 0, 0].
    flip_y_quat = torch.tensor([0.0, 1.0, 0.0, 0.0], device=env.device).view(1, 4)
    zero_pos = torch.zeros((1, 3), device=env.device)
    flipped_pos, flipped_quat = math_utils.combine_frame_transforms(
        env.fingertip_midpoint_pos,
        env.fingertip_midpoint_quat,
        zero_pos,
        flip_y_quat,
    )
    # Peg root is 50 mm above its tip; the physical finger pad already
    # occupies 17.608 mm of that distance.
    relative_pos = torch.zeros((1, 3), device=env.device)
    relative_pos[:, 2] = (
        float(env.cfg.task.held_asset_cfg.height)
        - float(env.cfg.task.robot_cfg.franka_fingerpad_length)
        + float(getattr(env.cfg, "pi0_peg_mount_depth_adjust_m", 0.0))
    )
    identity_quat = torch.tensor(
        [0.0, 0.0, 0.0, 1.0], device=env.device
    ).view(1, 4)
    held_pos, held_quat = math_utils.combine_frame_transforms(
        flipped_pos,
        flipped_quat,
        -relative_pos,
        identity_quat,
    )
    return held_quat, held_pos


def insertion_metrics(env: RealSimEnv) -> dict[str, float | bool]:
    """Measure GT replay against the environment's normal success predicate."""

    held_base_pos, _ = factory_utils.get_held_base_pose(
        env.held_pos,
        env.held_quat,
        env.cfg_task.name,
        env.cfg_task.fixed_asset_cfg,
        env.num_envs,
        env.device,
    )
    target_base_pos, _ = factory_utils.get_target_held_base_pose(
        env.fixed_pos,
        env.fixed_quat,
        env.cfg_task.name,
        env.cfg_task.fixed_asset_cfg,
        env.num_envs,
        env.device,
    )
    delta = target_base_pos[0] - held_base_pos[0]
    xy_error = torch.linalg.vector_norm(delta[:2])
    z_disp = held_base_pos[0, 2] - target_base_pos[0, 2]
    success = env._get_curr_successes(
        success_threshold=env.cfg_task.success_threshold,
        check_rot=False,
    )[0]
    return {
        "xy_error_m": float(xy_error.detach().cpu()),
        "z_disp_m": float(z_disp.detach().cpu()),
        "success": bool(success.detach().cpu()),
    }


def write_asset_pose(asset, pos: torch.Tensor, quat: torch.Tensor) -> None:
    state = asset.data.root_state_w.clone()
    state[:, :3] = pos
    state[:, 3:7] = quat
    state[:, 7:] = 0.0
    asset.write_root_pose_to_sim(state[:, :7])
    asset.write_root_velocity_to_sim(state[:, 7:])
    asset.reset()


def configure_pi0_cameras(cfg) -> None:
    """Apply the Pi0 camera configuration without creating a Pi0 policy."""

    # Match Pi0 exactly: RTX renders at 2x resolution and _resize_pi0_rgb()
    # Lanczos-downsamples to the 640x480 policy/video frame.  This avoids
    # a one-pixel aliasing seam at the physical tabletop patch boundary.
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
    # Reuse the current Pi0 wrist calibration copied by
    # configure_pi0_visual_scene(). Do not keep a second hard-coded replay pose.
    cfg.wrist_camera.offset.pos = tuple(cfg.pi0_wrist_camera_offset_pos)
    cfg.wrist_camera.offset.rot = _pi0_wrist_camera_rotation(cfg)
    cfg.wrist_camera.offset.convention = "opengl"


def configure_pi0_visual_scene(cfg) -> None:
    """Apply the same visual-only scene settings as the Pi0 evaluator."""

    # Retain the original PPO robot hierarchy and copy Pi0 visual parameters
    # only. Directly using the Pi0 task changes the left-arm bootstrap.
    pi0_visual_cfg = RealSimPi0PegInsertCfg()
    for name in dir(pi0_visual_cfg):
        if name.startswith("pi0_"):
            setattr(cfg, name, getattr(pi0_visual_cfg, name))
    cfg.override_held_asset_color = pi0_visual_cfg.override_held_asset_color
    cfg.held_asset_visual_color = pi0_visual_cfg.held_asset_visual_color
    cfg.override_fixed_asset_color = pi0_visual_cfg.override_fixed_asset_color
    cfg.fixed_asset_visual_color = pi0_visual_cfg.fixed_asset_visual_color
    # These right-arm visual parameters do not use the ``pi0_`` prefix, so
    # copy them explicitly into the actual RealSim/PPO config. Otherwise the
    # visual-only right arm falls back to RealSimEnv defaults and changes made
    # in RealSimPi0PegInsertCfg have no effect during rollout.
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

    # Keep the Isaac Lab 6-compatible Franka as the controlled articulation.
    # A separate physics-free visual shell, sourced from the same USD as the
    # right arm, follows its measured joint states during replay.
    cfg.remove_background_robot = True
    cfg.background_robot_visual_only = False
    cfg.replay_left_background_visual_shell = True
    # Keep the PPO left arm as the only articulated/controlled robot.  Add a
    # separate right-arm visual copy for the front camera only; RealSimEnv
    # strips its physics and never sends it actions.
    cfg.background_right_robot_visual_only = True
    cfg.background_fr3v2_right_robot_visual_only = False
    cfg.background_right_robot_copy_active_visual = False
    # Do not also keep the authored right-arm instance from the background
    # USD, otherwise the front camera would see two static right arms.
    cfg.remove_authored_right_robot = True
    # Use Pi0's calibrated physical RGB/depth table meshes.  They remain on
    # the static scene plane throughout the measured joint replay.
    # Keep the hole/base in RTX and match the green tabletop in front RGB.
    cfg.pi0_green_table_overlay_enabled = True
    cfg.pi0_green_table_overlay_z_offset = -0.005
    # Preserve the original sim_force visual contract.  The later enlarged
    # wrist-only overlay belongs to the continuous Pi0 evaluator and makes
    # this replay's dark-green tabletop appear bright cyan.
    cfg.pi0_green_table_wrist_overlay_enabled = False
    cfg.pi0_green_table_wrist_edge_cleanup_enabled = False
    cfg.pi0_green_table_front_composite_enabled = False
    # The raised 3-D patch creates a depth/material seam on the right side of
    # the tabletop. Replace it with a front-RGB-only soft blend so the table
    # stays continuous and temporally stable without changing scene physics.
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
    # Keep the physical green table fixed so the hole remains visible. Repair
    # only the thin saturated rim from the authored background table; keeping
    # this mask narrow avoids the wide dark strip produced by the old mask.
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
    # Appending the background USD fingertip meshes beneath a live
    # articulation changes its hierarchy after PhysX/Fabric has indexed it.
    # Keep the native PPO gripper so every left-arm link stays intact.
    cfg.pi0_black_gripper_visual = False
    # Pi0 imports the complete background USD with the Isaac Sim 6 XYZW
    # identity.  This affects the front camera and tabletop together, while
    # the live PPO left arm remains at its original /Robot/franka path and is
    # still driven exclusively by measured joint-state writes below.
    cfg.robot_base_rot = (0.0, 0.0, 0.0, 1.0)
    # The legacy Direct scene root previously supplied the X-180 flip itself.
    # With the Pi0 identity root, retain only the calibrated -30 degree
    # tabletop yaw (XYZW), otherwise the hole is inverted twice.
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


def configure_asset_collision_resolution(
    cfg,
    contact_offset_m: float | None = None,
    max_depenetration_velocity_m_s: float | None = None,
) -> None:
    """Enable peg/hole rigid-body contacts without letting the peg drift.

    Joint replay is pose-authoritative: the measured robot joints and the
    FK-derived peg pose must be restored for every recorded frame.  A free
    dynamic peg would accumulate contact impulses and fly away from the
    gripper, so the loop uses PhysX contacts as a per-frame probe and then
    reattaches the peg to the measured FK pose before rendering.
    """

    fixed_props = cfg.task.fixed_asset.spawn.rigid_props
    held_props = cfg.task.held_asset.spawn.rigid_props
    if fixed_props is None or held_props is None:
        raise RuntimeError("Peg/hole rigid-body properties are missing")
    fixed_props.rigid_body_enabled = True
    # The hole USD already contains a FixedJoint to the world.  Marking the
    # root kinematic makes both sides of that authored joint static, which
    # PhysX rejects ("cannot create a joint between static bodies").  Keep it
    # dynamic-but-world-fixed and let the authored joint hold it in place.
    fixed_props.kinematic_enabled = False
    fixed_props.disable_gravity = True
    held_props.rigid_body_enabled = True
    held_props.kinematic_enabled = False
    held_props.disable_gravity = True
    if max_depenetration_velocity_m_s is not None:
        fixed_props.max_depenetration_velocity = float(
            max_depenetration_velocity_m_s
        )
        held_props.max_depenetration_velocity = float(
            max_depenetration_velocity_m_s
        )
    if contact_offset_m is not None:
        fixed_collision = cfg.task.fixed_asset.spawn.collision_props
        held_collision = cfg.task.held_asset.spawn.collision_props
        if fixed_collision is None or held_collision is None:
            raise RuntimeError("Peg/hole collision properties are missing")
        fixed_collision.contact_offset = float(contact_offset_m)
        held_collision.contact_offset = float(contact_offset_m)
        fixed_collision.rest_offset = 0.0
        held_collision.rest_offset = 0.0


def disable_independent_held_asset_collisions(env: RealSimEnv) -> int:
    """Keep HeldAsset as a visual follower when the physical peg is on the robot."""
    from pxr import Usd, UsdPhysics

    held_root = str(env.cfg.task.held_asset.prim_path).replace("env_.*", "env_0")
    root = env.sim.stage.GetPrimAtPath(held_root)
    if not root.IsValid():
        raise RuntimeError(f"Missing HeldAsset root: {held_root}")
    disabled = 0
    for prim in Usd.PrimRange(root):
        if prim.HasAPI(UsdPhysics.CollisionAPI):
            UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Set(False)
            disabled += 1
    if disabled == 0:
        raise RuntimeError(f"No HeldAsset collision shapes found below {held_root}")
    print(
        f"[PhysicalWristForce] disabled {disabled} duplicate HeldAsset colliders; "
        "visual pose remains unchanged",
        flush=True,
    )
    return disabled


def set_integrated_peg_collision_enabled(env: RealSimEnv, enabled: bool) -> None:
    """Toggle only the replay peg collider for the no-contact counterfactual pass."""
    from pxr import UsdPhysics

    parent_path = str(getattr(env.cfg, "replay_peg_contact_body_path", ""))
    peg_path = f"{parent_path}/ReplayPeg/mesh"
    peg_prim = env.sim.stage.GetPrimAtPath(peg_path)
    if not peg_prim.IsValid() or not peg_prim.HasAPI(UsdPhysics.CollisionAPI):
        raise RuntimeError(f"Integrated replay peg collider is missing: {peg_path}")
    UsdPhysics.CollisionAPI(peg_prim).GetCollisionEnabledAttr().Set(bool(enabled))
    env.sim.forward()
    print(
        f"[CounterfactualWrench] integrated peg collision enabled={bool(enabled)}",
        flush=True,
    )


def reset_wrench_filter_state(env: RealSimEnv) -> None:
    """Clear stateful wrench filters between counterfactual and contact passes."""
    for name in (
        "force_sensor_parent_smooth",
        "wrench_tool_smooth",
        "wrench_tool_bias",
        "wrench_tool_bias_count",
        "wrench_tool_zeroed",
        "wrench_model_clean",
        "wrench_model",
        "force_sensor_smooth",
        "force_sensor_world_smooth",
    ):
        value = getattr(env, name, None)
        if isinstance(value, torch.Tensor):
            value.zero_()


def capture(
    env: RealSimEnv,
    domain_randomization: dict | None = None,
    image_rng: np.random.Generator | None = None,
):
    """Capture both supersampled cameras from one stable RTX render."""

    env.sim.render()
    env.tiled_camera.update(env.physics_dt, force_recompute=True)
    env.wrist_tiled_camera.update(env.physics_dt, force_recompute=True)
    front = env.tiled_camera.data.output["rgb"][0].detach().cpu().numpy().copy()
    wrist = (
        env.wrist_tiled_camera.data.output["rgb"][0]
        .detach()
        .cpu()
        .numpy()
        .copy()
    )

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
        and domain_randomization.get("enabled", False)
        and image_rng is not None
    ):
        front = apply_rgb_domain_randomization(front, domain_randomization, image_rng)
        wrist = apply_rgb_domain_randomization(wrist, domain_randomization, image_rng)
    return (
        cv2.cvtColor(np.ascontiguousarray(front[:, :, :3]), cv2.COLOR_RGB2BGR),
        cv2.cvtColor(np.ascontiguousarray(wrist[:, :, :3]), cv2.COLOR_RGB2BGR),
    )


def annotate_bgr_frame(frame: np.ndarray, lines: list[str]) -> np.ndarray:
    annotated = np.ascontiguousarray(frame.copy())
    for line_index, line in enumerate(lines):
        origin = (12, 28 + 24 * line_index)
        cv2.putText(
            annotated,
            line,
            origin,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.58,
            (0, 0, 0),
            3,
            cv2.LINE_AA,
        )
        cv2.putText(
            annotated,
            line,
            origin,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.58,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
    return annotated


def normalize_pi0_chunk(result: dict) -> np.ndarray:
    if "actions" not in result:
        raise KeyError("Pi0 response does not contain 'actions'")
    actions = np.asarray(result["actions"], dtype=np.float32)
    if actions.ndim == 3 and actions.shape[0] == 1:
        actions = actions[0]
    if actions.ndim != 2 or actions.shape[1] != 8:
        raise ValueError(f"Expected Pi0 actions with shape (H,8), got {actions.shape}")
    if not np.isfinite(actions).all():
        raise ValueError("Pi0 actions contain NaN or Inf")
    return actions


def pi0_action_error(predicted: np.ndarray, target: np.ndarray) -> dict[str, object]:
    error = np.asarray(predicted, dtype=np.float64) - np.asarray(target, dtype=np.float64)
    return {
        "mae_rad": float(np.mean(np.abs(error))),
        "rmse_rad": float(np.sqrt(np.mean(error**2))),
        "max_abs_rad": float(np.max(np.abs(error))),
        "per_joint_abs_rad": np.abs(error).tolist(),
    }


def load_probe_reference(path: Path | None) -> dict | None:
    if path is None:
        return None
    path = path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Real Pi0 probe summary not found: {path}")
    summary = json.loads(path.read_text(encoding="utf-8"))
    return {
        "path": str(path),
        "probed_frames": summary.get("probed_frames"),
        "selected_action_vs_gt_next": summary.get("selected_action_vs_gt_next"),
        "chunk_vs_gt_future": summary.get("chunk_vs_gt_future"),
    }


def close_pi0_client(client) -> None:
    if client is None:
        return
    close = getattr(client, "close", None)
    if callable(close):
        close()
        return
    websocket = getattr(client, "_ws", None)
    close_websocket = getattr(websocket, "close", None)
    if callable(close_websocket):
        close_websocket()


def summarize_pi0_probe(
    rows: list[dict],
    *,
    server_metadata,
    real_reference: dict | None,
    action_index: int,
) -> dict:
    selected_mae = np.asarray(
        [row["selected_vs_gt_next"]["mae_rad"] for row in rows], dtype=np.float64
    )
    selected_max = np.asarray(
        [row["selected_vs_gt_next"]["max_abs_rad"] for row in rows], dtype=np.float64
    )
    per_joint = np.asarray(
        [row["selected_vs_gt_next"]["per_joint_abs_rad"] for row in rows], dtype=np.float64
    )
    horizon_rows = [row for row in rows if row["chunk_vs_gt_future"] is not None]
    selected_summary = {
        "mean_mae_rad": float(selected_mae.mean()),
        "median_mae_rad": float(np.median(selected_mae)),
        "max_mae_rad": float(selected_mae.max()),
        "mean_max_abs_rad": float(selected_max.mean()),
        "per_joint_mean_abs_rad": per_joint.mean(axis=0).tolist(),
    }
    chunk_summary = {
        "probed_frames_with_future": len(horizon_rows),
        "mean_mae_rad": float(
            np.mean([row["chunk_vs_gt_future"]["mae_rad"] for row in horizon_rows])
        )
        if horizon_rows
        else None,
    }
    comparison = None
    if real_reference is not None:
        real_selected = real_reference.get("selected_action_vs_gt_next") or {}
        real_chunk = real_reference.get("chunk_vs_gt_future") or {}
        real_mae = real_selected.get("mean_mae_rad")
        real_chunk_mae = real_chunk.get("mean_mae_rad")
        comparison = {
            "real_probe": real_reference,
            "sim_minus_real_selected_mean_mae_rad": (
                float(selected_summary["mean_mae_rad"] - real_mae)
                if real_mae is not None
                else None
            ),
            "sim_minus_real_chunk_mean_mae_rad": (
                float(chunk_summary["mean_mae_rad"] - real_chunk_mae)
                if real_chunk_mae is not None and chunk_summary["mean_mae_rad"] is not None
                else None
            ),
        }
    return {
        "server_metadata": server_metadata,
        "probed_frames": len(rows),
        "action_index": action_index,
        "selected_action_vs_gt_next": selected_summary,
        "chunk_vs_gt_future": chunk_summary,
        "real_vs_sim_comparison": comparison,
        "note": (
            "The robot is held on the recorded GT joint trajectory. Only the rendered "
            "sim images are sent to Pi0; returned actions are never executed."
        ),
    }


def set_dynamic_green_table_hole_cutout(env: RealSimEnv, hole_pos: torch.Tensor) -> None:
    """Leave the current fixed-hole footprint uncovered by the green table."""

    from pxr import Gf, Usd, UsdGeom

    camera_prim = env.sim.stage.GetPrimAtPath(
        "/World/envs/env_0/franka_env/front_camera"
    )
    if not camera_prim.IsValid():
        raise RuntimeError("Cannot project hole cutout: front_camera prim is missing")
    camera = UsdGeom.Camera(camera_prim)
    width = float(env.cfg.pi0_visual_annotation_width)
    height = float(env.cfg.pi0_visual_annotation_height)
    focal_length = float(camera.GetFocalLengthAttr().Get())
    horizontal_aperture = float(camera.GetHorizontalApertureAttr().Get())
    vertical_aperture = float(camera.GetVerticalApertureAttr().Get())
    camera_to_world = UsdGeom.Xformable(camera_prim).ComputeLocalToWorldTransform(
        Usd.TimeCode.Default()
    )
    hole_world = Gf.Vec3d(*hole_pos[0].detach().cpu().tolist())
    hole_camera = camera_to_world.GetInverse().Transform(hole_world)
    depth = -float(hole_camera[2])
    if depth <= 1.0e-6:
        raise RuntimeError("Cannot project hole cutout: hole is behind front camera")
    fx = width * focal_length / horizontal_aperture
    fy = height * focal_length / vertical_aperture
    center_x = 0.5 * width + fx * float(hole_camera[0]) / depth
    center_y = 0.5 * height - fy * float(hole_camera[1]) / depth
    # Same 56x50 pixel clearance used by Pi0's calibrated traj_0 cutout.
    half_width_px, half_height_px = 28.0, 25.0
    env.cfg.pi0_green_table_hole_cutout_xyxy = (
        center_x - half_width_px,
        center_y - half_height_px,
        center_x + half_width_px,
        center_y + half_height_px,
    )


def main() -> None:
    if args.data_dir is not None:
        batch_data_dir = args.data_dir
        batch_output_root = args.output_root
        batch_start = int(args.trajectory_start)
        batch_count = int(args.trajectory_count)
        batch_indices = range(batch_start, batch_start + batch_count)
        batch_paths = [
            (index, batch_data_dir / f"traj_{index}" / "data.h5")
            for index in batch_indices
        ]
        missing = [str(path) for _, path in batch_paths if not path.is_file()]
        if missing:
            raise FileNotFoundError(
                "Missing batch trajectory H5 files: " + ", ".join(missing)
            )

        # Keep the Kit/Isaac Sim application alive and reuse this process for
        # all trajectories. Each recursive single-trajectory pass creates and
        # resets one RealSim scene, then writes its own traj_<i> directory.
        args.data_dir = None
        for index, h5_path in batch_paths:
            args.h5 = h5_path
            args.output_dir = batch_output_root / f"traj_{index}"
            print(
                f"[BatchReplay] traj_{index}: h5={h5_path} "
                f"output={args.output_dir}",
                flush=True,
            )
            main()
        print(
            f"[BatchReplay] completed {batch_start}.."
            f"{batch_start + batch_count - 1} in one Isaac Sim process",
            flush=True,
        )
        return

    q, gripper, real_ee_pose, timestamps = load_real_data(args.h5)
    if args.gripper_constant is not None:
        # Render every rollout with one grasp pose so the fingers look identical
        # across trajectories (some recordings stored gripper~0 -> closed fingers).
        gripper = np.full(len(gripper), float(args.gripper_constant), dtype=gripper.dtype)
    if args.max_frames is not None:
        if args.max_frames <= 0:
            raise ValueError("--max-frames must be greater than zero")
        frame_count = min(args.max_frames, len(q))
        q = q[:frame_count]
        gripper = gripper[:frame_count]
        real_ee_pose = real_ee_pose[:frame_count]
        timestamps = timestamps[:frame_count]
    cfg = parse_env_cfg(args.task, device=args.device, num_envs=1, use_fabric=True)
    cfg.scene.num_envs = 1
    cfg.policy_cfg = None
    cfg.teacher_policy_cfg = None
    cfg.enable_cameras = True
    cfg.data_collect_cfg["collect_data"] = False
    cfg.data_collect_cfg["immediate_stop"] = False
    cfg.data_collect_cfg["save_failed_trajectory"] = False
    cfg.data_collect_cfg["num_trajectories"] = 1_000_000
    cfg.teacher_eval_only = True
    cfg.episode_length_s = max(30.0, float(timestamps[-1] - timestamps[0] + 2.0))
    cfg.ctrl.reset_joints = q[0].tolist()
    # Use the Pi0 evaluator's visual scene without constructing its policy
    # client. Joint replay and control remain owned by this script.
    configure_pi0_visual_scene(cfg)
    cfg.replay_peg_in_articulation = bool(args.physical_wrist_force)
    cfg.replay_contact_compliant_stiffness_n_m = float(
        args.contact_compliant_stiffness_n_m
    )
    cfg.replay_contact_compliant_damping_n_s_m = float(
        args.contact_compliant_damping_n_s_m
    )
    cfg.replay_enable_compliant_contact = bool(args.enable_compliant_contact)
    if args.enable_compliant_contact:
        # The startup material randomizer assigns one rigid material to every
        # robot collision shape, including ReplayPeg, after the USD hook above.
        # Disable only that replay-time event so the peg's compliant material
        # remains bound. The held and fixed asset material events are unchanged.
        cfg.events.robot_physics_material = None
        print(
            "[PhysicalWristForce] disabled startup robot material randomization "
            "to preserve ReplayPeg compliant contact",
            flush=True,
        )
    if args.physical_wrist_force:
        args.incoming_wrench_body = "force_sensor"
        cfg.ft_parent_body_name = "panda_link7"
        cfg.ft_raw_wrench_frame = "parent_body"
        cfg.ft_raw_torque_reference = "joint_anchor"
        cfg.ft_apply_child_joint_frame_calibration = True
        print(
            "[PhysicalWristForce] enabled: direct joint replay + integrated peg "
            "+ calibrated force_sensor incoming wrench",
            flush=True,
        )
    if args.resolve_asset_collisions:
        configure_asset_collision_resolution(
            cfg,
            args.asset_contact_offset_m,
            args.asset_max_depenetration_velocity_m_s,
        )
        print(
            "[CollisionReplay] PhysX peg/hole contact resolution enabled "
            f"({args.collision_substeps} substeps/frame, "
            f"contact_offset_m={cfg.task.held_asset.spawn.collision_props.contact_offset}, "
            "max_depenetration_velocity_m_s="
            f"{cfg.task.held_asset.spawn.rigid_props.max_depenetration_velocity}); "
            "measured robot joints remain authoritative.",
            flush=True,
        )
    domain_randomization = {"enabled": False}
    if args.domain_randomization:
        domain_randomization = sample_visual_domain_randomization(
            cfg, args.h5, args.dr_seed
        )
        print(
            "[DomainRandomization] "
            + json.dumps(domain_randomization, sort_keys=True),
            flush=True,
        )
    image_rng = np.random.default_rng(
        int(domain_randomization.get("seed", 0)) ^ 0x51AACC
    )
    real_probe_reference = load_probe_reference(args.real_probe_summary)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    env_class = Pi0VisualJointReplayEnv if args.physical_wrist_force else RealSimEnv
    env = env_class(cfg, render_mode="rgb_array", output_dir=str(args.output_dir))
    if args.incoming_wrench_body is not None:
        body_names = list(env._robot.body_names)
        if args.incoming_wrench_body not in body_names:
            print(
                "[WrenchProbe] requested body is unavailable; keeping the default "
                f"force_sensor. requested={args.incoming_wrench_body!r} "
                f"available={body_names}",
                flush=True,
            )
        else:
            env._incoming_wrench_body_idx = body_names.index(args.incoming_wrench_body)
            print(
                "[WrenchProbe] incoming joint wrench body override: "
                f"{args.incoming_wrench_body!r} "
                f"(index={env._incoming_wrench_body_idx})",
                flush=True,
            )
    pi0_client = None
    try:
        print("[JointReplay] resetting RealSim/PPO scene", flush=True)
        env.reset()
        if args.physical_wrist_force:
            disable_independent_held_asset_collisions(env)
        # Do not rebind materials on the live PPO articulation after Fabric
        # has created its GPU buffers.  It causes Isaac Sim 6 CUDA readback
        # failures on the first camera capture.  Visual-only table/asset
        # matching has already been authored before scene initialization.

        inferred_gt_final_hole_pos = None
        if args.hole_reference in ("gt-final-xy-fixed-z", "gt-final"):
            # Infer the in-plane hole location from the final measured robot
            # pose.  Real holes stay on one physical table plane, so the
            # default paired-data path must not copy final-frame Z overshoot.
            set_robot_q(env, q[-1], gripper[-1])
            _, inferred_gt_final_hole_pos = held_peg_pose(env)
            hole_pos = inferred_gt_final_hole_pos.clone()
            if args.hole_reference == "gt-final-xy-fixed-z":
                hole_pos[:, 2] = float(cfg.pi0_hole_init_pos[2])
                print(
                    "[HolePlacement] Inferred XY from final peg FK and kept "
                    f"calibrated Z={float(cfg.pi0_hole_init_pos[2]):.9f} m; "
                    "final-frame Z overshoot is not applied to the hole",
                    flush=True,
                )
            hole_quat = torch.as_tensor(
                cfg.pi0_hole_init_rot, dtype=torch.float32, device=env.device
            ).view(1, 4)
            write_asset_pose(env._fixed_asset, hole_pos, hole_quat)
            cfg.task.fixed_asset.init_state.pos = tuple(
                float(value) for value in hole_pos[0].detach().cpu().tolist()
            )
            env._create_pi0_white_table_overlay()
        else:
            # Validation mode: do not infer or move the target from GT. Keep
            # exactly the fixed-hole pose used by the current Pi0 environment.
            # RealSim's legacy reset rewrites the quaternion, so mirror
            # Pi0RealSimEnv.randomize_initial_state() after reset instead of
            # trusting the resulting RealSim root state.
            env_ids = torch.arange(env.num_envs, dtype=torch.long, device=env.device)
            fixed_pose = env._fixed_asset.data.default_root_pose.torch.clone()[env_ids]
            fixed_vel = env._fixed_asset.data.default_root_vel.torch.clone()[env_ids]
            fixed_pose[:, :3] += env.scene.env_origins[env_ids]
            fixed_pose[:, 3:7] = torch.as_tensor(
                cfg.pi0_hole_init_rot,
                dtype=torch.float32,
                device=env.device,
            ).view(1, 4)
            fixed_vel.zero_()
            env._fixed_asset.write_root_pose_to_sim_index(
                root_pose=fixed_pose,
                env_ids=env_ids,
            )
            env._fixed_asset.write_root_velocity_to_sim_index(
                root_velocity=fixed_vel,
                env_ids=env_ids,
            )
            env._fixed_asset.reset()
            env.scene.write_data_to_sim()
            env.sim.forward()
            env.scene.update(dt=env.physics_dt)
            env._compute_intermediate_values(dt=env.physics_dt)
            hole_pos = fixed_pose[:, :3].clone()
            hole_quat = fixed_pose[:, 3:7].clone()
            print(
                "[GTValidation] Keeping configured fixed-hole pose: "
                f"pos={hole_pos[0].detach().cpu().tolist()} "
                f"quat={hole_quat[0].detach().cpu().tolist()}",
                flush=True,
            )

        counterfactual_wrench_streams = None
        if args.physical_wrist_force and args.counterfactual_wrench_subtraction:
            # Direct state writes create large pose-dependent articulation
            # reactions even in free space. Replay the identical joint writes
            # once with only the integrated peg collider disabled, then
            # subtract this no-contact reaction frame by frame from the loaded
            # pass. This preserves the measured trajectory and isolates the
            # peg/hole contribution without using the contact-point proxy.
            counterfactual_wrench_streams = {
                name: [] for name in _COUNTERFACTUAL_WRENCH_ATTRIBUTES
            }
            set_integrated_peg_collision_enabled(env, False)
            print(
                "[CounterfactualWrench] collecting no-contact direct-write baseline",
                flush=True,
            )
            for frame_index in range(len(q)):
                set_robot_q(env, q[frame_index], gripper[frame_index])
                frame_rows = []
                for _ in range(args.collision_substeps):
                    env.sim.step(render=False)
                    env.scene.update(dt=env.physics_dt)
                    env._compute_intermediate_values(dt=env.physics_dt)
                    frame_rows.append(
                        {
                            name: _wrench_row(env, name)
                            for name in _COUNTERFACTUAL_WRENCH_ATTRIBUTES
                        }
                    )
                for name in _COUNTERFACTUAL_WRENCH_ATTRIBUTES:
                    counterfactual_wrench_streams[name].append(
                        np.mean(
                            [row[name] for row in frame_rows],
                            axis=0,
                            dtype=np.float64,
                        ).astype(np.float32)
                    )
                set_robot_q(
                    env,
                    q[frame_index],
                    gripper[frame_index],
                    write_scene=False,
                )
                if frame_index == 0 or (frame_index + 1) % 50 == 0 or frame_index + 1 == len(q):
                    print(
                        f"[CounterfactualWrench] frame {frame_index + 1}/{len(q)}",
                        flush=True,
                    )
            set_robot_q(env, q[0], gripper[0])
            reset_wrench_filter_state(env)
            set_integrated_peg_collision_enabled(env, True)
            print(
                "[CounterfactualWrench] baseline complete; starting contact pass",
                flush=True,
            )

        pi0_probe_rows = []
        pi0_probe_front_frames = []
        pi0_probe_wrist_frames = []
        pi0_probe_metadata = None
        if args.probe_pi0:
            pi0_client = WebsocketClientPolicy(host=args.pi0_host, port=args.pi0_port)
            pi0_probe_metadata = pi0_client.get_server_metadata()
            pi0_client.reset()
            print(
                "[Pi0SimProbe] querying rendered sim observations with GT state: "
                f"host={args.pi0_host} port={args.pi0_port} action_index={args.pi0_action_index}",
                flush=True,
            )

        contact_probe = None
        if args.record_contact_pair:
            contact_probe = PhysxPegHoleContactProbe(env)

        front_frames, wrist_frames = [], []
        sim_ee = []
        replay_q = []
        replay_timestamps = []
        validation_rows = []
        sim_force_streams = (
            {
                "wrench_raw": [],
                "wrench_child_joint_parent": [],
                "wrench_anchor": [],
                "wrench_base": [],
                "wrench_corrected": [],
                "wrench_final": [],
                "wrench_tool_smooth": [],
                "wrench_model_clean": [],
                "wrench_model": [],
            }
            if args.save_sim_force
            else None
        )
        contact_pair_streams = (
            {
                "normal_world": [],
                "friction_world": [],
                "total_world": [],
                "normal_base": [],
                "friction_base": [],
                "total_base": [],
                "torque_k_world": [],
                "torque_k_base": [],
                "contact_count": [],
            }
            if contact_probe is not None
            else None
        )
        contact_pair_raw_streams = (
            {
                "normal_world": [],
                "friction_world": [],
                "total_world": [],
                "normal_base": [],
                "friction_base": [],
                "total_base": [],
                "torque_k_world": [],
                "torque_k_base": [],
                "contact_count": [],
            }
            if contact_probe is not None
            else None
        )
        previous_contact_row = None
        for index in range(len(q)):
            set_robot_q(env, q[index], gripper[index])
            peg_quat, peg_pos = held_peg_pose(env)
            write_asset_pose(env._held_asset, peg_pos, peg_quat)
            env.scene.write_data_to_sim()
            substep_contact_rows = []
            substep_wrench_rows = []
            if args.resolve_asset_collisions:
                # The peg is dynamic with gravity disabled.  Let PhysX build
                # contact impulses for this frame, then restore the measured
                # FK pose so those impulses cannot accumulate into visible
                # peg drift across the rollout.
                for _ in range(args.collision_substeps):
                    env.sim.step(render=False)
                    env.scene.update(dt=env.physics_dt)
                    env._compute_intermediate_values(dt=env.physics_dt)
                    if args.physical_wrist_force and sim_force_streams is not None:
                        substep_wrench_rows.append(
                            {
                                attribute: _wrench_row(env, attribute)
                                for attribute in sim_force_streams
                            }
                        )
                    if contact_probe is not None:
                        substep_contact_rows.append(
                            contact_probe.read(env, dt=float(env.physics_dt))
                        )
                write_asset_pose(env._held_asset, peg_pos, peg_quat)
                env._held_asset.write_data_to_sim()
                set_robot_q(env, q[index], gripper[index], write_scene=False)
            else:
                env.sim.forward()
                env.scene.update(dt=env.physics_dt)
                if contact_probe is not None:
                    substep_contact_rows.append(
                        contact_probe.read(env, dt=float(env.physics_dt))
                    )
            env._compute_intermediate_values(dt=env.physics_dt)
            if sim_force_streams is not None:
                if args.physical_wrist_force and substep_wrench_rows:
                    for attribute in sim_force_streams:
                        frame_wrench = np.mean(
                            [row[attribute] for row in substep_wrench_rows],
                            axis=0,
                            dtype=np.float64,
                        ).astype(np.float32)
                        sim_force_streams[attribute].append(frame_wrench)
                else:
                    for attribute in sim_force_streams:
                        sim_force_streams[attribute].append(_wrench_row(env, attribute))
            if contact_pair_streams is not None and substep_contact_rows:
                raw_contact_row = _average_contact_rows(substep_contact_rows)
                contact_row = _smooth_contact_row(
                    raw_contact_row,
                    previous_contact_row,
                    float(args.contact_force_smoothing_alpha),
                )
                previous_contact_row = contact_row
                for key, value in raw_contact_row.items():
                    contact_pair_raw_streams[key].append(value)
                for key, value in contact_row.items():
                    contact_pair_streams[key].append(value)
            metrics = insertion_metrics(env)
            validation_rows.append(
                {
                    "frame": index,
                    "timestamp_s": float(timestamps[index] - timestamps[0]),
                    **metrics,
                }
            )
            front, wrist = capture(env, domain_randomization, image_rng)
            front_frames.append(front)
            wrist_frames.append(wrist)
            if pi0_client is not None:
                sim_front_rgb = cv2.cvtColor(front, cv2.COLOR_BGR2RGB)
                sim_wrist_rgb = cv2.cvtColor(wrist, cv2.COLOR_BGR2RGB)
                probe_observation = {
                    "images": {
                        "cam_high": sim_front_rgb,
                        "cam_left_wrist": sim_wrist_rgb,
                        "cam_right_wrist": sim_wrist_rgb.copy(),
                    },
                    "state": np.concatenate((q[index], [gripper[index]])).astype(np.float32),
                    "prompt": "peg-in-hole",
                }
                probe_chunk = normalize_pi0_chunk(pi0_client.infer(probe_observation))
                if args.pi0_action_index >= len(probe_chunk):
                    raise ValueError(
                        f"--pi0-action-index {args.pi0_action_index} is outside action chunk length "
                        f"{len(probe_chunk)}"
                    )
                next_index = min(index + 1, len(q) - 1)
                future_count = min(
                    len(probe_chunk) - args.pi0_action_index,
                    len(q) - (index + 1),
                )
                future_gt = q[index + 1 : index + 1 + future_count]
                aligned_chunk = probe_chunk[
                    args.pi0_action_index : args.pi0_action_index + future_count,
                    :7,
                ]
                selected_error = pi0_action_error(
                    probe_chunk[args.pi0_action_index, :7], q[next_index]
                )
                probe_row = {
                    "frame": index,
                    "action_index": args.pi0_action_index,
                    "selected_vs_gt_next": selected_error,
                    "chunk_vs_gt_future": (
                        pi0_action_error(aligned_chunk.reshape(-1), future_gt.reshape(-1))
                        if future_count
                        else None
                    ),
                    "chunk_length": int(len(probe_chunk)),
                    "predicted_action": probe_chunk[args.pi0_action_index].tolist(),
                    "gt_next_q": q[next_index].tolist(),
                }
                if future_count:
                    probe_row["chunk_vs_gt_future_per_joint_mae_rad"] = np.mean(
                        np.abs(aligned_chunk - future_gt), axis=0
                    ).tolist()
                pi0_probe_rows.append(probe_row)
                probe_lines = [
                    f"SIM GT frame: {index:03d}",
                    f"selected action: [{args.pi0_action_index}]",
                    f"sim image vs GT next MAE: {1000.0 * selected_error['mae_rad']:.2f} mrad",
                    f"sim image vs GT next max: {1000.0 * selected_error['max_abs_rad']:.2f} mrad",
                ]
                if probe_row["chunk_vs_gt_future"] is not None:
                    probe_chunk_error = probe_row["chunk_vs_gt_future"]
                    probe_lines.append(
                        f"aligned chunk MAE: {1000.0 * probe_chunk_error['mae_rad']:.2f} mrad"
                    )
                pi0_probe_front_frames.append(annotate_bgr_frame(front, probe_lines))
                pi0_probe_wrist_frames.append(annotate_bgr_frame(wrist, probe_lines))
            sim_ee.append(
                np.concatenate((
                    env.fingertip_midpoint_pos[0].detach().cpu().numpy(),
                    env.fingertip_midpoint_quat[0].detach().cpu().numpy(),
                ))
            )
            replay_q.append(env.joint_pos[0, :7].detach().cpu().numpy())
            replay_timestamps.append(float(timestamps[index] - timestamps[0]))
            if index == 0 or index + 1 == len(q) or (index + 1) % 25 == 0:
                print(f"[JointReplay] frame {index + 1}/{len(q)}", flush=True)

        _write_h264_mp4(args.output_dir / "front_camera.mp4", front_frames, args.fps)
        _write_h264_mp4(args.output_dir / "wrist_camera.mp4", wrist_frames, args.fps)
        if pi0_client is not None:
            _write_h264_mp4(
                args.output_dir / "sim_front_pi0_probe.mp4",
                pi0_probe_front_frames,
                args.fps,
            )
            _write_h264_mp4(
                args.output_dir / "sim_wrist_pi0_probe.mp4",
                pi0_probe_wrist_frames,
                args.fps,
            )
            pi0_probe_summary = summarize_pi0_probe(
                pi0_probe_rows,
                server_metadata=pi0_probe_metadata,
                real_reference=real_probe_reference,
                action_index=args.pi0_action_index,
            )
            (args.output_dir / "sim_pi0_probe_summary.json").write_text(
                json.dumps(pi0_probe_summary, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            (args.output_dir / "sim_pi0_probe_steps.jsonl").write_text(
                "".join(
                    json.dumps(row, ensure_ascii=False) + "\n"
                    for row in pi0_probe_rows
                ),
                encoding="utf-8",
            )
            with (args.output_dir / "sim_pi0_probe_error.csv").open(
                "w", newline="", encoding="utf-8"
            ) as handle:
                writer = csv.writer(handle)
                writer.writerow(
                    [
                        "frame",
                        "selected_mae_rad",
                        "selected_max_abs_rad",
                        *[f"joint_{index}_abs_error_rad" for index in range(7)],
                    ]
                )
                for row in pi0_probe_rows:
                    error = row["selected_vs_gt_next"]
                    writer.writerow(
                        [
                            row["frame"],
                            error["mae_rad"],
                            error["max_abs_rad"],
                            *error["per_joint_abs_rad"],
                        ]
                    )
        sim_force_arrays = None
        if sim_force_streams is not None:
            sim_force_arrays = {
                name: np.asarray(values, dtype=np.float32).reshape(len(q), 6)
                for name, values in sim_force_streams.items()
            }
            wrench_for_calibration = sim_force_arrays["wrench_final"]
            if counterfactual_wrench_streams is not None:
                counterfactual_final = np.asarray(
                    counterfactual_wrench_streams["wrench_final"],
                    dtype=np.float32,
                ).reshape(len(q), 6)
                sim_force_arrays["wrench_counterfactual_no_contact"] = counterfactual_final
                wrench_for_calibration = (
                    sim_force_arrays["wrench_final"] - counterfactual_final
                )
                sim_force_arrays["wrench_contact_isolated_unfiltered"] = (
                    wrench_for_calibration
                )
            baseline_frames = min(
                len(q),
                max(1, int(round(float(args.wrench_baseline_seconds) * float(args.fps)))),
            )
            baseline = np.median(
                wrench_for_calibration[:baseline_frames], axis=0
            ).astype(np.float32)
            calibrated_unclipped = wrench_for_calibration - baseline[None, :]
            calibrated = _causal_ema(calibrated_unclipped, float(args.wrench_ema_alpha))
            sim_force_arrays["wrench_calibrated_unclipped"] = calibrated_unclipped
            sim_force_arrays["wrench_calibrated"] = calibrated
            sim_force_arrays["O_F_ext_hat_K"] = calibrated
            sim_force_arrays["incoming_joint_wrench_base_at_K"] = wrench_for_calibration
            calibrated_force_norm = np.linalg.norm(calibrated[:, :3], axis=1)
            nonzero_wrist_frames = int(np.count_nonzero(calibrated_force_norm > 1.0e-6))
            print(
                "[PhysicalWristForce] calibrated incoming wrench: "
                f"baseline_frames={baseline_frames} "
                f"nonzero={nonzero_wrist_frames}/{len(q)} "
                f"force_p95={np.percentile(calibrated_force_norm, 95):.6f}N",
                flush=True,
            )
            if (
                args.training_wrench_source == "incoming-joint"
                and args.physical_wrist_force
                and nonzero_wrist_frames == 0
            ):
                raise RuntimeError(
                    "Physical wrist-force replay produced an all-zero incoming wrench"
                )

        contact_force_base = None
        contact_wrench_base = None
        contact_wrench_base_raw = None
        contact_count_array = None
        contact_force_stats = None
        if contact_pair_streams is not None:
            contact_force_base = np.asarray(
                contact_pair_streams["total_base"], dtype=np.float32
            ).reshape(len(q), 3)
            if not np.isfinite(contact_force_base).all():
                raise FloatingPointError("Peg-hole contact force contains NaN or Inf")
            contact_force_norm = np.linalg.norm(contact_force_base, axis=1)
            nonzero_contact_frames = int(np.count_nonzero(contact_force_norm > 1.0e-6))
            contact_force_stats = {
                "nonzero_frames": nonzero_contact_frames,
                "force_norm_max_n": float(np.max(contact_force_norm)),
                "force_norm_std_n": float(np.std(contact_force_norm)),
            }
            print(
                "[TAVLAForce] source=contact_total_base "
                f"nonzero_frames={nonzero_contact_frames}/{len(q)} "
                f"max_norm_n={contact_force_stats['force_norm_max_n']:.6f} "
                f"std_norm_n={contact_force_stats['force_norm_std_n']:.6f}",
                flush=True,
            )
            if sim_force_arrays is not None and nonzero_contact_frames == 0:
                raise RuntimeError(
                    "Refusing to write an all-zero TAVLA training force: "
                    "the peg-hole contact probe reported no contact frames"
                )
            for key in _CONTACT_VECTOR_KEYS:
                _write_vector_csv(
                    args.output_dir / f"contact_{key}.csv",
                    np.asarray(contact_pair_streams[key], dtype=np.float32).reshape(len(q), 3),
                    key,
                    3,
                )
                _write_vector_csv(
                    args.output_dir / f"contact_{key}_raw.csv",
                    np.asarray(contact_pair_raw_streams[key], dtype=np.float32).reshape(len(q), 3),
                    f"{key}_raw",
                    3,
                )
            contact_wrench_base = np.concatenate(
                (
                    contact_force_base,
                    np.asarray(
                        contact_pair_streams["torque_k_base"], dtype=np.float32
                    ).reshape(len(q), 3),
                ),
                axis=1,
            )
            contact_wrench_base_raw = np.concatenate(
                (
                    np.asarray(
                        contact_pair_raw_streams["total_base"], dtype=np.float32
                    ).reshape(len(q), 3),
                    np.asarray(
                        contact_pair_raw_streams["torque_k_base"], dtype=np.float32
                    ).reshape(len(q), 3),
                ),
                axis=1,
            )
            contact_count_array = np.asarray(
                contact_pair_streams["contact_count"], dtype=np.float32
            ).reshape(len(q))
            if sim_force_arrays is not None:
                sim_force_arrays["contact_wrench_base_at_K"] = contact_wrench_base
                if args.training_wrench_source == "contact-equivalent":
                    # Direct joint-state replay is intentionally kinematic. Its
                    # incoming articulation wrench contains state-write impulses,
                    # so retain that stream only for diagnostics and use the
                    # external peg/hole contact wrench at K for training.
                    sim_force_arrays["incoming_joint_wrench_calibrated"] = (
                        sim_force_arrays["wrench_calibrated"].copy()
                    )
                    sim_force_arrays[
                        "contact_equivalent_wrench_base_at_K"
                    ] = contact_wrench_base.copy()
                    sim_force_arrays["wrench_calibrated_unclipped"] = (
                        contact_wrench_base_raw.copy()
                    )
                    sim_force_arrays["wrench_calibrated"] = contact_wrench_base.copy()
                    sim_force_arrays["O_F_ext_hat_K"] = contact_wrench_base.copy()
                    print(
                        "[TrainingForce] source=contact-equivalent wrench at K; "
                        "incoming joint wrench retained as diagnostic only",
                        flush=True,
                    )
            _write_wrench_csv(
                args.output_dir / "contact_wrench_base.csv",
                contact_wrench_base,
                "contact_wrench_base",
            )
            _write_wrench_csv(
                args.output_dir / "contact_wrench_base_raw.csv",
                contact_wrench_base_raw,
                "contact_wrench_base_raw",
            )
            _write_scalar_csv(
                args.output_dir / "contact_count.csv",
                contact_pair_streams["contact_count"],
                "contact_count",
            )
            if (
                args.training_wrench_source == "incoming-joint"
                and args.physical_wrist_force
                and sim_force_arrays is not None
            ):
                contact_mask = contact_count_array > 0.0
                wrist_force_norm = np.linalg.norm(
                    sim_force_arrays["wrench_calibrated"][:, :3], axis=1
                )
                if np.any(contact_mask) and not np.any(wrist_force_norm[contact_mask] > 1.0e-6):
                    raise RuntimeError(
                        "Peg-hole contact occurred but calibrated force_sensor incoming "
                        "wrench stayed zero"
                    )

        write_paired_cotraining_h5(
            args.h5,
            args.output_dir / "data.h5",
            len(q),
            domain_randomization,
            sim_force_arrays,
            contact_force_base=contact_force_base,
            contact_wrench_base=contact_wrench_base,
            contact_wrench_base_raw=contact_wrench_base_raw,
            contact_count=contact_count_array,
        )
        if contact_wrench_base is not None:
            contact_role = (
                "training source"
                if args.training_wrench_source == "contact-equivalent"
                else "diagnostic only"
            )
            print(
                "[H5Force] saved sim/force/contact_wrench_base, "
                "sim/force/contact_wrench_base_raw, sim/force/contact_count; "
                f"contact wrench role={contact_role}",
                flush=True,
            )
        if sim_force_arrays is not None:
            for name, values in sim_force_arrays.items():
                _write_wrench_csv(args.output_dir / f"{name}.csv", values, name)
        with (args.output_dir / "joint_pos_sim.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow([f"q{i}" for i in range(7)])
            writer.writerows(np.asarray(replay_q).tolist())
        with (args.output_dir / "ee_pose_sim.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["x", "y", "z", "qw", "qx", "qy", "qz"])
            writer.writerows(np.asarray(sim_ee).tolist())
        with (args.output_dir / "timestamps.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["timestamp"])
            writer.writerows([[value] for value in replay_timestamps])
        with (args.output_dir / "gt_replay_validation.csv").open(
            "w", newline="", encoding="utf-8"
        ) as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=("frame", "timestamp_s", "xy_error_m", "z_disp_m", "success"),
            )
            writer.writeheader()
            writer.writerows(validation_rows)

        actual_q = np.asarray(replay_q)
        successful_frames = [row["frame"] for row in validation_rows if row["success"]]
        best_xy_row = min(validation_rows, key=lambda row: row["xy_error_m"])
        lowest_z_row = min(validation_rows, key=lambda row: row["z_disp_m"])
        metadata = {
            "controller": "direct measured joint-state replay using RealSim/PPO scene",
            "ik_used": False,
            "ppo_checkpoint_used": False,
            "robot_articulation_path": "/World/envs/env_.*/franka_env/Robot/franka",
            "real_source": str(args.h5.resolve()),
            "frame_count": len(q),
            "fps": float(args.fps),
            "hole_position_sim_m": hole_pos[0].detach().cpu().tolist(),
            "hole_reference": args.hole_reference,
            "hole_position_gt_final_raw_m": (
                inferred_gt_final_hole_pos[0].detach().cpu().tolist()
                if inferred_gt_final_hole_pos is not None
                else None
            ),
            "hole_z_source": (
                "configured_pi0_hole_init_pos"
                if args.hole_reference in ("gt-final-xy-fixed-z", "configured")
                else "gt_final_peg_fk"
            ),
            "geometric_success": bool(successful_frames),
            "first_success_frame": successful_frames[0] if successful_frames else None,
            "successful_frame_count": len(successful_frames),
            "success_xy_threshold_m": float(cfg.task.success_xy_threshold),
            "success_z_threshold_m": float(
                cfg.task.fixed_asset_cfg.height * cfg.task.success_threshold
            ),
            "best_xy_error_m": best_xy_row["xy_error_m"],
            "best_xy_frame": best_xy_row["frame"],
            "minimum_z_disp_m": lowest_z_row["z_disp_m"],
            "minimum_z_frame": lowest_z_row["frame"],
            "final_xy_error_m": validation_rows[-1]["xy_error_m"],
            "final_z_disp_m": validation_rows[-1]["z_disp_m"],
            "peg_mode": (
                "integrated non-rendered peg collider on panda_fingertip_centered; "
                "independent HeldAsset is visual-only"
                if args.physical_wrist_force
                else (
                    "dynamic HeldAsset contact probe with per-frame FK reattachment"
                    if args.resolve_asset_collisions
                    else "per-frame kinematic HeldAsset pose from simulator FK"
                )
            ),
            "integrated_peg_mount_local_z_m": (
                float(cfg.task.held_asset_cfg.height)
                - float(cfg.task.robot_cfg.franka_fingerpad_length)
                + float(getattr(cfg, "pi0_peg_mount_depth_adjust_m", 0.0))
                if args.physical_wrist_force
                else None
            ),
            "physical_wrist_force": bool(args.physical_wrist_force),
            "counterfactual_no_contact_wrench_subtraction": bool(
                args.physical_wrist_force
                and args.counterfactual_wrench_subtraction
            ),
            "training_wrench_source": str(args.training_wrench_source),
            "contact_compliant_stiffness_n_m": float(
                args.contact_compliant_stiffness_n_m
            ),
            "contact_compliant_damping_n_s_m": float(
                args.contact_compliant_damping_n_s_m
            ),
            "contact_compliant_enabled": bool(args.enable_compliant_contact),
            "wrench_child_joint_frame_calibrated": bool(args.physical_wrist_force),
            "wrench_baseline_seconds": float(args.wrench_baseline_seconds),
            "wrench_ema_alpha": float(args.wrench_ema_alpha),
            "asset_collision_resolution": bool(args.resolve_asset_collisions),
            "collision_substeps": int(args.collision_substeps),
            "asset_contact_offset_m": (
                float(cfg.task.held_asset.spawn.collision_props.contact_offset)
                if args.resolve_asset_collisions
                else None
            ),
            "asset_max_depenetration_velocity_m_s": (
                float(
                    cfg.task.held_asset.spawn.rigid_props.max_depenetration_velocity
                )
                if args.resolve_asset_collisions
                else None
            ),
            "q_max_abs_error_rad": float(np.max(np.abs(actual_q - q))),
            "q_mean_abs_error_rad": float(np.mean(np.abs(actual_q - q))),
            "real_ee_pose_not_used_for_control": True,
            "real_ee_first": real_ee_pose[0].tolist(),
            "real_ee_last": real_ee_pose[-1].tolist(),
            "domain_randomization": domain_randomization,
            "cotraining_h5": "data.h5",
            "cotraining_state_action_source": "paired real H5 (unaltered labels/state)",
            "force_fields": (
                "contact-equivalent peg/hole wrench expressed in base at K; "
                "incoming-joint wrench is diagnostic under direct replay"
                if args.save_sim_force
                and args.training_wrench_source == "contact-equivalent"
                else (
                    "calibrated force_sensor incoming-joint wrench in base/K; "
                    "contact-point wrench is diagnostic only"
                    if args.save_sim_force
                    else "copied for schema compatibility; not simulated or adjusted"
                )
            ),
            "contact_training_force_stats": contact_force_stats,
            "sim_force_saved": bool(args.save_sim_force),
            "tavla_only_output": bool(args.tavla_only_output),
            "sim_force_streams": (
                [
                    "wrench_raw",
                    "wrench_child_joint_parent",
                    "wrench_anchor",
                    "wrench_base",
                    "wrench_corrected",
                    "wrench_final",
                    "wrench_tool_smooth",
                    "wrench_model_clean",
                    "wrench_model",
                    "wrench_calibrated_unclipped",
                    "wrench_calibrated",
                    "wrench_counterfactual_no_contact",
                    "wrench_contact_isolated_unfiltered",
                    "O_F_ext_hat_K",
                    "incoming_joint_wrench_base_at_K",
                    "incoming_joint_wrench_calibrated",
                    "contact_wrench_base_at_K",
                    "contact_equivalent_wrench_base_at_K",
                ]
                if args.save_sim_force
                else []
            ),
            "contact_pair_collection": {
                "enabled": bool(args.record_contact_pair),
                "sensor": "IsaacLab PhysX SimulationView.create_rigid_contact_view",
                "held_prim_path": str(
                    getattr(
                        cfg,
                        "replay_peg_contact_body_path",
                        "/World/envs/env_0/HeldAsset",
                    )
                ),
                "fixed_prim_path": "/World/envs/env_0/FixedAsset",
                "files": (
                    []
                    if args.tavla_only_output
                    else (
                        [
                            "contact_normal_world.csv",
                            "contact_normal_world_raw.csv",
                            "contact_friction_world.csv",
                            "contact_friction_world_raw.csv",
                            "contact_total_world.csv",
                            "contact_total_world_raw.csv",
                            "contact_normal_base.csv",
                            "contact_normal_base_raw.csv",
                            "contact_friction_base.csv",
                            "contact_friction_base_raw.csv",
                            "contact_total_base.csv",
                            "contact_total_base_raw.csv",
                            "contact_torque_k_world.csv",
                            "contact_torque_k_world_raw.csv",
                            "contact_torque_k_base.csv",
                            "contact_torque_k_base_raw.csv",
                            "contact_wrench_base.csv",
                            "contact_wrench_base_raw.csv",
                            "contact_count.csv",
                        ]
                        if args.record_contact_pair
                        else []
                    )
                ),
                "diagnostic_files_removed_after_replay": bool(
                    args.tavla_only_output
                ),
                "used_by_controller": False,
                "route_control_unchanged": True,
                "contact_wrench_base_semantics": (
                    "contact-point wrench [F_base, tau_about_K_base]; written to "
                    "obs/state/ee_wrench_base only when training_wrench_source is "
                    "contact-equivalent"
                ),
                "substep_aggregation": "mean of one contact read per physics substep",
                "smoothing": "same causal EMA on force and torque vectors",
                "smoothing_alpha": float(args.contact_force_smoothing_alpha),
            },
            "note": "The replay uses the same joint-space state path as the RealSim/PPO setup; Cartesian IK is bypassed.",
        }
        (args.output_dir / "replay_metadata.json").write_text(
            json.dumps(metadata, indent=2), encoding="utf-8"
        )
        if args.tavla_only_output:
            _cleanup_tavla_only_output(args.output_dir)
        print(f"[DONE] front: {args.output_dir / 'front_camera.mp4'}", flush=True)
        print(f"[DONE] wrist: {args.output_dir / 'wrist_camera.mp4'}", flush=True)
        print(f"[DONE] cotraining H5: {args.output_dir / 'data.h5'}", flush=True)
        print(
            "[GTValidation] "
            f"hole_reference={args.hole_reference} "
            f"success={bool(successful_frames)} "
            f"best_xy_mm={best_xy_row['xy_error_m'] * 1000.0:.3f} "
            f"minimum_z_mm={lowest_z_row['z_disp_m'] * 1000.0:.3f}",
            flush=True,
        )
    finally:
        close_pi0_client(pi0_client)
        env.close()


try:
    main()
finally:
    simulation_app.close()
