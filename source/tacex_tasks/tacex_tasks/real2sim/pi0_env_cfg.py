from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass

from .policy.configuration_pi0 import PI0RemoteJointConfig
from .realsim_env_cfg import RealSimTaskPegInsertCfg


@configclass
class RealSimPi0PegInsertCfg(RealSimTaskPegInsertCfg):
    """RealSim peg-in-hole configuration for image/state-only Pi0."""

    # IsaacLab 6 uses XYZW quaternions. The Pi0 scene root already provides
    # the required X orientation, so retain only the calibrated -30 degree
    # in-plane yaw. Adding another local 180 degree X flip inverts the hole.
    pi0_hole_init_pos: tuple[float, float, float] = (-0.144046545, -0.5311332345, 0.1239399314)
    pi0_hole_init_rot: tuple[float, float, float, float] = (
        0.0,
        0.0,
        -0.258819,
        0.965926,
    )
    pi0_robot_init_rot: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0)
    # Optional reset randomization used by the single-process Pi0 benchmark.
    # Values are symmetric half-ranges in meters. Defaults preserve the
    # calibrated deterministic traj_0 reset used by normal Pi0 evaluation.
    pi0_hole_position_noise_m: tuple[float, float, float] = (0.0, 0.0, 0.0)
    pi0_hand_position_noise_m: tuple[float, float, float] = (0.0, 0.0, 0.0)
    # Paired wrist-camera calibration. The second lateral correction reduces
    # the remaining far-hole/near-peg parallax without changing robot geometry.
    pi0_wrist_camera_offset_pos: tuple[float, float, float] = (
        0.07813,
        0.00105,
        -0.0285,
    )
    pi0_wrist_camera_offset_rot: tuple[float, float, float, float] = (
        0.71266,
        0.68644,
        0.07985,
        0.12057,
    )
    # Local OpenGL-camera correction: the positive yaw compensates the global
    # image shift from the lateral translation and keeps the near peg centered.
    pi0_wrist_camera_rotation_correction_deg: tuple[float, float, float] = (
        8.3,
        3.2,
        0.0,
    )
    # Restore the original Pi0 single-arm base frame under the ActiveRobot wrapper.
    active_robot_base_pos: tuple[float, float, float] = (-0.23579, -0.087822794, 0.071018631)
    active_robot_base_rot: tuple[float, float, float, float] = (0.0, 0.0, -0.70710677, 0.70710677)
    # Do not add a separate rollout gripper visual over the native gripper.
    pi0_black_gripper_visual: bool = False
    # Gray-white Franka shell matched to the real traj_0 front image.  The
    # value is linear RGB because it is passed to UsdPreviewSurface.
    pi0_robot_visual_color: tuple[float, float, float] = (0.34, 0.35, 0.34)
    pi0_robot_visual_roughness: float = 0.38
    # The recorded physical opening remains 6.92 mm.  This optional
    # No visualization-only gripper opening compensation.
    pi0_visual_gripper_extra_width_m: float = 0.0
    # Move the peg farther into the closed fingers so its visible length in
    # the front camera matches the real traj_0 frame.  This changes only the
    # peg mount pose, not the gripper state or opening.
    pi0_peg_mount_depth_adjust_m: float = 0.005

    # Isaac6 UsdPreviewSurface expects linear color inputs.
    # The peg keeps the PPO orange. The default hole color matches the front
    # camera; the wrist render temporarily uses its camera-specific color
    # below because the two real cameras have different color responses.
    # RealSimEnv binds them to every mesh below HeldAsset/FixedAsset.
    override_held_asset_color: bool = True
    held_asset_visual_color: tuple[float, float, float] = (0.70, 0.10, 0.004)
    override_fixed_asset_color: bool = True
    fixed_asset_visual_color: tuple[float, float, float] = (0.040, 0.048, 0.042)
    pi0_front_hole_visual_color: tuple[float, float, float] = (0.040, 0.048, 0.042)
    pi0_wrist_hole_visual_color: tuple[float, float, float] = (0.115, 0.030, 0.016)

    # First visual-matching pass: use one explicit light and fixed per-camera
    # exposure. These affect rendering only; policy state and control are
    # unchanged.
    pi0_disable_background_lights: bool = True
    pi0_dome_light_intensity: float = 2200.0
    pi0_dome_light_color: tuple[float, float, float] = (0.75, 0.75, 0.75)
    pi0_front_camera_exposure: float = -1.5
    pi0_wrist_camera_exposure: float = -2.0
    # Deterministic camera-response match for the real wrist sensor. The real
    # frame is brighter and less saturated on the left, then falls off toward
    # the right. This operates on RGB only and introduces no scene geometry.
    pi0_wrist_visual_match_enabled: bool = True
    pi0_wrist_visual_gain_left: float = 0.72
    pi0_wrist_visual_gain_right: float = 0.48
    # The real wrist sensor has a broad illumination lobe in the lower-left;
    # reproduce it as a smooth response term instead of adding scene geometry.
    pi0_wrist_visual_gain_bottom_left: float = 0.32
    pi0_wrist_visual_left_desaturation: float = 0.35
    # Narrow the wrist-camera field of view so the gripper and fixture match
    # their apparent scale in the real traj_0 wrist image.
    pi0_wrist_camera_focal_length: float = 24.0
    # Shift the wrist-camera principal point so the complete image moves down
    # without changing near/far perspective or the gripper scale.
    pi0_wrist_camera_vertical_aperture_offset: float = 0.65
    # The camera is mounted about 78 mm from the hand frame.  The shared
    # 100 mm near plane clips most of the nearby fingers from the image.
    pi0_wrist_camera_clipping_range: tuple[float, float] = (0.01, 1.0e5)
    # Render Pi0 cameras at 2x resolution, then Lanczos-downsample before policy
    # inference and video output. Existing visual annotations stay in 640x480.
    pi0_camera_render_width: int = 1280
    pi0_camera_render_height: int = 960
    pi0_camera_output_width: int = 640
    pi0_camera_output_height: int = 480
    pi0_visual_annotation_width: int = 640
    pi0_visual_annotation_height: int = 480
    # Green tabletop annotated directly in the 640x480 Pi0 front view. The
    # six pixels are back-projected onto one horizontal visual-only surface.
    # Use the same shared 3-D green tabletop as the sim_force replay so both
    # the front and wrist cameras observe the same material/color.
    pi0_green_table_overlay_enabled: bool = True
    pi0_green_table_overlay_image_polygon: tuple[tuple[float, float], ...] = (
        (77.0, 239.0),
        (605.0, 237.0),
        (640.0, 286.0),
        (640.0, 490.0),
        (0.0, 490.0),
        (0.0, 365.0),
    )
    # Keep the green alignment in the front RGB composite. A 3-D tabletop
    # overlay can occlude the physical hole base in the PPO scene.
    pi0_green_table_overlay_z_offset: float = -0.005
    pi0_green_table_overlay_color: tuple[float, float, float] = (
        0.002,
        0.025,
        0.018,
    )
    pi0_green_table_overlay_roughness: float = 0.85
    # The front-camera polygon is intentionally kept tight for its image
    # match.  The wrist camera can see farther forward as the arm reaches, so
    # use one larger static tabletop mesh for the wrist render only.
    pi0_green_table_wrist_overlay_enabled: bool = True
    pi0_green_table_wrist_overlay_margin_m: float = 0.35
    pi0_green_table_wrist_overlay_z_offset: float = -0.006
    # Remove the high-chroma cyan fringe left by the static wrist mesh. This
    # is an image-only cleanup shared by policy input and continuous video.
    pi0_green_table_wrist_edge_cleanup_enabled: bool = True
    pi0_green_table_wrist_edge_cleanup_rgb: tuple[int, int, int] = (24, 75, 65)
    pi0_green_table_wrist_edge_cleanup_hue_range: tuple[int, int] = (75, 105)
    pi0_green_table_wrist_edge_cleanup_min_saturation: int = 80
    pi0_green_table_wrist_edge_cleanup_min_value: int = 90
    pi0_green_table_wrist_edge_cleanup_min_green: int = 100
    pi0_green_table_wrist_edge_cleanup_min_blue: int = 90
    pi0_green_table_wrist_edge_cleanup_max_red: int = 70
    # The shared 3-D tabletop now provides the color for both cameras. Keep
    # the front-only RGB recoloring disabled to avoid double-processing.
    pi0_green_table_front_composite_enabled: bool = False
    pi0_green_table_front_composite_rgb: tuple[int, int, int] = (24, 75, 65)
    pi0_green_table_front_composite_alpha: float = 0.92
    pi0_green_table_front_composite_hue_range: tuple[int, int] = (40, 110)
    pi0_green_table_front_composite_min_saturation: int = 25
    # Do not use a fixed image-space hole cutout. Its old traj_0 pixel box no
    # longer follows the projected hole and exposes the bright background.
    # RTX depth keeps the physical hole visible above the tabletop overlay.
    pi0_green_table_hole_cutout_xyxy: tuple[float, ...] = ()
    # Raise only the far-right area that must occlude the old bright table.
    # This region is away from the task hole and reuses the same material.
    pi0_green_table_right_patch_enabled: bool = False
    pi0_green_table_right_patch_image_polygon: tuple[tuple[float, float], ...] = (
        (540.0, 237.0),
        (625.0, 237.0),
        (680.0, 342.0),
        (680.0, 500.0),
        (540.0, 500.0),
    )
    # Retain the old hole-relative value for compatibility if a replay needs
    # to re-enable the 3-D patch; the active path uses the RGB patch below.
    pi0_green_table_right_patch_z_offset: float = -0.001
    # Rollout front-image repair for the small bright background remnant at
    # the far-right table edge.  This is deliberately separate from the
    # physical tabletop so the dynamic hole remains depth-rendered.
    pi0_green_table_right_edge_front_repair_enabled: bool = True
    pi0_green_table_right_edge_front_repair_alpha: float = 0.92
    pi0_green_table_right_edge_front_repair_feather_px: float = 8.0
    # Static visual reference visible on the right side of the real front
    # image: a black hollow tube on a silver flange.  The base contact point
    # is specified in the annotated 640x480 front image and back-projected to
    # the green tabletop.  It has no rigid-body or collision APIs.
    # Do not insert the reference prop into the Gaussian/RTX scene: adding
    # geometry there creates non-local splat artifacts in the wrist camera.
    pi0_reference_cylinder_enabled: bool = False
    # The prop is static real-scene context, so composite it only into the
    # fixed front RGB after rendering. The wrist observation remains intact.
    pi0_reference_cylinder_front_composite_enabled: bool = True
    pi0_reference_cylinder_source_video: str = "real_data/traj_0/front_camera.mp4"
    # Source crop in the real 640x480 frame: x0, y0, x1, y1. The destination
    # center keeps the real object's original image location.
    pi0_reference_cylinder_source_crop_xyxy: tuple[int, int, int, int] = (
        548,
        278,
        610,
        338,
    )
    pi0_reference_cylinder_front_center_pixel: tuple[float, float] = (579.0, 308.0)
    pi0_reference_cylinder_composite_feather_px: int = 7
    pi0_reference_cylinder_base_pixel: tuple[float, float] = (572.0, 317.0)
    pi0_reference_cylinder_surface_z_offset: float = -0.0012
    pi0_reference_cylinder_base_radius_m: float = 0.028
    pi0_reference_cylinder_base_height_m: float = 0.007
    pi0_reference_cylinder_collar_radius_m: float = 0.019
    pi0_reference_cylinder_collar_height_m: float = 0.009
    pi0_reference_cylinder_tube_radius_m: float = 0.015
    pi0_reference_cylinder_tube_height_m: float = 0.046
    pi0_reference_cylinder_inner_radius_m: float = 0.010
    pi0_reference_cylinder_metal_color: tuple[float, float, float] = (
        0.18,
        0.19,
        0.18,
    )
    pi0_reference_cylinder_black_color: tuple[float, float, float] = (
        0.006,
        0.006,
        0.005,
    )
    pi0_reference_cylinder_inner_color: tuple[float, float, float] = (
        0.0005,
        0.0005,
        0.0005,
    )
    # Do not add these references to the RTX/Gaussian scene. As with the
    # black reference tube, use real-frame crops in the fixed front image so
    # their size, color and location exactly follow the training domain.
    pi0_cable_grommets_enabled: bool = False
    pi0_cable_grommet_front_composite_enabled: bool = True
    pi0_cable_grommet_source_video: str = "real_data/traj_0/front_camera.mp4"
    pi0_cable_grommet_source_crop_xyxy: tuple[int, int, int, int] = (
        493,
        194,
        543,
        244,
    )
    pi0_cable_grommet_center_pixels: tuple[tuple[float, float], ...] = (
        (478.0, 210.0),
    )
    pi0_cable_grommet_composite_radius_px: float = 16.0
    pi0_cable_grommet_composite_feather_px: int = 2
    # Composite only onto pixels matching the local white tabletop. This
    # preserves the robot, black fingers, orange peg and green foreground.
    pi0_cable_grommet_table_color_threshold: float = 42.0
    # The white tabletop is a separate, lower surface, matching the physical
    # step between the white table and the raised green mat.  These are the
    # four corners of the red trapezoid marked on the real 640x480 front view,
    # in clockwise image order: top-left, top-right, bottom-right,
    # bottom-left.  The runtime converts these pixels back to a horizontal
    # USD plane, so the simulated tabletop follows the real camera boundary
    # instead of covering the whole rear background.
    pi0_white_table_overlay_enabled: bool = True
    pi0_white_table_overlay_image_polygon: tuple[tuple[float, float], ...] = (
        (171.0, 105.0),
        (472.0, 103.0),
        (566.0, 239.0),
        (84.0, 240.0),
    )
    # Fallback only when the front camera prim is unavailable.
    pi0_white_table_overlay_bounds: tuple[float, float, float, float] = (
        -0.38,
        0.85,
        -0.335,
        0.40,
    )
    pi0_white_table_overlay_z_offset: float = -0.015
    pi0_white_table_overlay_color: tuple[float, float, float] = (
        0.230,
        0.230,
        0.224,
    )
    pi0_white_table_overlay_roughness: float = 0.90
    # Keep the second Franka authored in the background USD as a static
    # visual-only robot. Pi0 controls a separate ActiveRobot articulation.
    remove_background_robot: bool = True
    background_robot_visual_only: bool = False
    # Use the already-authored right arm in franka_env.usd. Do not add a
    # second RightRobot reference: that would duplicate the real background arm.
    background_right_robot_visual_only: bool = True
    background_fr3v2_right_robot_visual_only: bool = False
    # Authored /World/fr3v2_01 base transform from the real-data background USD.
    background_fr3v2_right_robot_pos: tuple[float, float, float] = (
        0.6658084946,
        -0.08782,
        0.0991799997,
    )
    background_fr3v2_right_robot_rot: tuple[float, float, float, float] = (
        0.70710677,
        0.0,
        0.0,
        -0.70710677,
    )
    background_right_robot_offset_pos: tuple[float, float, float] = (0.86, 0.0, 0.0)
    # Keep the configured right base fixed. This is the desired end-effector
    # position in the copied Franka prim frame.  The final adjustment moves
    # the visual endpoint further toward image-left and image-down while the
    # base translation remains unchanged.
    background_right_robot_endpoint_reference_offset_pos: tuple[float, float, float] = (0.48, -0.30, 0.2)
    # End-effector orientation copied from the active rollout/real pose. The
    # pose IK uses this together with the position above to prevent wrist drift.
    background_right_robot_endpoint_target_rot: tuple[float, ...] = (
        1.0,
        0.0,
        0.0,
        0.0,
        -1.0,
        0.0,
        0.0,
        0.0,
        -1.0,
    )
    # Optional user-facing right-arm endpoint target. Fill this with three
    # meters in the selected frame; an empty tuple keeps the nominal pose.
    # "franka_env" is an absolute position in the environment frame.
    # Direct-copy mode does not use an IK endpoint target.
    background_right_robot_ee_target_pos: tuple[float, ...] = ()
    background_right_robot_ee_target_frame: str = "right_base"
    # IK seed only; this is no longer the final pose that users need to tune.
    # IK seed only. The base remains fixed during an episode; only this static
    # visual arm pose is solved from the endpoint target above.
    background_right_robot_joint_pos: tuple[float, ...] = (
        -0.3501465281,
        -0.1548610135,
        -0.0804703683,
        -2.7613992887,
        -0.2190701977,
        2.8177658174,
        1.0439091736,
    )
    background_right_robot_gripper_open: float = 0.04

    # Keep the external Gym action shape inherited from RealSim. The Pi0
    # policy is evaluated inside the environment and the external action is
    # ignored; this avoids routing 8-D joint targets through the old Cartesian
    # PPO interface.
    policy_cfg = None
    pi0_policy_cfg = PI0RemoteJointConfig()
    pi0_action_start_index: int = 2
    pi0_replan_actions: int = 3

    # 1/120 physics with decimation 12 gives the 10 Hz control period used by
    # the real-data policy.
    decimation: int = 12
    scene = InteractiveSceneCfg(
        num_envs=1,
        env_spacing=2.0,
        replicate_physics=True,
        filter_collisions=True,
        clone_in_fabric=False,
    )
    enable_cameras: bool = True
