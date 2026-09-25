"""Build a visual-only right Franka from an end-effector position.

This module is Pi0-specific. It reads the Franka joint frames from the USD,
solves a damped least-squares position IK, applies the resulting FK pose to a
flattened copy of the asset, and removes all physics schemas before export.
The exported USD is meant to be referenced as a static visual distractor.
"""
from __future__ import annotations

import math
import os
import tempfile
from pathlib import Path

import numpy as np


JOINT_LIMITS = np.asarray(
    [
        (-2.8973, 2.8973),
        (-1.7628, 1.7628),
        (-2.8973, 2.8973),
        (-3.0718, -0.0698),
        (-2.8973, 2.8973),
        (-0.0175, 3.7525),
        (-2.8973, 2.8973),
    ],
    dtype=np.float64,
)


def _pose_matrix(Gf, position, quat):
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


def _axis_rotation(Gf, axis, angle_rad):
    rotation = Gf.Rotation(
        Gf.Vec3d(*[float(value) for value in axis]),
        math.degrees(float(angle_rad)),
    )
    return Gf.Matrix4d(Gf.Matrix3d(rotation), Gf.Vec3d(0.0, 0.0, 0.0))


def _axis_translation(Gf, axis, distance):
    matrix = Gf.Matrix4d(1.0)
    matrix.SetTranslate(
        Gf.Vec3d(*[float(value) * float(distance) for value in axis])
    )
    return matrix


def _local_matrix(UsdGeom, stage, path):
    prim = stage.GetPrimAtPath(path)
    if not prim.IsValid():
        raise RuntimeError(f"Missing USD prim while solving right-arm IK: {path}")
    return UsdGeom.Xformable(prim).GetLocalTransformation()


def _endpoint_source_in_robot(stage, robot_path: str):
    """Return (path, source transform in robot frame, direct-child flag)."""
    from pxr import UsdGeom

    direct_path = f"{robot_path}/panda_fingertip_centered"
    if stage.GetPrimAtPath(direct_path).IsValid():
        return direct_path, _local_matrix(UsdGeom, stage, direct_path), True

    endpoint_path = (
        f"{robot_path}/panda_link7/panda_link8/panda_hand/panda_ee"
    )
    endpoint = stage.GetPrimAtPath(endpoint_path)
    robot = stage.GetPrimAtPath(robot_path)
    if not endpoint.IsValid():
        raise RuntimeError(
            "Missing USD end-effector prim while solving right-arm IK: "
            f"{direct_path} or {endpoint_path}"
        )

    # panda_ee is nested below link7 in fr3v2.usd. Convert its authored
    # local-to-world transform into the Franka prim frame before applying the
    # current link7 pose; using only panda_ee's local offset drops link8/hand.
    cache = UsdGeom.XformCache()
    endpoint_world = cache.GetLocalToWorldTransform(endpoint)
    robot_world = cache.GetLocalToWorldTransform(robot)
    return endpoint_path, endpoint_world * robot_world.GetInverse(), False


def _fk_fingertip(stage, robot_path: str, q: np.ndarray):
    """Return the centered fingertip position in the Franka prim frame."""
    from pxr import Gf, UsdGeom, UsdPhysics

    link7_source = _local_matrix(
        UsdGeom, stage, f"{robot_path}/panda_link7"
    )
    parent_pose = Gf.Matrix4d(1.0)
    link7_pose = None
    for joint_index, joint_angle in enumerate(q, start=1):
        joint_path = f"{robot_path}/joints/panda_joint{joint_index}"
        joint = UsdPhysics.RevoluteJoint(stage.GetPrimAtPath(joint_path))
        if not joint.GetPrim().IsValid():
            raise RuntimeError(f"Missing right-arm joint: {joint_path}")
        frame0 = _pose_matrix(
            Gf,
            joint.GetLocalPos0Attr().Get(),
            joint.GetLocalRot0Attr().Get(),
        )
        frame1 = _pose_matrix(
            Gf,
            joint.GetLocalPos1Attr().Get(),
            joint.GetLocalRot1Attr().Get(),
        )
        parent_pose = (
            frame1.GetInverse()
            * _axis_rotation(Gf, (0.0, 0.0, 1.0), joint_angle)
            * frame0
            * parent_pose
        )
        link7_pose = parent_pose

    _, fingertip_source, _ = _endpoint_source_in_robot(stage, robot_path)
    wrist_delta = link7_source.GetInverse() * link7_pose
    fingertip_pose = fingertip_source * wrist_delta
    position = fingertip_pose.ExtractTranslation()
    return np.asarray([float(position[i]) for i in range(3)], dtype=np.float64), link7_pose


def solve_position_ik(
    stage,
    robot_path: str,
    target_pos,
    initial_q,
    *,
    max_iterations: int = 120,
    position_tolerance_m: float = 1.0e-4,
    damping: float = 2.0e-3,
    nominal_weight: float = 2.0e-3,
    finite_difference_rad: float = 1.0e-4,
    max_step_rad: float = 0.12,
):
    """Solve 3-D position IK while staying near the supplied nominal pose."""
    target = np.asarray(target_pos, dtype=np.float64).reshape(3)
    nominal = np.asarray(initial_q, dtype=np.float64).reshape(7)
    nominal = np.clip(nominal, JOINT_LIMITS[:, 0], JOINT_LIMITS[:, 1])
    q = nominal.copy()

    for _ in range(max_iterations):
        current, _ = _fk_fingertip(stage, robot_path, q)
        error = target - current
        if float(np.linalg.norm(error)) <= position_tolerance_m:
            return q, float(np.linalg.norm(error))

        jacobian = np.zeros((3, 7), dtype=np.float64)
        for index in range(7):
            q_plus = q.copy()
            q_minus = q.copy()
            q_plus[index] = min(
                JOINT_LIMITS[index, 1], q_plus[index] + finite_difference_rad
            )
            q_minus[index] = max(
                JOINT_LIMITS[index, 0], q_minus[index] - finite_difference_rad
            )
            plus, _ = _fk_fingertip(stage, robot_path, q_plus)
            minus, _ = _fk_fingertip(stage, robot_path, q_minus)
            denominator = q_plus[index] - q_minus[index]
            if denominator > 0.0:
                jacobian[:, index] = (plus - minus) / denominator

        lhs = jacobian.T @ jacobian
        lhs += (damping + nominal_weight) * np.eye(7, dtype=np.float64)
        rhs = jacobian.T @ error + nominal_weight * (nominal - q)
        delta = np.linalg.solve(lhs, rhs)
        delta_norm = float(np.linalg.norm(delta))
        if delta_norm > max_step_rad:
            delta *= max_step_rad / delta_norm
        q = np.clip(q + delta, JOINT_LIMITS[:, 0], JOINT_LIMITS[:, 1])

    final_pos, _ = _fk_fingertip(stage, robot_path, q)
    residual = float(np.linalg.norm(target - final_pos))
    if residual > max(position_tolerance_m * 5.0, 1.0e-3):
        raise ValueError(
            "Right-arm position IK did not converge: "
            f"target={target.tolist()} final={final_pos.tolist()} "
            f"residual={residual:.6f} m"
        )
    return q, residual


def _matrix_to_numpy(matrix):
    return np.asarray(
        [[float(matrix[row][col]) for col in range(4)] for row in range(4)],
        dtype=np.float64,
    )


def _rotation_log(rotation):
    """Return the axis-angle vector for a 3x3 relative rotation matrix."""
    rotation = np.asarray(rotation, dtype=np.float64).reshape(3, 3)
    cosine = float(np.clip((np.trace(rotation) - 1.0) * 0.5, -1.0, 1.0))
    angle = math.acos(cosine)
    skew = np.asarray(
        [
            rotation[2, 1] - rotation[1, 2],
            rotation[0, 2] - rotation[2, 0],
            rotation[1, 0] - rotation[0, 1],
        ],
        dtype=np.float64,
    )
    sine = float(np.linalg.norm(skew) * 0.5)
    if sine < 1.0e-8:
        return 0.5 * skew
    return (angle / (2.0 * sine)) * skew


def _axis_rotation_numpy(angle_rad):
    """Return the USD/Gf row-vector rotation about the joint z axis."""
    cosine = math.cos(float(angle_rad))
    sine = math.sin(float(angle_rad))
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = np.asarray(
        [
            [cosine, sine, 0.0],
            [-sine, cosine, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    return matrix


def _make_fk_cache(stage, robot_path: str):
    from pxr import Gf, UsdPhysics, UsdGeom

    link7_source = _matrix_to_numpy(
        _local_matrix(UsdGeom, stage, f"{robot_path}/panda_link7")
    )
    joint_frames = []
    for joint_index in range(1, 8):
        joint_path = f"{robot_path}/joints/panda_joint{joint_index}"
        if not stage.GetPrimAtPath(joint_path).IsValid():
            joint_path = (
                f"{robot_path}/panda_link{joint_index - 1}/"
                f"panda_joint{joint_index}"
            )
        joint = UsdPhysics.RevoluteJoint(stage.GetPrimAtPath(joint_path))
        if not joint.GetPrim().IsValid():
            raise RuntimeError(f"Missing right-arm joint: {joint_path}")
        frame0 = _pose_matrix(
            Gf,
            joint.GetLocalPos0Attr().Get(),
            joint.GetLocalRot0Attr().Get(),
        )
        frame1 = _pose_matrix(
            Gf,
            joint.GetLocalPos1Attr().Get(),
            joint.GetLocalRot1Attr().Get(),
        )
        joint_frames.append(
            (
                _matrix_to_numpy(frame0),
                np.linalg.inv(_matrix_to_numpy(frame1)),
            )
        )

    _, fingertip_source, _ = _endpoint_source_in_robot(stage, robot_path)
    return (
        np.linalg.inv(link7_source),
        _matrix_to_numpy(fingertip_source),
        tuple(joint_frames),
    )


def _fk_endpoint_pose_cached(fk_cache, q):
    link7_source_inverse, fingertip_source, joint_frames = fk_cache
    q = np.asarray(q, dtype=np.float64).reshape(7)
    parent_pose = np.eye(4, dtype=np.float64)
    for joint_angle, (frame0, frame1) in zip(q, joint_frames):
        parent_pose = (
            frame1
            @ _axis_rotation_numpy(joint_angle)
            @ frame0
            @ parent_pose
        )
    wrist_delta = link7_source_inverse @ parent_pose
    fingertip_pose = fingertip_source @ wrist_delta
    return fingertip_pose[3, :3].copy(), fingertip_pose[:3, :3].copy()


def fk_endpoint_pose(stage, robot_path: str, q):
    """Return the visual fingertip position and rotation in the robot frame."""
    return _fk_endpoint_pose_cached(_make_fk_cache(stage, robot_path), q)


def _solve_small_linear_system(matrix, vector):
    """Solve the 7x7 DLS system without entering NumPy LAPACK from Isaac."""
    augmented = [
        [float(value) for value in row] + [float(vector[row_index])]
        for row_index, row in enumerate(np.asarray(matrix, dtype=np.float64))
    ]
    size = len(augmented)
    for column in range(size):
        pivot = max(
            range(column, size),
            key=lambda row_index: abs(augmented[row_index][column]),
        )
        if abs(augmented[pivot][column]) < 1.0e-12:
            raise np.linalg.LinAlgError("singular damped IK system")
        if pivot != column:
            augmented[column], augmented[pivot] = (
                augmented[pivot],
                augmented[column],
            )
        pivot_value = augmented[column][column]
        augmented[column] = [
            value / pivot_value for value in augmented[column]
        ]
        for row_index in range(size):
            if row_index == column:
                continue
            factor = augmented[row_index][column]
            if factor == 0.0:
                continue
            augmented[row_index] = [
                value - factor * pivot_value
                for value, pivot_value in zip(
                    augmented[row_index], augmented[column]
                )
            ]
    return np.asarray(
        [augmented[row_index][-1] for row_index in range(size)],
        dtype=np.float64,
    )


def solve_pose_ik(
    stage,
    robot_path: str,
    target_pos,
    target_rot,
    initial_q,
    *,
    max_iterations: int = 240,
    position_tolerance_m: float = 1.0e-4,
    rotation_tolerance_rad: float = 1.0e-3,
    damping: float = 2.0e-3,
    nominal_weight: float = 1.0e-3,
    finite_difference_rad: float = 1.0e-4,
    max_step_rad: float = 0.12,
):
    """Solve a 6-D end-effector pose IK near the supplied initial joint pose."""
    target_pos = np.asarray(target_pos, dtype=np.float64).reshape(3)
    target_rot = np.asarray(target_rot, dtype=np.float64).reshape(3, 3)
    nominal = np.asarray(initial_q, dtype=np.float64).reshape(7)
    nominal = np.clip(nominal, JOINT_LIMITS[:, 0], JOINT_LIMITS[:, 1])
    q = nominal.copy()
    fk_cache = _make_fk_cache(stage, robot_path)
    task_weights = np.asarray([1.0, 1.0, 1.0, 0.15, 0.15, 0.15])

    for _ in range(max_iterations):
        current_pos, current_rot = _fk_endpoint_pose_cached(fk_cache, q)
        position_error = target_pos - current_pos
        rotation_error = _rotation_log(current_rot.T @ target_rot)
        error = np.concatenate((position_error, rotation_error))
        if (
            float(np.linalg.norm(position_error)) <= position_tolerance_m
            and float(np.linalg.norm(rotation_error)) <= rotation_tolerance_rad
        ):
            return q, float(np.linalg.norm(position_error)), float(
                np.linalg.norm(rotation_error)
            )

        jacobian = np.zeros((6, 7), dtype=np.float64)
        for index in range(7):
            q_plus = q.copy()
            q_minus = q.copy()
            q_plus[index] = min(
                JOINT_LIMITS[index, 1], q_plus[index] + finite_difference_rad
            )
            q_minus[index] = max(
                JOINT_LIMITS[index, 0], q_minus[index] - finite_difference_rad
            )
            plus_pos, plus_rot = _fk_endpoint_pose_cached(fk_cache, q_plus)
            minus_pos, minus_rot = _fk_endpoint_pose_cached(fk_cache, q_minus)
            denominator = q_plus[index] - q_minus[index]
            if denominator > 0.0:
                jacobian[:3, index] = (plus_pos - minus_pos) / denominator
                plus_rotation_delta = _rotation_log(current_rot.T @ plus_rot)
                minus_rotation_delta = _rotation_log(current_rot.T @ minus_rot)
                jacobian[3:, index] = (
                    plus_rotation_delta - minus_rotation_delta
                ) / denominator
        weighted_jacobian = jacobian * task_weights[:, None]
        weighted_error = error * task_weights
        lhs = weighted_jacobian.T @ weighted_jacobian
        lhs += (damping + nominal_weight) * np.eye(7, dtype=np.float64)
        rhs = weighted_jacobian.T @ weighted_error + nominal_weight * (nominal - q)
        delta = _solve_small_linear_system(lhs, rhs)
        delta_norm = float(np.linalg.norm(delta))
        if delta_norm > max_step_rad:
            delta *= max_step_rad / delta_norm
        q = np.clip(q + delta, JOINT_LIMITS[:, 0], JOINT_LIMITS[:, 1])

    final_pos, final_rot = _fk_endpoint_pose_cached(fk_cache, q)
    position_residual = float(np.linalg.norm(target_pos - final_pos))
    rotation_residual = float(
        np.linalg.norm(_rotation_log(final_rot.T @ target_rot))
    )
    if (
        position_residual > max(position_tolerance_m * 5.0, 1.0e-3)
        or rotation_residual > max(rotation_tolerance_rad * 5.0, 0.02)
    ):
        raise ValueError(
            "Right-arm pose IK did not converge: "
            f"target_pos={target_pos.tolist()} final_pos={final_pos.tolist()} "
            f"position_residual={position_residual:.6f} m "
            f"rotation_residual={rotation_residual:.6f} rad"
        )
    return q, position_residual, rotation_residual


def _write_matrix(stage, path: str, matrix):
    from pxr import UsdGeom

    prim = stage.OverridePrim(path)
    xform = UsdGeom.Xformable(prim)
    xform.ClearXformOpOrder()
    xform.AddTransformOp(
        precision=UsdGeom.XformOp.PrecisionDouble,
        opSuffix="pi0_right_static_pose",
    ).Set(matrix)


def _apply_fk_pose(stage, robot_path: str, q, gripper_open: float):
    from pxr import Gf, UsdGeom, UsdPhysics

    q = np.asarray(q, dtype=np.float64).reshape(7)
    link7_source = _local_matrix(
        UsdGeom, stage, f"{robot_path}/panda_link7"
    )
    parent_pose = Gf.Matrix4d(1.0)
    link_poses = []
    for joint_index, joint_angle in enumerate(q, start=1):
        joint_path = f"{robot_path}/joints/panda_joint{joint_index}"
        joint = UsdPhysics.RevoluteJoint(stage.GetPrimAtPath(joint_path))
        if not joint.GetPrim().IsValid():
            raise RuntimeError(f"Missing right-arm joint: {joint_path}")
        frame0 = _pose_matrix(
            Gf,
            joint.GetLocalPos0Attr().Get(),
            joint.GetLocalRot0Attr().Get(),
        )
        frame1 = _pose_matrix(
            Gf,
            joint.GetLocalPos1Attr().Get(),
            joint.GetLocalRot1Attr().Get(),
        )
        parent_pose = (
            frame1.GetInverse()
            * _axis_rotation(Gf, (0.0, 0.0, 1.0), joint_angle)
            * frame0
            * parent_pose
        )
        link_poses.append(parent_pose)
        _write_matrix(
            stage, f"{robot_path}/panda_link{joint_index}", parent_pose
        )

    link7_pose = link_poses[-1]
    wrist_delta = link7_source.GetInverse() * link7_pose
    for finger_index in (1, 2):
        joint_path = f"{robot_path}/joints/panda_finger_joint{finger_index}"
        joint = UsdPhysics.PrismaticJoint(stage.GetPrimAtPath(joint_path))
        if not joint.GetPrim().IsValid():
            raise RuntimeError(f"Missing right-arm joint: {joint_path}")
        finger_path = (
            f"{robot_path}/panda_leftfinger"
            if finger_index == 1
            else f"{robot_path}/panda_rightfinger"
        )
        frame0 = _pose_matrix(
            Gf,
            joint.GetLocalPos0Attr().Get(),
            joint.GetLocalRot0Attr().Get(),
        )
        frame1 = _pose_matrix(
            Gf,
            joint.GetLocalPos1Attr().Get(),
            joint.GetLocalRot1Attr().Get(),
        )
        finger_pose = (
            frame1.GetInverse()
            * _axis_translation(Gf, (0.0, 1.0, 0.0), gripper_open)
            * frame0
            * link7_pose
        )
        _write_matrix(stage, finger_path, finger_pose)

    fingertip_path, fingertip_source, fingertip_is_direct = (
        _endpoint_source_in_robot(stage, robot_path)
    )
    if fingertip_is_direct:
        _write_matrix(stage, fingertip_path, fingertip_source * wrist_delta)


def _strip_physics(stage):
    from pxr import UsdPhysics

    physics_apis = (
        UsdPhysics.RigidBodyAPI,
        UsdPhysics.CollisionAPI,
        UsdPhysics.MassAPI,
        UsdPhysics.ArticulationRootAPI,
    )
    for prim in list(stage.Traverse()):
        if prim.IsA(UsdPhysics.Joint):
            prim.SetActive(False)
            continue
        for schema in physics_apis:
            if prim.HasAPI(schema):
                prim.RemoveAPI(schema)


def _transform_point(matrix, point):
    from pxr import Gf

    value = matrix.Transform(Gf.Vec3d(*[float(v) for v in point]))
    return np.asarray([float(value[i]) for i in range(3)], dtype=np.float64)


def build_static_right_arm_usd(
    source_usd: str | Path,
    *,
    target_pos=(),
    target_frame: str = "franka_env",
    nominal_q=(),
    base_pos=(0.6658084946, -0.08782, 0.0991799997),
    base_rot=(0.70710677, 0.0, 0.0, -0.70710677),
    gripper_open: float = 0.04,
) -> tuple[Path, np.ndarray, float | None]:
    """Create a temporary visual-only USD and return (path, q, residual)."""
    from pxr import Gf, Usd, UsdGeom

    source_usd = Path(source_usd).resolve()
    if not source_usd.is_file():
        raise FileNotFoundError(f"Right-arm source USD not found: {source_usd}")
    stage = Usd.Stage.Open(str(source_usd))
    if stage is None:
        raise RuntimeError(f"Could not open right-arm source USD: {source_usd}")
    robot_path = "/Root/franka"
    if not stage.GetPrimAtPath(robot_path).IsValid():
        raise RuntimeError(f"Right-arm source has no {robot_path}: {source_usd}")

    nominal = np.asarray(nominal_q, dtype=np.float64)
    if nominal.size == 0:
        nominal = np.asarray(
            [0.0, -0.7853981634, 0.0, -2.3561944902, 0.0, 1.5707963268, 0.0],
            dtype=np.float64,
        )
    nominal = nominal.reshape(7)

    target = np.asarray(target_pos, dtype=np.float64)
    residual = None
    if target.size == 0:
        q = nominal.copy()
    else:
        if target.size != 3:
            raise ValueError(
                "background_right_robot_ee_target_pos must contain 3 values in meters"
            )
        root_frame = _local_matrix(UsdGeom, stage, robot_path)
        base_quat = Gf.Quatd(
            float(base_rot[0]),
            Gf.Vec3d(float(base_rot[1]), float(base_rot[2]), float(base_rot[3])),
        )
        base_rotation = Gf.Matrix4d(
            Gf.Matrix3d(base_quat), Gf.Vec3d(0.0, 0.0, 0.0)
        )
        base_matrix = Gf.Matrix4d(
            Gf.Matrix3d(base_quat),
            Gf.Vec3d(*[float(v) for v in base_pos]),
        )
        frame = str(target_frame).lower()
        if frame in {"franka_model", "robot", "model"}:
            ik_target = target
        elif frame in {"right_base", "base"}:
            # Relative coordinates from the calibrated right-arm base. The
            # base translation is intentionally omitted because the input is
            # relative; the calibrated base rotation is still applied.
            target_asset = _transform_point(base_rotation, target)
            ik_target = _transform_point(root_frame.GetInverse(), target_asset)
        elif frame in {"franka_env", "world", "env"}:
            target_asset = _transform_point(base_matrix.GetInverse(), target)
            ik_target = _transform_point(root_frame.GetInverse(), target_asset)
        else:
            raise ValueError(
                "background_right_robot_ee_target_frame must be "
                "'franka_env', 'right_base', or 'franka_model'"
            )
        q, residual = solve_position_ik(stage, robot_path, ik_target, nominal)

    _apply_fk_pose(stage, robot_path, q, float(np.clip(gripper_open, 0.0, 0.04)))
    root = UsdGeom.Xformable(stage.GetPrimAtPath("/Root"))
    root.ClearXformOpOrder()
    root.AddTransformOp(precision=UsdGeom.XformOp.PrecisionDouble).Set(
        Gf.Matrix4d(
            Gf.Matrix3d(
                Gf.Quatd(
                    float(base_rot[0]),
                    Gf.Vec3d(
                        float(base_rot[1]), float(base_rot[2]), float(base_rot[3])
                    ),
                )
            ),
            Gf.Vec3d(0.0, 0.0, 0.0),
        )
    )
    _strip_physics(stage)

    flat_layer = stage.Flatten()
    fd, filename = tempfile.mkstemp(
        prefix="tacex_pi0_static_right_", suffix=".usd"
    )
    os.close(fd)
    output_path = Path(filename)
    flat_layer.Export(str(output_path))
    print(
        "[Pi0RightIK] static right arm: "
        f"q={np.array2string(q, precision=6)} "
        f"target_frame={target_frame} residual_m={residual}",
        flush=True,
    )
    return output_path, q.astype(np.float32), residual
