from __future__ import annotations

import numpy as np
import pytest

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

_HELPER_PATH = Path(__file__).parents[1] / "source/tacex_tasks/tacex_tasks/real2sim/replay_utils.py"
_SPEC = spec_from_file_location("replay_utils_under_test", _HELPER_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)

compose_pose = _MODULE.compose_pose
interpolate_pose = _MODULE.interpolate_pose
make_quaternion_sign_continuous = _MODULE.make_quaternion_sign_continuous
relative_pose = _MODULE.relative_pose
robust_pose_mean = _MODULE.robust_pose_mean
validate_replay_inputs = _MODULE.validate_replay_inputs


def assert_pose_close(first: np.ndarray, second: np.ndarray, atol: float = 1.0e-6) -> None:
    np.testing.assert_allclose(first[:3], second[:3], atol=atol)
    assert abs(float(np.dot(first[3:], second[3:]))) > 1.0 - atol


def test_relative_pose_reintegrates_to_second_pose() -> None:
    first = np.array([0.2, -0.1, 0.4, 0.9238795, 0.0, 0.3826834, 0.0])
    second = np.array([0.4, 0.0, 0.3, 0.9659258, 0.0, 0.2588190, 0.0])
    assert_pose_close(compose_pose(first, relative_pose(first, second)), second)


def test_unknown_left_base_transform_cancels_from_body_delta() -> None:
    source_first = np.array([0.1, 0.2, 0.3, 1.0, 0.0, 0.0, 0.0])
    source_second = np.array([0.2, 0.1, 0.5, 0.9659258, 0.0, 0.0, 0.2588190])
    unknown_base = np.array([-1.0, 0.7, 0.2, 0.9238795, 0.0, 0.0, 0.3826834])
    transformed_first = compose_pose(unknown_base, source_first)
    transformed_second = compose_pose(unknown_base, source_second)
    assert_pose_close(relative_pose(source_first, source_second), relative_pose(transformed_first, transformed_second))


def test_quaternion_sign_continuity_and_slerp() -> None:
    quaternions = np.array([[1.0, 0.0, 0.0, 0.0], [-1.0, 0.0, 0.0, 0.0]])
    continuous = make_quaternion_sign_continuous(quaternions)
    assert float(np.dot(continuous[0], continuous[1])) > 0.0
    midpoint = interpolate_pose(
        np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]),
        np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]),
        0.5,
    )
    np.testing.assert_allclose(midpoint[:3], [0.5, 0.0, 0.0])
    np.testing.assert_allclose(np.linalg.norm(midpoint[3:]), 1.0)


def test_robust_pose_mean_uses_median_position_and_quaternion_mean() -> None:
    poses = np.array(
        [
            [0.10, 0.20, 0.30, 1.0, 0.0, 0.0, 0.0],
            [0.11, 0.20, 0.30, -1.0, 0.0, 0.0, 0.0],
            [0.10, 0.21, 0.30, 1.0, 0.0, 0.0, 0.0],
        ]
    )
    mean = robust_pose_mean(poses)
    np.testing.assert_allclose(mean[:3], [0.10, 0.20, 0.30])
    np.testing.assert_allclose(np.abs(mean[3:]), [1.0, 0.0, 0.0, 0.0])


def test_validate_replay_inputs_rejects_bad_timestamps_and_joint_jumps() -> None:
    q = np.zeros((3, 7))
    g = np.zeros(3)
    with pytest.raises(ValueError, match="strictly increasing"):
        validate_replay_inputs(q, g, [0.0, 1.0, 1.0])
    q[1, 0] = 2.0
    with pytest.raises(ValueError, match="safety limit"):
        validate_replay_inputs(q, g, [0.0, 1.0, 2.0], max_joint_step=1.0)
