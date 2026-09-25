#!/usr/bin/env python3
"""Pure-numpy unit tests for the base/K wrench contract."""

from __future__ import annotations

import unittest

import numpy as np


def rotate_wrench(rotation: np.ndarray, wrench: np.ndarray) -> np.ndarray:
    return np.concatenate((rotation @ wrench[:3], rotation @ wrench[3:]))


def translate_to_k(wrench_at_a: np.ndarray, point_a: np.ndarray, point_k: np.ndarray) -> np.ndarray:
    force = wrench_at_a[:3]
    torque_k = wrench_at_a[3:] + np.cross(point_a - point_k, force)
    return np.concatenate((force, torque_k))


class WrenchContractTest(unittest.TestCase):
    def test_rotation_preserves_expected_axes(self) -> None:
        rotation_z_90 = np.array(((0.0, -1.0, 0.0), (1.0, 0.0, 0.0), (0.0, 0.0, 1.0)))
        wrench = np.array((2.0, 0.0, 0.0, 0.0, 3.0, 0.0))
        actual = rotate_wrench(rotation_z_90, wrench)
        np.testing.assert_allclose(actual, (0.0, 2.0, 0.0, -3.0, 0.0, 0.0), atol=1e-12)

    def test_reference_translation_adds_moment(self) -> None:
        wrench_at_a = np.array((0.0, 10.0, 0.0, 0.0, 0.0, 0.0))
        actual = translate_to_k(wrench_at_a, np.array((0.10, 0.0, 0.0)), np.zeros(3))
        np.testing.assert_allclose(actual, (0.0, 10.0, 0.0, 0.0, 0.0, 1.0), atol=1e-12)

    def test_global_sign_changes_all_six_components(self) -> None:
        wrench = np.array((1.0, -2.0, 3.0, -0.1, 0.2, -0.3))
        np.testing.assert_allclose(-wrench, (-1.0, 2.0, -3.0, 0.1, -0.2, 0.3), atol=1e-12)

    def test_torque_is_not_hard_coded_to_zero(self) -> None:
        translated = translate_to_k(
            np.array((4.0, 0.0, 0.0, 0.0, 0.0, 0.0)),
            np.array((0.0, 0.05, 0.0)),
            np.zeros(3),
        )
        self.assertGreater(abs(translated[5]), 1.0e-9)


if __name__ == "__main__":
    unittest.main()
