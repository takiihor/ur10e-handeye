import math

import cv2
import numpy as np
import pytest

from ur_handeye.transforms import (
    angle_between_deg,
    describe,
    invert,
    matrix,
    matrix_to_pose,
    mean_transform,
    pose_to_matrix,
    quaternion_xyzw,
    rotation_to_rotvec,
    rotvec_to_rotation,
    rpy_deg,
)


@pytest.mark.parametrize("rotvec", [[1e-9, -2e-9, 3e-9], [0.3, -0.2, 0.1], [math.pi - 1e-7, 0, 0],
                                    [0, 2.0, -1.5], [0, 0, math.pi]])
def test_rotation_vector_round_trip_including_tiny_and_half_turn(rotvec):
    rotation = rotvec_to_rotation(rotvec)
    back = rotation_to_rotvec(rotation)
    np.testing.assert_allclose(rotvec_to_rotation(back), rotation, atol=1e-12)
    assert np.linalg.norm(back) <= math.pi + 1e-12


def test_tiny_rotations_are_not_zeroed_like_cv2_rodrigues():
    tiny = np.array([2e-7, -1e-7, 3e-7])
    assert np.allclose(cv2.Rodrigues(rotvec_to_rotation(tiny))[0].ravel(), 0)  # the trap
    np.testing.assert_allclose(rotation_to_rotvec(rotvec_to_rotation(tiny)), tiny, rtol=1e-6)


def test_pose_matrix_inverse_and_mean(rng):
    pose = [0.4, -0.2, 0.3, 0.1, 2.0, -0.3]
    transform = pose_to_matrix(pose)
    np.testing.assert_allclose(matrix_to_pose(transform), pose, atol=1e-12)
    np.testing.assert_allclose(transform @ invert(transform), np.eye(4), atol=1e-12)
    noisy = [transform @ matrix(rotvec_to_rotation(rng.normal(0, 1e-3, 3)), rng.normal(0, 1e-4, 3)) for _ in range(50)]
    assert angle_between_deg(mean_transform(noisy), transform) < 0.02
    with pytest.raises(ValueError):
        pose_to_matrix([0, 0, 0, 0, 0])


def test_rpy_matches_polyscope_convention_and_quaternion():
    roll, pitch, yaw = 10.0, -20.0, 30.0
    r = lambda axis, deg: rotvec_to_rotation(np.eye(3)[axis] * math.radians(deg))  # noqa: E731
    rotation = r(2, yaw) @ r(1, pitch) @ r(0, roll)
    np.testing.assert_allclose(rpy_deg(rotation), [roll, pitch, yaw], atol=1e-9)
    x, y, z, w = quaternion_xyzw(rotation_about_z := r(2, 90.0))
    np.testing.assert_allclose([x, y, z, w], [0, 0, math.sqrt(0.5), math.sqrt(0.5)], atol=1e-12)
    assert set(describe(matrix(rotation_about_z, [0.1, 0, 0]))) == {"matrix", "pose_ur", "xyz_mm", "rpy_deg",
                                                                     "quaternion_xyzw"}
