"""Rigid transforms in Universal Robots conventions.

A UR pose is ``[x, y, z, rx, ry, rz]``: metres and a rotation vector in radians. Matrices are 4x4
homogeneous transforms named ``A_from_B`` in prose (``T_base_flange`` maps flange coordinates into
the base frame), matching how the robot reports ``getActualTCPPose``.
"""
import math

import cv2
import numpy as np


def check_transform(value, name: str = "transform") -> np.ndarray:
    transform = np.asarray(value, float)
    if transform.shape != (4, 4) or not np.isfinite(transform).all():
        raise ValueError(f"{name} must be a finite 4x4 rigid transform")
    rotation = transform[:3, :3]
    if (not np.allclose(transform[3], [0, 0, 0, 1], atol=1e-6, rtol=0)
            or not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-6, rtol=0)
            or abs(np.linalg.det(rotation) - 1) > 1e-6):
        raise ValueError(f"{name} must be a rigid transform with bottom row [0,0,0,1]")
    return transform


def matrix(rotation: np.ndarray, translation) -> np.ndarray:
    result = np.eye(4)
    result[:3, :3] = np.asarray(rotation, float)
    result[:3, 3] = np.asarray(translation, float).ravel()
    return result


def rotvec_to_rotation(rotvec) -> np.ndarray:
    return cv2.Rodrigues(np.asarray(rotvec, float).reshape(3, 1))[0]


def rotation_to_rotvec(rotation: np.ndarray) -> np.ndarray:
    """Log map via the unit quaternion: exact for tiny angles and stable near pi.

    cv2.Rodrigues returns zero below about 1e-5 rad, which silently zeroes numerical
    derivatives in IK and least-squares refinement.
    """
    x, y, z, w = quaternion_xyzw(rotation)
    vector = np.array([x, y, z])
    sine = np.linalg.norm(vector)
    if sine < 1e-12:
        return 2.0 * vector
    return vector / sine * 2.0 * math.atan2(sine, w)


def pose_to_matrix(pose) -> np.ndarray:
    values = np.asarray(pose, float).ravel()
    if values.shape != (6,) or not np.isfinite(values).all():
        raise ValueError(f"A UR pose needs six finite values, got {pose!r}")
    return matrix(rotvec_to_rotation(values[3:]), values[:3])


def matrix_to_pose(transform: np.ndarray) -> list[float]:
    transform = np.asarray(transform, float)
    return [*transform[:3, 3].tolist(), *rotation_to_rotvec(transform[:3, :3]).tolist()]


def invert(transform: np.ndarray) -> np.ndarray:
    rotation = transform[:3, :3]
    return matrix(rotation.T, -rotation.T @ transform[:3, 3])


def rotation_angle(rotation: np.ndarray) -> float:
    """Angle of a rotation matrix in radians, robust near 0 and pi."""
    cosine = (np.trace(rotation) - 1.0) / 2.0
    return float(math.acos(min(1.0, max(-1.0, cosine))))


def angle_between_deg(a: np.ndarray, b: np.ndarray) -> float:
    return math.degrees(rotation_angle(a[:3, :3].T @ b[:3, :3]))


def axis_angle_rotation(axis, angle_rad: float) -> np.ndarray:
    axis = np.asarray(axis, float)
    norm = np.linalg.norm(axis)
    if norm < 1e-12:
        return np.eye(3)
    return rotvec_to_rotation(axis / norm * angle_rad)


def project_to_rotation(values: np.ndarray) -> np.ndarray:
    """Nearest proper rotation (SVD), used for rotation means and noisy estimates."""
    u, _, vt = np.linalg.svd(values)
    rotation = u @ vt
    if np.linalg.det(rotation) < 0:
        u[:, -1] *= -1
        rotation = u @ vt
    return rotation


def mean_transform(transforms) -> np.ndarray:
    transforms = list(transforms)
    rotation = project_to_rotation(sum(t[:3, :3] for t in transforms))
    return matrix(rotation, np.mean([t[:3, 3] for t in transforms], axis=0))


def quaternion_xyzw(rotation: np.ndarray) -> list[float]:
    r = np.asarray(rotation, float)
    trace = np.trace(r)
    if trace > 0:
        s = math.sqrt(trace + 1.0) * 2
        q = [(r[2, 1] - r[1, 2]) / s, (r[0, 2] - r[2, 0]) / s, (r[1, 0] - r[0, 1]) / s, 0.25 * s]
    elif r[0, 0] > r[1, 1] and r[0, 0] > r[2, 2]:
        s = math.sqrt(1.0 + r[0, 0] - r[1, 1] - r[2, 2]) * 2
        q = [0.25 * s, (r[0, 1] + r[1, 0]) / s, (r[0, 2] + r[2, 0]) / s, (r[2, 1] - r[1, 2]) / s]
    elif r[1, 1] > r[2, 2]:
        s = math.sqrt(1.0 + r[1, 1] - r[0, 0] - r[2, 2]) * 2
        q = [(r[0, 1] + r[1, 0]) / s, 0.25 * s, (r[1, 2] + r[2, 1]) / s, (r[0, 2] - r[2, 0]) / s]
    else:
        s = math.sqrt(1.0 + r[2, 2] - r[0, 0] - r[1, 1]) * 2
        q = [(r[0, 2] + r[2, 0]) / s, (r[1, 2] + r[2, 1]) / s, 0.25 * s, (r[1, 0] - r[0, 1]) / s]
    q = np.asarray(q) / np.linalg.norm(q)
    return (q if q[3] >= 0 else -q).tolist()


def rpy_deg(rotation: np.ndarray) -> list[float]:
    """PolyScope RPY: fixed-axis roll about X, pitch about Y, yaw about Z (R = Rz Ry Rx)."""
    r = np.asarray(rotation, float)
    pitch = math.asin(max(-1.0, min(1.0, -r[2, 0])))
    if abs(math.cos(pitch)) > 1e-9:
        roll, yaw = math.atan2(r[2, 1], r[2, 2]), math.atan2(r[1, 0], r[0, 0])
    else:  # gimbal lock: put everything into yaw
        roll, yaw = 0.0, math.atan2(-r[0, 1], r[1, 1])
    return [math.degrees(roll), math.degrees(pitch), math.degrees(yaw)]


def describe(transform: np.ndarray) -> dict:
    """Every representation an integrator may need, from one matrix."""
    transform = np.asarray(transform, float)
    return {
        "matrix": transform.tolist(),
        "pose_ur": matrix_to_pose(transform),
        "xyz_mm": (transform[:3, 3] * 1000).tolist(),
        "rpy_deg": rpy_deg(transform[:3, :3]),
        "quaternion_xyzw": quaternion_xyzw(transform[:3, :3]),
    }
