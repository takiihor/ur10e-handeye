"""Calibration view planning around a fixed board, from a single starting view.

Views rotate the whole flange-camera body about the observed board centre (so the board stays
in view) with tilts about varied axes, rolls about the viewing direction and distance changes.
That gives the rotation diversity hand-eye calibration needs; pure translations do not constrain
the camera rotation in the flange at all.

The pivot depends on the camera's orientation in the flange, which is unknown before
calibration. ``estimate_camera_rotation`` measures it from three small pure translations of the
flange (the board's apparent motion in the camera is the flange motion rotated into the camera
frame), so no hand-entered mounting guess is needed.
"""
import math
from dataclasses import dataclass

import numpy as np

from .transforms import axis_angle_rotation, matrix, project_to_rotation


@dataclass(frozen=True)
class PlanOptions:
    tilt_deg: float = 25.0
    roll_deg: float = 30.0
    #: Fractions of the start distance used for near/far views.
    near_scale: float = 0.9
    far_scale: float = 1.1
    max_views: int = 14
    probe_mm: float = 30.0

    def __post_init__(self):
        if not 5 <= self.tilt_deg <= 40:
            raise ValueError("tilt_deg must be between 5 and 40 degrees")
        if not 0 <= self.roll_deg <= 60:
            raise ValueError("roll_deg must be between 0 and 60 degrees")
        if not 0.6 <= self.near_scale <= 1.0 <= self.far_scale <= 1.5:
            raise ValueError("near_scale must be in [0.6, 1] and far_scale in [1, 1.5]")
        if not 6 <= self.max_views <= 40:
            raise ValueError("max_views must be between 6 and 40")
        if not 5 <= self.probe_mm <= 100:
            raise ValueError("probe_mm must be between 5 and 100")


@dataclass(frozen=True)
class View:
    label: str
    flange: np.ndarray  # target T_base_flange


def probe_targets(start_flange: np.ndarray, probe_m: float) -> list["View"]:
    """Pure translations of the flange along its own X, Y, Z axes (no rotation)."""
    targets = []
    for name, axis in zip("xyz", np.eye(3), strict=True):
        target = start_flange.copy()
        target[:3, 3] += start_flange[:3, :3] @ axis * probe_m
        targets.append(View(f"probe-{name}", target))
    return targets


def estimate_camera_rotation(start_flange: np.ndarray, start_board_in_camera: np.ndarray,
                             probes: list[tuple[np.ndarray, np.ndarray]]) -> tuple[np.ndarray, float]:
    """Camera rotation in the flange from pure flange translations; returns (R, fit residual in mm).

    For a flange moved by d (expressed in the flange frame) without rotation, the board origin
    seen by the camera moves by -R^T d, where R is the camera rotation in the flange.
    """
    if len(probes) < 2:
        raise ValueError("At least two non-parallel probe translations are needed")
    flange_moves, camera_moves = [], []
    for flange, board_in_camera in probes:
        flange_moves.append(start_flange[:3, :3].T @ (flange[:3, 3] - start_flange[:3, 3]))
        camera_moves.append(-(board_in_camera[:3, 3] - start_board_in_camera[:3, 3]))
    d, a = np.asarray(flange_moves), np.asarray(camera_moves)
    # Kabsch: R^T maps flange moves onto (negated) camera moves.
    rotation_transpose = project_to_rotation(a.T @ d)
    residual = float(np.sqrt(np.mean(np.sum((d @ rotation_transpose.T - a) ** 2, axis=1))) * 1000)
    return rotation_transpose.T, residual


def plan_views(start_flange: np.ndarray, camera_in_flange: np.ndarray, start_board_in_camera: np.ndarray,
               board_centre: np.ndarray, options: PlanOptions = PlanOptions()) -> list[View]:
    camera = start_flange @ camera_in_flange
    pivot = (camera @ start_board_in_camera @ np.append(board_centre, 1.0))[:3]
    offset = pivot - camera[:3, 3]
    distance = float(np.linalg.norm(offset))
    if distance < 0.05:
        raise ValueError("Board centre is less than 5 cm from the camera; start further away")
    view = offset / distance
    # Tilt axes perpendicular to the viewing direction, anchored to the camera's X axis.
    first = camera[:3, 0] - view * float(camera[:3, 0] @ view)
    first /= np.linalg.norm(first)
    second = np.cross(view, first)

    def rigid(azimuth_deg: float, tilt_deg: float, roll_deg: float, scale: float) -> np.ndarray:
        azimuth = math.radians(azimuth_deg)
        axis = math.cos(azimuth) * first + math.sin(azimuth) * second
        rotation = axis_angle_rotation(axis, math.radians(tilt_deg)) @ axis_angle_rotation(view, math.radians(roll_deg))
        approach = matrix(np.eye(3), view * (1.0 - scale) * distance)  # move along the view ray
        about_pivot = matrix(np.eye(3), pivot) @ matrix(rotation, np.zeros(3)) @ matrix(np.eye(3), -pivot)
        return about_pivot @ approach @ start_flange

    tilt, roll = options.tilt_deg, options.roll_deg
    specs = []
    for k in range(8):  # ring of tilts in eight directions, alternating roll sign and distance
        specs.append((f"tilt{k * 45:03d}", k * 45.0, tilt, roll if k % 2 == 0 else -roll,
                      options.near_scale if k % 2 == 0 else options.far_scale))
    for k in range(4):  # inner ring with larger rolls
        specs.append((f"inner{45 + k * 90:03d}", 45.0 + k * 90.0, tilt / 2, 1.5 * roll * (1 if k % 2 else -1), 1.0))
    specs += [("roll+", 0.0, 0.0, 1.5 * roll, 0.95), ("roll-", 0.0, 0.0, -1.5 * roll, 1.05)]
    return [View(label, rigid(azimuth, tilt_deg, roll_deg, scale))
            for label, azimuth, tilt_deg, roll_deg, scale in specs[:options.max_views]]
