"""A complete simulated cell: UR10e, wrist camera with a known mount, and a fixed board."""
from dataclasses import dataclass

import numpy as np

from .robot import SimulatedRobot
from .transforms import matrix, rotvec_to_rotation
from .vision import SIMULATED_INTRINSICS, BoardTarget, SimulatedCamera

#: A realistic wrist mount: optical axis roughly along tool Z, rotated 90 deg about it, offset sideways.
DEFAULT_CAMERA_IN_FLANGE = matrix(rotvec_to_rotation([0.03, -0.02, np.pi / 2]), [0.035, -0.075, 0.06])


@dataclass
class Scenario:
    robot: SimulatedRobot
    camera: SimulatedCamera
    target: BoardTarget
    camera_in_flange: np.ndarray
    board_in_base: np.ndarray


def calibration_board_target() -> BoardTarget:
    """The charuco-generator A4 calibration board: 5 x 7 squares of 30 mm, DICT_6X6_250, IDs 0-16."""
    from charuco_generator.board import create_board
    from charuco_generator.config import BoardSpec, Geometry
    geometry = Geometry("DICT_6X6_250", 5, 7, 30.0, 22.5, False, 1)
    return BoardTarget.from_board(create_board(geometry, BoardSpec("calibration", "calibration", tuple(range(17)))),
                                  "calibration")


def build_scenario(*, target: BoardTarget | None = None, camera_in_flange: np.ndarray | None = None,
                   distance_m: float = 0.42, pose_noise_mm: float = 0.02, pose_noise_deg: float = 0.005,
                   image_noise: float = 2.0, seed: int = 0) -> Scenario:
    target = target or calibration_board_target()
    x = DEFAULT_CAMERA_IN_FLANGE if camera_in_flange is None else camera_in_flange
    robot = SimulatedRobot(pose_noise_m=pose_noise_mm / 1000, pose_noise_rad=np.radians(pose_noise_deg), seed=seed)
    # Board fixed in the cell, seen roughly centred and slightly oblique from the start pose.
    rng = np.random.default_rng(seed)
    view = matrix(rotvec_to_rotation(rng.uniform(-0.08, 0.08, 3)), -target.centre + [0, 0, distance_m])
    board_in_base = robot.true_flange_pose() @ x @ view
    camera = SimulatedCamera(robot, target, x, board_in_base, SIMULATED_INTRINSICS, noise_level=image_noise, seed=seed)
    return Scenario(robot, camera, target, x, board_in_base)
