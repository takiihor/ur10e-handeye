import cv2
import numpy as np
import pytest
from charuco_realsense.calibration import opencv_calibration
from charuco_realsense.detect import CharucoDetector

from ur_handeye.simulation import build_scenario
from ur_handeye.transforms import angle_between_deg, matrix, rotvec_to_rotation
from ur_handeye.vision import BoardObserver


@pytest.fixture(scope="module")
def scenario():
    return build_scenario(seed=1, image_noise=0.0)


def test_simulated_view_recovers_true_board_pose(scenario):
    observation = BoardObserver(scenario.camera, scenario.target).observe(3)
    truth = scenario.camera.board_in_camera()
    assert observation.usable(0.3) == (True, [])
    assert len(observation.charuco_ids) == 24
    assert np.linalg.norm(observation.board_in_camera[:3, 3] - truth[:3, 3]) < 5e-5  # 0.05 mm at 0.42 m
    assert angle_between_deg(observation.board_in_camera, truth) < 0.02


def test_opencv_charuco_corner_offset_is_corrected(scenario):
    """OpenCV 4.11 detectBoard returns ChArUco corners ~+0.5 px off; the observer re-refines them.

    If this starts failing on the raw detector, OpenCV fixed the offset and the extra refinement
    can be reconsidered (it remains harmless).
    """
    gray = cv2.cvtColor(scenario.camera.read().bgr, cv2.COLOR_BGR2GRAY)
    intrinsics = scenario.camera.intrinsics
    detection = CharucoDetector(scenario.target.board, refine=False).detect(gray, intrinsics)
    truth = scenario.camera.board_in_camera()
    matrix_k, coeffs = opencv_calibration(intrinsics)
    ids = list(detection.charuco_ids)
    projected, _ = cv2.projectPoints(scenario.target.corners[ids], cv2.Rodrigues(truth[:3, :3])[0], truth[:3, 3],
                                     matrix_k, coeffs)
    raw = (detection.charuco_corners.reshape(-1, 2) - projected.reshape(-1, 2)).mean(axis=0)
    assert raw == pytest.approx([0.5, 0.5], abs=0.1)
    from ur_handeye.vision import refine_corners
    refined = (refine_corners(gray, detection.charuco_corners) - projected.reshape(-1, 2)).mean(axis=0)
    assert np.abs(refined).max() < 0.02


def test_unrefined_observer_shows_the_bias(scenario):
    truth = scenario.camera.board_in_camera()
    raw = BoardObserver(scenario.camera, scenario.target, refine=False).observe(1).board_in_camera
    fixed = BoardObserver(scenario.camera, scenario.target).observe(1).board_in_camera
    assert np.linalg.norm((raw - truth)[:2, 3]) > 1e-4 > np.linalg.norm((fixed - truth)[:2, 3])


class ShakyCamera:
    """Each frame comes from a slightly different flange pose, like a mount still ringing."""

    def __init__(self, camera, robot, amplitude_m):
        self.camera, self.robot, self.amplitude = camera, robot, amplitude_m
        self.rng = np.random.default_rng(0)

    def read(self):
        original = self.robot.true_flange_pose
        shake = matrix(np.eye(3), self.rng.normal(0, self.amplitude, 3))
        self.robot.true_flange_pose = lambda: original() @ shake
        try:
            return self.camera.read()
        finally:
            self.robot.true_flange_pose = original


def test_vibration_is_rejected_as_corner_jitter(scenario):
    shaky = ShakyCamera(scenario.camera, scenario.robot, 0.0015)
    observation = BoardObserver(shaky, scenario.target).observe(5)
    usable, reasons = observation.usable(0.3)
    assert not usable and "corner_jitter" in reasons


def test_board_out_of_view_has_no_pose():
    scenario = build_scenario(seed=2)
    scenario.camera.board_in_base = scenario.camera.board_in_base @ matrix(rotvec_to_rotation([0, 0, 0]), [1.0, 0, 0])
    observation = BoardObserver(scenario.camera, scenario.target).observe(2)
    assert observation.pose is None and observation.usable(0.3) == (False, ["no_pose"])


def test_changing_intrinsics_within_capture_is_rejected(scenario):
    from dataclasses import replace

    from ur_handeye.vision import VisionError
    class Camera:
        count = 0
        def read(self):
            self.count += 1
            frame = scenario.camera.read()
            return replace(frame, intrinsics=dict(frame.intrinsics, fx=910 + self.count))
    with pytest.raises(VisionError, match='changed during capture'):
        BoardObserver(Camera(), scenario.target).observe(3)


def test_repeated_camera_frame_does_not_count_as_multiple_samples(scenario):
    class Camera:
        def read(self):
            return frame
    frame = scenario.camera.read()
    observation = BoardObserver(Camera(), scenario.target).observe(5)
    assert observation.frames_used == 1
    assert not observation.usable(0.3)[0]
