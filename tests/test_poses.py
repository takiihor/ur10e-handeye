import numpy as np
import pytest

from ur_handeye.poses import PlanOptions, estimate_camera_rotation, plan_views, probe_targets
from ur_handeye.robot import SimulatedRobot
from ur_handeye.simulation import DEFAULT_CAMERA_IN_FLANGE
from ur_handeye.transforms import angle_between_deg, invert, matrix


def test_probe_translations_recover_camera_rotation_exactly():
    start = SimulatedRobot().true_flange_pose()
    x = DEFAULT_CAMERA_IN_FLANGE
    board_in_base = start @ x @ matrix(np.eye(3), [0, 0, 0.4])
    seen = lambda flange: invert(x) @ invert(flange) @ board_in_base  # noqa: E731
    probes = [(view.flange, seen(view.flange)) for view in probe_targets(start, 0.03)]
    for view in probe_targets(start, 0.03):
        np.testing.assert_allclose(view.flange[:3, :3], start[:3, :3])  # pure translations
    rotation, residual = estimate_camera_rotation(start, seen(start), probes)
    np.testing.assert_allclose(rotation, x[:3, :3], atol=1e-9)
    assert residual < 1e-6


def test_views_rotate_about_the_board_and_keep_it_centred():
    start = SimulatedRobot().true_flange_pose()
    x = DEFAULT_CAMERA_IN_FLANGE
    centre = np.array([0.075, 0.105, 0.0])
    board_in_camera = matrix(np.eye(3), -centre + [0, 0, 0.4])
    board_in_base = start @ x @ board_in_camera
    views = plan_views(start, x, board_in_camera, centre, PlanOptions())
    assert len(views) == 14 and len({v.label for v in views}) == 14
    for view in views:
        in_camera = invert(x) @ invert(view.flange) @ board_in_base @ np.append(centre, 1)
        # The board centre stays on the optical axis; only the distance changes.
        assert np.hypot(in_camera[0], in_camera[1]) < 1e-9
        assert 0.4 * 0.9 - 1e-9 <= in_camera[2] <= 0.4 * 1.1 + 1e-9
    rotations = [angle_between_deg(start, view.flange) for view in views]
    # Every view rotates the rig; the largest combines the 25 deg tilt with the 30 deg roll.
    assert min(rotations) > 10 and 35 < max(rotations) < 50


@pytest.mark.parametrize("kwargs", [{"tilt_deg": 2}, {"roll_deg": 90}, {"near_scale": 1.2}, {"max_views": 3},
                                    {"probe_mm": 500}])
def test_plan_options_are_bounded(kwargs):
    with pytest.raises(ValueError):
        PlanOptions(**kwargs)
