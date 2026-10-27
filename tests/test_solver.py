import cv2
import numpy as np
import pytest
from charuco_realsense.calibration import opencv_calibration

from ur_handeye.solver import CLOSED_FORM_METHODS, Sample, SolveError, closed_form, solve
from ur_handeye.transforms import angle_between_deg, invert, matrix, rotvec_to_rotation
from ur_handeye.vision import SIMULATED_INTRINSICS as INTRINSICS

CORNERS = np.array([[x * 0.03, y * 0.03, 0] for y in range(1, 7) for x in range(1, 5)], float)
X = matrix(rotvec_to_rotation([0.02, -0.03, np.pi / 2]), [0.035, -0.07, 0.055])
Y = matrix(rotvec_to_rotation([0.01, 0.02, 0.3]), [0.8, -0.2, 0.0])


def synthetic(rng, *, n=15, pixel_noise=0.0, single_axis=False):
    camera_matrix, coeffs = opencv_calibration(INTRINSICS)
    samples = []
    for index in range(n):
        axis = np.array([1.0, 0, 0]) if single_axis else rng.normal(size=3)
        rotation = rotvec_to_rotation(axis / np.linalg.norm(axis) * np.radians(rng.uniform(10, 30)))
        view = matrix(rotation, -rotation @ CORNERS.mean(0) + [*rng.uniform(-0.03, 0.03, 2), rng.uniform(0.35, 0.5)])
        flange = Y @ invert(view) @ invert(X)
        pixels, _ = cv2.projectPoints(CORNERS, cv2.Rodrigues(view[:3, :3])[0], view[:3, 3], camera_matrix, coeffs)
        pixels = pixels.reshape(-1, 2) + rng.normal(0, pixel_noise, (len(CORNERS), 2))
        _, rvec, tvec = cv2.solvePnP(CORNERS, pixels, camera_matrix, coeffs)
        samples.append(Sample(index, flange, matrix(cv2.Rodrigues(rvec)[0], tvec), list(range(len(CORNERS))), pixels))
    return samples


def test_every_closed_form_method_is_exact_on_exact_data(rng):
    candidates = closed_form(synthetic(rng))
    assert set(CLOSED_FORM_METHODS) | {"shah_robot_world"} == set(candidates)
    for name, (x, y) in candidates.items():
        assert np.linalg.norm(x[:3, 3] - X[:3, 3]) < 1e-6, name
        assert angle_between_deg(x, X) < 1e-4, name
        assert angle_between_deg(y, Y) < 1e-4, name


def test_refined_solution_with_pixel_noise_matches_its_uncertainty(rng):
    solution = solve(synthetic(rng, pixel_noise=0.2), CORNERS, INTRINSICS, monte_carlo_runs=0)
    error_mm = np.linalg.norm(solution.camera_in_flange[:3, 3] - X[:3, 3]) * 1000
    sigma_mm = np.linalg.norm(solution.metrics["camera_in_flange_std"]["translation_mm"])
    assert error_mm < 4 * sigma_mm < 1.0
    assert solution.metrics["reprojection_rmse_px"] == pytest.approx(0.2 * np.sqrt(2), rel=0.25)
    assert solution.verdict == "good" and not solution.warnings
    assert solution.metrics["leave_one_out"]["position_rms_mm"] < 0.5


def test_robust_loss_ignores_a_bad_corner(rng):
    samples = synthetic(rng, pixel_noise=0.1)
    samples[4].pixels[7] += [15.0, -12.0]  # e.g. a specular highlight on one corner
    solution = solve(samples, CORNERS, INTRINSICS, monte_carlo_runs=0, leave_one_out=False)
    assert np.linalg.norm(solution.camera_in_flange[:3, 3] - X[:3, 3]) * 1000 < 0.5


def test_single_rotation_axis_is_flagged_unobservable(rng):
    solution = solve(synthetic(rng, pixel_noise=0.2, single_axis=True), CORNERS, INTRINSICS, monte_carlo_runs=0)
    assert solution.verdict == "poor"
    assert any("nearly parallel" in warning for warning in solution.warnings)
    # The covariance exposes the unobservable translation direction honestly.
    assert max(solution.metrics["camera_in_flange_std"]["translation_mm"]) > 5


def test_robot_noise_uncertainty_is_reported_separately(rng):
    solution = solve(synthetic(rng, pixel_noise=0.2), CORNERS, INTRINSICS, leave_one_out=False, monte_carlo_runs=8)
    uncertainty = solution.metrics["camera_in_flange_uncertainty"]
    assert uncertainty["from_robot_noise"]["assumed_position_sigma_mm"] == 0.05
    combined = np.asarray(uncertainty["combined_translation_mm"])
    assert np.all(combined >= np.asarray(uncertainty["from_pixel_noise"]["translation_mm"]))


def test_too_few_samples_is_an_error(rng):
    with pytest.raises(SolveError, match="at least 6"):
        solve(synthetic(rng, n=5), CORNERS, INTRINSICS)
