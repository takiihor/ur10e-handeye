"""Eye-in-hand calibration: camera pose in the flange frame, and board pose in the base frame.

The chain for every sample i is  T_base_board = T_base_flange(i) @ X @ T_cam_board(i),  with
X = T_flange_camera unknown and T_base_board = Y fixed. OpenCV's closed-form solvers give
starting values; the answer is then refined by minimising corner reprojection error over X and Y
together (the maximum-likelihood problem for pixel noise), with a robust loss so one bad corner
cannot pull the solution. Quality is reported by measures an integrator can act on: board-in-base
consistency, leave-one-out prediction error, parameter uncertainty and rotation diversity.
"""
import math
from dataclasses import dataclass, field

import cv2
import numpy as np
from charuco_realsense.calibration import opencv_calibration

from .transforms import (
    angle_between_deg,
    describe,
    invert,
    matrix,
    mean_transform,
    rotation_angle,
    rotation_to_rotvec,
    rotvec_to_rotation,
)

CLOSED_FORM_METHODS = {"tsai": cv2.CALIB_HAND_EYE_TSAI, "park": cv2.CALIB_HAND_EYE_PARK,
                       "horaud": cv2.CALIB_HAND_EYE_HORAUD, "andreff": cv2.CALIB_HAND_EYE_ANDREFF,
                       "daniilidis": cv2.CALIB_HAND_EYE_DANIILIDIS}
MIN_SAMPLES = 6


class SolveError(ValueError):
    pass


@dataclass
class Sample:
    index: int
    flange: np.ndarray  # T_base_flange (robot-reported)
    board_in_camera: np.ndarray  # T_cam_board (per-image PnP)
    charuco_ids: list[int]
    pixels: np.ndarray  # (M, 2)


@dataclass(frozen=True)
class Thresholds:
    """Acceptance limits for the verdict; tighten them to the application's error budget."""
    good_reprojection_px: float = 0.5
    max_reprojection_px: float = 1.0
    good_loo_mm: float = 1.0
    max_loo_mm: float = 2.5
    min_rotation_range_deg: float = 30.0
    min_axis_spread: float = 0.15


@dataclass
class Solution:
    camera_in_flange: np.ndarray
    board_in_base: np.ndarray
    metrics: dict
    methods: dict
    warnings: list[str] = field(default_factory=list)
    verdict: str = "unknown"


def _check(samples: list[Sample]) -> None:
    if len(samples) < MIN_SAMPLES:
        raise SolveError(f"Need at least {MIN_SAMPLES} usable samples, got {len(samples)}")


def closed_form(samples: list[Sample]) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Every OpenCV closed-form estimate as (X, Y); Y is the mean board pose implied by X."""
    _check(samples)
    rotations = [s.flange[:3, :3] for s in samples]
    translations = [s.flange[:3, 3] for s in samples]
    board_rotations = [s.board_in_camera[:3, :3] for s in samples]
    board_translations = [s.board_in_camera[:3, 3] for s in samples]
    results = {}
    for name, method in CLOSED_FORM_METHODS.items():
        try:
            rotation, translation = cv2.calibrateHandEye(rotations, translations, board_rotations,
                                                         board_translations, method=method)
        except cv2.error:
            continue
        x = matrix(rotation, translation)
        if np.isfinite(x).all() and abs(np.linalg.det(rotation) - 1) < 1e-3:
            results[name] = (x, mean_transform(board_poses(samples, x)))
    try:
        base_rotations = [s.flange[:3, :3].T for s in samples]
        base_translations = [-s.flange[:3, :3].T @ s.flange[:3, 3] for s in samples]
        r_base_world, t_base_world, r_gripper_cam, t_gripper_cam = cv2.calibrateRobotWorldHandEye(
            board_rotations, board_translations, base_rotations, base_translations,
            method=cv2.CALIB_ROBOT_WORLD_HAND_EYE_SHAH)
        x = invert(matrix(r_gripper_cam, t_gripper_cam))
        if np.isfinite(x).all():
            results["shah_robot_world"] = (x, invert(matrix(r_base_world, t_base_world)))
    except cv2.error:
        pass
    if not results:
        raise SolveError("Every closed-form hand-eye method failed; the samples are degenerate")
    return results


def board_poses(samples: list[Sample], camera_in_flange: np.ndarray) -> list[np.ndarray]:
    return [s.flange @ camera_in_flange @ s.board_in_camera for s in samples]


def consistency(samples: list[Sample], camera_in_flange: np.ndarray, board_in_base: np.ndarray,
                centre: np.ndarray) -> dict:
    """How well each sample's observation agrees with one fixed board pose, at the board centre."""
    target = board_in_base @ np.append(centre, 1.0)
    distances, angles = [], []
    for pose in board_poses(samples, camera_in_flange):
        distances.append(float(np.linalg.norm(pose @ np.append(centre, 1.0) - target) * 1000))
        angles.append(angle_between_deg(pose, board_in_base))
    return {"position_rms_mm": float(np.sqrt(np.mean(np.square(distances)))),
            "position_max_mm": float(max(distances)),
            "rotation_rms_deg": float(np.sqrt(np.mean(np.square(angles)))),
            "rotation_max_deg": float(max(angles))}


def _perturb(transform: np.ndarray, delta: np.ndarray) -> np.ndarray:
    # Rotation perturbed in the local frame (no singularity at any orientation);
    # translation perturbed additively in the parent frame, so its uncertainty reads in mm per axis.
    return matrix(transform[:3, :3] @ rotvec_to_rotation(delta[:3]), transform[:3, 3] + delta[3:])


class _Problem:
    def __init__(self, samples: list[Sample], corners: np.ndarray, camera_matrix, coeffs):
        self.samples, self.corners, self.camera_matrix, self.coeffs = samples, corners, camera_matrix, coeffs
        self.points = [corners[s.charuco_ids] for s in samples]
        self.observed = np.concatenate([s.pixels.reshape(-1, 2) for s in samples])
        self.flange_inverse = [invert(s.flange) for s in samples]

    def project(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        camera_from_flange = invert(x)
        projected = []
        for flange_inverse, points in zip(self.flange_inverse, self.points, strict=True):
            pose = camera_from_flange @ flange_inverse @ y
            camera_points = points @ pose[:3, :3].T + pose[:3, 3]
            pixels, _ = cv2.projectPoints(camera_points, np.zeros(3), np.zeros(3), self.camera_matrix, self.coeffs)
            projected.append(pixels.reshape(-1, 2))
        return np.concatenate(projected)

    def residuals(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        return (self.project(x, y) - self.observed).ravel()


def refine(problem: _Problem, x: np.ndarray, y: np.ndarray, *, huber_px: float = 1.0,
           iterations: int = 100) -> tuple[np.ndarray, np.ndarray, dict]:
    """Robust Levenberg-Marquardt on reprojection error over X and Y (12 parameters)."""
    def weights(residual):
        norms = np.repeat(np.linalg.norm(residual.reshape(-1, 2), axis=1), 2)
        return np.where(norms <= huber_px, 1.0, huber_px / np.maximum(norms, 1e-12))

    def apply(x, y, delta):
        return _perturb(x, delta[:6]), _perturb(y, delta[6:])

    def jacobian(x, y, residual):
        columns = []
        for k in range(12):
            delta = np.zeros(12)
            delta[k] = 1e-6
            columns.append((problem.residuals(*apply(x, y, delta)) - residual) / 1e-6)
        return np.column_stack(columns)

    damping = 1e-3
    residual = problem.residuals(x, y)
    w = weights(residual)
    cost = float(np.sum(w * residual ** 2))
    converged = False
    for _ in range(iterations):
        jac = jacobian(x, y, residual)
        normal = jac.T @ (w[:, None] * jac)
        gradient = jac.T @ (w * residual)
        improved = False
        while damping < 1e10:
            step = np.linalg.solve(normal + damping * np.diag(np.diag(normal) + 1e-12), -gradient)
            x_new, y_new = apply(x, y, step)
            residual_new = problem.residuals(x_new, y_new)
            w_new = weights(residual_new)
            cost_new = float(np.sum(w_new * residual_new ** 2))
            if cost_new < cost:
                x, y, residual, w = x_new, y_new, residual_new, w_new
                improved, damping = True, max(damping / 10, 1e-12)
                converged = np.linalg.norm(step) < 1e-10 or (cost - cost_new) < 1e-12 * max(cost, 1e-12)
                cost = cost_new
                break
            damping *= 10
        if not improved or converged:
            break
    jac = jacobian(x, y, residual)
    normal = jac.T @ (w[:, None] * jac)
    dof = max(1, len(residual) - 12)
    sigma2 = float(np.sum(w * residual ** 2) / dof)
    try:
        covariance = sigma2 * np.linalg.inv(normal)
        std = np.sqrt(np.clip(np.diag(covariance), 0, None))
    except np.linalg.LinAlgError:
        std = np.full(12, np.inf)
    per_point = np.linalg.norm(residual.reshape(-1, 2), axis=1)
    stats = {"reprojection_rmse_px": float(np.sqrt(np.mean(per_point ** 2))),
             "reprojection_max_px": float(per_point.max()),
             "corner_count": len(per_point),
             "camera_in_flange_std": {"rotation_deg": np.degrees(std[:3]).tolist(),
                                      "translation_mm": (std[3:6] * 1000).tolist()},
             "board_in_base_std": {"rotation_deg": np.degrees(std[6:9]).tolist(),
                                   "translation_mm": (std[9:12] * 1000).tolist()},
             "converged": bool(converged or not improved)}
    return x, y, stats


def robot_noise_std(problem: _Problem, x: np.ndarray, y: np.ndarray, *, position_sigma_m: float,
                    rotation_sigma_rad: float, runs: int = 20, seed: int = 0) -> dict:
    """Spread of X when the reported flange poses carry the given random error (Monte Carlo).

    The covariance from the fit covers pixel noise only. On a real arm the flange pose error
    usually dominates: 0.01 deg at a 0.4 m lever already moves the board 0.07 mm, and X's
    translation is amplified further when the tilts between views are small.
    """
    rng = np.random.default_rng(seed)
    offsets, angles = [], []
    for _ in range(runs):
        noisy = []
        for sample in problem.samples:
            flange = sample.flange.copy()
            flange[:3, 3] += rng.normal(0, position_sigma_m, 3)
            flange[:3, :3] = flange[:3, :3] @ rotvec_to_rotation(rng.normal(0, rotation_sigma_rad, 3))
            noisy.append(Sample(sample.index, flange, sample.board_in_camera, sample.charuco_ids, sample.pixels))
        x_run, _, _ = refine(_Problem(noisy, problem.corners, problem.camera_matrix, problem.coeffs), x, y)
        offsets.append((x_run[:3, 3] - x[:3, 3]) * 1000)
        angles.append(angle_between_deg(x_run, x))
    return {"translation_mm": np.std(offsets, axis=0, ddof=1).tolist(),
            "rotation_deg": float(np.sqrt(np.mean(np.square(angles)))),
            "assumed_position_sigma_mm": position_sigma_m * 1000,
            "assumed_rotation_sigma_deg": math.degrees(rotation_sigma_rad), "runs": runs}


def rotation_diversity(samples: list[Sample]) -> dict:
    """Hand-eye needs rotations about at least two non-parallel axes; report how far apart they are."""
    axes, angles = [], []
    for i in range(len(samples)):
        for j in range(i + 1, len(samples)):
            relative = samples[i].flange[:3, :3].T @ samples[j].flange[:3, :3]
            angle = rotation_angle(relative)
            angles.append(math.degrees(angle))
            if angle > math.radians(2):
                axes.append(rotation_to_rotvec(relative) / angle * math.sin(min(angle, math.pi / 2)))
    spread = 0.0
    if len(axes) >= 2:
        singular = np.linalg.svd(np.asarray(axes), compute_uv=False)
        spread = float(singular[1] / singular[0]) if singular[0] > 0 else 0.0
    return {"max_relative_rotation_deg": float(max(angles, default=0.0)),
            "mean_relative_rotation_deg": float(np.mean(angles)) if angles else 0.0,
            "axis_spread": spread,
            "note": "axis_spread is the ratio of the second to first singular value of the relative "
                    "rotation axes; near 0 means all rotations share one axis and X is unobservable"}


def _intrinsics_matrix(intrinsics: dict):
    size = (intrinsics["width"], intrinsics["height"]) if "width" in intrinsics else None
    return opencv_calibration(intrinsics, size)


def solve(samples: list[Sample], corners: np.ndarray, intrinsics: dict, *, thresholds: Thresholds = Thresholds(),
          leave_one_out: bool = True, robot_position_sigma_mm: float = 0.05,
          robot_rotation_sigma_deg: float = 0.01, monte_carlo_runs: int = 20) -> Solution:
    """Calibrate X (camera in flange) and Y (board in base) from at least MIN_SAMPLES views.

    The robot noise sigmas are assumptions for the uncertainty report only (defaults are of the
    order of UR10e repeatability); they do not influence the estimate itself.
    """
    _check(samples)
    camera_matrix, coeffs = _intrinsics_matrix(intrinsics)
    problem = _Problem(samples, corners, camera_matrix, coeffs)
    centre = corners.mean(axis=0)
    candidates = closed_form(samples)
    methods = {}
    for name, (x, y) in candidates.items():
        rmse = float(np.sqrt(np.mean(problem.residuals(x, y).reshape(-1, 2) ** 2) * 2))
        methods[name] = {"camera_in_flange": describe(x), "reprojection_rmse_px": rmse,
                         "consistency": consistency(samples, x, y, centre)}
    seed = min(candidates, key=lambda n: methods[n]["reprojection_rmse_px"])
    x, y, stats = refine(problem, *candidates[seed])
    for name, (cx, _) in candidates.items():
        methods[name]["difference_from_refined"] = {
            "translation_mm": float(np.linalg.norm(cx[:3, 3] - x[:3, 3]) * 1000),
            "rotation_deg": angle_between_deg(cx, x)}
    metrics = {"sample_count": len(samples), "sample_indices": [s.index for s in samples],
               "seed_method": seed, **stats,
               "per_sample_reprojection_rmse_px": per_sample_rmse(problem, x, y),
               "consistency": consistency(samples, x, y, centre),
               "rotation_diversity": rotation_diversity(samples)}
    if monte_carlo_runs > 1 and (robot_position_sigma_mm > 0 or robot_rotation_sigma_deg > 0):
        robot = robot_noise_std(problem, x, y, position_sigma_m=robot_position_sigma_mm / 1000,
                                rotation_sigma_rad=math.radians(robot_rotation_sigma_deg), runs=monte_carlo_runs)
        pixel = stats["camera_in_flange_std"]
        metrics["camera_in_flange_uncertainty"] = {
            "from_pixel_noise": pixel, "from_robot_noise": robot,
            "combined_translation_mm": np.hypot(pixel["translation_mm"], robot["translation_mm"]).tolist(),
            "note": "1-sigma; pixel part from the fit covariance, robot part from Monte Carlo with the "
                    "assumed flange pose noise. Leave-one-out error includes both, empirically."}
    if leave_one_out and len(samples) > MIN_SAMPLES:
        metrics["leave_one_out"] = _leave_one_out(samples, problem, x, y, centre)
    solution = Solution(x, y, metrics, methods)
    _verdict(solution, thresholds)
    return solution


def per_sample_rmse(problem: _Problem, x: np.ndarray, y: np.ndarray) -> list[float]:
    residual = np.linalg.norm(problem.residuals(x, y).reshape(-1, 2), axis=1)
    result, start = [], 0
    for points in problem.points:
        result.append(float(np.sqrt(np.mean(residual[start:start + len(points)] ** 2))))
        start += len(points)
    return result


def _leave_one_out(samples, problem, x, y, centre) -> dict:
    """Refit without each sample, then predict that sample: the honest out-of-sample error."""
    position_errors, pixel_errors = [], []
    for held in range(len(samples)):
        kept = [s for i, s in enumerate(samples) if i != held]
        sub = _Problem(kept, problem.corners, problem.camera_matrix, problem.coeffs)
        x_held, y_held, _ = refine(sub, x, y)
        sample = samples[held]
        predicted = sample.flange @ x_held @ sample.board_in_camera @ np.append(centre, 1.0)
        position_errors.append(float(np.linalg.norm(predicted - y_held @ np.append(centre, 1.0)) * 1000))
        single = _Problem([sample], problem.corners, problem.camera_matrix, problem.coeffs)
        pixel_errors.append(float(np.sqrt(np.mean(single.residuals(x_held, y_held).reshape(-1, 2) ** 2) * 2)))
    return {"position_rms_mm": float(np.sqrt(np.mean(np.square(position_errors)))),
            "position_max_mm": float(max(position_errors)),
            "reprojection_rms_px": float(np.sqrt(np.mean(np.square(pixel_errors)))),
            "reprojection_max_px": float(max(pixel_errors)),
            "per_sample_position_mm": position_errors}


def _verdict(solution: Solution, limits: Thresholds) -> None:
    m, warnings = solution.metrics, solution.warnings
    diversity = m["rotation_diversity"]
    if diversity["max_relative_rotation_deg"] < limits.min_rotation_range_deg:
        warnings.append(f"Rotations span only {diversity['max_relative_rotation_deg']:.0f} deg; use larger tilts "
                        f"(at least {limits.min_rotation_range_deg:.0f} deg between views)")
    if diversity["axis_spread"] < limits.min_axis_spread:
        warnings.append("Rotation axes are nearly parallel; tilt the camera about two different axes")
    rmse = m["reprojection_rmse_px"]
    loo = m.get("leave_one_out", {}).get("position_rms_mm")
    if rmse > limits.max_reprojection_px:
        warnings.append(f"Reprojection RMSE {rmse:.2f} px exceeds {limits.max_reprojection_px} px: check intrinsics, "
                        "board flatness, robot pose timing and the printed scale")
    if loo is not None and loo > limits.max_loo_mm:
        warnings.append(f"Leave-one-out error {loo:.2f} mm exceeds {limits.max_loo_mm} mm")
    worst = int(np.argmax(m["per_sample_reprojection_rmse_px"]))
    if m["per_sample_reprojection_rmse_px"][worst] > 3 * max(rmse, 0.1):
        warnings.append(f"Sample {m['sample_indices'][worst]} fits much worse than the rest; inspect it or "
                        "exclude it with --exclude")
    if loo is None:
        warnings.append("Too few samples for leave-one-out validation")
    if warnings:
        solution.verdict = "poor" if (rmse > limits.max_reprojection_px or (loo or 0) > limits.max_loo_mm
                                      or diversity["axis_spread"] < limits.min_axis_spread) else "acceptable"
    elif rmse <= limits.good_reprojection_px and loo is not None and loo <= limits.good_loo_mm:
        solution.verdict = "good"
    else:
        solution.verdict = "acceptable"
