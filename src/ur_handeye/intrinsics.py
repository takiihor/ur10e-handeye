"""Explicit offline RGB intrinsic calibration from recorded ChArUco corners."""
import json
from pathlib import Path

import cv2
import numpy as np
from charuco_realsense.calibration import opencv_calibration

from .solver import SolveError, _Problem, rotation_diversity


def load_intrinsics(path) -> dict:
    """Accept a camera calibration or the camera model embedded in a hand-eye result."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(data, dict) and data.get("type") == "eye_in_hand":
        from .dataset import load_result
        result = load_result(path)
        data = dict(result["intrinsics"])
        serial = result.get("camera", {}).get("device", {}).get("serial_number")
        if serial:
            data["camera_serial"] = serial
        if (result.get("simulation_truth") or result.get("camera", {}).get("simulated")
                or result.get("robot", {}).get("host") == "simulated"):
            data["simulated"] = True
    if not isinstance(data, dict) or "width" not in data or "height" not in data:
        raise ValueError("Intrinsics must specify the calibrated image width and height")
    if "estimation" in data and not isinstance(data["estimation"], dict):
        raise ValueError("Intrinsic estimation metadata must be an object")
    opencv_calibration(data)
    return data


def fit_intrinsics(samples, corners, recorded: dict, *, fix_camera_matrix=False) -> dict:
    """Fit focal lengths, principal point and two radial terms; never change source observations.

    Keep k3 and tangential distortion fixed to avoid an unnecessarily flexible model on a small
    hand-eye dataset. The output is an estimate to validate on fresh, broadly distributed views.
    """
    if len(samples) < 8:
        raise SolveError("Intrinsic calibration needs at least 8 usable views; "
                         "capture varied tilts and image positions")
    k, distortion = opencv_calibration(recorded)
    width, height = int(recorded["width"]), int(recorded["height"])
    problem = _Problem(samples, corners, k, distortion)
    diversity = rotation_diversity(samples)
    if diversity["max_relative_rotation_deg"] < 30 or diversity["axis_spread"] < 0.15:
        raise SolveError("Intrinsic calibration needs varied tilts about two axes; repeated or roll-only views "
                         "cannot constrain the camera model")
    # Discard old distortion when fitting a new forward OpenCV model.
    flags = cv2.CALIB_USE_INTRINSIC_GUESS | cv2.CALIB_ZERO_TANGENT_DIST | cv2.CALIB_FIX_K3
    if fix_camera_matrix:
        flags |= cv2.CALIB_FIX_FOCAL_LENGTH | cv2.CALIB_FIX_PRINCIPAL_POINT
    try:
        rmse, k, distortion, rotations, translations = cv2.calibrateCamera(
            [np.asarray(p, np.float32) for p in problem.points],
            [np.asarray(s.pixels, np.float32) for s in samples], (width, height), k, np.zeros(5), flags=flags,
            criteria=(cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 100, 1e-9))
    except cv2.error as exc:
        raise SolveError(f"Intrinsic calibration failed: {exc}") from exc
    if (not np.isfinite(k).all() or not np.isfinite(distortion).all() or not np.isfinite(rmse)
            or min(k[0, 0], k[1, 1]) <= 0 or not 0 <= k[0, 2] < width or not 0 <= k[1, 2] < height):
        raise SolveError("Intrinsic fit is invalid; collect more varied views across the image")
    per_view = []
    for sample, points, rotation, translation in zip(samples, problem.points, rotations, translations, strict=True):
        projected, _ = cv2.projectPoints(points, rotation, translation, k, distortion)
        error = projected.reshape(-1, 2) - sample.pixels
        per_view.append({"index": sample.index,
                         "reprojection_rmse_px": float(np.sqrt(np.mean(np.sum(error ** 2, axis=1))))})
    pixels = np.concatenate([s.pixels for s in samples]).astype(np.float32)
    coverage = float(cv2.contourArea(cv2.convexHull(pixels)) / (width * height))
    centres = np.array([s.pixels.mean(axis=0) for s in samples])
    centre_span = np.ptp(centres, axis=0) / [width, height]
    warnings = []
    if coverage < 0.25 or min(centre_span) < 0.25:
        warnings.append("Limited image coverage: collect additional views near all image edges and corners, "
                        "with different tilts and distances; this estimate may not generalize across the image")
    if rmse > 0.5:
        warnings.append(f"Intrinsic reprojection RMSE is {rmse:.2f} px; inspect board flatness, printed dimensions "
                        "and corner quality")
    return {"width": width, "height": height, "fx": float(k[0, 0]), "fy": float(k[1, 1]),
            "ppx": float(k[0, 2]), "ppy": float(k[1, 2]), "model": "opencv",
            "coeffs": distortion.ravel().tolist(),
            "estimation": {"validated": False, "method": "charuco_radial_k1_k2",
                           "camera_matrix_fixed": fix_camera_matrix, "samples": len(samples),
                           "sample_indices": [s.index for s in samples], "reprojection_rmse_px": float(rmse),
                           "image_coverage_fraction": coverage, "view_centre_span_fraction": centre_span.tolist(),
                           "per_sample": per_view, "warnings": warnings,
                           "note": "Training image fit does not establish physical accuracy. Validate with fresh "
                                   "views before using this estimated camera model on hardware."}}
