"""Board observations from the wrist camera, built on charuco-realsense detection and pose.

A calibration sample is taken with the robot at rest, so several frames are averaged per
corner ID: this lowers pixel noise and exposes residual vibration (corner jitter) that would
otherwise corrupt the sample silently. A camera simulator renders the real generated board
through a distorted pinhole model so the full pipeline can run without hardware.
"""
import math
import time
from collections import defaultdict
from dataclasses import dataclass, field

import cv2
import numpy as np
from charuco_realsense.calibration import opencv_calibration
from charuco_realsense.camera import Frame
from charuco_realsense.detect import CharucoDetector, Detection
from charuco_realsense.pose import PoseOptions, estimate_pose

from .transforms import invert, matrix

#: Calibration-grade gates: stricter than live monitoring, since one bad sample biases the solution.
CALIBRATION_POSE_OPTIONS = PoseOptions(refine=True, min_corners=10, min_coverage=0.4, max_rmse_px=1.0,
                                       max_error_px=3.0, min_inlier_fraction=0.9)


class VisionError(RuntimeError):
    pass


def refine_corners(image_gray: np.ndarray, pixels: np.ndarray) -> np.ndarray | None:
    """Refine intersections, marking individual unreliable corners as NaN.

    Match charuco-realsense's window and 1 px displacement guard, including the
    correction of OpenCV 4.11's half-pixel offset. One unstable intersection must
    not discard all other corners in the frame. The observer removes NaN rows
    before averaging, then applies the usual temporal, coverage and pose gates.
    """
    points = np.asarray(pixels, np.float32).reshape(-1, 1, 2)
    if len(points) < 2 or not np.isfinite(points).all():
        return None
    spacing = math.sqrt(max(cv2.contourArea(cv2.convexHull(points)), 1.0) / len(points))
    half = int(min(8, max(2, spacing * 0.25)))
    try:
        refined = cv2.cornerSubPix(image_gray, points.copy(), (half, half), (-1, -1),
                                  (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 100, 1e-4))
    except cv2.error:
        return None
    refined = refined.reshape(-1, 2)
    valid = (np.isfinite(refined).all(axis=1)
             & (np.linalg.norm(refined - points.reshape(-1, 2), axis=1) <= 1.0))
    refined[~valid] = np.nan
    return refined


class BoardTarget:
    """A ChArUco board reconstructed and verified from a charuco-generator manifest."""

    def __init__(self, manifest_path, name: str = "calibration"):
        from charuco_generator.board import reconstruct_board, sha256
        from charuco_generator.validation import load_manifest
        try:
            manifest = load_manifest(manifest_path)
            self.board = reconstruct_board(manifest, name)
        except (ValueError, KeyError, OSError) as exc:
            raise VisionError(f"Cannot load board {name!r} from {manifest_path}: {exc}") from exc
        self.name = name
        self.manifest_path = str(manifest_path)
        self.manifest_sha256 = sha256(manifest_path)
        self.record = next(b for b in manifest["boards"] if b["name"] == name)
        self.border_bits = int(self.record["geometry"]["border_bits"])

    @classmethod
    def from_board(cls, board, name: str, border_bits: int = 1) -> "BoardTarget":
        """Wrap an already constructed board (simulation and tests)."""
        target = cls.__new__(cls)
        target.board, target.name, target.border_bits = board, name, border_bits
        target.manifest_path = target.manifest_sha256 = None
        target.record = {"name": name, "marker_ids": board.getIds().ravel().tolist()}
        return target

    @property
    def corners(self) -> np.ndarray:
        """Internal chessboard corners in board coordinates (metres, Z=0), indexed by ChArUco ID."""
        return np.asarray(self.board.getChessboardCorners(), float)

    @property
    def centre(self) -> np.ndarray:
        return self.corners.mean(axis=0)

    def describe(self) -> dict:
        columns, rows = self.board.getChessboardSize()
        return {"name": self.name, "manifest": self.manifest_path, "manifest_sha256": self.manifest_sha256,
                "squares": [columns, rows], "square_length_m": float(self.board.getSquareLength()),
                "marker_length_m": float(self.board.getMarkerLength()),
                "marker_ids": self.board.getIds().ravel().tolist(), "corner_count": len(self.corners)}


@dataclass
class Observation:
    charuco_ids: list[int]
    pixels: np.ndarray  # (M, 2) mean corner positions
    jitter_px: float | None  # RMS per-corner standard deviation across frames
    frames_used: int
    frames_requested: int
    pose: dict | None  # charuco-realsense pose result for the averaged corners
    intrinsics: dict
    image_bgr: np.ndarray | None
    frame_rejections: dict[str, int] = field(default_factory=dict)
    frames: list = field(default_factory=list)

    @property
    def board_in_camera(self) -> np.ndarray | None:
        if self.pose is None:
            return None
        return matrix(np.asarray(self.pose["rotation_matrix"]), self.pose["tvec_m"])

    def usable(self, max_jitter_px: float) -> tuple[bool, list[str]]:
        reasons = []
        if self.pose is None:
            reasons.append("no_pose")
        elif not self.pose["quality_passed"]:
            reasons.extend(self.pose["quality_reasons"])
        if self.jitter_px is not None and self.jitter_px > max_jitter_px:
            reasons.append("corner_jitter")
        if self.pose is not None and self.frames_used < (
                max(2, math.ceil(self.frames_requested / 2)) if self.frames_requested > 1 else 1):
            reasons.append("insufficient_frames")
        if ((self.pose is None or "insufficient_frames" in reasons)
                and self.frame_rejections.get("corner_refinement_failed")):
            reasons.append("corner_refinement_failed")
        return not reasons, reasons

    def to_record(self) -> dict:
        pose = self.pose
        return {"charuco_ids": self.charuco_ids, "pixels": self.pixels.tolist(), "jitter_px": self.jitter_px,
                "frames_used": self.frames_used, "frames_requested": self.frames_requested,
                "frame_rejections": self.frame_rejections,
                "pose": None if pose is None else {
                    key: pose[key] for key in ("rvec", "tvec_m", "rotation_matrix", "quality_passed",
                                               "quality_reasons", "reprojection_rmse_px", "reprojection_max_px",
                                               "board_coverage_fraction", "image_coverage_fraction",
                                               "inlier_count", "corner_count")}}



class BoardObserver:
    def __init__(self, camera, target: BoardTarget, pose_options: PoseOptions = CALIBRATION_POSE_OPTIONS,
                 intrinsics: dict | None = None, *, refine: bool = True):
        self.camera, self.target, self.pose_options = camera, target, pose_options
        self.intrinsics_override = intrinsics
        self.refine = refine
        self.detector = CharucoDetector(target.board, target.border_bits, refine=False)

    def observe(self, frames: int = 5) -> Observation:
        if frames < 1:
            raise ValueError("frames must be at least 1")
        samples: dict[int, list[np.ndarray]] = defaultdict(list)
        image, intrinsics, used, size = None, None, 0, None
        first_intrinsics, first_size = None, None
        frame_numbers = set()
        captured = self.captured_frames = []
        for _ in range(frames):
            frame: Frame = self.camera.read()
            detail = {"camera_frame_number": frame.frame_number, "camera_timestamp_ms": frame.timestamp_ms,
                      "camera_timestamp_domain": frame.timestamp_domain, "skip_reason": None}
            captured.append((frame, detail))
            intrinsics = self.intrinsics_override or frame.intrinsics
            gray = cv2.cvtColor(frame.bgr, cv2.COLOR_BGR2GRAY)
            size = (gray.shape[1], gray.shape[0])
            if first_intrinsics is None:
                first_intrinsics, first_size = dict(intrinsics), size
            elif intrinsics != first_intrinsics or size != first_size:
                raise VisionError("Camera intrinsics or image size changed during capture")
            if frame.frame_number:
                if frame.frame_number in frame_numbers:
                    detail["skip_reason"] = "duplicate_camera_frame"
                    continue
                frame_numbers.add(frame.frame_number)
            detection = self.detector.detect(gray, intrinsics)
            detail.update(marker_ids=list(detection.marker_ids), marker_corners=detection.marker_corners.tolist(),
                          charuco_ids=list(detection.charuco_ids),
                          charuco_corners=detection.charuco_corners.reshape(-1, 2).tolist())
            if image is None:
                image = frame.bgr
            if len(detection.charuco_ids) != len(set(detection.charuco_ids)):
                detail["skip_reason"] = "duplicate_charuco_ids"
                continue  # ambiguous association; never average it
            if len(detection.marker_ids) != len(set(detection.marker_ids)):
                detail["skip_reason"] = "duplicate_marker_ids"
                continue
            if detection.charuco_ids:
                corners = detection.charuco_corners.reshape(-1, 2)
                corner_ids = np.asarray(detection.charuco_ids)
                if self.refine:
                    corners = refine_corners(gray, corners)
                    if corners is None:
                        detail["skip_reason"] = "corner_refinement_failed"
                        continue
                    valid = np.isfinite(corners).all(axis=1)
                    detail["refinement_rejected_ids"] = corner_ids[~valid].tolist()
                    if not valid.any():
                        detail["skip_reason"] = "corner_refinement_failed"
                        continue
                    corner_ids, corners = corner_ids[valid], corners[valid]
                detail["used_charuco_ids"] = corner_ids.tolist()
                detail["used_charuco_corners"] = corners.tolist()
                used += 1
                for corner_id, pixel in zip(corner_ids.tolist(), corners, strict=True):
                    samples[corner_id].append(pixel.astype(float))
            else:
                detail["skip_reason"] = "no_charuco_corners"
        # Keep corners seen in most frames, so one flickering edge corner cannot shift the mean.
        needed = max(1, math.ceil(frames / 2))
        ids = sorted(i for i, points in samples.items() if len(points) >= needed)
        pixels = np.array([np.mean(samples[i], axis=0) for i in ids]).reshape(-1, 2)
        variances = [np.var(samples[i], axis=0, ddof=1).sum() for i in ids if len(samples[i]) > 1]
        jitter = float(math.sqrt(np.mean(variances))) if variances else None
        pose = None
        if len(ids) >= 4 and size is not None:
            detection = Detection(size, (), np.empty((0, 4, 2), np.float32), tuple(ids),
                                  pixels.reshape(-1, 1, 2).astype(np.float32), 0)
            pose = estimate_pose(self.target.board, detection, intrinsics, self.pose_options)
        rejections = {}
        for _, detail in captured:
            if detail["skip_reason"]:
                reason = detail["skip_reason"]
                rejections[reason] = rejections.get(reason, 0) + 1
        return Observation(ids, pixels, jitter, used, frames, pose, intrinsics or {}, image, rejections, captured)


class SimulatedCamera:
    """Renders the board as seen from a camera rigidly mounted on a simulated robot's flange."""

    def __init__(self, robot, target: BoardTarget, camera_in_flange: np.ndarray, board_in_base: np.ndarray,
                 intrinsics: dict, *, px_per_mm: float = 4.0, noise_level: float = 2.0, blur_sigma_px: float = 0.7,
                 seed: int = 0):
        self.robot, self.target = robot, target
        self.camera_in_flange, self.board_in_base = camera_in_flange, board_in_base
        self.intrinsics = dict(intrinsics)
        self.noise_level = noise_level
        self._rng = np.random.default_rng(seed)
        self.frame_number = 0
        board = target.board
        columns, rows = board.getChessboardSize()
        square_px = round(board.getSquareLength() * 1000 * px_per_mm)
        margin = square_px  # quiet zone around the printed pattern
        size = (columns * square_px + 2 * margin, rows * square_px + 2 * margin)
        texture = board.generateImage(size, marginSize=margin, borderBits=target.border_bits)
        if blur_sigma_px > 0:  # optical blur also avoids aliasing when the texture is minified
            texture = cv2.GaussianBlur(texture, (0, 0), blur_sigma_px * square_px / 60)
        self._texture = texture
        scale = square_px / board.getSquareLength()
        # Board metres -> texture pixel centres.
        self._texture_from_board = np.array([[scale, 0, margin - 0.5], [0, scale, margin - 0.5], [0, 0, 1]])
        width, height = int(intrinsics["width"]), int(intrinsics["height"])
        self._size = (width, height)
        camera_matrix, coeffs = opencv_calibration(self.intrinsics, self._size)
        grid = np.stack(np.meshgrid(np.arange(width), np.arange(height)), axis=-1).reshape(-1, 1, 2).astype(np.float64)
        # More iterations than undistortPoints' default 5, which leaves error in the image corners.
        normalized = cv2.undistortPointsIter(grid, camera_matrix, coeffs, None, None,
                                             (cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 30, 1e-9))
        self._rays = np.concatenate([normalized.reshape(-1, 2), np.ones((width * height, 1))], axis=1).T

    def board_in_camera(self) -> np.ndarray:
        return invert(self.camera_in_flange) @ invert(self.robot.true_flange_pose()) @ self.board_in_base

    def read(self) -> Frame:
        pose = self.board_in_camera()
        width, height = self._size
        homography = np.column_stack([pose[:3, 0], pose[:3, 1], pose[:3, 3]])  # board plane -> camera rays
        image = np.full((height, width), 150, np.uint8)
        if np.linalg.cond(homography) < 1e8:
            board_points = np.linalg.solve(homography, self._rays)
            # Rays meeting the plane behind the camera have a negative plane scale.
            valid = board_points[2] > 1e-9
            texture = self._texture_from_board @ (board_points / np.where(valid, board_points[2], 1.0))
            map_x = np.where(valid, texture[0], -1e4).reshape(height, width).astype(np.float32)
            map_y = np.where(valid, texture[1], -1e4).reshape(height, width).astype(np.float32)
            image = cv2.remap(self._texture, map_x, map_y, cv2.INTER_LINEAR,
                              borderMode=cv2.BORDER_CONSTANT, borderValue=150)
        if self.noise_level > 0:
            noisy = image.astype(np.float32) + self._rng.normal(0, self.noise_level, image.shape)
            image = np.clip(noisy, 0, 255).astype(np.uint8)
        self.frame_number += 1
        return Frame(bgr=cv2.cvtColor(image, cv2.COLOR_GRAY2BGR), intrinsics=dict(self.intrinsics),
                     timestamp_ms=time.monotonic() * 1000, frame_number=self.frame_number,
                     timestamp_domain="simulated")

    def close(self) -> None:
        pass


#: Typical D435i colour intrinsics at 1280x720 with mild Brown-Conrady distortion (simulation default).
SIMULATED_INTRINSICS = {"width": 1280, "height": 720, "fx": 910.0, "fy": 908.0, "ppx": 645.0, "ppy": 362.0,
                        "model": "brown_conrady", "coeffs": [0.08, -0.15, 0.0005, -0.0004, 0.05]}
