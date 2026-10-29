"""Re-run vision on recorded raw frames without changing the source evidence."""
import hashlib
import math
from pathlib import Path

import cv2
from charuco_realsense.camera import Frame

from .dataset import DatasetError, Session, load_session
from .vision import BoardObserver


class _RecordedCamera:
    def __init__(self, source, frames, intrinsics):
        self.source, self.frames, self.intrinsics = source, iter(frames), intrinsics

    def read(self):
        detail = next(self.frames)
        path = (self.source / detail["image"]).resolve()
        if not path.is_relative_to(self.source):
            raise DatasetError("Recorded image path escapes the source session")
        image = cv2.imread(str(path))
        if image is None:
            raise DatasetError(f"Cannot read recorded image {path}")
        if image.shape[:2] != (self.intrinsics["height"], self.intrinsics["width"]):
            raise DatasetError(f"Recorded image {path} differs from the session resolution")
        return Frame(image, self.intrinsics, detail.get("camera_timestamp_ms", 0),
                     detail.get("camera_frame_number", 0), detail.get("camera_timestamp_domain", "unknown"))


def reprocess_session(source, output, target, *, max_jitter_px=0.3, log=print):
    """Create a derived session; retain robot poses and any non-vision rejections.

    Replay every original frame, including rejected ones, with the recorded camera
    model. Intrinsic estimation and solving remain explicit, separate operations.
    """
    if not math.isfinite(max_jitter_px) or max_jitter_px <= 0:
        raise ValueError("max_jitter_px must be finite and positive")
    source = Path(source).resolve()
    metadata, records = load_session(source)
    if not records or "intrinsics" not in metadata:
        raise DatasetError("Reprocessing needs recorded samples and camera intrinsics")
    for record in records:
        frames = record.get("camera_frames", [])
        requested = record["observation"].get("frames_requested")
        if not frames or len(frames) != requested:
            raise DatasetError(f"Sample {record['index']}: complete raw frames are required for reprocessing")
    provenance = {"source_session": str(source),
                  "source_samples_sha256": hashlib.sha256((source / "samples.jsonl").read_bytes()).hexdigest(),
                  "max_jitter_px": max_jitter_px, "method": "per_corner_refinement"}
    derived = {k: v for k, v in metadata.items() if k not in ("created_utc", "software", "schema_version")}
    session = Session(output, {**derived, "reprocessing": provenance})
    accepted = 0
    for record in records:
        camera = _RecordedCamera(source, record["camera_frames"], metadata["intrinsics"])
        observation = BoardObserver(camera, target).observe(len(record["camera_frames"]))
        _, reasons = observation.usable(max_jitter_px)
        old_pose = record["observation"].get("pose") or {}
        old_vision_reasons = {"no_pose", "corner_jitter", "insufficient_frames", "corner_refinement_failed",
                              *old_pose.get("quality_reasons", [])}
        # A better image fit cannot overturn a robot-motion, profile-change or unknown fault.
        reasons += [r for r in record.get("reasons", []) if r not in old_vision_reasons and r not in reasons]
        if record["status"] != "ok" and not record.get("reasons"):
            reasons.append("source_rejected_without_reason")
        updated = {k: v for k, v in record.items() if k not in ("index", "image", "camera_frames")}
        updated.update(status="rejected" if reasons else "ok", reasons=reasons,
                       observation=observation.to_record(),
                       reprocessing={"source_index": record["index"], "source_status": record["status"],
                                     "source_reasons": record.get("reasons", [])})
        index = session.add(updated, observation.image_bgr, observation.frames)
        accepted += not reasons
        log(f"[{index:02d}] {record['label']}: {record['status']} -> {updated['status']} "
            f"({len(observation.charuco_ids)} corners, {observation.frames_used}/{observation.frames_requested} frames)"
            + (f"; {', '.join(reasons)}" if reasons else ""))
    return accepted
