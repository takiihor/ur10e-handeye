"""Calibration sessions on disk, and the calibration result file other programs consume.

A session is everything needed to re-solve offline: ``session.json`` (robot, TCP offset, camera,
intrinsics, board), ``samples.jsonl`` (one line per captured view, flushed as it is taken) and the
raw images. Solving never needs the robot or camera again, so methods and thresholds can be
compared later on exactly the same data.
"""
import json
import os
import time
from datetime import UTC, datetime
from pathlib import Path

import cv2
import numpy as np

from . import __version__
from .solver import Sample, Solution
from .transforms import describe, invert, matrix, pose_to_matrix

SESSION_FILE, SAMPLES_FILE, IMAGE_DIR = "session.json", "samples.jsonl", "images"
RESULT_SCHEMA_VERSION = 1


class DatasetError(ValueError):
    pass


def write_json(path: Path, data: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


class Session:
    """An append-only recording directory; existing sessions are never overwritten."""

    def __init__(self, directory, metadata: dict):
        self.directory = Path(directory)
        if self.directory.exists() and any(self.directory.iterdir()):
            raise DatasetError(f"{self.directory} already exists and is not empty; choose a new session folder")
        (self.directory / IMAGE_DIR).mkdir(parents=True, exist_ok=True)
        self.metadata = {"schema_version": 1, "software": f"ur10e-handeye {__version__}",
                         "created_utc": datetime.now(UTC).isoformat(), **metadata}
        write_json(self.directory / SESSION_FILE, self.metadata)
        self.count = 0

    def update_metadata(self, **values) -> None:
        self.metadata.update(values)
        write_json(self.directory / SESSION_FILE, self.metadata)

    def add(self, record: dict, image: np.ndarray | None) -> int:
        index = self.count
        record = {"index": index, "time_unix": time.time(), **record}
        if image is not None:
            name = f"{IMAGE_DIR}/{index:03d}.png"
            if not cv2.imwrite(str(self.directory / name), image):
                raise DatasetError(f"Cannot write {name}")
            record["image"] = name
        with open(self.directory / SAMPLES_FILE, "a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, allow_nan=False) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        self.count += 1
        return index


def load_session(directory) -> tuple[dict, list[dict]]:
    directory = Path(directory)
    try:
        metadata = json.loads((directory / SESSION_FILE).read_text(encoding="utf-8"))
        lines = (directory / SAMPLES_FILE).read_text(encoding="utf-8").splitlines()
    except (OSError, json.JSONDecodeError) as exc:
        raise DatasetError(f"Cannot read session {directory}: {exc}") from exc
    records = []
    for number, line in enumerate(lines, 1):
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            # A power cut can truncate only the line being written; earlier lines are complete.
            if number != len(lines):
                raise DatasetError(f"{directory / SAMPLES_FILE}: corrupt line {number}") from None
    return metadata, records


def samples_from_records(records: list[dict], exclude=()) -> list[Sample]:
    excluded = set(exclude)
    samples = []
    for record in records:
        if record.get("status") != "ok" or record["index"] in excluded:
            continue
        pose = record["observation"]["pose"]
        samples.append(Sample(record["index"], pose_to_matrix(record["flange_pose"]),
                              matrix(np.asarray(pose["rotation_matrix"]), pose["tvec_m"]),
                              list(record["observation"]["charuco_ids"]),
                              np.asarray(record["observation"]["pixels"], float).reshape(-1, 2)))
    return samples


def build_result(solution: Solution, metadata: dict, session_dir) -> dict:
    """The calibration file contract shared with the pick-and-place software (schema 1)."""
    tcp_offset = metadata["tcp_offset"]
    camera_in_tcp = invert(pose_to_matrix(tcp_offset)) @ solution.camera_in_flange
    return {
        "schema_version": RESULT_SCHEMA_VERSION,
        "type": "eye_in_hand",
        "software": f"ur10e-handeye {__version__}",
        "created_utc": datetime.now(UTC).isoformat(),
        "camera_in_flange": describe(solution.camera_in_flange),
        "camera_in_tcp": {**describe(camera_in_tcp), "tcp_offset": list(tcp_offset)},
        "board_in_base": describe(solution.board_in_base),
        "intrinsics": metadata["intrinsics"],
        "camera": metadata.get("camera", {}),
        "robot": metadata.get("robot", {}),
        "board": metadata.get("board", {}),
        "session": str(session_dir),
        "verdict": solution.verdict,
        "warnings": solution.warnings,
        "metrics": solution.metrics,
        "methods": solution.methods,
        "conventions": {
            "units": "metres; rotation vectors in radians; *_mm and *_deg fields are for reading",
            "camera_in_flange": "maps camera optical coordinates (X right, Y down, Z forward) into the flange "
                                "(tool0) frame. As a PolyScope TCP it puts the TCP at the camera optical centre",
            "camera_in_tcp": "the same camera relative to the TCP that was active during calibration "
                             "(tcp_offset); valid only while that TCP is active",
            "board_in_base": "calibration board origin (outer top-left pattern corner, X right, Y down, Z into "
                             "the board) in the robot base frame; valid only while the board is not moved",
            "point_in_base": "p_base = T_base_flange @ camera_in_flange.matrix @ p_camera",
        },
    }


def load_result(path) -> dict:
    try:
        result = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise DatasetError(f"Cannot read calibration {path}: {exc}") from exc
    if not isinstance(result, dict) or result.get("schema_version") != RESULT_SCHEMA_VERSION \
            or result.get("type") != "eye_in_hand":
        raise DatasetError(f"{path} is not a schema {RESULT_SCHEMA_VERSION} eye-in-hand calibration")
    transform = np.asarray(result["camera_in_flange"]["matrix"], float)
    rotation = transform[:3, :3]
    if transform.shape != (4, 4) or not np.allclose(rotation @ rotation.T, np.eye(3), atol=1e-6) \
            or abs(np.linalg.det(rotation) - 1) > 1e-6:
        raise DatasetError(f"{path}: camera_in_flange is not a rigid transform")
    return result
