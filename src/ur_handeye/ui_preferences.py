"""Remember an explicitly selected camera model; never restore motion approvals."""
import json
import os
from pathlib import Path

from .dataset import write_json
from .intrinsics import load_intrinsics


def preference_path():
    return Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config")) / "ur10e-handeye" / "camera.json"


def _camera_path(value):
    if not isinstance(value, str) or not value.strip():
        return ""
    path = Path(value).expanduser().resolve()
    model = load_intrinsics(path)
    return "" if model.get("simulated") else str(path)


def load_camera_preference(path):
    try:
        data = json.loads(Path(path).read_text())
        return _camera_path(data.get("intrinsics")) if isinstance(data, dict) else ""
    except (ValueError, OSError, KeyError, TypeError):
        return ""  # missing, corrupt or removed camera models do not prevent startup


def save_camera_preference(path, value):
    if not isinstance(value, str):
        return False
    try:
        selected = _camera_path(value) if value.strip() else ""
        if value.strip() and not selected:
            return False  # do not replace a real model with simulation data or a partial typed path
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        write_json(path, {"intrinsics": selected})
    except (ValueError, OSError, KeyError, TypeError):
        return False
    return True
