import json

import pytest

from ur_handeye.ui_preferences import load_camera_preference, save_camera_preference


def test_camera_selection_survives_restart_and_explicit_clear(tmp_path):
    model = {"width": 1280, "height": 720, "fx": 900, "fy": 900, "ppx": 640, "ppy": 360,
             "model": "opencv", "coeffs": [0.1, 0, 0, 0, 0], "camera_serial": "123"}
    calibration = tmp_path / "rgb.json"
    calibration.write_text(json.dumps(model))
    preference = tmp_path / "config" / "camera.json"
    assert save_camera_preference(preference, str(calibration))
    assert load_camera_preference(preference) == str(calibration)
    # A partially typed/missing path must not replace the previously valid selection.
    assert not save_camera_preference(preference, str(tmp_path / "missing.json"))
    assert load_camera_preference(preference) == str(calibration)
    assert save_camera_preference(preference, "")
    assert load_camera_preference(preference) == ""


@pytest.mark.parametrize("data", [None, [], {}, {"intrinsics": 123}, {"intrinsics": "missing.json"}])
def test_corrupt_or_missing_preference_does_not_block_startup(tmp_path, data):
    preference = tmp_path / "camera.json"
    assert load_camera_preference(preference) == ""
    preference.write_text(json.dumps(data))
    assert load_camera_preference(preference) == ""


@pytest.mark.parametrize("model", [{"width": 1280, "height": 720}, {"type": "eye_in_hand"}, []])
def test_invalid_model_is_not_restored_or_saved(tmp_path, model):
    calibration = tmp_path / "bad.json"
    calibration.write_text(json.dumps(model))
    preference = tmp_path / "camera.json"
    preference.write_text(json.dumps({"intrinsics": str(calibration)}))
    assert load_camera_preference(preference) == ""
    assert not save_camera_preference(preference, str(calibration))


def test_simulated_model_is_not_remembered_for_live_capture(tmp_path):
    from ur_handeye.vision import SIMULATED_INTRINSICS

    calibration = tmp_path / "simulated.json"
    calibration.write_text(json.dumps({**SIMULATED_INTRINSICS, "simulated": True}))
    preference = tmp_path / "camera.json"
    assert not save_camera_preference(preference, str(calibration))
    assert not preference.exists()
