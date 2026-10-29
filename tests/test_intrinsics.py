import json

import cv2
import numpy as np
import pytest
from charuco_realsense.calibration import opencv_calibration

from ur_handeye.cli import main
from ur_handeye.dataset import Session, load_result
from ur_handeye.intrinsics import fit_intrinsics, load_intrinsics
from ur_handeye.simulation import calibration_board_target
from ur_handeye.solver import Sample, SolveError, reestimate_board_poses
from ur_handeye.transforms import angle_between_deg, invert, matrix, matrix_to_pose, rotvec_to_rotation


@pytest.fixture
def camera_data():
    """Known physical geometry, observed through a lens whose factory model omits distortion."""
    target = calibration_board_target()
    truth = {"width": 1280, "height": 720, "fx": 900.0, "fy": 902.0,
             "ppx": 640.0, "ppy": 360.0, "model": "opencv", "coeffs": [0.15, -0.3, 0, 0, 0]}
    factory = {**truth, "coeffs": [0, 0, 0, 0, 0]}
    k, d = opencv_calibration(truth)
    x = matrix(rotvec_to_rotation([0.1, -0.05, 0.3]), [0.03, -0.07, 0.05])
    y = matrix(rotvec_to_rotation([0.02, 0.01, 0.2]), [0.7, -0.2, 0.0])
    rng = np.random.default_rng(25)
    samples = []
    for index in range(32):
        rotation = rotvec_to_rotation(rng.uniform([-0.5, -0.5, -0.8], [0.5, 0.5, 0.8]))
        distance = rng.uniform(0.4, 0.55)
        centre = [rng.uniform(-0.17, 0.17), rng.uniform(-0.07, 0.07), distance]
        view = matrix(rotation, np.asarray(centre) - rotation @ target.centre)
        pixels, _ = cv2.projectPoints(target.corners, cv2.Rodrigues(rotation)[0], view[:3, 3], k, d)
        samples.append(Sample(index, y @ invert(view) @ invert(x), view,
                              list(range(len(target.corners))), pixels.reshape(-1, 2)))
    return target, truth, factory, samples, x, y


def test_intrinsic_fit_recovers_known_distortion_and_camera_matrix(camera_data):
    target, truth, factory, samples, _, _ = camera_data
    model = fit_intrinsics(samples, target.corners, factory)
    assert [model[k] for k in ("fx", "fy", "ppx", "ppy")] == pytest.approx(
        [truth[k] for k in ("fx", "fy", "ppx", "ppy")], abs=0.01)
    assert model["coeffs"] == pytest.approx(truth["coeffs"], abs=0.0001)
    assert model["estimation"]["reprojection_rmse_px"] < 0.001
    assert model["estimation"]["validated"] is False


def test_recomputing_saved_poses_removes_the_wrong_camera_model_bias(camera_data):
    target, truth, factory, samples, _, _ = camera_data
    wrong = reestimate_board_poses(samples, target.corners, factory)
    fixed = reestimate_board_poses(wrong, target.corners, truth)
    assert max(angle_between_deg(a.board_in_camera, b.board_in_camera)
               for a, b in zip(samples, wrong, strict=True)) > 1
    assert max(angle_between_deg(a.board_in_camera, b.board_in_camera)
               for a, b in zip(samples, fixed, strict=True)) < 1e-4


def test_intrinsic_fit_requires_enough_diverse_views(camera_data):
    target, _, factory, samples, _, _ = camera_data
    with pytest.raises(SolveError, match="at least 8"):
        fit_intrinsics(samples[:7], target.corners, factory)
    for sample in samples:
        sample.flange = samples[0].flange.copy()
    with pytest.raises(SolveError, match="two axes"):
        fit_intrinsics(samples, target.corners, factory)


def test_offline_intrinsic_solve_verify_workflow_preserves_source_data(camera_data, tmp_path, capsys):
    target, truth, factory, samples, x, _ = camera_data

    def save_session(name, selected):
        session = Session(tmp_path / name, {"board": target.describe(), "intrinsics": factory,
                                          "tcp_offset": [0] * 6, "camera": {"device": {"serial_number": "123"}}})
        wrong = reestimate_board_poses(selected, target.corners, factory)
        for sample in wrong:
            session.add({"label": f"view{sample.index}", "status": "ok",
                         "flange_pose": matrix_to_pose(sample.flange), "observation": {
                             "charuco_ids": sample.charuco_ids, "pixels": sample.pixels.tolist(),
                             "pose": {"rotation_matrix": sample.board_in_camera[:3, :3].tolist(),
                                      "tvec_m": sample.board_in_camera[:3, 3].tolist()}}}, None)
        return session.directory

    calibration = save_session("calibration", samples[:22])
    verification = save_session("verification", samples[22:])
    originals = {p: p.read_bytes() for folder in (calibration, verification) for p in folder.iterdir() if p.is_file()}
    intrinsic_path = tmp_path / "intrinsics.json"
    assert main(["intrinsics", "--session", str(calibration), "--output", str(intrinsic_path)]) == 0
    model = load_intrinsics(intrinsic_path)
    assert model["camera_serial"] == "123"
    assert model["coeffs"] == pytest.approx(truth["coeffs"], abs=0.001)
    result_path = tmp_path / "handeye.json"
    assert main(["solve", "--session", str(calibration), "--intrinsics", str(intrinsic_path),
                 "--output", str(result_path)]) == 0
    result = load_result(result_path)
    assert result["intrinsics_source"]["board_poses_reestimated"] is True
    assert np.linalg.norm(np.array(result["camera_in_flange"]["matrix"])[:3, 3] - x[:3, 3]) < 1e-4
    # The separate recording still contains poses estimated using the wrong factory model.
    capsys.readouterr()
    assert main(["verify", "--calibration", str(result_path), "--session", str(verification)]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["board_poses_reestimated"] is True
    assert report["consistency"]["rotation_max_deg"] < 0.01
    assert not report["failures"]
    assert all(p.read_bytes() == original for p, original in originals.items())
    assert main(["solve", "--session", str(calibration), "--output", str(result_path)]) == 2
    assert main(["intrinsics", "--session", str(calibration), "--output", str(intrinsic_path)]) == 2


def test_combined_rgb_calibration_solves_with_its_model_and_preserves_evidence(collected, tmp_path, monkeypatch):
    _, directory, _ = collected
    originals = {p: p.read_bytes() for p in directory.iterdir() if p.is_file()}
    output, model_path = tmp_path / "handeye.json", tmp_path / "rgb.json"

    def connect(*args, **kwargs):
        pytest.fail("offline calibration opened robot control")

    monkeypatch.setattr("ur_handeye.robot.URRobot", connect)
    args = ["calibrate", "--session", str(directory), "--output", str(output),
            "--intrinsics-output", str(model_path)]
    assert main(args) == 0
    result = load_result(output)
    assert result["intrinsics"] == load_intrinsics(model_path)
    assert result["intrinsics_source"]["board_poses_reestimated"] is True
    assert result["verdict"] == "acceptable"  # a good training fit does not validate the estimated model
    assert any("fresh views" in warning for warning in result["warnings"])
    assert all(p.read_bytes() == data for p, data in originals.items())
    model_path.unlink()
    assert main(args) == 2  # refuse an existing hand-eye output before writing a new model
    assert not model_path.exists()


def test_override_for_a_different_image_size_is_refused(collected, tmp_path, capsys):
    _, directory, _ = collected
    model = {"width": 640, "height": 480, "fx": 450, "fy": 450,
             "ppx": 320, "ppy": 240, "model": "opencv", "coeffs": [0] * 5}
    path = tmp_path / "wrong-profile.json"
    path.write_text(json.dumps(model))
    assert main(["solve", "--session", str(directory), "--intrinsics", str(path),
                 "--output", str(tmp_path / "unused.json")]) == 2
    assert "exact resolution" in capsys.readouterr().err


@pytest.mark.parametrize("mutation", ["serial", "simulated"])
def test_incompatible_intrinsics_are_refused_before_robot_control(tmp_path, monkeypatch, mutation):
    model = {"width": 1280, "height": 720, "fx": 900, "fy": 900,
             "ppx": 640, "ppy": 360, "model": "opencv", "coeffs": [0] * 5, "camera_serial": "123"}
    if mutation == "simulated":
        model["simulated"] = True
    path = tmp_path / "rgb.json"
    path.write_text(json.dumps(model))

    def connect(*args, **kwargs):
        pytest.fail("incompatible intrinsic model opened robot control")

    monkeypatch.setattr("ur_handeye.robot.URRobot", connect)
    serial = "other-camera" if mutation == "serial" else "123"
    assert main(["collect", "--robot-ip", "192.0.2.1", "--serial", serial, "--intrinsics", str(path)]) == 2
