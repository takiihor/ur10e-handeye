import json
import shutil

import numpy as np
import pytest

from ur_handeye.cli import main
from ur_handeye.dataset import DatasetError, load_result, load_session
from ur_handeye.transforms import angle_between_deg, describe, invert, pose_to_matrix

CONTRACT_KEYS = {"schema_version", "type", "camera_in_flange", "camera_in_tcp", "board_in_base", "intrinsics",
                 "verdict", "metrics"}


def test_solve_writes_the_shared_contract_and_matches_truth(collected, tmp_path, capsys):
    scenario, directory, _ = collected
    output = tmp_path / "calibration.json"
    assert main(["solve", "--session", str(directory), "--output", str(output)]) == 0
    assert "Verdict: GOOD" in capsys.readouterr().out
    result = load_result(output)
    assert CONTRACT_KEYS <= set(result) and result["type"] == "eye_in_hand"
    x = np.asarray(result["camera_in_flange"]["matrix"])
    assert np.linalg.norm(x[:3, 3] - scenario.camera_in_flange[:3, 3]) < 1e-3
    assert angle_between_deg(x, scenario.camera_in_flange) < 0.2
    np.testing.assert_allclose(pose_to_matrix(result["camera_in_flange"]["pose_ur"]), x, atol=1e-9)
    # camera_in_tcp is the same camera seen from the TCP active at calibration.
    tcp = pose_to_matrix(result["camera_in_tcp"]["tcp_offset"])
    np.testing.assert_allclose(tcp @ np.asarray(result["camera_in_tcp"]["matrix"]), x, atol=1e-12)
    assert result["metrics"]["leave_one_out"]["position_rms_mm"] < 0.5


def test_verify_passes_for_the_true_calibration_and_fails_for_a_wrong_one(collected, tmp_path, capsys):
    _, directory, _ = collected
    calibration = tmp_path / "calibration.json"
    assert main(["solve", "--session", str(directory), "--output", str(calibration), "--no-loo"]) == 0
    assert main(["verify", "--calibration", str(calibration), "--session", str(directory)]) == 0
    result = json.loads(calibration.read_text())
    x = np.asarray(result["camera_in_flange"]["matrix"])
    x[:3, 3] += [0.004, 0, 0]  # a 4 mm error in the camera mount
    result["camera_in_flange"] = describe(x)
    offset = pose_to_matrix(result["camera_in_tcp"]["tcp_offset"])
    result["camera_in_tcp"].update(describe(invert(offset) @ x))
    calibration.write_text(json.dumps(result))
    capsys.readouterr()
    assert main(["verify", "--calibration", str(calibration), "--session", str(directory)]) == 1
    assert json.loads(capsys.readouterr().out)["passed"] is False


def test_exclude_and_rejection_of_bad_files(collected, tmp_path):
    _, directory, _ = collected
    assert main(["solve", "--session", str(directory), "--output", str(tmp_path / "a.json"),
                 "--exclude", "0", "1", "--no-loo"]) == 0
    solved = json.loads((tmp_path / "a.json").read_text())
    assert solved["metrics"]["sample_count"] == len(load_session(directory)[1]) - 2
    (tmp_path / "bad.json").write_text(json.dumps({"schema_version": 2, "type": "eye_in_hand"}))
    with pytest.raises(DatasetError):
        load_result(tmp_path / "bad.json")
    assert main(["verify", "--calibration", str(tmp_path / "bad.json"), "--session", str(directory)]) == 2


def test_truncated_last_line_is_tolerated(collected, tmp_path):
    _, directory, _ = collected
    copy = tmp_path / "copy"
    shutil.copytree(directory, copy)
    with open(copy / "samples.jsonl", "a", encoding="utf-8") as stream:
        stream.write('{"index": 99, "status": "o')  # power cut mid-write
    _, records = load_session(copy)
    assert len(records) == len(load_session(directory)[1])


def test_simulate_command_end_to_end(tmp_path, capsys):
    assert main(["simulate", "--output", str(tmp_path / "sim"), "--seed", "11", "--views", "10"]) == 0
    out = capsys.readouterr().out
    assert "Ground truth error" in out
    truth = json.loads((tmp_path / "sim" / "calibration.json").read_text())["simulation_truth"]
    assert truth["camera_translation_error_mm"] < 1.0


def test_hardware_commands_need_a_robot_address():
    with pytest.raises(SystemExit):
        main(["collect"])


@pytest.mark.parametrize("flags", [["--mode", "manual", "--dry-run", "--freedrive"],
                                   ["--mode", "replay"], ["--freedrive"], ["--frames", "0"]])
def test_incompatible_collection_options_fail_before_connecting(monkeypatch, flags):
    def connect(*args, **kwargs):
        pytest.fail("invalid options opened RTDE control")
    monkeypatch.setattr("ur_handeye.robot.URRobot", connect)
    assert main(["collect", "--robot-ip", "192.0.2.1", *flags]) == 2


def test_default_manifest_points_at_the_sibling_generator():
    from pathlib import Path

    from ur_handeye.cli import DEFAULT_MANIFEST
    repo = Path(__file__).resolve().parents[1]
    expected = repo.parent / "charuco-generator" / "output" / "calibration" / "board_manifest.json"
    assert Path(DEFAULT_MANIFEST) == expected


def test_simulation_with_a_real_generator_manifest(tmp_path, capsys):
    calibration = pytest.importorskip("charuco_generator.calibration")
    from charuco_generator.generator import generate

    from ur_handeye.vision import BoardTarget
    config = calibration.calibration_config(calibration.CalibrationTarget(dpi=150), str(tmp_path / "board"))
    generate(config)
    manifest = tmp_path / "board" / "board_manifest.json"
    target = BoardTarget(manifest)
    assert target.describe()["squares"] == [5, 7] and len(target.corners) == 24
    assert main(["simulate", "--manifest", str(manifest), "--output", str(tmp_path / "sim"), "--views", "8"]) == 0
    session = json.loads((tmp_path / "sim" / "session.json").read_text())
    assert session["board"]["manifest_sha256"] == target.manifest_sha256


@pytest.mark.parametrize('mutation', ['nan_translation', 'bottom_row', 'redundant_pose'])
def test_calibration_loader_rejects_corrupt_transforms(tmp_path, mutation):
    from ur_handeye.dataset import build_result
    from ur_handeye.solver import Solution
    from ur_handeye.vision import SIMULATED_INTRINSICS
    result = build_result(Solution(np.eye(4), np.eye(4), {}, {}, verdict='good'),
                          {'tcp_offset': [0]*6, 'intrinsics': SIMULATED_INTRINSICS}, 'offline')
    if mutation == 'nan_translation':
        result['camera_in_flange']['matrix'][0][3] = float('nan')
    elif mutation == 'bottom_row':
        result['camera_in_flange']['matrix'][3][0] = 0.1
    else:
        result['camera_in_flange']['pose_ur'][0] = 0.01
    path = tmp_path / 'bad.json'
    path.write_text(json.dumps(result))
    with pytest.raises(DatasetError, match='Invalid calibration'):
        load_result(path)


def test_complete_corrupt_session_line_is_not_treated_as_power_cut(collected, tmp_path):
    _, original, _ = collected
    target = tmp_path / 'copy'
    shutil.copytree(original, target)
    with (target / 'samples.jsonl').open('a') as stream:
        stream.write('{bad}\n')
    with pytest.raises(DatasetError, match='corrupt line'):
        load_session(target)


def test_changed_manifest_cannot_silently_resolve_a_session(tmp_path):
    from charuco_generator.calibration import CalibrationTarget, calibration_config
    from charuco_generator.generator import generate

    from ur_handeye.cli import _target_from_metadata
    from ur_handeye.vision import BoardTarget
    generate(calibration_config(CalibrationTarget(dpi=150), str(tmp_path / 'board')))
    path = tmp_path / 'board' / 'board_manifest.json'
    metadata = {'board': BoardTarget(path).describe()}
    path.write_text(path.read_text() + ' ')
    with pytest.raises(DatasetError, match='changed since collection'):
        _target_from_metadata(metadata)


def test_refit_verification_refuses_a_single_view(collected, tmp_path):
    from ur_handeye.dataset import build_result
    from ur_handeye.solver import Solution
    scenario, directory, _ = collected
    metadata, records = load_session(directory)
    result = build_result(Solution(scenario.camera_in_flange, scenario.board_in_base, {}, {}, verdict="good"),
                          metadata, directory)
    path = tmp_path / "calibration.json"
    path.write_text(json.dumps(result))
    session = tmp_path / "one-view"
    session.mkdir()
    (session / "session.json").write_text(json.dumps(metadata))
    (session / "samples.jsonl").write_text(json.dumps(records[0]) + "\n")
    assert main(["verify", "--refit-board", "--calibration", str(path), "--session", str(session)]) == 2


@pytest.mark.parametrize('active', [None, [0, 0, 0.2, 0, 0, 0]])
def test_read_only_locate_requires_the_recorded_tcp_before_connect(tmp_path, monkeypatch, active):
    from ur_handeye.dataset import build_result
    from ur_handeye.solver import Solution
    from ur_handeye.vision import SIMULATED_INTRINSICS
    result = build_result(Solution(np.eye(4), np.eye(4), {}, {}, verdict='good'),
                          {'tcp_offset': [0]*6, 'intrinsics': SIMULATED_INTRINSICS}, 'offline')
    path = tmp_path / 'calibration.json'
    path.write_text(json.dumps(result))
    def connect(*args, **kwargs):
        pytest.fail('invalid TCP declaration opened hardware')
    monkeypatch.setattr('ur_handeye.robot.URRobot', connect)
    flags = [] if active is None else ['--active-tcp', *map(str, active)]
    assert main(['locate', '--robot-ip', '192.0.2.1', '--calibration', str(path), *flags]) == 2
