"""Fixed-camera geometry, persisted contracts and UI routing."""
import json

import cv2
import numpy as np
import pytest

from ur_handeye.cli import main
from ur_handeye.dataset import DatasetError, Session, build_result, load_result
from ur_handeye.intrinsics import load_intrinsics
from ur_handeye.simulation import calibration_board_target
from ur_handeye.solver import Sample, solve
from ur_handeye.transforms import angle_between_deg, describe, invert, matrix, matrix_to_pose, rotvec_to_rotation
from ur_handeye.ui import build_command
from ur_handeye.vision import SIMULATED_INTRINSICS

CAMERA = matrix(rotvec_to_rotation([0.2, -0.1, 0.5]), [0.6, -0.2, 0.3])
MOUNT = matrix(rotvec_to_rotation([0.1, 0.2, -0.3]), [0.025, -0.02, 0.08])


def fixed_views(seed, count=14, noise=0.05, single_axis=False):
    rng = np.random.default_rng(seed)
    target = calibration_board_target()
    corners = target.corners
    k = np.array([[SIMULATED_INTRINSICS['fx'], 0, SIMULATED_INTRINSICS['ppx']],
                  [0, SIMULATED_INTRINSICS['fy'], SIMULATED_INTRINSICS['ppy']], [0, 0, 1.]])
    samples = []
    for i in range(count):
        vector = rng.uniform(-0.4, 0.4, 3)
        if single_axis:
            vector[1:] = 0
        rotation = rotvec_to_rotation(vector)
        view = matrix(rotation, -rotation @ target.centre + [*rng.uniform(-0.04, 0.04, 2), rng.uniform(.35, .55)])
        # Physical closure: base_from_flange @ flange_from_board = base_from_camera @ camera_from_board.
        flange = CAMERA @ view @ invert(MOUNT)
        pixels, _ = cv2.projectPoints(corners, cv2.Rodrigues(rotation)[0], view[:3, 3],
                                     k, np.array(SIMULATED_INTRINSICS['coeffs']))
        pixels = pixels.reshape(-1, 2) + rng.normal(0, noise, (len(corners), 2))
        ok, rvec, tvec = cv2.solvePnP(corners, pixels, k, np.array(SIMULATED_INTRINSICS['coeffs']))
        assert ok
        samples.append(Sample(i, flange, matrix(cv2.Rodrigues(rvec)[0], tvec), list(range(len(corners))), pixels))
    return target, samples


def record(directory, seed):
    target, samples = fixed_views(seed)
    metadata = {"calibration_type": "eye_to_hand", "intrinsics": SIMULATED_INTRINSICS,
                "tcp_offset": [0.] * 6, "board": target.describe(), "simulated": True}
    session = Session(directory, metadata)
    for s in samples:
        session.add({"status": "ok", "label": f"manual{s.index}", "flange_pose": matrix_to_pose(s.flange),
                     "joints": [0.] * 6, "observation": {
                         "pose": {"rotation_matrix": s.board_in_camera[:3, :3].tolist(),
                                  "tvec_m": s.board_in_camera[:3, 3].tolist()},
                         "charuco_ids": s.charuco_ids, "pixels": s.pixels.tolist()}}, None)
    return target, samples, metadata


def test_fixed_camera_solution_recovers_camera_and_unknown_board_mount(tmp_path):
    target, samples, metadata = record(tmp_path / 'data', 12)
    solution = solve(samples, target.corners, SIMULATED_INTRINSICS, calibration_type='eye_to_hand',
                     monte_carlo_runs=3)
    result = build_result(solution, metadata, tmp_path / 'data')
    assert solution.verdict == 'good'
    assert result['type'] == 'eye_to_hand'
    assert 'camera_in_flange' not in result and 'camera_in_tcp' not in result and 'board_in_base' not in result
    for key, truth in [('camera_in_base', CAMERA), ('board_in_flange', MOUNT)]:
        estimated = np.array(result[key]['matrix'])
        assert np.linalg.norm(estimated[:3, 3] - truth[:3, 3]) < .001
        assert angle_between_deg(estimated, truth) < .1
    assert 'camera_in_base_uncertainty' in result['metrics']
    path = tmp_path / 'result.json'
    path.write_text(json.dumps(result))
    assert load_result(path)['type'] == 'eye_to_hand'
    assert load_intrinsics(path)['fx'] == SIMULATED_INTRINSICS['fx']
    result['camera_in_base']['matrix'][3][0] = .1
    path.write_text(json.dumps(result))
    with pytest.raises(DatasetError, match='rigid transform'):
        load_result(path)


def test_cli_solve_and_independent_fixed_camera_verification(tmp_path, capsys):
    calibration_session = tmp_path / 'cal'
    verification_session = tmp_path / 'fresh'
    record(calibration_session, 12)
    record(verification_session, 44)
    path = tmp_path / 'calibration.json'
    assert main(['solve', '--session', str(calibration_session), '--output', str(path)]) == 0
    assert 'Camera in base' in capsys.readouterr().out
    argv = ['verify', '--calibration', str(path), '--session', str(verification_session)]
    assert main(argv) == 0
    report = json.loads(capsys.readouterr().out)
    assert report['passed']
    result = json.loads(path.read_text())
    wrong = CAMERA.copy()
    wrong[:3, 3] += [.01, 0, 0]
    result['camera_in_base'] = describe(wrong)
    path.write_text(json.dumps(result))
    assert main(argv) == 1
    assert not json.loads(capsys.readouterr().out)['passed']
    metadata_path = verification_session / 'session.json'
    metadata = json.loads(metadata_path.read_text())
    metadata['calibration_type'] = 'eye_in_hand'
    metadata_path.write_text(json.dumps(metadata))
    assert main(argv) == 2
    assert 'calibration type differs' in capsys.readouterr().err


def test_fixed_camera_auto_and_mismatched_replay_refused_before_hardware(tmp_path, capsys):
    assert main(['collect', '--robot-ip', 'invalid', '--calibration-type', 'eye_to_hand']) == 2
    assert 'automatic wrist-camera planning' in capsys.readouterr().err
    record(tmp_path / 'recording', 12)
    assert main(['collect', '--robot-ip', 'invalid', '--mode', 'replay',
                 '--replay', str(tmp_path / 'recording')]) == 2
    assert 'calibration type differs' in capsys.readouterr().err


def test_ui_routes_fixed_camera_collection_to_manual(tmp_path):
    manifest = tmp_path / 'board.json'
    manifest.write_text('{}')
    settings = dict(calibration_type='eye_to_hand', session=str(tmp_path / 'new'), robot_ip='invalid',
                    manifest=str(manifest), board='calibration', serial='', speed='0.1', views='14')
    command = build_command('collect', settings)
    assert command[command.index('--mode') + 1] == 'manual'
    assert command[command.index('--calibration-type') + 1] == 'eye_to_hand'
    assert '--freedrive' not in command and '--yes' not in command
    with pytest.raises(ValueError, match='no automatic plan'):
        build_command('plan', settings)


def test_manual_fixed_camera_capture_records_setup_without_commanding_motion(tmp_path, monkeypatch):
    from contextlib import nullcontext

    from ur_handeye import cli
    from ur_handeye.dataset import load_session
    from ur_handeye.simulation import build_scenario

    scenario = build_scenario(seed=5)
    scenario.camera.metadata = {'simulated': True}
    monkeypatch.setattr('ur_handeye.robot.LocalCaptureRobot', lambda *args: scenario.robot)
    monkeypatch.setattr(cli, 'BoardTarget', lambda *args: scenario.target)
    monkeypatch.setattr(cli, '_camera', lambda args: nullcontext(scenario.camera))
    args = cli.parser().parse_args(['collect', '--robot-ip', 'simulated', '--mode', 'manual',
                                   '--calibration-type', 'eye_to_hand', '--output', str(tmp_path / 'capture')])
    replies = iter(['', 'q'])
    # One view is saved but is insufficient for solving.
    assert cli.cmd_collect(args, read_line=lambda prompt: next(replies)) == 1
    metadata, records = load_session(tmp_path / 'capture')
    assert metadata['calibration_type'] == 'eye_to_hand'
    assert metadata['mode'] == 'manual'
    assert len(records) == 1 and records[0]['status'] == 'ok'
    assert scenario.robot.moves == 0


def test_fixed_camera_locate_uses_base_transform_without_tcp_or_control(tmp_path, monkeypatch, capsys):
    from contextlib import nullcontext

    from ur_handeye import cli
    from ur_handeye.simulation import build_scenario
    from ur_handeye.solver import Solution

    scenario = build_scenario(seed=5)
    result = build_result(Solution(CAMERA, MOUNT, {}, {}, verdict='good', calibration_type='eye_to_hand'),
                          {'tcp_offset': [0.] * 6, 'intrinsics': SIMULATED_INTRINSICS}, 'offline')
    path = tmp_path / 'calibration.json'
    path.write_text(json.dumps(result))

    def connect(host, *, control):
        assert not control
        return scenario.robot

    monkeypatch.setattr('ur_handeye.robot.URRobot', connect)
    monkeypatch.setattr(cli, 'BoardTarget', lambda *args: scenario.target)
    monkeypatch.setattr(cli, '_camera', lambda args: nullcontext(scenario.camera))
    assert main(['locate', '--robot-ip', 'fake', '--calibration', str(path), '--use-fk']) == 0
    report = json.loads(capsys.readouterr().out)
    expected = CAMERA @ scenario.camera.board_in_camera()
    np.testing.assert_allclose(report['board_in_base']['matrix'], expected, atol=.002)
    assert report['frame_source'] == 'fixed camera in base'
    assert scenario.robot.moves == 0


def test_rendered_fixed_camera_manual_capture_solve_and_fresh_verification(tmp_path, monkeypatch, capsys):
    """Exercise real image rendering/detection and reachable UR kinematics, not supplied PnP corners."""
    from contextlib import nullcontext

    from ur_handeye import cli
    from ur_handeye.robot import SimulatedRobot
    from ur_handeye.vision import SimulatedCamera

    robot = SimulatedRobot(tcp_offset=[.03, -.02, .18, .1, -.2, .3])
    target = calibration_board_target()
    initial_view = matrix(rotvec_to_rotation([.03, -.04, .05]), -target.centre + [0, 0, .45])
    base_camera = robot.true_flange_pose() @ MOUNT @ invert(initial_view)

    class FixedCamera(SimulatedCamera):
        def board_in_camera(self):
            return invert(base_camera) @ robot.true_flange_pose() @ MOUNT

    camera = FixedCamera(robot, target, np.eye(4), np.eye(4), SIMULATED_INTRINSICS, noise_level=1, seed=15)
    camera.metadata = {'simulated': True}
    rng = np.random.default_rng(15)
    targets = []
    for _ in range(22):
        rotation = rotvec_to_rotation(rng.uniform(-.4, .4, 3))
        view = matrix(rotation, -rotation @ target.centre +
                      [*rng.uniform(-.025, .025, 2), rng.uniform(.4, .5)])
        targets.append(robot.plan(base_camera @ view @ invert(MOUNT), robot.joints()))
    monkeypatch.setattr('ur_handeye.robot.LocalCaptureRobot', lambda *args: robot)
    monkeypatch.setattr(cli, 'BoardTarget', lambda *args: target)
    monkeypatch.setattr(cli, '_camera', lambda args: nullcontext(camera))

    for name, poses in [('cal', targets[:14]), ('fresh', targets[14:])]:
        pending = iter(poses)

        def jog(prompt, pending=pending):
            joints = next(pending, None)
            if joints is None:
                return 'q'
            robot._q = joints  # Pendant positioning; Collector must issue no motion commands.
            return ''

        args = cli.parser().parse_args(['collect', '--robot-ip', 'simulated', '--mode', 'manual',
                                       '--calibration-type', 'eye_to_hand', '--frames', '2',
                                       '--output', str(tmp_path / name)])
        assert cli.cmd_collect(args, read_line=jog) == 0
    assert robot.moves == 0
    path = tmp_path / 'calibration.json'
    assert main(['solve', '--session', str(tmp_path / 'cal'), '--output', str(path)]) == 0
    result = load_result(path)
    assert result['verdict'] == 'good'
    for key, truth in [('camera_in_base', base_camera), ('board_in_flange', MOUNT)]:
        estimated = np.array(result[key]['matrix'])
        assert np.linalg.norm(estimated[:3, 3] - truth[:3, 3]) < .0005
        assert angle_between_deg(estimated, truth) < .1
    capsys.readouterr()
    assert main(['verify', '--calibration', str(path), '--session', str(tmp_path / 'fresh')]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report['passed'] and report['samples'] == 8
    assert report['consistency']['position_max_mm'] < .5
    assert report['consistency']['rotation_max_deg'] < .2


@pytest.mark.parametrize('seed', [0, 5, 9])
def test_all_fixed_camera_closed_form_methods_match_exact_geometry(seed):
    from ur_handeye.solver import CLOSED_FORM_METHODS, calibration_samples, closed_form

    _, samples = fixed_views(seed, noise=0)
    candidates = closed_form(calibration_samples(samples, 'eye_to_hand'))
    assert set(candidates) == set(CLOSED_FORM_METHODS) | {'shah_robot_world'}
    for camera, board in candidates.values():
        np.testing.assert_allclose(camera, CAMERA, atol=1e-7)
        np.testing.assert_allclose(board, MOUNT, atol=1e-7)


@pytest.mark.parametrize('changed', ['camera_in_base', 'board_in_flange'])
def test_fixed_camera_verification_rejects_rotation_errors(tmp_path, capsys, changed):
    from ur_handeye.solver import Solution

    _, _, metadata = record(tmp_path / 'fresh', 44)
    result = build_result(Solution(CAMERA, MOUNT, {}, {}, verdict='good', calibration_type='eye_to_hand'),
                          metadata, tmp_path / 'cal')
    wrong = np.array(result[changed]['matrix'])
    wrong[:3, :3] = wrong[:3, :3] @ rotvec_to_rotation([0, 0, np.radians(3)])
    result[changed] = describe(wrong)
    path = tmp_path / 'result.json'
    path.write_text(json.dumps(result))
    assert main(['verify', '--calibration', str(path), '--session', str(tmp_path / 'fresh')]) == 1
    report = json.loads(capsys.readouterr().out)
    assert not report['passed']
    assert any('rotation' in row['failed_limits'] for row in report['failures'])


def test_fixed_camera_verification_recomputes_pnp_with_calibrated_intrinsics(tmp_path, capsys):
    from ur_handeye.dataset import load_session
    from ur_handeye.solver import Solution

    directory = tmp_path / 'fresh'
    _, _, metadata = record(directory, 44)
    result = build_result(Solution(CAMERA, MOUNT, {}, {}, verdict='good', calibration_type='eye_to_hand'),
                          metadata, tmp_path / 'cal')
    recorded, samples = load_session(directory)
    recorded['intrinsics'] = {**SIMULATED_INTRINSICS, 'fx': SIMULATED_INTRINSICS['fx'] * 1.05}
    (directory / 'session.json').write_text(json.dumps(recorded))
    # Stale poses must be discarded: they cannot agree with the saved pixels and calibrated camera model.
    for sample in samples:
        sample['observation']['pose']['tvec_m'][0] += .03
    (directory / 'samples.jsonl').write_text(''.join(json.dumps(s) + '\n' for s in samples))
    path = tmp_path / 'result.json'
    path.write_text(json.dumps(result))
    assert main(['verify', '--calibration', str(path), '--session', str(directory)]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report['passed'] and report['board_poses_reestimated']
    assert report['consistency']['position_max_mm'] < .5



def test_fixed_camera_single_axis_motion_is_rejected_as_unobservable():
    target, samples = fixed_views(15, noise=0, single_axis=True)
    result = solve(samples, target.corners, SIMULATED_INTRINSICS, calibration_type='eye_to_hand',
                   monte_carlo_runs=0)
    assert result.verdict == 'poor'
    assert any('nearly parallel' in warning and 'board' in warning for warning in result.warnings)
