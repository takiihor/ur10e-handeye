"""Binary-protocol and ownership checks for Local Control eye-to-hand capture."""
import socket
import struct
import sys
import time
import types
from dataclasses import replace

import numpy as np
import pytest

from ur_handeye.primary import PrimaryMonitor, decode_state
from ur_handeye.robot import LocalCaptureRobot, RobotError, SimulatedRobot, ur10e_forward
from ur_handeye.transforms import matrix_to_pose, pose_to_matrix

Q = tuple(SimulatedRobot.HOME)
OFFSET = (.03, -.02, .18, .1, -.2, .3)
# Deliberately differ from nominal FK: controller-calibrated Cartesian data is authoritative.
FLANGE = ur10e_forward(Q)
FLANGE[:3, 3] += [.003, -.002, .001]
TCP = tuple(matrix_to_pose(FLANGE @ pose_to_matrix(OFFSET)))


def package(kind, payload):
    return struct.pack('!IB', len(payload) + 5, kind) + payload


def state_message(timestamp=1, *, q=Q, velocity=(0.,) * 6, offset=OFFSET, tcp=TCP, future=False):
    joints = b''.join(struct.pack('!dddffffB', a, a, v, 0, 0, 0, 0, 253)
                      for a, v in zip(q, velocity, strict=True))
    parts = (package(0, struct.pack('!Q', timestamp)) + package(1, joints)
             + package(4, struct.pack('!12d', *tcp, *offset) + (b'future-field' if future else b'')))
    if future:
        parts += package(250, b'unknown-package')
    return package(16, parts)


def test_primary_state_uses_one_packet_and_calibrated_pose_with_nonzero_tcp():
    state = decode_state(state_message(future=True), time.monotonic())
    assert state.joints == Q and state.tcp_offset == OFFSET and state.tcp_pose == TCP
    np.testing.assert_allclose(state.flange, FLANGE, atol=1e-12)
    assert np.linalg.norm(state.flange[:3, 3] - ur10e_forward(Q)[:3, 3]) > .003
    # An incomplete state never borrows fields from a previous packet.
    assert decode_state(package(16, package(4, struct.pack('!12d', *TCP, *OFFSET))), 0) is None
    assert decode_state(package(20, b'version-message'), 0) is None


@pytest.mark.parametrize('message', [
    b'bad',
    package(16, struct.pack('!IB', 5000, 4)),
    package(16, package(0, b'a') + package(1, b'a') + package(4, b'a')),
    package(16, package(0, b'a') + package(0, b'a')),
    state_message(tcp=(float('nan'), *TCP[1:])),
])
def test_invalid_primary_state_is_refused(message):
    with pytest.raises(ValueError):
        decode_state(message, 0)


@pytest.fixture
def stream(monkeypatch):
    receiver, sender = socket.socketpair()
    sender.settimeout(1)

    def connect(address, timeout):
        assert address == ('robot-test', 30011)
        return receiver

    monkeypatch.setattr('ur_handeye.primary.socket.create_connection', connect)
    yield receiver, sender
    receiver.close()
    sender.close()


def test_read_only_stream_handles_fragmentation_coalescing_and_freshness(stream):
    _, sender = stream
    first = state_message(1)
    # Fragmented header/body followed by two messages in one write.
    sender.sendall(first[:2])
    sender.sendall(first[2:11])
    sender.sendall(first[11:])
    monitor = PrimaryMonitor('robot-test')
    try:
        assert monitor.snapshot().timestamp_us == 1
        after = time.monotonic()
        sender.sendall(package(20, b'version') + state_message(2) + state_message(3))
        fresh = monitor.snapshot(after=after)
        assert fresh.timestamp_us >= 2
        assert fresh.received_at > after
        # Wait for the last frame deterministically, then repeated controller timestamps
        # must not refresh it, even when fresh bytes keep arriving over the socket.
        while monitor.snapshot().timestamp_us != 3:
            monitor.snapshot(after=fresh.received_at)
        with monitor._condition:
            monitor._state = replace(monitor._state, received_at=time.monotonic() - 2)
        sender.sendall(state_message(3))
        with pytest.raises(TimeoutError, match='fresh calibrated robot pose'):
            monitor.snapshot(timeout=.03)
    finally:
        monitor.close()
    assert not monitor._thread.is_alive()
    assert sender.recv(1) == b''  # No commands of any kind were sent to the controller.


def test_primary_bad_packet_fails_without_fallback(stream):
    _, sender = stream
    sender.sendall(state_message())
    monitor = PrimaryMonitor('robot-test')
    try:
        sender.sendall(struct.pack('!I', 2))
        with pytest.raises(ConnectionError, match='Invalid primary message size'):
            monitor.snapshot(after=time.monotonic())
    finally:
        monitor.close()


def test_primary_startup_failure_closes_the_socket(stream):
    _, sender = stream
    sender.sendall(struct.pack('!I', 2))
    with pytest.raises(ConnectionError):
        PrimaryMonitor('robot-test')
    assert sender.recv(1) == b''


@pytest.fixture
def local_robot(monkeypatch):
    class Receive:
        def __init__(self, *args):
            self.q = list(Q)
            self.qd = [0.] * 6
            self.disconnected = False

        def isConnected(self): return not self.disconnected
        def getTimestamp(self): return time.monotonic()
        def getSafetyMode(self): return 1
        def getRobotMode(self): return 7
        def isProtectiveStopped(self): return False
        def isEmergencyStopped(self): return False
        def getSpeedScalingCombined(self): return 1.
        def getActualQ(self): return self.q
        def getActualQd(self): return self.qd
        def getActualTCPPose(self): return TCP
        def disconnect(self): self.disconnected = True

    class Primary:
        def __init__(self, host):
            self.state = decode_state(state_message(), time.monotonic())
            self.closed = False
            self.after = []

        def snapshot(self, *, after=None):
            assert not self.closed
            self.after.append(after)
            return self.state

        def close(self): self.closed = True

    def forbidden_control(*args, **kwargs):
        pytest.fail('Local capture constructed RTDE control')

    monkeypatch.setitem(sys.modules, 'rtde_receive', types.SimpleNamespace(RTDEReceiveInterface=Receive))
    monkeypatch.setitem(sys.modules, 'rtde_control', types.SimpleNamespace(RTDEControlInterface=forbidden_control))
    monkeypatch.setattr('ur_handeye.primary.PrimaryMonitor', Primary)
    robot = LocalCaptureRobot('robot-test')
    yield robot
    robot.close()


def test_local_robot_never_takes_control_and_does_not_use_nominal_fk(local_robot):
    robot = local_robot
    assert robot.control is None
    assert robot.tcp_offset() == list(OFFSET)
    np.testing.assert_allclose(robot.flange_pose(), FLANGE, atol=1e-12)
    for operation in (lambda: robot.move_joints(Q), lambda: robot.set_freedrive(True)):
        with pytest.raises(RobotError, match='connected read-only'):
            operation()
    primary, receive = robot.primary, robot.receive
    robot.stop()  # No remote stop is sent: the pendant owns motion.
    robot.close()
    assert primary.closed and receive.disconnected


def test_local_primary_failure_releases_receive_connection(monkeypatch, local_robot):
    receive = local_robot.receive
    monkeypatch.setattr('rtde_receive.RTDEReceiveInterface', lambda *args: receive)

    def fail(host):
        raise TimeoutError('No fresh calibrated robot pose')

    monkeypatch.setattr('ur_handeye.primary.PrimaryMonitor', fail)
    with pytest.raises(RobotError, match='30011'):
        LocalCaptureRobot('robot-test')
    assert receive.disconnected


@pytest.mark.parametrize('disturbance', [None, 'velocity', 'tcp'])
def test_local_capture_records_coherent_flange_and_rejects_transient_motion(local_robot, tmp_path, disturbance):
    from ur_handeye.collect import CollectOptions, Collector
    from ur_handeye.dataset import Session
    from ur_handeye.simulation import build_scenario
    from ur_handeye.vision import BoardObserver

    scenario = build_scenario(seed=5)
    observation = BoardObserver(scenario.camera, scenario.target).observe(1)
    robot = local_robot

    class Observer:
        def observe(self, frames, *, monitor):
            original = robot.primary.state
            if disturbance == 'velocity':
                robot.receive.qd[0] = .003
            elif disturbance == 'tcp':
                robot.primary.state = replace(original, tcp_offset=(.2, *OFFSET[1:]))
            monitor()
            robot.receive.qd = [0.] * 6
            robot.primary.state = original
            monitor()  # Returning to the same pose must not hide a transient disturbance.
            return observation

    session = Session(tmp_path / 'capture', {'calibration_type': 'eye_to_hand'})
    collector = Collector(robot, Observer(), session, CollectOptions(frames=1), log=lambda *_: None)
    record, _ = collector.capture('manual')
    np.testing.assert_allclose(pose_to_matrix(record['flange_pose']), FLANGE, atol=1e-12)
    assert record['joints'] == list(Q)
    assert record['robot_state']['tcp_offset'] == list(OFFSET)
    assert record['robot_state']['source'] == 'primary_read_only'
    assert sum(after is not None for after in robot.primary.after) == 2
    if disturbance:
        assert record['status'] == 'rejected'
        reason = 'robot_moved_during_capture' if disturbance == 'velocity' else 'tcp_changed_during_capture'
        assert reason in record['reasons']
    else:
        assert record['status'] == 'ok'


def test_eye_to_hand_cli_manual_capture_and_check_only_use_local_robot(local_robot, tmp_path, monkeypatch, capsys):
    from contextlib import nullcontext

    from ur_handeye import cli
    from ur_handeye.dataset import load_session
    from ur_handeye.simulation import build_scenario

    robot = local_robot
    scenario = build_scenario(seed=5)
    scenario.camera.metadata = {'simulated': True}
    monkeypatch.setattr('ur_handeye.robot.LocalCaptureRobot', lambda *args: robot)

    def forbidden_remote(*args, **kwargs):
        pytest.fail('Eye-to-hand Local Control path opened the Remote Control robot')

    monkeypatch.setattr('ur_handeye.robot.URRobot', forbidden_remote)
    monkeypatch.setattr(cli, 'BoardTarget', lambda *args: scenario.target)
    monkeypatch.setattr(cli, '_camera', lambda args: nullcontext(scenario.camera))
    # Check must preflight the same calibrated, read-only pose source used by capture.
    primary, receive = robot.primary, robot.receive
    monkeypatch.setattr(robot, 'close', lambda: None)
    assert cli.main(['check', '--robot-ip', 'robot-test', '--calibration-type', 'eye_to_hand', '--no-camera']) == 0
    assert 'Calibrated flange pose available' in capsys.readouterr().out
    args = cli.parser().parse_args(['collect', '--robot-ip', 'robot-test', '--mode', 'manual',
                                   '--calibration-type', 'eye_to_hand', '--frames', '1',
                                   '--output', str(tmp_path / 'capture')])
    replies = iter(['', 'q'])
    assert cli.cmd_collect(args, read_line=lambda _: next(replies)) == 1  # One view is intentionally too few.
    metadata, records = load_session(tmp_path / 'capture')
    assert metadata['pose_source'] == 'primary_read_only'
    assert records[0]['status'] == 'ok'
    np.testing.assert_allclose(pose_to_matrix(records[0]['flange_pose']), FLANGE, atol=1e-12)
    assert not primary.closed and not receive.disconnected
    # Close through the class because this test replaced the instance cleanup method.
    LocalCaptureRobot.close(robot)


def test_eye_to_hand_remote_freedrive_is_rejected_before_connecting(monkeypatch):
    from ur_handeye.cli import main

    def forbidden(*args, **kwargs):
        pytest.fail('Invalid eye-to-hand options connected to the robot')

    monkeypatch.setattr('ur_handeye.robot.LocalCaptureRobot', forbidden)
    monkeypatch.setattr('ur_handeye.robot.URRobot', forbidden)
    assert main(['collect', '--robot-ip', 'robot-test', '--calibration-type', 'eye_to_hand',
                 '--mode', 'manual', '--freedrive']) == 2


@pytest.mark.parametrize('fault', ['disconnect', 'backwards_timestamp'])
def test_primary_stream_fault_cannot_reuse_an_old_pose(stream, fault):
    _, sender = stream
    sender.sendall(state_message(3))
    monitor = PrimaryMonitor('robot-test')
    try:
        after = time.monotonic()
        if fault == 'disconnect':
            sender.close()
        else:
            sender.sendall(state_message(2))
        with pytest.raises(ConnectionError, match=r'disconnected|backwards'):
            monitor.snapshot(after=after)
    finally:
        monitor.close()


def test_primary_startup_interruption_cleans_up_reader_and_socket(stream, monkeypatch):
    _, sender = stream

    def interrupt(*args, **kwargs):
        raise KeyboardInterrupt

    monkeypatch.setattr(PrimaryMonitor, 'snapshot', interrupt)
    with pytest.raises(KeyboardInterrupt):
        PrimaryMonitor('robot-test')
    assert sender.recv(1) == b''
