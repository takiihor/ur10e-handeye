import math
import sys
import time
import types

import numpy as np
import pytest

from ur_handeye.robot import (
    MotionLimits,
    RobotError,
    SimulatedRobot,
    URRobot,
    flange_from_tcp,
    ur10e_forward,
    ur10e_inverse,
)
from ur_handeye.transforms import matrix, matrix_to_pose, pose_to_matrix, rotvec_to_rotation

HOME = SimulatedRobot.HOME


def test_ur10e_kinematics_match_nominal_reach_and_invert():
    np.testing.assert_allclose(ur10e_forward([0] * 6)[:3, 3], [-1.18425, -0.2907, 0.06085], atol=1e-9)
    target = ur10e_forward(HOME) @ matrix(rotvec_to_rotation([0.2, -0.3, 0.4]), [0.05, -0.03, 0.04])
    q = ur10e_inverse(target, HOME)
    np.testing.assert_allclose(ur10e_forward(q), target, atol=1e-9)


def test_simulated_robot_refuses_configuration_flips():
    robot = SimulatedRobot()
    flipped = ur10e_forward(HOME) @ matrix(rotvec_to_rotation([0, 0, math.radians(120)]), [0, 0, 0])
    with pytest.raises(RobotError, match="joint by"):
        robot.plan(flipped, HOME)


@pytest.mark.parametrize("kwargs", [{"joint_speed_rad_s": 2.0}, {"joint_accel_rad_s2": 3.0},
                                    {"max_travel_m": 0}, {"settle_s": -1}])
def test_motion_limits_are_bounded(kwargs):
    with pytest.raises(ValueError):
        MotionLimits(**kwargs)


class FakeControl:
    def __init__(self, host, frequency):
        self.calls = []
        self.tcp = [0, 0, 0.15, 0, 0, 0]
        self.ik_ok = self.pose_ok = self.joints_ok = self.move_ok = True
        self.ik_offset = 0.0
        self.q = list(HOME)
        self.progress = -1

    def isConnected(self):
        return True

    def getTCPOffset(self):
        return self.tcp

    def getForwardKinematics(self, q, tcp=None):
        assert tcp is not None and any(tcp), "zero or omitted TCP triggers ur_rtde 1.6.5 FK bugs"
        return matrix_to_pose(ur10e_forward(q) @ pose_to_matrix(tcp))

    def isPoseWithinSafetyLimits(self, pose):
        return self.pose_ok

    def getInverseKinematicsHasSolution(self, pose, q_near):
        return self.ik_ok

    def getInverseKinematics(self, pose, q_near):
        flange = pose_to_matrix(pose) @ np.linalg.inv(pose_to_matrix(self.tcp))
        q = ur10e_inverse(flange, q_near)
        return [q[0] + self.ik_offset, *q[1:]]

    def isJointsWithinSafetyLimits(self, q):
        return self.joints_ok

    def getAsyncOperationProgress(self):
        return self.progress

    def moveJ(self, q, speed, accel, asynchronous=False):
        assert asynchronous is True
        self.calls.append(("moveJ", speed, accel))
        if self.move_ok:
            self.q = list(q)
            self.progress = -2 if self.progress == -1 else -1
        return self.move_ok

    def isSteady(self):
        return True

    def teachMode(self):
        self.calls.append("teach")
        return True

    def endTeachMode(self):
        self.calls.append("endteach")
        return True

    def stopJ(self, a):
        self.calls.append("stopJ")
        return True

    def stopScript(self):
        self.calls.append("stopScript")
        raise RuntimeError("socket closed")

    def disconnect(self):
        self.calls.append("disconnect")


class FakeReceive:
    def __init__(self, host, frequency):
        self.safety, self.mode, self.protective = 1, 7, False
        self.control = None

    def isConnected(self):
        return True

    def getTimestamp(self):
        return time.monotonic()

    def getSafetyMode(self):
        return self.safety

    def getRobotMode(self):
        return self.mode

    def isProtectiveStopped(self):
        return self.protective

    def isEmergencyStopped(self):
        return False

    def getSpeedScalingCombined(self):
        return 1.0

    def getActualQ(self):
        return self.control.q

    def getActualTCPPose(self):
        return matrix_to_pose(ur10e_forward(self.control.q) @ pose_to_matrix(self.control.tcp))

    def disconnect(self):
        pass


@pytest.fixture
def robot(monkeypatch):
    monkeypatch.setitem(sys.modules, "rtde_control", types.SimpleNamespace(RTDEControlInterface=FakeControl))
    monkeypatch.setitem(sys.modules, "rtde_receive", types.SimpleNamespace(RTDEReceiveInterface=FakeReceive))
    instance = URRobot("192.0.2.10", MotionLimits(settle_s=0))
    instance.receive.control = instance.control
    return instance


def test_ur_robot_plans_in_flange_frame_independent_of_tcp(robot):
    target = ur10e_forward(HOME) @ matrix(rotvec_to_rotation([0.1, 0, 0]), [0.02, 0, 0])
    q = robot.plan(target, HOME)
    np.testing.assert_allclose(ur10e_forward(q), target, atol=1e-9)
    np.testing.assert_allclose(robot.flange_pose(), ur10e_forward(HOME), atol=1e-12)
    np.testing.assert_allclose(flange_from_tcp(robot.tcp_pose(), robot.tcp_offset()), ur10e_forward(HOME), atol=1e-12)
    robot.move_joints(q)
    assert robot.control.calls == [("moveJ", 0.35, 0.6)]


@pytest.mark.parametrize(("attribute", "message"), [("pose_ok", "safety configuration"), ("ik_ok", "out of reach"),
                                                    ("joints_ok", "joint safety limits")])
def test_ur_robot_refuses_unsafe_plans(robot, attribute, message):
    setattr(robot.control, attribute, False)
    with pytest.raises(RobotError, match=message):
        robot.plan(ur10e_forward(HOME), HOME)


def test_ur_robot_cross_checks_ik_with_forward_kinematics(robot):
    robot.control.ik_offset = 0.01  # IK disagrees with FK by ~0.6 deg on the base joint
    with pytest.raises(RobotError, match="misses the target"):
        robot.plan(ur10e_forward(HOME), HOME)


@pytest.mark.parametrize(("field", "value", "message"), [("safety", 3, "protective stop"),
                                                         ("protective", True, "protective"),
                                                         ("mode", 5, "not RUNNING")])
def test_ur_robot_will_not_move_unless_ready(robot, field, value, message):
    setattr(robot.receive, field, value)
    with pytest.raises(RobotError, match=message):
        robot.move_joints(HOME)
    assert not any(call[0] == "moveJ" for call in robot.control.calls if isinstance(call, tuple))


def test_rejected_move_and_large_jump_raise(robot):
    robot.control.move_ok = False
    with pytest.raises(RobotError, match="moveJ was rejected"):
        robot.move_joints(HOME)
    far = list(HOME)
    far[5] += math.radians(100)
    with pytest.raises(RobotError, match="joint by"):
        robot.move_joints(far)


@pytest.mark.parametrize("q", [[0] * 5, [0, 0, 0, 0, 0, float("nan")]])
def test_invalid_joint_targets_never_move(robot, q):
    with pytest.raises(RobotError, match="six finite"):
        robot.move_joints(q)
    assert robot.control.calls == []


def test_move_rechecks_controller_limits(robot):
    robot.control.joints_ok = False
    with pytest.raises(RobotError, match="joint safety limits"):
        robot.move_joints(HOME)
    assert robot.control.calls == []


def test_interrupt_during_async_move_sends_stop(robot, monkeypatch):
    def progress():
        if robot.control.calls:
            raise KeyboardInterrupt
        return -1
    monkeypatch.setattr(robot.control, "getAsyncOperationProgress", progress)
    with pytest.raises(KeyboardInterrupt):
        robot.move_joints(HOME)
    assert robot.control.calls[-1] == "stopJ"


def test_close_runs_every_step_and_reports_failures(robot):
    control = robot.control
    with pytest.raises(RobotError, match="socket closed"):
        robot.close()
    assert control.calls[-2:] == ["stopScript", "disconnect"]


def test_read_only_connection_never_takes_control(monkeypatch):
    monkeypatch.setitem(sys.modules, "rtde_receive", types.SimpleNamespace(RTDEReceiveInterface=FakeReceive))
    monkeypatch.setitem(sys.modules, "rtde_control", None)  # importing control would fail
    robot = URRobot("192.0.2.10", control=False)
    with pytest.raises(RobotError, match="read-only"):
        robot.tcp_offset()
