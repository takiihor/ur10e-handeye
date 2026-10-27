"""Robot access over RTDE (ur_rtde), plus a kinematic UR10e simulator with the same interface.

Everything is planned and recorded in the **flange** frame (tool0), never the active TCP: the
hand-eye result must not change meaning when someone edits the TCP in PolyScope. Every motion is
a joint move to an IK solution chosen near the current joints and checked against the controller's
safety limits and a per-move joint-step limit before it is sent.
"""
import math
import time
from dataclasses import dataclass
from typing import Protocol

import numpy as np

from .transforms import invert, matrix_to_pose, pose_to_matrix

# RTDE enumerations (UR RTDE guide).
ROBOT_MODE_RUNNING = 7
SAFETY_NORMAL, SAFETY_REDUCED = 1, 2
SAFETY_MODES = {1: "normal", 2: "reduced", 3: "protective stop", 4: "recovery", 5: "safeguard stop",
                6: "system emergency stop", 7: "robot emergency stop", 8: "violation", 9: "fault",
                10: "validate joint id", 11: "undefined", 12: "automatic mode safeguard stop",
                13: "system three position enabling stop"}


class RobotError(RuntimeError):
    """The robot refused, cannot reach, or is not in a state to perform a request."""


@dataclass(frozen=True)
class MotionLimits:
    """Conservative defaults for a camera on a UR10e working near a fixed calibration board."""
    joint_speed_rad_s: float = 0.35
    joint_accel_rad_s2: float = 0.6
    #: Largest change of any joint in one move; stops IK from choosing a different configuration.
    max_joint_step_rad: float = math.radians(75)
    #: Largest flange travel from the start pose; calibration views stay near the board.
    max_travel_m: float = 0.45
    #: Time to wait after the controller reports steady, so the camera mount stops vibrating.
    settle_s: float = 0.8
    steady_timeout_s: float = 10.0

    def __post_init__(self):
        for name in ("joint_speed_rad_s", "joint_accel_rad_s2", "max_joint_step_rad", "max_travel_m",
                     "steady_timeout_s"):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if not math.isfinite(self.settle_s) or self.settle_s < 0:
            raise ValueError("settle_s must be finite and nonnegative")
        if self.joint_speed_rad_s > 1.05 or self.joint_accel_rad_s2 > 1.5:
            raise ValueError("Calibration moves are limited to 1.05 rad/s and 1.5 rad/s^2")


class Robot(Protocol):
    limits: MotionLimits

    def joints(self) -> list[float]: ...
    def flange_pose(self) -> np.ndarray: ...
    def tcp_offset(self) -> list[float]: ...
    def plan(self, flange_target: np.ndarray, q_near: list[float]) -> list[float]: ...
    def move_joints(self, q: list[float]) -> None: ...
    def check_ready(self) -> dict: ...
    def set_freedrive(self, enabled: bool) -> None: ...
    def stop(self) -> None: ...
    def close(self) -> None: ...


def _joint_step(q: list[float], q_near: list[float], limits: MotionLimits) -> None:
    step = max(abs(a - b) for a, b in zip(q, q_near, strict=True))
    if step > limits.max_joint_step_rad:
        raise RobotError(f"Planned move changes a joint by {math.degrees(step):.0f} deg (limit "
                         f"{math.degrees(limits.max_joint_step_rad):.0f}); the IK solution switches configuration")


def _flange_matches(actual: np.ndarray, target: np.ndarray, what: str) -> None:
    position = np.linalg.norm(actual[:3, 3] - target[:3, 3])
    angle = math.degrees(math.acos(min(1.0, max(-1.0, (np.trace(actual[:3, :3].T @ target[:3, :3]) - 1) / 2))))
    if position > 1e-3 or angle > 0.1:
        raise RobotError(f"{what}: IK solution misses the target by {position * 1000:.2f} mm / {angle:.2f} deg")


class URRobot:
    """A UR e-Series arm through ur_rtde. Control is optional so read-only tools never take over."""

    def __init__(self, host: str, limits: MotionLimits = MotionLimits(), *, control: bool = True,
                 frequency: float = 125.0):
        try:
            from rtde_receive import RTDEReceiveInterface
            if control:
                from rtde_control import RTDEControlInterface
        except ImportError as exc:
            raise RobotError("ur_rtde is not installed; run: pip install ur_rtde") from exc
        self.host, self.limits = host, limits
        self.receive = self.control = None
        try:
            self.receive = RTDEReceiveInterface(host, frequency)
            # Connecting control uploads the RTDE control script and stops the running program.
            self.control = RTDEControlInterface(host, frequency) if control else None
        except RuntimeError as exc:
            self.close()
            raise RobotError(f"Cannot connect to the robot at {host} over RTDE: {exc}. Check the IP, that "
                             "the robot is powered on with brakes released, and Remote Control on the pendant") from exc
        except BaseException:
            self.close()
            raise

    def _require_control(self):
        if self.control is None:
            raise RobotError("This operation needs RTDE control; it was connected read-only")
        return self.control

    def check_ready(self) -> dict:
        if not self.receive.isConnected() or (self.control is not None and not self.control.isConnected()):
            raise RobotError("RTDE connection lost")
        safety = int(self.receive.getSafetyMode())
        mode = int(self.receive.getRobotMode())
        state = {"robot_mode": mode, "safety_mode": safety, "safety": SAFETY_MODES.get(safety, str(safety)),
                 "protective_stop": bool(self.receive.isProtectiveStopped()),
                 "emergency_stop": bool(self.receive.isEmergencyStopped()),
                 "speed_scaling": float(self.receive.getSpeedScalingCombined())}
        if state["protective_stop"] or state["emergency_stop"] or safety not in (SAFETY_NORMAL, SAFETY_REDUCED):
            reason = ("emergency stop" if state["emergency_stop"] else
                      "protective stop" if state["protective_stop"] else state["safety"])
            raise RobotError(f"Robot is in {reason}; clear it on the pendant before continuing")
        if mode != ROBOT_MODE_RUNNING:
            raise RobotError(f"Robot mode is {mode}, not RUNNING (7): power on and release the brakes")
        return state

    def joints(self) -> list[float]:
        return list(self.receive.getActualQ())

    def tcp_pose(self) -> np.ndarray:
        return pose_to_matrix(self.receive.getActualTCPPose())

    def tcp_offset(self) -> list[float]:
        return list(self._require_control().getTCPOffset())

    def flange_pose(self) -> np.ndarray:
        # The controller's calibrated kinematics with a zero TCP give tool0 exactly.
        return pose_to_matrix(self._require_control().getForwardKinematics(self.joints(), [0.0] * 6))

    def plan(self, flange_target: np.ndarray, q_near: list[float]) -> list[float]:
        control = self._require_control()
        tcp_target = matrix_to_pose(flange_target @ pose_to_matrix(self.tcp_offset()))
        if not control.isPoseWithinSafetyLimits(tcp_target):
            raise RobotError("Target violates the robot safety configuration (planes, tool limits)")
        if not control.getInverseKinematicsHasSolution(tcp_target, q_near):
            raise RobotError("Target is out of reach (no inverse kinematics solution)")
        q = list(control.getInverseKinematics(tcp_target, q_near))
        if len(q) != 6 or not all(math.isfinite(v) for v in q):
            raise RobotError("Inverse kinematics returned no valid solution")
        if not control.isJointsWithinSafetyLimits(q):
            raise RobotError("Planned joints violate the robot's joint safety limits")
        _joint_step(q, q_near, self.limits)
        _flange_matches(pose_to_matrix(control.getForwardKinematics(q, [0.0] * 6)), flange_target, "Plan check")
        return q

    def move_joints(self, q: list[float]) -> None:
        control = self._require_control()
        self.check_ready()
        _joint_step(q, self.joints(), self.limits)
        if control.moveJ(q, self.limits.joint_speed_rad_s, self.limits.joint_accel_rad_s2) is not True:
            self.check_ready()  # report a protective stop rather than a generic failure
            raise RobotError("moveJ was rejected or did not complete")
        deadline = time.monotonic() + self.limits.steady_timeout_s
        while not control.isSteady():
            if time.monotonic() > deadline:
                raise RobotError("Robot did not come to rest after the move")
            time.sleep(0.02)
        time.sleep(self.limits.settle_s)
        self.check_ready()
        reached = max(abs(a - b) for a, b in zip(self.joints(), q, strict=True))
        if reached > math.radians(0.2):
            raise RobotError(f"Robot stopped {math.degrees(reached):.2f} deg from the target joints")

    def set_freedrive(self, enabled: bool) -> None:
        control = self._require_control()
        ok = control.teachMode() if enabled else control.endTeachMode()
        if ok is not True:
            raise RobotError(f"Cannot {'enter' if enabled else 'leave'} freedrive (teach) mode")

    def stop(self) -> None:
        if self.control is not None:
            self.control.stopJ(2.0)

    def close(self) -> None:
        errors = []
        operations = []
        if self.control is not None:
            operations += [self.control.stopScript, self.control.disconnect]
        if self.receive is not None:
            operations.append(self.receive.disconnect)
        for operation in operations:
            try:
                operation()
            except Exception as exc:
                errors.append(str(exc))
        self.control = self.receive = None
        if errors:
            raise RobotError("Robot cleanup failed: " + "; ".join(errors))


# Nominal UR10e standard DH parameters (Universal Robots). The real controller uses its own
# factory-calibrated kinematics; the simulator only needs realistic reach and joint behaviour.
UR10E_DH = {"d": (0.1807, 0.0, 0.0, 0.17415, 0.11985, 0.11655),
            "a": (0.0, -0.6127, -0.57155, 0.0, 0.0, 0.0),
            "alpha": (math.pi / 2, 0.0, 0.0, math.pi / 2, -math.pi / 2, 0.0)}


def ur10e_forward(q) -> np.ndarray:
    transform = np.eye(4)
    for theta, d, a, alpha in zip(q, UR10E_DH["d"], UR10E_DH["a"], UR10E_DH["alpha"], strict=True):
        ct, st, ca, sa = math.cos(theta), math.sin(theta), math.cos(alpha), math.sin(alpha)
        transform = transform @ np.array([[ct, -st * ca, st * sa, a * ct], [st, ct * ca, -ct * sa, a * st],
                                          [0, sa, ca, d], [0, 0, 0, 1]])
    return transform


def _pose_error(current: np.ndarray, target: np.ndarray) -> np.ndarray:
    from .transforms import rotation_to_rotvec
    return np.concatenate([target[:3, 3] - current[:3, 3],
                           current[:3, :3] @ rotation_to_rotvec(current[:3, :3].T @ target[:3, :3])])


def ur10e_inverse(target: np.ndarray, q_near, iterations: int = 200) -> list[float] | None:
    """Damped least-squares IK from q_near: the nearest solution, which is what the controller picks."""
    q = np.asarray(q_near, float).copy()
    for _ in range(iterations):
        error = _pose_error(ur10e_forward(q), target)
        if np.linalg.norm(error[:3]) < 1e-7 and np.linalg.norm(error[3:]) < 1e-7:
            return q.tolist()
        jacobian = np.zeros((6, 6))
        for joint in range(6):
            delta = np.zeros(6)
            delta[joint] = 1e-6
            jacobian[:, joint] = (_pose_error(ur10e_forward(q), ur10e_forward(q + delta))) / 1e-6
        q += jacobian.T @ np.linalg.solve(jacobian @ jacobian.T + 1e-6 * np.eye(6), error)
    return None


class SimulatedRobot:
    """Kinematic UR10e: same planning checks as URRobot, instantaneous exact motion, optional noise."""

    #: A camera-friendly start: flange about 0.5 m above the base plane, tool pointing down.
    HOME = (0.0, -1.9, -1.6, -1.2, 1.5708, 0.0)

    def __init__(self, q=None, limits: MotionLimits = MotionLimits(settle_s=0.0), *,
                 tcp_offset=(0.0, 0.0, 0.2, 0.0, 0.0, 0.0), pose_noise_m: float = 0.0,
                 pose_noise_rad: float = 0.0, seed: int = 0):
        self.limits = limits
        self._q = list(self.HOME if q is None else q)
        self._tcp = list(tcp_offset)
        self._noise = (pose_noise_m, pose_noise_rad)
        self._rng = np.random.default_rng(seed)
        self.freedrive = False
        self.moves = 0
        self.fail_on_move: int | None = None  # test hook: raise on this move number

    def check_ready(self) -> dict:
        return {"robot_mode": ROBOT_MODE_RUNNING, "safety_mode": SAFETY_NORMAL, "safety": "normal",
                "protective_stop": False, "emergency_stop": False, "speed_scaling": 1.0, "simulated": True}

    def joints(self) -> list[float]:
        return list(self._q)

    def true_flange_pose(self) -> np.ndarray:
        return ur10e_forward(self._q)

    def flange_pose(self) -> np.ndarray:
        """Reported pose, with optional encoder/kinematic noise like a real controller."""
        transform = self.true_flange_pose()
        position_noise, rotation_noise = self._noise
        if position_noise or rotation_noise:
            from .transforms import rotvec_to_rotation
            transform = transform.copy()
            transform[:3, 3] += self._rng.normal(0, position_noise, 3)
            transform[:3, :3] = transform[:3, :3] @ rotvec_to_rotation(self._rng.normal(0, rotation_noise, 3))
        return transform

    def tcp_offset(self) -> list[float]:
        return list(self._tcp)

    def tcp_pose(self) -> np.ndarray:
        return self.flange_pose() @ pose_to_matrix(self._tcp)

    def plan(self, flange_target: np.ndarray, q_near: list[float]) -> list[float]:
        q = ur10e_inverse(flange_target, q_near)
        if q is None:
            raise RobotError("Target is out of reach (no inverse kinematics solution)")
        if any(abs(v) > 2 * math.pi for v in q):
            raise RobotError("Planned joints violate the robot's joint safety limits")
        _joint_step(q, q_near, self.limits)
        _flange_matches(ur10e_forward(q), flange_target, "Plan check")
        return q

    def move_joints(self, q: list[float]) -> None:
        if self.freedrive:
            raise RobotError("Cannot move while freedrive is active")
        _joint_step(q, self._q, self.limits)
        self.moves += 1
        if self.fail_on_move == self.moves:
            raise RobotError("Simulated protective stop")
        self._q = list(q)

    def set_freedrive(self, enabled: bool) -> None:
        self.freedrive = enabled

    def stop(self) -> None:
        pass

    def close(self) -> None:
        pass


def flange_from_tcp(tcp_pose: np.ndarray, tcp_offset) -> np.ndarray:
    """Flange pose from a reported TCP pose and the TCP offset that was active."""
    return tcp_pose @ invert(pose_to_matrix(tcp_offset))
