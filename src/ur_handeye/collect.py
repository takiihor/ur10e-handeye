"""Calibration data collection: automatic views, operator-positioned views, or a replayed path.

Safety model: every view is planned (IK near the previous joints, controller safety limits,
per-move joint-step limit, travel limit from the start pose) *before* the first motion, and the
operator confirms the complete plan. Any robot error or Ctrl+C stops the arm where it is; the
collector never makes a recovery motion on its own.
"""
import itertools
import math
from dataclasses import dataclass, field

import numpy as np

from .poses import PlanOptions, View, estimate_camera_rotation, plan_views, probe_targets
from .robot import RobotError
from .solver import MIN_SAMPLES, SolveError, _Problem, closed_form, refine
from .transforms import matrix, matrix_to_pose
from .vision import Observation


class CollectError(RuntimeError):
    pass


@dataclass(frozen=True)
class CollectOptions:
    frames: int = 5
    max_jitter_px: float = 0.3
    plan: PlanOptions = field(default_factory=PlanOptions)
    #: Re-plan the remaining views with an interim calibration after this many usable samples (0: never).
    bootstrap_after: int = 6
    #: Camera translation in the flange used for planning before calibration (metres); rotation is probed.
    translation_guess_m: tuple[float, float, float] = (0.0, 0.0, 0.0)
    #: Largest joint motion allowed while a sample is captured (robot must be at rest).
    max_capture_motion_deg: float = 0.02


@dataclass
class PlannedView:
    label: str
    flange: np.ndarray
    joints: list[float]
    #: Go back to the start pose first, because a direct move would exceed the joint-step limit.
    via_start: bool = False


def _step(a, b) -> float:
    return max(abs(x - y) for x, y in zip(a, b, strict=True))


class Collector:
    def __init__(self, robot, observer, session, options: CollectOptions = CollectOptions(), *,
                 log=print, confirm=lambda summary: False):
        self.robot, self.observer, self.session, self.options = robot, observer, session, options
        self.log, self.confirm = log, confirm
        self.usable = 0
        self.intrinsics = None

    # -- capture -----------------------------------------------------------------------------
    def capture(self, label: str) -> tuple[dict, Observation]:
        self.robot.check_ready()
        joints = self.robot.joints()
        flange = self.robot.flange_pose()
        observation = self.observer.observe(self.options.frames)
        after = self.robot.joints()
        usable, reasons = observation.usable(self.options.max_jitter_px)
        motion = max(abs(a - b) for a, b in zip(joints, after, strict=True))
        if motion > math.radians(self.options.max_capture_motion_deg):
            usable, reasons = False, [*reasons, "robot_moved_during_capture"]
        if self.intrinsics is None and observation.intrinsics:
            self.intrinsics = observation.intrinsics
            self.session.update_metadata(intrinsics=self.intrinsics)
        elif observation.intrinsics and observation.intrinsics != self.intrinsics:
            usable, reasons = False, [*reasons, "intrinsics_changed"]
        record = {"label": label, "status": "ok" if usable else "rejected", "reasons": reasons,
                  "joints": joints, "flange_pose": matrix_to_pose(flange),
                  "observation": observation.to_record()}
        index = self.session.add(record, observation.image_bgr)
        record["index"] = index
        self.usable += usable
        pose = observation.pose
        detail = (f"{len(observation.charuco_ids)} corners, rmse {pose['reprojection_rmse_px']:.2f} px, "
                  f"jitter {observation.jitter_px or 0:.2f} px" if pose else f"{len(observation.charuco_ids)} corners")
        self.log(f"  [{index:02d}] {label:<10} {'ok' if usable else 'REJECTED ' + ','.join(reasons)}  ({detail})")
        return record, observation

    def _move(self, joints: list[float]) -> None:
        try:
            self.robot.move_joints(joints)
        except BaseException:
            self._halt()
            raise

    def _halt(self) -> None:
        try:
            self.robot.stop()
        except Exception as exc:
            self.log(f"WARNING: stop command failed: {exc}")
        self.log("Robot stopped where it is. Check the cell before moving it again.")

    # -- planning ----------------------------------------------------------------------------
    def plan(self, views: list[View], start_flange: np.ndarray, q_start: list[float], *,
             q_from: list[float] | None = None) -> tuple[list[PlannedView], list]:
        """IK-check every view from the start joints, then order them to keep each move short.

        Solving every view from q_start keeps one arm configuration throughout (chained IK lets wrist
        changes accumulate). Views are visited nearest-first in joint space; where a direct move
        would still exceed the joint-step limit, the path goes back through the start pose, which
        every planned view is within one step of by construction.
        """
        planned, skipped = [], []
        for view in views:
            travel = float(np.linalg.norm(view.flange[:3, 3] - start_flange[:3, 3]))
            try:
                if travel > self.robot.limits.max_travel_m:
                    raise RobotError(f"{travel * 1000:.0f} mm from the start exceeds the "
                                     f"{self.robot.limits.max_travel_m * 1000:.0f} mm travel limit")
                q = self.robot.plan(view.flange, q_start)
            except RobotError as exc:
                skipped.append((view.label, str(exc)))
                continue
            planned.append(PlannedView(view.label, view.flange, q))
        ordered, current, remaining = [], q_from or q_start, planned
        while remaining:
            nearest = min(remaining, key=lambda view: _step(current, view.joints))
            remaining = [view for view in remaining if view is not nearest]
            nearest.via_start = _step(current, nearest.joints) > self.robot.limits.max_joint_step_rad
            ordered.append(nearest)
            current = nearest.joints
        return ordered, skipped

    def _summary(self, planned: list[PlannedView], skipped, start_flange, q_start) -> str:
        lines = [f"{len(planned)} views planned, {len(skipped)} skipped."]
        for label, reason in skipped:
            lines.append(f"  skipped {label}: {reason}")
        if planned:
            travel = max(np.linalg.norm(p.flange[:3, 3] - start_flange[:3, 3]) for p in planned) * 1000
            step = max(_step(p.joints, q_start) for p in planned)
            via = sum(p.via_start for p in planned)
            lines.append(f"  max flange travel from start {travel:.0f} mm; max joint change from start "
                         f"{math.degrees(step):.0f} deg; joint speed {self.robot.limits.joint_speed_rad_s:.2f} rad/s"
                         + (f"; {via} view(s) reached through the start pose" if via else ""))
        return "\n".join(lines)

    def _interim(self, samples_records, corners) -> np.ndarray | None:
        from .dataset import samples_from_records
        samples = samples_from_records(samples_records)
        if len(samples) < MIN_SAMPLES:
            return None
        try:
            from charuco_realsense.calibration import opencv_calibration
            camera_matrix, coeffs = opencv_calibration(self.intrinsics)
            candidates = closed_form(samples)
            x, y = candidates.get("shah_robot_world") or next(iter(candidates.values()))
            x, _, _ = refine(_Problem(samples, corners, camera_matrix, coeffs), x, y)
            return x
        except (SolveError, ValueError):
            return None

    # -- modes -------------------------------------------------------------------------------
    def run_auto(self, *, dry_run: bool = False) -> int:
        robot, options = self.robot, self.options
        robot.check_ready()
        q_start, start_flange = robot.joints(), robot.flange_pose()
        self.session.update_metadata(start_joints=q_start, start_flange_pose=matrix_to_pose(start_flange),
                                     mode="auto")
        self.log("Start view:")
        record, start = self.capture("start")
        if record["status"] != "ok":
            raise CollectError("The board is not fully and sharply visible from the start pose "
                               f"({', '.join(record['reasons'])}). Jog the robot so the camera sees the whole "
                               "board from 0.3-0.6 m, then start again")
        records = [record]
        corners = self.observer.target.corners
        if dry_run:
            rotation = np.eye(3)
            self.log("Dry run: planning with the camera optical axis assumed along flange Z (no probe moves)")
        else:
            probe_plan, skipped = self.plan(probe_targets(start_flange, options.plan.probe_mm / 1000),
                                            start_flange, q_start)
            if len(probe_plan) < 3 or skipped:
                raise CollectError(f"Probe moves of {options.plan.probe_mm:g} mm are not possible here: {skipped}")
            summary = (f"Probe: 3 straight {options.plan.probe_mm:g} mm flange moves to measure the camera "
                       "orientation, returning to the start after each.")
            if not self.confirm(summary):
                raise CollectError("Cancelled by the operator before any motion")
            probes = []
            for planned in probe_plan:
                self._move(planned.joints)
                probe_record, observation = self.capture(planned.label)
                records.append(probe_record)
                if observation.board_in_camera is None:
                    self._move(q_start)
                    raise CollectError("Board lost during a probe move; start closer to the board centre")
                probes.append((robot.flange_pose(), observation.board_in_camera))
                self._move(q_start)
            rotation, residual = estimate_camera_rotation(start_flange, start.board_in_camera, probes)
            self.log(f"Camera orientation measured from probes (fit residual {residual:.2f} mm)")
            if residual > 0.2 * options.plan.probe_mm:
                raise CollectError("Probe motion and camera motion disagree; is the camera rigidly mounted "
                                   "on the tool, and is the board fixed?")
        guess = matrix(rotation, options.translation_guess_m)
        views = plan_views(start_flange, guess, start.board_in_camera, self.observer.target.centre, options.plan)
        planned, skipped = self.plan(views, start_flange, q_start)
        summary = self._summary(planned, skipped, start_flange, q_start)
        self.log(summary)
        self.session.update_metadata(plan={"views": [p.label for p in planned], "skipped": skipped})
        if dry_run:
            return len(planned)
        if len(planned) + self.usable < MIN_SAMPLES + 2:
            raise CollectError("Too few reachable views for a reliable calibration; start from a pose with more "
                               "room around it, or reduce --tilt-deg")
        if not self.confirm(summary + "\nMove through these views now?"):
            raise CollectError("Cancelled by the operator before the calibration moves")
        bootstrapped = options.bootstrap_after == 0
        position = 0
        while position < len(planned):
            view = planned[position]
            if view.via_start:
                self._move(q_start)
            self._move(view.joints)
            record, _ = self.capture(view.label)
            records.append(record)
            position += 1
            if not bootstrapped and self.usable >= options.bootstrap_after:
                bootstrapped = True
                interim = self._interim(records, corners)
                if interim is not None:
                    remaining = {p.label for p in planned[position:]}
                    better = [v for v in plan_views(start_flange, interim, start.board_in_camera,
                                                    self.observer.target.centre, options.plan)
                              if v.label in remaining]
                    replanned, skipped_again = self.plan(better, start_flange, q_start, q_from=robot.joints())
                    planned = planned[:position] + replanned
                    self.log(f"Re-planned {len(replanned)} remaining views with the interim calibration"
                             + (f" ({len(skipped_again)} now unreachable)" if skipped_again else ""))
        self.log("Returning to the start pose")
        self._move(q_start)
        return self.usable

    def run_manual(self, read_line=input, *, freedrive: bool = False) -> int:
        """Operator positions the robot (pendant jog or freedrive); Enter captures, q finishes."""
        self.session.update_metadata(mode="manual")
        self.log("Manual mode: vary the view direction by 20-40 degrees about different axes between captures.")
        try:
            while True:
                if freedrive:
                    self.robot.set_freedrive(True)
                answer = read_line(f"[{self.usable} usable] Position the robot, then Enter to capture (q to finish): ")
                if freedrive:
                    self.robot.set_freedrive(False)  # capture with the arm held still, not compliant
                if answer.strip().lower() in ("q", "quit", "done"):
                    break
                self.capture(f"manual{self.session.count:02d}")
        finally:
            if freedrive:
                try:
                    self.robot.set_freedrive(False)
                except RobotError as exc:
                    self.log(f"WARNING: {exc}")
        return self.usable

    def run_replay(self, joint_targets: list[list[float]]) -> int:
        """Repeat a previous session's views, e.g. for a periodic recalibration check."""
        self.robot.check_ready()
        q_start = self.robot.joints()
        self.session.update_metadata(mode="replay", start_joints=q_start)
        path = [q_start, *joint_targets, q_start]
        for a, b in itertools.pairwise(path):
            step = max(abs(x - y) for x, y in zip(a, b, strict=True))
            if step > self.robot.limits.max_joint_step_rad:
                raise CollectError(f"Replay path has a {math.degrees(step):.0f} deg joint step; start from the "
                                   "recorded start pose or recollect")
        if not self.confirm(f"Replay {len(joint_targets)} recorded views and return to the current pose?"):
            raise CollectError("Cancelled by the operator before any motion")
        for number, joints in enumerate(joint_targets):
            self._move(joints)
            self.capture(f"replay{number:02d}")
        self._move(q_start)
        return self.usable
