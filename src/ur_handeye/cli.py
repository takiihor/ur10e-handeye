"""Command line: check, collect, solve, verify, locate and simulate."""
import argparse
import json
import math
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

from . import __version__
from .collect import CollectError, CollectOptions, Collector
from .dataset import DatasetError, Session, build_result, load_result, load_session, samples_from_records, write_json
from .poses import PlanOptions
from .robot import MotionLimits, RobotError
from .solver import SolveError, Thresholds, _Problem, board_poses, consistency, solve
from .transforms import angle_between_deg, describe, mean_transform
from .vision import BoardObserver, BoardTarget, VisionError

#: The sibling charuco-generator checkout's calibration board (src/ur_handeye/cli.py -> repo -> its parent).
DEFAULT_MANIFEST = str(Path(__file__).resolve().parents[3] / "charuco-generator" / "output" / "calibration"
                       / "board_manifest.json")


def _confirm(args):
    if getattr(args, "yes", False):
        return lambda summary: True

    def ask(summary: str) -> bool:
        print(summary)
        return input("Type 'yes' to continue (keep a hand on the emergency stop): ").strip().lower() == "yes"
    return ask


def _camera(args):
    from charuco_realsense.camera import RealsenseCamera, StreamConfig
    camera = RealsenseCamera(args.serial, StreamConfig(args.width, args.height, args.fps))
    try:
        if args.lock_exposure:
            deadline = time.monotonic() + 2.0  # let auto exposure settle before freezing it
            while time.monotonic() < deadline:
                camera.read()
            camera.lock_exposure()
        else:
            for _ in range(15):
                camera.read()
    except BaseException:
        camera.close()
        raise
    return camera


def _intrinsics(args) -> dict | None:
    if not getattr(args, "intrinsics", None):
        return None
    from charuco_realsense.calibration import load_calibration
    return load_calibration(args.intrinsics)


def _limits(args) -> MotionLimits:
    return MotionLimits(joint_speed_rad_s=args.joint_speed, max_travel_m=args.max_travel_mm / 1000,
                        settle_s=args.settle_s, max_joint_step_rad=math.radians(args.max_joint_step_deg))


def _print_solution(result: dict) -> None:
    m = result["metrics"]
    x = result["camera_in_flange"]
    print(f"Camera in flange: xyz {np.round(x['xyz_mm'], 2).tolist()} mm, rpy {np.round(x['rpy_deg'], 3).tolist()} deg")
    print(f"  UR pose (TCP for the camera): {np.round(x['pose_ur'], 6).tolist()}")
    print(f"Samples {m['sample_count']}, seed {m['seed_method']}, reprojection RMSE {m['reprojection_rmse_px']:.3f} px "
          f"(max {m['reprojection_max_px']:.2f})")
    c = m["consistency"]
    print(f"Board-in-base consistency: {c['position_rms_mm']:.3f} mm RMS (max {c['position_max_mm']:.3f}), "
          f"{c['rotation_rms_deg']:.3f} deg RMS")
    if "leave_one_out" in m:
        loo = m["leave_one_out"]
        print(f"Leave-one-out prediction: {loo['position_rms_mm']:.3f} mm RMS (max {loo['position_max_mm']:.3f})")
    if "camera_in_flange_uncertainty" in m:
        u = m["camera_in_flange_uncertainty"]
        combined = np.round(u["combined_translation_mm"], 3).tolist()
        print(f"1-sigma camera translation (pixel + robot noise): {combined} mm")
    d = m["rotation_diversity"]
    print(f"Rotation diversity: max {d['max_relative_rotation_deg']:.0f} deg between views, "
          f"axis spread {d['axis_spread']:.2f}")
    for warning in result["warnings"]:
        print(f"WARNING: {warning}")
    print(f"Verdict: {result['verdict'].upper()}")


def _solve_session(session_dir, *, exclude=(), thresholds=Thresholds(), loo=True, sigma_mm=0.05, sigma_deg=0.01,
                   manifest=None) -> dict:
    metadata, records = load_session(session_dir)
    samples = samples_from_records(records, exclude)
    if "intrinsics" not in metadata:
        raise DatasetError("Session has no camera intrinsics; no sample was captured")
    target = _target_from_metadata(metadata, manifest)
    solution = solve(samples, target.corners, metadata["intrinsics"], thresholds=thresholds, leave_one_out=loo,
                     robot_position_sigma_mm=sigma_mm, robot_rotation_sigma_deg=sigma_deg)
    return build_result(solution, metadata, session_dir)


def _target_from_metadata(metadata: dict, manifest=None) -> BoardTarget:
    board = metadata["board"]
    path = manifest or board.get("manifest")
    if path:
        return BoardTarget(path, board["name"])
    from .simulation import calibration_board_target
    return calibration_board_target()


def cmd_check(args) -> int:
    ok = True
    if args.robot_ip:
        from .robot import URRobot
        robot = URRobot(args.robot_ip, control=False)
        try:
            state = robot.check_ready()
            print(f"Robot {args.robot_ip}: {state['safety']} mode, speed scaling {state['speed_scaling']:.2f}")
            print(f"  TCP pose {np.round(robot.receive.getActualTCPPose(), 4).tolist()}")
        except RobotError as exc:
            ok = False
            print(f"Robot NOT ready: {exc}")
        finally:
            robot.close()
    if not args.no_camera:
        target = BoardTarget(args.manifest, args.board)
        with _camera(args) as camera:
            observation = BoardObserver(camera, target, intrinsics=_intrinsics(args)).observe(args.frames)
        usable, reasons = observation.usable(args.max_jitter_px)
        pose = observation.pose
        print(f"Camera: {len(observation.charuco_ids)}/{len(target.corners)} corners"
              + (f", board at {np.round(pose['position_mm'], 1).tolist()} mm, "
                 f"rmse {pose['reprojection_rmse_px']:.2f} px" if pose else ""))
        print("  Usable calibration view" if usable else f"  Not usable: {', '.join(reasons)}")
        ok &= usable
    return 0 if ok else 1


def cmd_collect(args) -> int:
    from .robot import URRobot
    target = BoardTarget(args.manifest, args.board)
    output = Path(args.output or f"sessions/{datetime.now():%Y%m%d-%H%M%S}")
    robot = URRobot(args.robot_ip, _limits(args))
    try:
        with _camera(args) as camera:
            observer = BoardObserver(camera, target, intrinsics=_intrinsics(args))
            session = Session(output, {"robot": {"host": args.robot_ip, "model": "UR10e"},
                                       "tcp_offset": robot.tcp_offset(), "camera": camera.metadata,
                                       "board": target.describe(), "limits": vars(robot.limits)})
            options = CollectOptions(frames=args.frames, max_jitter_px=args.max_jitter_px,
                                     plan=PlanOptions(args.tilt_deg, args.roll_deg, max_views=args.views,
                                                      probe_mm=args.probe_mm))
            collector = Collector(robot, observer, session, options, confirm=_confirm(args))
            if args.mode == "manual":
                collector.run_manual(freedrive=args.freedrive)
            elif args.mode == "replay":
                _, records = load_session(args.replay)
                collector.run_replay([r["joints"] for r in records if r.get("status") == "ok"
                                      and r.get("label") not in ("start", "probe")])
            else:
                collector.run_auto(dry_run=args.dry_run)
    finally:
        robot.close()
    print(f"Session saved to {output.resolve()} ({collector.usable} usable samples)")
    if args.dry_run:
        return 0
    if collector.usable < 8:
        print("Fewer than 8 usable samples; collect more before solving.")
        return 1
    print(f"Next: python -m ur_handeye solve --session {output}")
    return 0


def cmd_solve(args) -> int:
    thresholds = Thresholds(max_loo_mm=args.max_loo_mm, max_reprojection_px=args.max_reprojection_px)
    result = _solve_session(args.session, exclude=args.exclude, thresholds=thresholds, loo=not args.no_loo,
                            sigma_mm=args.robot_sigma_mm, sigma_deg=args.robot_sigma_deg, manifest=args.manifest)
    output = Path(args.output or Path(args.session) / "calibration.json")
    write_json(output, result)
    _print_solution(result)
    print(f"Calibration written to {output.resolve()}")
    return 0 if result["verdict"] != "poor" else 1


def cmd_verify(args) -> int:
    result = load_result(args.calibration)
    metadata, records = load_session(args.session)
    samples = samples_from_records(records)
    if not samples:
        raise DatasetError("The verification session has no usable samples")
    if metadata.get("intrinsics") != result["intrinsics"]:
        print("WARNING: verification intrinsics differ from the calibration's (other camera profile?)")
    target = _target_from_metadata(metadata, args.manifest)
    x = np.asarray(result["camera_in_flange"]["matrix"])
    y = (mean_transform(board_poses(samples, x)) if args.refit_board
         else np.asarray(result["board_in_base"]["matrix"]))
    from charuco_realsense.calibration import opencv_calibration
    problem = _Problem(samples, target.corners, *opencv_calibration(metadata["intrinsics"]))
    residual = np.linalg.norm(problem.residuals(x, y).reshape(-1, 2), axis=1)
    report = {"calibration": str(args.calibration), "session": str(args.session), "samples": len(samples),
              "board_refit": args.refit_board, "consistency": consistency(samples, x, y, target.centre),
              "reprojection_rmse_px": float(np.sqrt(np.mean(residual ** 2)))}
    report["passed"] = report["consistency"]["position_max_mm"] <= args.max_mm
    print(json.dumps(report, indent=2))
    return 0 if report["passed"] else 1


def cmd_locate(args) -> int:
    from .robot import URRobot
    result = load_result(args.calibration)
    target = BoardTarget(args.manifest, args.board)
    robot = URRobot(args.robot_ip, control=args.use_fk)
    try:
        robot.check_ready()
        with _camera(args) as camera:
            intrinsics = _intrinsics(args) or result["intrinsics"]
            observation = BoardObserver(camera, target, intrinsics=intrinsics).observe(args.frames)
        if args.use_fk:
            camera_in_base = robot.flange_pose() @ np.asarray(result["camera_in_flange"]["matrix"])
        else:
            # Read-only: the controller's active TCP must be the one recorded at calibration.
            camera_in_base = robot.tcp_pose() @ np.asarray(result["camera_in_tcp"]["matrix"])
    finally:
        robot.close()
    usable, reasons = observation.usable(args.max_jitter_px)
    if observation.board_in_camera is None:
        print(f"Board {args.board} not located: {', '.join(reasons)}")
        return 1
    board = camera_in_base @ observation.board_in_camera
    report = {"board": args.board, "board_in_base": describe(board),
              "quality_passed": usable, "quality_reasons": reasons,
              "corners": len(observation.charuco_ids), "reprojection_rmse_px": observation.pose["reprojection_rmse_px"],
              "calibration_verdict": result["verdict"],
              "frame_source": "flange FK" if args.use_fk else "actual TCP pose and calibration-time TCP offset"}
    print(json.dumps(report, indent=2))
    return 0 if usable else 1


def cmd_simulate(args) -> int:
    from .simulation import build_scenario
    target = BoardTarget(args.manifest, args.board) if args.manifest else None
    scenario = build_scenario(target=target, seed=args.seed, pose_noise_mm=args.robot_noise_mm,
                              pose_noise_deg=args.robot_noise_deg)
    output = Path(args.output or f"sessions/sim-{datetime.now():%Y%m%d-%H%M%S}")
    observer = BoardObserver(scenario.camera, scenario.target)
    session = Session(output, {"robot": {"host": "simulated", "model": "UR10e"},
                               "tcp_offset": scenario.robot.tcp_offset(), "camera": {"simulated": True},
                               "board": scenario.target.describe(), "simulated": True})
    options = CollectOptions(plan=PlanOptions(args.tilt_deg, args.roll_deg, max_views=args.views))
    Collector(scenario.robot, observer, session, options, confirm=lambda summary: True).run_auto()
    result = _solve_session(output, manifest=args.manifest)
    x = np.asarray(result["camera_in_flange"]["matrix"])
    error_mm = float(np.linalg.norm(x[:3, 3] - scenario.camera_in_flange[:3, 3]) * 1000)
    error_deg = angle_between_deg(x, scenario.camera_in_flange)
    result["simulation_truth"] = {"camera_in_flange": describe(scenario.camera_in_flange),
                                  "board_in_base": describe(scenario.board_in_base),
                                  "camera_translation_error_mm": error_mm, "camera_rotation_error_deg": error_deg}
    write_json(output / "calibration.json", result)
    _print_solution(result)
    print(f"Ground truth error: {error_mm:.3f} mm, {error_deg:.4f} deg   (session {output.resolve()})")
    return 0 if error_mm <= args.max_error_mm and error_deg <= args.max_error_deg else 1


def _camera_options(parser):
    parser.add_argument("--serial", help="RealSense serial (required with several cameras)")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--intrinsics",
                        help="Calibration JSON overriding factory intrinsics (charuco-realsense format)")
    parser.add_argument("--lock-exposure", action="store_true", help="Freeze auto exposure after it settles")
    parser.add_argument("--frames", type=int, default=5, help="Frames averaged per view (robot at rest)")
    parser.add_argument("--max-jitter-px", type=float, default=0.3,
                        help="Reject views whose corners move more than this")


def _board_options(parser, default_board="calibration"):
    parser.add_argument("--manifest", default=DEFAULT_MANIFEST, help="charuco-generator board_manifest.json")
    parser.add_argument("--board", default=default_board)


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(prog="ur_handeye", description="UR10e eye-in-hand camera calibration over RTDE")
    root.add_argument("--version", action="version", version=f"ur10e-handeye {__version__}")
    commands = root.add_subparsers(dest="command", required=True)

    check = commands.add_parser("check", help="Read-only: robot state and whether the camera sees the board")
    check.add_argument("--robot-ip")
    check.add_argument("--no-camera", action="store_true")
    _board_options(check)
    _camera_options(check)

    collect = commands.add_parser("collect", help="Move through calibration views and record a session")
    collect.add_argument("--robot-ip", required=True)
    collect.add_argument("--output", help="New session folder (default sessions/YYYYMMDD-HHMMSS)")
    collect.add_argument("--mode", choices=("auto", "manual", "replay"), default="auto")
    collect.add_argument("--replay", help="Session whose views to repeat (with --mode replay)")
    collect.add_argument("--freedrive", action="store_true", help="Manual mode: hand-guide the arm between captures")
    collect.add_argument("--dry-run", action="store_true", help="Plan and check every view without moving")
    collect.add_argument("--yes", action="store_true", help="Do not ask before moving (attended cells only)")
    collect.add_argument("--views", type=int, default=14)
    collect.add_argument("--tilt-deg", type=float, default=25.0)
    collect.add_argument("--roll-deg", type=float, default=30.0)
    collect.add_argument("--probe-mm", type=float, default=30.0)
    collect.add_argument("--joint-speed", type=float, default=0.35, help="rad/s (max 1.05)")
    collect.add_argument("--max-travel-mm", type=float, default=450.0)
    collect.add_argument("--max-joint-step-deg", type=float, default=75.0)
    collect.add_argument("--settle-s", type=float, default=0.8)
    _board_options(collect)
    _camera_options(collect)

    solve_cmd = commands.add_parser("solve", help="Compute the calibration from a recorded session (offline)")
    solve_cmd.add_argument("--session", required=True)
    solve_cmd.add_argument("--output", help="Calibration JSON (default SESSION/calibration.json)")
    solve_cmd.add_argument("--manifest", help="Override the board manifest recorded in the session")
    solve_cmd.add_argument("--exclude", type=int, nargs="*", default=[], help="Sample indices to leave out")
    solve_cmd.add_argument("--no-loo", action="store_true", help="Skip leave-one-out validation")
    solve_cmd.add_argument("--robot-sigma-mm", type=float, default=0.05, help="Assumed flange position noise")
    solve_cmd.add_argument("--robot-sigma-deg", type=float, default=0.01, help="Assumed flange rotation noise")
    solve_cmd.add_argument("--max-loo-mm", type=float, default=2.5)
    solve_cmd.add_argument("--max-reprojection-px", type=float, default=1.0)

    verify = commands.add_parser("verify", help="Check a calibration against a separately collected session")
    verify.add_argument("--calibration", required=True)
    verify.add_argument("--session", required=True)
    verify.add_argument("--manifest", help="Override the board manifest recorded in the session")
    verify.add_argument("--refit-board", action="store_true", help="The board was moved since calibration")
    verify.add_argument("--max-mm", type=float, default=2.0, help="Pass limit on the worst board position error")

    locate = commands.add_parser("locate", help="Read-only: a board's pose in the robot base frame")
    locate.add_argument("--robot-ip", required=True)
    locate.add_argument("--calibration", required=True)
    locate.add_argument("--use-fk", action="store_true",
                        help="Use flange kinematics (takes RTDE control) instead of the active TCP")
    _board_options(locate, default_board="front")
    _camera_options(locate)

    sim = commands.add_parser("simulate", help="Full pipeline on a simulated UR10e and camera, against ground truth")
    sim.add_argument("--output")
    sim.add_argument("--seed", type=int, default=0)
    sim.add_argument("--views", type=int, default=14)
    sim.add_argument("--tilt-deg", type=float, default=25.0)
    sim.add_argument("--roll-deg", type=float, default=30.0)
    sim.add_argument("--robot-noise-mm", type=float, default=0.02)
    sim.add_argument("--robot-noise-deg", type=float, default=0.005)
    sim.add_argument("--max-error-mm", type=float, default=1.0)
    sim.add_argument("--max-error-deg", type=float, default=0.2)
    sim.add_argument("--manifest", help="Simulate a board from a charuco-generator manifest (default: the A4 "
                                        "calibration board built in memory)")
    sim.add_argument("--board", default="calibration")
    return root


COMMANDS = {"check": cmd_check, "collect": cmd_collect, "solve": cmd_solve, "verify": cmd_verify,
            "locate": cmd_locate, "simulate": cmd_simulate}


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        return COMMANDS[args.command](args)
    except (RobotError, CollectError, SolveError, DatasetError, VisionError, ValueError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("\nInterrupted. The robot was told to stop where it is.", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
