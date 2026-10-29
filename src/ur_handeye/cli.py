"""Command line: check, collect, solve, verify, locate, simulate and the desktop UI."""
import argparse
import json
import math
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import cv2
import numpy as np
from charuco_realsense.camera import CameraError

from . import __version__
from .collect import CollectError, CollectOptions, Collector
from .dataset import (
    DatasetError,
    Session,
    build_result,
    load_result,
    load_session,
    samples_from_records,
    write_json,
    write_new_json,
)
from .intrinsics import fit_intrinsics, load_intrinsics
from .poses import PlanOptions
from .robot import MotionLimits, RobotError, ur10e_forward
from .solver import (
    MIN_SAMPLES,
    SolveError,
    Thresholds,
    _Problem,
    board_poses,
    consistency,
    per_sample_rmse,
    reestimate_board_poses,
    rotation_diversity,
    solve,
)
from .transforms import angle_between_deg, describe, mean_transform, pose_to_matrix
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
    model = load_intrinsics(args.intrinsics)
    if model.get("simulated"):
        raise DatasetError("Simulated RGB intrinsics cannot be used for live capture")
    serial = model.get("camera_serial")
    if serial and getattr(args, "serial", None) and args.serial != serial:
        raise DatasetError("Requested camera serial differs from the selected RGB intrinsics")
    if serial and hasattr(args, "serial"):
        args.serial = serial
    return model


def _check_intrinsics_profile(intrinsics, metadata):
    recorded = metadata["intrinsics"]
    if any(intrinsics.get(key) != recorded.get(key) for key in ("width", "height")):
        raise DatasetError("Intrinsics image size differs from the recorded camera profile; use a calibration "
                           "for this exact resolution")
    serial = metadata.get("camera", {}).get("device", {}).get("serial_number")
    if serial and intrinsics.get("camera_serial") and intrinsics["camera_serial"] != serial:
        raise DatasetError("Intrinsics belong to a different camera serial")


def _same_camera_model(first, second):
    from charuco_realsense.calibration import opencv_calibration
    models = zip(opencv_calibration(first), opencv_calibration(second), strict=True)
    return (all(first.get(k) == second.get(k) for k in ("width", "height"))
            and all(np.array_equal(a, b) for a, b in models))


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
    worst = sorted(zip(m["sample_indices"], m["per_sample_reprojection_rmse_px"], strict=True),
                   key=lambda pair: pair[1], reverse=True)[:3]
    print("Largest view reprojection errors (sample: px): "
          + ", ".join(f"{index}: {error:.3f}" for index, error in worst))
    c = m["consistency"]
    print(f"Board-in-base consistency: {c['position_rms_mm']:.3f} mm RMS (max {c['position_max_mm']:.3f}), "
          f"{c['rotation_rms_deg']:.3f} deg RMS (max {c['rotation_max_deg']:.3f})")
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
                   manifest=None, intrinsics=None) -> dict:
    metadata, records = load_session(session_dir)
    samples = samples_from_records(records, exclude)
    if "intrinsics" not in metadata:
        raise DatasetError("Session has no camera intrinsics; no sample was captured")
    target = _target_from_metadata(metadata, manifest)
    if intrinsics:
        override = load_intrinsics(intrinsics)
        simulated = (metadata.get("simulated") or metadata.get("camera", {}).get("simulated")
                     or metadata.get("robot", {}).get("host") == "simulated")
        if override.get("simulated") and not simulated:
            raise DatasetError("Simulated RGB intrinsics cannot calibrate a hardware recording")
        _check_intrinsics_profile(override, metadata)
        samples = reestimate_board_poses(samples, target.corners, override)
        metadata = {**metadata, "intrinsics": override,
                    "intrinsics_source": {"type": "override", "path": str(Path(intrinsics).resolve()),
                                          "board_poses_reestimated": True,
                                          "recorded_intrinsics": metadata["intrinsics"]}}
    solution = solve(samples, target.corners, metadata["intrinsics"], thresholds=thresholds, leave_one_out=loo,
                     robot_position_sigma_mm=sigma_mm, robot_rotation_sigma_deg=sigma_deg)
    estimation = metadata["intrinsics"].get("estimation", {})
    if estimation and not estimation.get("validated"):
        solution.warnings.append("RGB intrinsics were estimated from recorded views; validate with fresh views "
                                 "covering the image before use")
        solution.warnings.extend(estimation.get("warnings", []))
        if solution.verdict == "good":
            solution.verdict = "acceptable"
    return build_result(solution, metadata, session_dir)


def _target_from_metadata(metadata: dict, manifest=None) -> BoardTarget:
    board = metadata["board"]
    path = manifest or board.get("manifest")
    if path:
        target = BoardTarget(path, board["name"])
        if board.get("manifest_sha256") and target.manifest_sha256 != board["manifest_sha256"]:
            raise DatasetError("Board manifest changed since collection; restore the recorded manifest")
        return target
    from .simulation import calibration_board_target
    target = calibration_board_target()
    current = target.describe()
    if any(board.get(key) != current[key] for key in
           ("name", "squares", "square_length_m", "marker_length_m", "marker_ids")):
        raise DatasetError("Session board differs from the built-in board; provide its recorded manifest")
    return target


def cmd_check(args, *, on_observation=None) -> int:
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
        intrinsics = _intrinsics(args)
        with _camera(args) as camera:
            if intrinsics:
                _check_intrinsics_profile(intrinsics, {"intrinsics": camera.intrinsics, "camera": camera.metadata})
            observation = BoardObserver(camera, target, intrinsics=intrinsics).observe(args.frames)
        usable, reasons = observation.usable(args.max_jitter_px)
        if on_observation is not None:
            on_observation(observation)
        pose = observation.pose
        print(f"Camera: {len(observation.charuco_ids)}/{len(target.corners)} corners"
              + (f", board at {np.round(pose['position_mm'], 1).tolist()} mm, "
                 f"rmse {pose['reprojection_rmse_px']:.2f} px" if pose else ""))
        print("  Usable calibration view" if usable else f"  Not usable: {', '.join(reasons)}")
        ok &= usable
    return 0 if ok else 1


def cmd_collect(args, *, confirm=None, read_line=input, on_capture=None) -> int:
    from .robot import URRobot
    # Reject incompatible options before opening RTDE control or a camera.
    if args.mode == "manual" and args.dry_run:
        raise ValueError("--dry-run is supported in auto and replay modes; manual mode requires operator positioning")
    if args.freedrive and args.mode != "manual":
        raise ValueError("--freedrive requires --mode manual")
    if (args.mode == "replay") != bool(args.replay):
        raise ValueError("--mode replay and --replay SESSION must be provided together")
    options = CollectOptions(frames=args.frames, max_jitter_px=args.max_jitter_px,
                             plan=PlanOptions(args.tilt_deg, args.roll_deg, max_views=args.views,
                                              probe_mm=args.probe_mm))
    replay_targets = None
    if args.mode == "replay":
        _, records = load_session(args.replay)
        replay_targets = [r["joints"] for r in records if r.get("status") == "ok"
                          and not r.get("label", "").startswith(("start", "probe"))]
        if not replay_targets:
            raise ValueError("Replay session has no usable calibration views")
    intrinsics = _intrinsics(args)
    target = BoardTarget(args.manifest, args.board)
    output = Path(args.output or f"sessions/{datetime.now():%Y%m%d-%H%M%S}")
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise DatasetError(f"{output} already contains data; choose a new session folder")
    robot = URRobot(args.robot_ip, _limits(args))
    try:
        with _camera(args) as camera:
            if intrinsics:
                _check_intrinsics_profile(intrinsics, {"intrinsics": camera.intrinsics, "camera": camera.metadata})
            observer = BoardObserver(camera, target, intrinsics=intrinsics)
            session = Session(output, {"robot": {"host": args.robot_ip, "model": "UR10e"},
                                       "tcp_offset": robot.tcp_offset(), "camera": camera.metadata,
                                       "board": target.describe(), "limits": vars(robot.limits),
                                       "intrinsics_source": {"type": "override" if intrinsics else "factory",
                                                             "path": args.intrinsics}})
            collector = Collector(robot, observer, session, options,
                                  confirm=confirm if confirm is not None else _confirm(args), on_capture=on_capture)
            if args.mode == "manual":
                collector.run_manual(read_line=read_line, freedrive=args.freedrive)
            elif args.mode == "replay":
                collector.run_replay(replay_targets, dry_run=args.dry_run)
            else:
                planned_count = collector.run_auto(dry_run=args.dry_run)
    finally:
        robot.close()
    print(f"Session saved to {output.resolve()} ({collector.usable} usable samples)")
    if args.dry_run:
        if args.mode == "auto" and planned_count + collector.usable < MIN_SAMPLES + 2:
            print("Too few reachable views for calibration; adjust the start pose and inspect the plan again.")
            return 1
        return 0
    if collector.usable < 8:
        print("Fewer than 8 usable samples; collect more before solving.")
        return 1
    print(f"Next: python -m ur_handeye solve --session {output}")
    return 0


def cmd_solve(args) -> int:
    output = Path(args.output or Path(args.session) / "calibration.json")
    if output.exists():
        raise DatasetError(f"{output} already exists; choose --output with a new filename to preserve it")
    thresholds = Thresholds(max_loo_mm=args.max_loo_mm, max_reprojection_px=args.max_reprojection_px,
                            max_consistency_mm=args.max_consistency_mm, max_consistency_deg=args.max_consistency_deg)
    result = _solve_session(args.session, exclude=args.exclude, thresholds=thresholds, loo=not args.no_loo,
                            sigma_mm=args.robot_sigma_mm, sigma_deg=args.robot_sigma_deg, manifest=args.manifest,
                            intrinsics=args.intrinsics)
    write_new_json(output, result)
    _print_solution(result)
    print(f"Calibration written to {output.resolve()}")
    return 0 if result["verdict"] != "poor" else 1


def cmd_reprocess(args) -> int:
    from .reprocess import reprocess_session
    metadata, _ = load_session(args.session)
    target = _target_from_metadata(metadata, args.manifest)
    accepted = reprocess_session(args.session, args.output, target, max_jitter_px=args.max_jitter_px)
    print(f"Reprocessed session written to {Path(args.output).resolve()}; {accepted} usable captures")
    print("This is derived calibration data, not an independent verification session.")
    return 0


def cmd_calibrate(args, *, on_intrinsics=None) -> int:
    """Explicit offline RGB calibration followed by a hand-eye solve with that model."""
    output, intrinsic_output = Path(args.output), Path(args.intrinsics_output)
    if output.resolve() == intrinsic_output.resolve():
        raise DatasetError("Intrinsic and hand-eye outputs must be different files")
    for path in (output, intrinsic_output):
        if path.exists():
            raise DatasetError(f"{path} already exists; choose new output filenames to preserve evidence")
    cmd_intrinsics(argparse.Namespace(session=args.session, output=intrinsic_output, manifest=args.manifest,
                                     exclude=args.exclude, fix_camera_matrix=False), on_result=on_intrinsics)
    result = _solve_session(args.session, exclude=args.exclude, manifest=args.manifest, intrinsics=intrinsic_output)
    write_new_json(output, result)
    _print_solution(result)
    print(f"Calibration written to {output.resolve()}")
    return 0 if result["verdict"] != "poor" else 1


def cmd_intrinsics(args, *, on_result=None) -> int:
    output = Path(args.output)
    if output.exists():
        raise DatasetError(f"{output} already exists; choose a new intrinsic calibration filename")
    metadata, records = load_session(args.session)
    if "intrinsics" not in metadata:
        raise DatasetError("Session has no camera intrinsics; no usable camera recording is available")
    target = _target_from_metadata(metadata, args.manifest)
    model = fit_intrinsics(samples_from_records(records, args.exclude), target.corners, metadata["intrinsics"],
                           fix_camera_matrix=args.fix_camera_matrix)
    serial = metadata.get("camera", {}).get("device", {}).get("serial_number")
    if serial:
        model["camera_serial"] = serial
    model.update(software=f"ur10e-handeye {__version__}", created_utc=datetime.now(UTC).isoformat())
    model["estimation"].update(source_session=str(Path(args.session).resolve()),
                               manifest_sha256=metadata["board"].get("manifest_sha256"))
    if (metadata.get("simulated") or metadata.get("camera", {}).get("simulated")
            or metadata.get("robot", {}).get("host") == "simulated"):
        model["simulated"] = True
    write_new_json(output, model)
    print(f"RGB intrinsic estimate written to {output.resolve()}")
    print(f"Training reprojection RMSE {model['estimation']['reprojection_rmse_px']:.3f} px")
    print(model["estimation"]["note"])
    for warning in model["estimation"]["warnings"]:
        print(f"WARNING: {warning}")
    if on_result:
        on_result(model, str(output.resolve()))
    return 0


def cmd_verify(args, *, on_report=None) -> int:
    if not math.isfinite(args.max_mm) or args.max_mm <= 0 \
            or not math.isfinite(args.max_deg) or args.max_deg <= 0:
        raise ValueError("Verification limits must be finite and positive")
    if args.output and Path(args.output).exists():
        raise DatasetError(f"{args.output} already exists; choose a new verification report filename")
    result = load_result(args.calibration)
    metadata, records = load_session(args.session)
    samples = samples_from_records(records)
    if not samples:
        raise DatasetError("The verification session has no usable samples")
    if args.refit_board:
        diversity = rotation_diversity(samples)
        if len(samples) < MIN_SAMPLES or diversity["max_relative_rotation_deg"] < 30 \
                or diversity["axis_spread"] < 0.15:
            raise DatasetError("Refitting the board for verification needs at least six diverse views "
                               "with rotations about two axes; one pose can absorb any camera error")
    target = _target_from_metadata(metadata, args.manifest)
    _check_intrinsics_profile(result["intrinsics"], metadata)
    serial = result.get("camera", {}).get("device", {}).get("serial_number")
    observed_serial = metadata.get("camera", {}).get("device", {}).get("serial_number")
    if serial and observed_serial and serial != observed_serial:
        raise DatasetError("Verification camera serial differs from the calibration camera")
    # Use one camera model for both the PnP consistency and reprojection checks.
    reestimated = not _same_camera_model(metadata["intrinsics"], result["intrinsics"])
    if reestimated:
        samples = reestimate_board_poses(samples, target.corners, result["intrinsics"])
    x = np.asarray(result["camera_in_flange"]["matrix"])
    y = (mean_transform(board_poses(samples, x)) if args.refit_board
         else np.asarray(result["board_in_base"]["matrix"]))
    from charuco_realsense.calibration import opencv_calibration
    problem = _Problem(samples, target.corners, *opencv_calibration(result["intrinsics"]))
    residual = np.linalg.norm(problem.residuals(x, y).reshape(-1, 2), axis=1)
    report = {"calibration": str(args.calibration), "session": str(args.session), "samples": len(samples),
              "software": f"ur10e-handeye {__version__}", "created_utc": datetime.now(UTC).isoformat(),
              "board_refit": args.refit_board, "consistency": consistency(samples, x, y, target.centre),
              "reprojection_rmse_px": float(np.sqrt(np.mean(residual ** 2))),
              "limits": {"position_max_mm": args.max_mm, "rotation_max_deg": args.max_deg},
              "board_poses_reestimated": reestimated, "failures": []}
    labels = {r["index"]: r.get("label", str(r["index"])) for r in records}
    for row, rmse in zip(report["consistency"]["per_sample"], per_sample_rmse(problem, x, y), strict=True):
        row.update(label=labels[row["index"]], reprojection_rmse_px=rmse)
        row["failed_limits"] = (["position"] if row["position_mm"] > args.max_mm else []) + (
            ["rotation"] if row["rotation_deg"] > args.max_deg else [])
        if row["failed_limits"]:
            report["failures"].append(row.copy())
    report["passed"] = (report["consistency"]["position_max_mm"] <= args.max_mm
                        and report["consistency"]["rotation_max_deg"] <= args.max_deg)
    if args.output:
        write_new_json(Path(args.output), report)
    if on_report is not None:
        on_report(report)
    print(json.dumps(report, indent=2))
    return 0 if report["passed"] else 1


def cmd_locate(args) -> int:
    from .robot import URRobot
    result = load_result(args.calibration)
    if result["verdict"] == "poor":
        raise DatasetError("Poor calibration refused for locating; collect a better calibration")
    if result.get("simulation_truth") or result.get("camera", {}).get("simulated"):
        raise DatasetError("Simulated calibration cannot locate boards on live hardware")
    camera_serial = result.get("camera", {}).get("device", {}).get("serial_number")
    if camera_serial:
        if args.serial and args.serial != camera_serial:
            raise DatasetError("Requested camera serial differs from the calibration camera")
        args.serial = camera_serial
    if not args.use_fk:
        if args.active_tcp is None:
            raise ValueError("Read-only locate requires --active-tcp with the calibration-time TCP offset")
        declared, recorded = pose_to_matrix(args.active_tcp), pose_to_matrix(result["camera_in_tcp"]["tcp_offset"])
        if np.linalg.norm(declared[:3, 3] - recorded[:3, 3]) > 1e-6 \
                or angle_between_deg(declared, recorded) > 1e-4:
            raise ValueError("Declared active TCP differs from the calibration-time TCP; use --use-fk")
    target = BoardTarget(args.manifest, args.board)
    robot = URRobot(args.robot_ip, control=args.use_fk)
    try:
        robot.check_ready()
        before = robot.joints()
        with _camera(args) as camera:
            intrinsics = _intrinsics(args) or result["intrinsics"]
            observation = BoardObserver(camera, target, intrinsics=intrinsics).observe(args.frames)
        if args.use_fk:
            camera_in_base = robot.flange_pose() @ np.asarray(result["camera_in_flange"]["matrix"])
        else:
            # Read-only: the controller's active TCP must be the one recorded at calibration.
            actual = robot.tcp_pose()
            nominal = ur10e_forward(before) @ declared
            if np.linalg.norm(actual[:3, 3] - nominal[:3, 3]) > 0.01 \
                    or angle_between_deg(actual, nominal) > 2:
                raise ValueError("Actual TCP disagrees with the declared TCP and nominal UR10e kinematics")
            camera_in_base = actual @ np.asarray(result["camera_in_tcp"]["matrix"])
        robot.check_ready()
        if max(abs(a - b) for a, b in zip(before, robot.joints(), strict=True)) > math.radians(0.02):
            raise VisionError("Robot moved during locate capture; observation discarded")
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


def cmd_simulate(args, *, on_capture=None) -> int:
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
    Collector(scenario.robot, observer, session, options,
              confirm=lambda summary: True, on_capture=on_capture).run_auto()
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


def cmd_ui(args) -> int:
    from .ui import launch
    return launch(calibration=args.calibration)


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(prog="ur_handeye", description="UR10e eye-in-hand camera calibration over RTDE")
    root.add_argument("--version", action="version", version=f"ur10e-handeye {__version__}")
    commands = root.add_subparsers(dest="command", required=True)
    ui = commands.add_parser("ui", help="Open the guided desktop calibration interface")
    ui.add_argument("--calibration", help="Open a saved calibration in Results on startup")

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
    solve_cmd.add_argument("--max-consistency-mm", type=float, default=2.0)
    solve_cmd.add_argument("--max-consistency-deg", type=float, default=1.0)
    solve_cmd.add_argument("--intrinsics", help="RGB camera calibration; recompute saved board poses with this model")

    reprocess = commands.add_parser("reprocess", help="Re-run vision on saved raw frames into a new session (offline)")
    reprocess.add_argument("--session", required=True)
    reprocess.add_argument("--output", required=True, help="New session directory; preserves the original recording")
    reprocess.add_argument("--manifest", help="Override the recorded board manifest")
    reprocess.add_argument("--max-jitter-px", type=float, default=0.3)

    calibrate = commands.add_parser("calibrate", help="Estimate RGB intrinsics and solve hand-eye together (offline)")
    calibrate.add_argument("--session", required=True)
    calibrate.add_argument("--output", required=True, help="New hand-eye calibration JSON")
    calibrate.add_argument("--intrinsics-output", required=True, help="New RGB intrinsic calibration JSON")
    calibrate.add_argument("--manifest", help="Override the recorded board manifest")
    calibrate.add_argument("--exclude", type=int, nargs="*", default=[])

    intrinsic = commands.add_parser("intrinsics", help="Estimate RGB intrinsics from recorded views (offline)")
    intrinsic.add_argument("--session", required=True)
    intrinsic.add_argument("--output", required=True, help="New camera calibration JSON; preserves existing files")
    intrinsic.add_argument("--manifest", help="Override the recorded board manifest")
    intrinsic.add_argument("--exclude", type=int, nargs="*", default=[])
    intrinsic.add_argument("--fix-camera-matrix", action="store_true",
                           help="Fit only radial distortion; keep recorded focal lengths and principal point")

    verify = commands.add_parser("verify", help="Check a calibration against a separately collected session")
    verify.add_argument("--calibration", required=True)
    verify.add_argument("--session", required=True)
    verify.add_argument("--output", help="Save a new JSON verification report, including per-view failures")
    verify.add_argument("--manifest", help="Override the board manifest recorded in the session")
    verify.add_argument("--refit-board", action="store_true", help="The board was moved since calibration")
    verify.add_argument("--max-mm", type=float, default=2.0, help="Pass limit on the worst board position error")
    verify.add_argument("--max-deg", type=float, default=1.0, help="Pass limit on the worst board rotation error")

    locate = commands.add_parser("locate", help="Read-only: a board's pose in the robot base frame")
    locate.add_argument("--robot-ip", required=True)
    locate.add_argument("--calibration", required=True)
    locate.add_argument("--use-fk", action="store_true",
                        help="Use flange kinematics (takes RTDE control) instead of the active TCP")
    locate.add_argument("--active-tcp", type=float, nargs=6,
                        help="Declare the active TCP offset for read-only use (metres/radians)")
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
            "locate": cmd_locate, "simulate": cmd_simulate, "ui": cmd_ui, "intrinsics": cmd_intrinsics,
            "reprocess": cmd_reprocess, "calibrate": cmd_calibrate}


def main(argv=None):
    from .shutdown import stop_on_termination
    with stop_on_termination():
        return _main(argv)


def _main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        return COMMANDS[args.command](args)
    except (RobotError, CollectError, SolveError, DatasetError, VisionError, CameraError,
            ValueError, OSError, cv2.error) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("\nInterrupted. The robot was told to stop where it is.", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
