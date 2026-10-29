
import numpy as np
import pytest

from ur_handeye.collect import CollectError, CollectOptions, Collector
from ur_handeye.dataset import Session, load_session
from ur_handeye.robot import RobotError
from ur_handeye.simulation import build_scenario
from ur_handeye.vision import BoardObserver


def make(tmp_path, scenario=None, confirm=lambda summary: True):
    scenario = scenario or build_scenario(seed=5)
    session = Session(tmp_path / "s", {"tcp_offset": scenario.robot.tcp_offset(), "board": scenario.target.describe()})
    logs = []
    collector = Collector(scenario.robot, BoardObserver(scenario.camera, scenario.target), session,
                          CollectOptions(frames=2), log=logs.append, confirm=confirm)
    return scenario, collector, logs


def test_auto_collection_records_every_view_and_returns_home(collected):
    scenario, directory, collector = collected
    metadata, records = load_session(directory)
    labels = [r["label"] for r in records]
    assert labels[0] == "start" and sorted(labels[1:4]) == ["probe-x", "probe-y", "probe-z"]
    assert sum(label.startswith("probe") for label in labels) == 3
    assert collector.usable == len(records) >= 16
    assert all(r["status"] == "ok" for r in records)
    assert scenario.robot.joints() == pytest.approx(metadata["start_joints"])
    assert (directory / records[-1]["image"]).is_file()
    assert metadata["intrinsics"]["width"] == 1280
    assert metadata["plan"]["targets"] and len(metadata["plan"]["targets"][0]["joints"]) == 6
    assert len(records[0]["camera_frames"]) == 5
    assert all((directory / frame["image"]).is_file() for frame in records[0]["camera_frames"])


def test_dry_run_never_moves(tmp_path):
    scenario, collector, logs = make(tmp_path)
    planned = collector.run_auto(dry_run=True)
    assert planned >= 10 and scenario.robot.moves == 0
    assert any("Dry run" in line for line in logs)


def test_operator_refusal_means_no_motion(tmp_path):
    scenario, collector, _ = make(tmp_path, confirm=lambda summary: False)
    with pytest.raises(CollectError, match="Cancelled"):
        collector.run_auto()
    assert scenario.robot.moves == 0


def test_robot_fault_stops_without_recovery_motion(tmp_path):
    scenario, collector, logs = make(tmp_path)
    stops = []
    scenario.robot.stop = lambda: stops.append(True)
    scenario.robot.fail_on_move = 9
    with pytest.raises(RobotError, match="protective stop"):
        collector.run_auto()
    assert stops == [True] and scenario.robot.moves == 9  # nothing sent after the fault
    assert any("stopped where it is" in line for line in logs)
    _, records = load_session(tmp_path / "s")
    assert len(records) >= 4  # samples captured before the fault are kept


def test_start_without_board_is_refused_before_motion(tmp_path):
    scenario = build_scenario(seed=5)
    scenario.camera.board_in_base = scenario.camera.board_in_base.copy()
    scenario.camera.board_in_base[:3, 3] += [2.0, 0, 0]
    scenario, collector, _ = make(tmp_path, scenario)
    with pytest.raises(CollectError, match="not fully and sharply visible"):
        collector.run_auto()
    assert scenario.robot.moves == 0


def test_manual_mode_captures_with_freedrive_released(tmp_path):
    scenario, collector, _ = make(tmp_path)
    states = []
    original = collector.capture

    def capture(label):
        states.append(scenario.robot.freedrive)
        return original(label)
    collector.capture = capture
    answers = iter(["", "", "q"])
    assert collector.run_manual(lambda prompt: next(answers), freedrive=True) == 2
    assert states == [False, False] and scenario.robot.freedrive is False


def test_replay_repeats_views_and_rejects_big_jumps(tmp_path, collected):
    _, directory, _ = collected
    _, records = load_session(directory)
    targets = [r["joints"] for r in records if not r["label"].startswith(("start", "probe"))][:4]
    scenario, collector, _ = make(tmp_path)
    assert collector.run_replay(targets) == 4
    assert scenario.robot.moves == 5
    far = [list(np.add(targets[0], [0, 0, 0, 0, 0, 2.0]))]
    _, collector, _ = make(tmp_path / "b")
    with pytest.raises(CollectError, match="joint step"):
        collector.run_replay(far)


def test_replay_dry_run_checks_every_target_without_motion(tmp_path):
    scenario, collector, _ = make(tmp_path)
    seen = []
    validate = scenario.robot.validate_joints
    scenario.robot.validate_joints = lambda q: (seen.append(q), validate(q))[1]
    assert collector.run_replay([scenario.robot.joints()], dry_run=True) == 1
    assert len(seen) == 2  # target and return pose
    assert scenario.robot.moves == 0


def test_replay_routes_large_transitions_through_start(tmp_path):
    scenario, collector, _ = make(tmp_path)
    start = scenario.robot.joints()
    targets = [list(np.add(start, [0, 0, 0, 0, 0, offset])) for offset in (0.8, -0.8)]
    assert collector.run_replay(targets, dry_run=True) == 2
    assert scenario.robot.moves == 0
    assert collector.run_replay(targets) == 2
    assert scenario.robot.moves == 4
    assert scenario.robot.joints() == pytest.approx(start)


def test_unsafe_replay_target_refused_before_first_move(tmp_path):
    scenario, collector, _ = make(tmp_path)
    def unsafe(q):
        raise RobotError("unsafe saved target")
    scenario.robot.validate_joints = unsafe
    with pytest.raises(RobotError, match="unsafe saved"):
        collector.run_replay([scenario.robot.joints()])
    assert scenario.robot.moves == 0


def test_operator_can_refuse_updated_plan_without_return_motion(tmp_path):
    answers = iter([True, True, False])
    scenario, collector, _ = make(tmp_path, confirm=lambda summary: next(answers))
    with pytest.raises(CollectError, match="updated calibration"):
        collector.run_auto()
    assert scenario.robot.joints() != pytest.approx(scenario.robot.HOME)


def test_session_folder_is_never_reused(tmp_path):
    make(tmp_path)
    with pytest.raises(Exception, match="not empty"):
        make(tmp_path)
