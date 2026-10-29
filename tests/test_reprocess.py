import json

import pytest

from ur_handeye.cli import main
from ur_handeye.dataset import Session, load_session
from ur_handeye.simulation import build_scenario
from ur_handeye.transforms import matrix_to_pose
from ur_handeye.vision import BoardObserver


@pytest.fixture
def recorded(tmp_path):
    scenario = build_scenario(seed=1, image_noise=0.0)
    observation = BoardObserver(scenario.camera, scenario.target).observe(5)
    source = tmp_path / "source"
    session = Session(source, {"board": scenario.target.describe(), "intrinsics": observation.intrinsics,
                               "simulated": True, "tcp_offset": [0] * 6})
    for reasons in (["no_pose", "corner_refinement_failed"], ["robot_moved_during_capture"], ["unknown_fault"]):
        session.add({"label": "view", "status": "rejected", "reasons": reasons,
                     "flange_pose": matrix_to_pose(scenario.robot.flange_pose()),
                     "observation": observation.to_record()}, observation.image_bgr, observation.frames)
    return source


def test_reprocess_recovers_vision_only_failure_and_preserves_physical_faults(recorded, tmp_path, monkeypatch):
    def connect(*args, **kwargs):
        pytest.fail("offline reprocessing opened robot control")

    monkeypatch.setattr("ur_handeye.robot.URRobot", connect)
    originals = {p: p.read_bytes() for p in recorded.rglob("*") if p.is_file()}
    output = tmp_path / "derived"
    argv = ["reprocess", "--session", str(recorded), "--output", str(output)]
    assert main(argv) == 0
    metadata, records = load_session(output)
    _, sources = load_session(recorded)
    assert [r["status"] for r in records] == ["ok", "rejected", "rejected"]
    assert records[1]["reasons"] == ["robot_moved_during_capture"]
    assert records[2]["reasons"] == ["unknown_fault"]
    assert metadata["reprocessing"]["source_session"] == str(recorded)
    assert metadata["reprocessing"]["source_samples_sha256"]
    for source, result in zip(sources, records, strict=True):
        assert source["flange_pose"] == result["flange_pose"]
        assert source["time_unix"] == result["time_unix"]
        assert len(result["camera_frames"]) == 5
        assert result["reprocessing"]["source_status"] == "rejected"
    assert all(p.read_bytes() == original for p, original in originals.items())
    assert main(argv) == 2  # never replace a previous result


def test_reprocess_refuses_incomplete_raw_recording(recorded, tmp_path, capsys):
    path = recorded / "samples.jsonl"
    lines = [json.loads(line) for line in path.read_text().splitlines()]
    lines[0]["camera_frames"].pop()
    path.write_text("".join(json.dumps(r) + "\n" for r in lines))
    output = tmp_path / "derived"
    assert main(["reprocess", "--session", str(recorded), "--output", str(output)]) == 2
    assert "complete raw frames" in capsys.readouterr().err
    assert not output.exists()
