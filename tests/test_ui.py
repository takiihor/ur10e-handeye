import io
import json
import os
import subprocess
import sys
import time

import pytest

from ur_handeye.ui import CalibrationUI, build_command
from ur_handeye.ui_jobs import JobRunner
from ur_handeye.ui_worker import Channel, LogStream, run


@pytest.fixture
def settings(tmp_path):
    manifest = tmp_path / "board_manifest.json"
    manifest.write_text("{}")
    return {
        "robot_ip": "192.0.2.1",
        "manifest": str(manifest),
        "board": "calibration",
        "serial": "camera-123",
        "session": str(tmp_path / "calibration"),
        "speed": "0.1",
        "views": "14",
    }


def test_ui_commands_keep_approvals_and_isolate_dry_run_data(settings):
    collect = build_command("collect", settings)
    plan = build_command("plan", settings)
    assert "--yes" not in collect and "--freedrive" not in collect
    assert "--dry-run" in plan
    assert plan[plan.index("--output") + 1] != settings["session"]
    assert collect[collect.index("--joint-speed") + 1] == "0.1"
    assert collect[collect.index("--serial") + 1] == "camera-123"
    assert build_command("solve", settings) == ["solve", "--session", settings["session"]]


@pytest.mark.parametrize("action", ["collect", "simulate", "manual", "replay"])
def test_existing_data_is_refused_before_starting_a_worker(settings, tmp_path, action):
    directory = tmp_path / "existing"
    directory.mkdir()
    (directory / "samples.jsonl").write_text("saved data")
    settings["session"] = str(directory)
    with pytest.raises(ValueError, match="already contains data"):
        build_command(action, settings, verification=str(directory))


@pytest.mark.parametrize(
    "key,value",
    [
        ("speed", "nan"),
        ("speed", "1.1"),
        ("views", "40"),
        ("views", "2"),
        ("robot_ip", ""),
        ("manifest", "/missing.json"),
    ],
)
def test_invalid_hardware_settings_are_refused(settings, key, value):
    settings[key] = value
    with pytest.raises(ValueError):
        build_command("collect", settings)


def test_verification_uses_a_separate_session(settings, tmp_path):
    with pytest.raises(ValueError, match="separate"):
        build_command("verify", settings, calibration="result.json", verification=settings["session"])
    verification = tmp_path / "check"
    with pytest.raises(ValueError, match="Collect fresh verification"):
        build_command("verify", settings, calibration="result.json", verification=str(verification))
    verification.mkdir()
    for name in ("session.json", "samples.jsonl"):
        (verification / name).write_text("{}")
    command = build_command("verify", settings, calibration="result.json", verification=str(verification))
    assert command[0] == "verify" and "--refit-board" not in command


def test_ui_intrinsics_apply_to_check_capture_and_offline_solving(settings, tmp_path):
    from ur_handeye.vision import SIMULATED_INTRINSICS
    path = tmp_path / "rgb.json"
    path.write_text(json.dumps(SIMULATED_INTRINSICS))
    settings["intrinsics"] = str(path)
    for action in ("check", "plan", "collect", "solve"):
        command = build_command(action, settings)
        assert command[command.index("--intrinsics") + 1] == str(path)
    command = build_command("intrinsics", settings)
    assert command[0] == "intrinsics"
    assert "--robot-ip" not in command


def test_ui_resolving_creates_a_new_result_filename(settings):
    from pathlib import Path
    folder = Path(settings["session"])
    folder.mkdir()
    (folder / "calibration.json").write_text("old evidence")
    command = build_command("solve", settings)
    assert command[command.index("--output") + 1] != str(folder / "calibration.json")


@pytest.mark.parametrize("reply,expected", [(True, True), (False, False), ("yes", False)])
def test_only_a_current_explicit_boolean_can_approve_motion(reply, expected):
    output = io.StringIO()
    channel = Channel(output, io.StringIO(json.dumps({"id": 1, "value": reply}) + "\n"))
    assert channel.request("motion", "Review targets") is expected
    event = json.loads(output.getvalue())
    assert event["kind"] == "motion" and event["summary"] == "Review targets"


def test_disconnected_and_stale_prompt_replies_never_approve():
    with pytest.raises(KeyboardInterrupt):
        Channel(io.StringIO(), io.StringIO()).request("motion", "plan")
    with pytest.raises(ValueError, match="Stale"):
        Channel(io.StringIO(), io.StringIO('{"id": 9, "value": true}\n')).request("motion", "plan")


def test_worker_never_accepts_unattended_collection_or_other_commands():
    channel = Channel(io.StringIO(), io.StringIO())
    with pytest.raises(ValueError, match="requires motion approval"):
        run(["collect", "--robot-ip", "192.0.2.1", "--yes"], channel)
    with pytest.raises(ValueError, match="Unsupported"):
        run(["locate", "--robot-ip", "192.0.2.1", "--calibration", "a.json"], channel)


def test_log_stream_preserves_partial_lines():
    output = io.StringIO()
    stream = LogStream(Channel(output, io.StringIO()))
    stream.write("first")
    stream.write(" line\nlast line")
    stream.flush()
    assert [json.loads(line)["message"] for line in output.getvalue().splitlines()] == ["first line", "last line"]


def wait_events(runner, predicate, timeout=30):
    events = []
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        events.extend(runner.drain())
        if predicate(events):
            return events
        time.sleep(0.02)
    runner.stop()
    pytest.fail("UI worker did not respond before the test deadline")


def test_runner_stop_keeps_ownership_until_cleanup_acknowledges(monkeypatch):
    # A real child process waits for approval, then for interruption. Its finally block is the
    # important contract: cleanup must finish before another UI operation becomes available.
    child = """
import sys, time
from ur_handeye.ui_worker import Channel
channel = Channel(sys.stdout, sys.stdin)
try:
    channel.request('motion', 'fake target; no hardware')
    while True:
        time.sleep(0.02)
except KeyboardInterrupt:
    channel.emit('log', message='stopped in place')
finally:
    time.sleep(0.1)
    channel.emit('log', message='cleanup finished')
"""
    original = subprocess.Popen

    def launch(argv, **kwargs):
        return original([sys.executable, "-u", "-c", child], **kwargs)

    monkeypatch.setattr("ur_handeye.ui_jobs.subprocess.Popen", launch)
    runner = JobRunner()
    try:
        runner.start([])
        request = wait_events(runner, lambda es: any(e["type"] == "request" for e in es))[-1]
        with pytest.raises(RuntimeError, match="still owns"):
            runner.start([])
        with pytest.raises(ValueError, match="no longer active"):
            runner.answer(request["id"] + 1, True)
        runner.answer(request["id"], True)
        with pytest.raises(ValueError, match="no longer active"):
            runner.answer(request["id"], True)
        runner.stop()
        runner.stop()  # repeated Stop must not interrupt cleanup again
        assert runner.active and runner.stopping
        events = wait_events(runner, lambda es: any(e["type"] == "done" for e in es))
        assert [e["message"] for e in events if e["type"] == "log"] == ["stopped in place", "cleanup finished"]
        assert not runner.active
    finally:
        if runner.active:
            runner.stop()
            runner.process.wait(timeout=5)


def test_lost_ui_interrupts_worker_without_approval(tmp_path):
    process = subprocess.Popen(
        [sys.executable, "-u", "-m", "ur_handeye.ui_worker", "simulate", "--output", str(tmp_path / "simulation")],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    try:
        # Wait until its signal/watchdog setup and first real capture have completed.
        while True:
            line = process.stdout.readline()
            assert line, "worker exited before the first simulated observation"
            if json.loads(line).get("type") == "observation":
                break
        process.stdin.close()
        process.stdin = None
        output, _ = process.communicate(timeout=20)
        assert process.returncode == 130
        assert "interrupted" in output
        assert not (tmp_path / "simulation" / "calibration.json").exists()
    finally:
        if process.poll() is None:
            process.send_signal(2)
            process.wait(timeout=5)


@pytest.fixture
def desktop():
    tk = pytest.importorskip("tkinter")
    if not os.environ.get("DISPLAY"):
        pytest.skip("A desktop display is needed for widget checks; worker checks run headlessly")
    root = tk.Tk()
    root.withdraw()
    app = CalibrationUI(root)
    try:
        yield root, app
    finally:
        if app.runner.active:
            app.runner.stop()
            app.runner.process.wait(timeout=10)
        try:
            root.destroy()
        except tk.TclError:
            pass


def test_desktop_simulation_displays_results_and_refuses_live_replay(desktop, tmp_path, monkeypatch):
    root, app = desktop
    app.settings["session"].set(str(tmp_path / "sim"))
    app.start("simulate")
    deadline = time.monotonic() + 40
    while app.runner.active and time.monotonic() < deadline:
        root.update()
        time.sleep(0.03)
    assert not app.runner.active
    assert app.result["verdict"] == "good"
    assert "SIMULATION" in app.result_status.get()
    assert len(app.metrics.get_children()) >= 9
    assert app.preview_photo is not None
    assert app.accepted >= 16
    assert app.pose.get().startswith("[")
    for value in app.ready:
        value.set(True)
    errors = []
    monkeypatch.setattr(app, "error", errors.append)
    app.start("replay")
    assert not app.runner.active
    assert "simulated calibration" in str(errors[-1])


def test_desktop_hardware_steps_require_check_and_plan(desktop, settings, monkeypatch):
    _, app = desktop
    for key, value in settings.items():
        app.settings[key].set(value)
    for value in app.ready:
        value.set(True)
    errors = []
    monkeypatch.setattr(app, "error", errors.append)
    app.start("collect")
    assert "Check view" in str(errors[-1]) and not app.runner.active
    app.checked = app.hardware_key()
    app.start("collect")
    assert "preliminary plan" in str(errors[-1]) and not app.runner.active


def test_desktop_intrinsic_estimation_selects_model_and_resolves_without_overwriting(desktop, collected, monkeypatch):
    from pathlib import Path
    root, app = desktop
    _, directory, _ = collected
    app.settings["session"].set(str(directory))
    monkeypatch.setattr(app, "error", lambda message: pytest.fail(str(message)))
    app.start("intrinsics")
    deadline = time.monotonic() + 30
    while app.runner.active and time.monotonic() < deadline:
        root.update()
        time.sleep(0.02)
    assert not app.runner.active
    intrinsic_path = Path(app.settings["intrinsics"].get())
    assert intrinsic_path.is_file()
    assert "solve the session again" in app.result_status.get()
    assert app.step_lights["intrinsics"].cget("text") == "● Review warnings"
    # A pre-existing solve must survive a new camera-model solve.
    original = directory / "calibration.json"
    original.write_text("existing calibration evidence")
    app.start("solve")
    deadline = time.monotonic() + 30
    while app.runner.active and time.monotonic() < deadline:
        root.update()
        time.sleep(0.02)
    assert not app.runner.active
    assert app.result is not None
    assert app.result["intrinsics_source"]["board_poses_reestimated"] is True
    assert Path(app.result_path) != original
    assert original.read_text() == "existing calibration evidence"
    original.unlink()


def test_step_lights_follow_completion_failure_and_settings(desktop, settings, monkeypatch):
    _, app = desktop
    for key, value in settings.items():
        app.settings[key].set(value)
    for value in app.ready:
        value.set(True)
    assert app.step_lights["setup"].cget("text") == "● Ready"
    monkeypatch.setattr(app.runner, "start", lambda argv: None)
    monkeypatch.setattr(app, "error", lambda message: pytest.fail(str(message)))
    app.start("check")
    assert app.step_lights["check"].cget("text") == "● Running"
    app.runner.events.put({"type": "done", "code": 0})
    app._poll_events()
    assert app.step_lights["check"].cget("text") == "● Ready"
    assert app.checked == app.hardware_key()
    app.start("plan")
    app.runner.events.put({"type": "done", "code": 2})
    app._poll_events()
    assert app.step_lights["plan"].cget("text") == "● Failed"
    app.settings["robot_ip"].set("192.0.2.2")
    assert app.step_lights["check"].cget("text") == "● Pending"
    assert app.step_lights["plan"].cget("text") == "● Pending"
    assert app.checked is None and app.planned is None


def test_verification_light_requires_pass_and_clears_for_new_folder(desktop):
    _, app = desktop
    app.action = "verify"
    app.verification_passed = False
    for passed, expected in ((False, "● Failed"), (True, "● Ready")):
        app.runner.events.put({"type": "verification", "report": {
            "passed": passed, "consistency": {"position_max_mm": 1, "rotation_max_deg": 0.2}}})
        app.runner.events.put({"type": "done", "code": 0})
        app._poll_events()
        assert app.step_lights["verify"].cget("text") == expected
    app.verification.set("sessions/new-verification")
    assert app.step_lights["verify"].cget("text") == "● Pending"
    app.action = "manual"
    app.runner.events.put({"type": "done", "code": 130})
    app._poll_events()
    assert app.step_lights["manual"].cget("text") == "● Stopped"


def test_display_error_requests_stop_and_keeps_processing_events(desktop, monkeypatch):
    _, app = desktop
    stops = []
    monkeypatch.setattr(app.runner, "stop", lambda: stops.append(True))
    app.runner.events.put({"type": "observation"})  # corrupt display payload
    app.poll()
    assert stops == [True]
    assert "Display update failed" in app.result_status.get()
    app.runner.events.put({"type": "log", "message": "cleanup still monitored"})
    app.poll()
    assert "cleanup still monitored" in app.log.get("1.0", "end")


def test_close_waits_for_running_worker_cleanup(desktop, tmp_path):
    root, app = desktop
    directory = tmp_path / "interrupted"
    app.settings["session"].set(str(directory))
    app.start("simulate")
    deadline = time.monotonic() + 20
    while app.accepted == 0 and app.runner.active and time.monotonic() < deadline:
        root.update()
        time.sleep(0.02)
    assert app.accepted >= 1 and app.runner.active
    app.close()
    assert app.runner.active and root.winfo_exists()
    while app.runner.active and time.monotonic() < deadline:
        root.update()
        time.sleep(0.02)
    assert not app.runner.active
    assert not (directory / "calibration.json").exists()
