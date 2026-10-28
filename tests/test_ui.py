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
    command = build_command("verify", settings, calibration="result.json", verification=str(tmp_path / "check"))
    assert command[0] == "verify" and "--refit-board" not in command


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
