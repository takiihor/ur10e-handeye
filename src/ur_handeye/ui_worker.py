"""Run one calibration command in a child process with explicit UI messages.

Hardware ownership and SIGINT cleanup stay in the same process as the existing CLI.
Stdout carries JSON messages; stdin carries replies to numbered operator requests.
"""

import base64
import contextlib
import io
import json
import queue
import signal
import sys
import threading
from pathlib import Path

import cv2

from . import cli
from .dataset import load_result
from .shutdown import stop_on_termination


class Channel:
    def __init__(self, output, replies):
        self.output, self.replies = output, replies
        self.request_id = 0
        self.inbox = None
        self.closed = threading.Event()
        self.on_disconnect = None

    def watch_parent(self):
        """Notice a vanished UI even during motion, rather than only at the next prompt."""
        self.inbox = queue.Queue()

        def read():
            for line in self.replies:
                self.inbox.put(line)
            self.inbox.put(None)
            if not self.closed.is_set():
                # Signal the main thread so the normal move/RTDE cleanup path runs.
                signal.raise_signal(signal.SIGINT)

        threading.Thread(target=read, daemon=True).start()

    def emit(self, event_type, **values):
        try:
            self.output.write(json.dumps({"type": event_type, **values}, allow_nan=False) + "\n")
            self.output.flush()
        except BrokenPipeError:
            pass  # the stdin watchdog interrupts the command when the UI disappears

    def request(self, kind, summary):
        self.request_id += 1
        self.emit("request", id=self.request_id, kind=kind, summary=summary)
        line = self.inbox.get() if self.inbox is not None else self.replies.readline()
        if not line:
            if self.on_disconnect is not None:
                self.on_disconnect()
            raise KeyboardInterrupt  # the UI disappeared; never approve by default
        reply = json.loads(line)
        if reply.get("id") != self.request_id:
            raise ValueError("Stale operator response refused")
        value = reply.get("value")
        if kind == "motion":
            return value is True
        if value not in ("", "q"):
            raise ValueError("Invalid manual capture response")
        return value

    def observation(self, observation, record=None):
        usable, reasons = observation.usable(0.3)
        image = None
        if observation.image_bgr is not None:
            picture = observation.image_bgr.copy()
            for x, y in observation.pixels:
                cv2.circle(picture, (round(float(x)), round(float(y))), 5, (70, 210, 70), 2)
            height, width = picture.shape[:2]
            if width > 800:
                picture = cv2.resize(picture, (800, round(height * 800 / width)))
            ok, encoded = cv2.imencode(".jpg", picture, [cv2.IMWRITE_JPEG_QUALITY, 85])
            if ok:
                image = base64.b64encode(encoded).decode("ascii")
        self.emit(
            "observation",
            image=image,
            corners=len(observation.charuco_ids),
            jitter_px=observation.jitter_px,
            rmse_px=observation.pose["reprojection_rmse_px"] if observation.pose else None,
            label=record["label"] if record else "Board check",
            status=record["status"] if record else ("ok" if usable else "rejected"),
            reasons=record["reasons"] if record else reasons,
        )


class LogStream(io.TextIOBase):
    def __init__(self, channel):
        self.channel, self.pending = channel, ""

    def write(self, value):
        self.pending += value
        while "\n" in self.pending:
            line, self.pending = self.pending.split("\n", 1)
            self.channel.emit("log", message=line)
        return len(value)

    def flush(self):
        if self.pending:
            self.channel.emit("log", message=self.pending)
            self.pending = ""


def run(argv, channel):
    args = cli.parser().parse_args(argv)
    if args.command not in ("check", "collect", "solve", "verify", "simulate", "intrinsics"):
        raise ValueError("Unsupported UI operation")
    if args.command == "collect":
        if args.yes or args.freedrive:
            raise ValueError("The UI requires motion approval and pendant jogging for manual captures")
        code = cli.cmd_collect(
            args,
            confirm=lambda summary: channel.request("motion", summary),
            read_line=lambda summary: channel.request("manual", summary),
            on_capture=lambda record, observation: channel.observation(observation, record),
        )
    elif args.command == "check":
        code = cli.cmd_check(args, on_observation=channel.observation)
    elif args.command == "simulate":
        code = cli.cmd_simulate(args, on_capture=lambda record, obs: channel.observation(obs, record))
    elif args.command == "verify":
        code = cli.cmd_verify(args, on_report=lambda report: channel.emit("verification", report=report))
    elif args.command == "intrinsics":
        code = cli.cmd_intrinsics(
            args, on_result=lambda model, path: channel.emit("intrinsics", model=model, path=path))
    else:
        code = cli.cmd_solve(args)
    if args.command in ("solve", "simulate"):
        path = (
            Path(args.output) / "calibration.json"
            if args.command == "simulate"
            else Path(args.output or Path(args.session) / "calibration.json")
        )
        channel.emit("result", path=str(path.resolve()), result=load_result(path))
    return code


def main(argv=None):
    channel = Channel(sys.stdout, sys.stdin)
    log = LogStream(channel)
    with contextlib.redirect_stdout(log), contextlib.redirect_stderr(log), stop_on_termination():
        previous = signal.getsignal(signal.SIGINT)
        previous_term = signal.getsignal(signal.SIGTERM)
        interrupted = False

        def interrupt(number, frame):
            nonlocal interrupted
            if not interrupted:
                interrupted = True
                raise KeyboardInterrupt

        signal.signal(signal.SIGINT, interrupt)
        signal.signal(signal.SIGTERM, interrupt)
        channel.on_disconnect = lambda: interrupt(None, None)
        try:
            channel.watch_parent()
            return run(sys.argv[1:] if argv is None else argv, channel)
        except KeyboardInterrupt:
            channel.emit("log", message="Operation interrupted; hardware cleanup requested. Check the robot state.")
            return 130
        except Exception as exc:
            channel.emit("log", message=f"error: {exc}")
            return 2
        finally:
            channel.closed.set()
            signal.signal(signal.SIGINT, previous)
            signal.signal(signal.SIGTERM, previous_term)
            log.flush()


if __name__ == "__main__":
    raise SystemExit(main())
