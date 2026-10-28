"""Nonblocking desktop job runner. One child owns hardware until cleanup finishes."""

import json
import queue
import signal
import subprocess
import sys
import threading


class JobRunner:
    def __init__(self):
        self.process = None
        self.events = queue.Queue()
        self.pending = None
        self.stopping = False

    @property
    def active(self):
        # Cleared only after stdout closes and the exit event reaches the UI.
        return self.process is not None

    def start(self, argv):
        if self.active:
            raise RuntimeError("Another operation still owns the robot/camera")
        self.pending, self.stopping = None, False
        self.process = subprocess.Popen(
            [sys.executable, "-u", "-m", "ur_handeye.ui_worker", *argv],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )
        threading.Thread(target=self._read, args=(self.process,), daemon=True).start()

    def _read(self, process):
        try:
            for line in process.stdout:
                try:
                    event = json.loads(line)
                    if not isinstance(event, dict) or "type" not in event:
                        raise ValueError("Not a UI event")
                except (ValueError, TypeError):
                    event = {"type": "log", "message": line.rstrip()}
                self.events.put(event)
        except OSError as exc:
            self.events.put({"type": "log", "message": f"Worker communication failed: {exc}"})
            self.stop()
        finally:
            code = process.wait()
            process.stdout.close()
            process.stdin.close()
            self.events.put({"type": "done", "code": code})

    def drain(self):
        while True:
            try:
                event = self.events.get_nowait()
            except queue.Empty:
                return
            if event["type"] == "request":
                self.pending = event["id"]
            elif event["type"] == "done":
                self.process, self.pending = None, None
            yield event

    def answer(self, request_id, value):
        if not self.active or self.stopping or request_id != self.pending:
            raise ValueError("This operator request is no longer active")
        self.process.stdin.write(json.dumps({"id": request_id, "value": value}) + "\n")
        self.process.stdin.flush()
        self.pending = None

    def stop(self):
        if self.active and not self.stopping:
            self.stopping, self.pending = True, None
            if self.process.poll() is None:
                self.process.send_signal(signal.SIGINT)
