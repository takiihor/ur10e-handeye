"""Translate SIGTERM into the CLI's normal interruption and cleanup path."""
import signal
import threading
from contextlib import contextmanager


def _terminate(number, frame):
    raise KeyboardInterrupt


@contextmanager
def stop_on_termination():
    if threading.current_thread() is not threading.main_thread():
        yield
        return
    previous = signal.signal(signal.SIGTERM, _terminate)
    try:
        yield
    finally:
        signal.signal(signal.SIGTERM, previous)
