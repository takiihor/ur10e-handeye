"""Read-only UR state on port 30011, available in Local and Remote Control.

Cartesian Info contains controller-calibrated TCP pose and active TCP offset in
one Robot State message. Removing that offset yields tool0 without nominal FK,
URScript, or an RTDE control connection. Protocol:
https://docs.universal-robots.com/tutorials/communication-protocol-tutorials/primary-secondary-guide.html
"""
import math
import socket
import struct
import threading
import time
from dataclasses import dataclass

from .transforms import invert, pose_to_matrix

MAX_MESSAGE_BYTES = 1024 * 1024
MAX_STATE_AGE_S = 0.5


@dataclass(frozen=True)
class PrimaryState:
    timestamp_us: int
    received_at: float
    joints: tuple[float, ...]
    velocities: tuple[float, ...]
    tcp_pose: tuple[float, ...]
    tcp_offset: tuple[float, ...]

    @property
    def flange(self):
        return pose_to_matrix(self.tcp_pose) @ invert(pose_to_matrix(self.tcp_offset))


def decode_state(message: bytes, received_at: float) -> PrimaryState | None:
    """Decode complete length-prefixed messages; ignore unknown fields and message types."""
    if len(message) < 5 or struct.unpack_from('!I', message)[0] != len(message):
        raise ValueError('Invalid primary message length')
    if message[4] != 16:  # ROBOT_STATE
        return None
    parts = {}
    offset = 5
    while offset < len(message):
        if len(message) - offset < 5:
            raise ValueError('Truncated primary subpackage')
        length, kind = struct.unpack_from('!IB', message, offset)
        if length < 5 or offset + length > len(message):
            raise ValueError('Invalid primary subpackage length')
        if kind in (0, 1, 4):
            if kind in parts:
                raise ValueError('Duplicate primary state subpackage')
            parts[kind] = message[offset + 5:offset + length]
        offset += length
    if not all(kind in parts for kind in (0, 1, 4)):
        return None  # Never combine fields from different messages.
    if len(parts[0]) < 8 or len(parts[1]) < 6 * 41 or len(parts[4]) < 12 * 8:
        raise ValueError('Truncated primary robot state')
    timestamp = struct.unpack_from('!Q', parts[0])[0]
    # JointData: q_actual, q_target, qd_actual, four floats, jointMode (41 bytes per joint).
    joints = tuple(struct.unpack_from('!d', parts[1], i * 41)[0] for i in range(6))
    velocities = tuple(struct.unpack_from('!d', parts[1], i * 41 + 16)[0] for i in range(6))
    cartesian = struct.unpack_from('!12d', parts[4])
    if not all(math.isfinite(v) for v in (*joints, *velocities, *cartesian)):
        raise ValueError('Non-finite primary robot state')
    return PrimaryState(timestamp, received_at, joints, velocities, cartesian[:6], cartesian[6:])


class PrimaryMonitor:
    """Drain the read-only stream continuously so captures cannot use buffered old poses."""

    def __init__(self, host: str):
        self._condition = threading.Condition()
        self._closed = False
        self._state = self._error = None
        self._socket = self._thread = None
        try:
            self._socket = socket.create_connection((host, 30011), timeout=3)
            self._socket.settimeout(0.25)
            self._thread = threading.Thread(target=self._read, daemon=True)
            self._thread.start()
            self.snapshot(timeout=3)
        except BaseException:
            self.close()
            raise

    def _read(self):
        buffer = bytearray()
        try:
            while not self._closed:
                try:
                    chunk = self._socket.recv(65536)
                except TimeoutError:
                    continue
                if not chunk:
                    raise ConnectionError('Read-only primary state stream disconnected')
                buffer.extend(chunk)
                while len(buffer) >= 4:
                    length = struct.unpack_from('!I', buffer)[0]
                    if not 5 <= length <= MAX_MESSAGE_BYTES:
                        raise ValueError('Invalid primary message size')
                    if len(buffer) < length:
                        break
                    state = decode_state(bytes(buffer[:length]), time.monotonic())
                    del buffer[:length]
                    if state is None:
                        continue
                    with self._condition:
                        if self._state is not None:
                            if state.timestamp_us < self._state.timestamp_us:
                                raise ValueError('Primary state timestamp moved backwards; reconnect')
                            if state.timestamp_us == self._state.timestamp_us:
                                continue  # Repeated timestamps cannot refresh stale feedback.
                        self._state = state
                        self._condition.notify_all()
        except Exception as exc:
            with self._condition:
                self._error = exc
                self._condition.notify_all()

    def snapshot(self, *, after: float | None = None, timeout: float = 1) -> PrimaryState:
        deadline = time.monotonic() + timeout
        with self._condition:
            while True:
                if self._closed:
                    raise ConnectionError('Read-only primary state monitor is closed')
                if self._error:
                    raise ConnectionError(f'Read-only primary state failed: {self._error}') from self._error
                state = self._state
                now = time.monotonic()
                if state is not None and now - state.received_at <= MAX_STATE_AGE_S:
                    if after is None or state.received_at > after:
                        return state
                remaining = deadline - now
                if remaining <= 0:
                    raise TimeoutError('No fresh calibrated robot pose on read-only port 30011')
                self._condition.wait(min(remaining, 0.1))

    def close(self):
        with self._condition:
            self._closed = True
            self._condition.notify_all()
        if self._socket is not None:
            try:
                self._socket.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            self._socket.close()
        if self._thread is not None:
            self._thread.join(timeout=1)
