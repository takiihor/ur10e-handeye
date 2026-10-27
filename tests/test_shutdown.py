import signal

import pytest

from ur_handeye.shutdown import stop_on_termination


def test_sigterm_uses_interrupt_path_and_restores_handler():
    previous = signal.getsignal(signal.SIGTERM)
    with pytest.raises(KeyboardInterrupt), stop_on_termination():
        signal.raise_signal(signal.SIGTERM)
    assert signal.getsignal(signal.SIGTERM) == previous
