import pytest
import serial

from astro.devices.handset import Handset, parse_reply


@pytest.mark.parametrize("data,expected", [
    (b"+00123\t-04567\r", (123, -4567)),
    (b"junk+09215\t+00000\r", (9215, 0)),
    (b"+00123\t-045", None),  # incomplete
    (b"", None),
])
def test_parse_reply(data, expected):
    assert parse_reply(data) == expected


class FakePort:
    def __init__(self, replies):
        self.replies, self.written = list(replies), []

    def reset_input_buffer(self):
        pass

    def write(self, data):
        self.written.append(data)

    def read_until(self, terminator, size):
        return self.replies.pop(0) if self.replies else b""

    def close(self):
        pass


def test_counts_queries_and_keeps_last_good_reading():
    port = FakePort([b"+00010\t+00020\r", b"", b"+00011\t+00021\r"])
    h = Handset("/dev/ttyUSB0", serial_factory=lambda *a, **k: port)
    assert h.counts() == (10, 20)
    assert h.counts() == (10, 20) and h.failures == 1  # no answer: last good reading
    assert h.counts() == (11, 21) and h.failures == 0
    assert port.written == [b"Q"] * 3


class UnpluggedPort(FakePort):
    def write(self, data):
        raise serial.SerialException("device disconnected")


def test_unplugged_adapter_counts_as_failure():
    h = Handset("/dev/ttyUSB0", serial_factory=lambda *a, **k: UnpluggedPort([]))
    assert h.counts() == (0, 0) and h.failures == 1


def test_read_allows_leading_noise():
    port = FakePort([b"junk+09215\t+00000\r"])
    sizes = []
    real_read = port.read_until
    port.read_until = lambda term, size: (sizes.append(size), real_read(term, size))[1]
    assert Handset("p", serial_factory=lambda *a, **k: port).counts() == (9215, 0)
    assert sizes[0] >= len(b"junk+09215\t+00000\r")
