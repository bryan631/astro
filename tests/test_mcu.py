import math

import pytest

from astro.devices.dew import heater_percent
from astro.devices.mcu_protocol import (
    Environment,
    McuError,
    Position,
    Version,
    dew_point_c,
    frame,
    heat,
    parse,
)


@pytest.mark.parametrize("body,expected", [
    ("POS 1234 -56", Position(1234, -56)),
    ("ENV 24.5 81.0 nan", Environment(24.5, 81.0, math.nan)),
    ("VER astro-mcu 0.1", Version("astro-mcu", "0.1")),
    ("ERR bme280 missing", McuError("bme280 missing")),
])
def test_roundtrip(body, expected):
    got = parse(frame(body))
    if isinstance(expected, Environment):
        assert got.temp_c == 24.5 and math.isnan(got.optic_c)
    else:
        assert got == expected


@pytest.mark.parametrize("line", [b"POS 1234 -56*00\n", b"POS 1234 -56\n", b"POS 12x4 -56*" +
                                  frame("POS 12x4 -56")[-3:-1] + b"\n", b"\xff\xfe*AA\n", b""])
def test_garbled_lines_dropped(line):
    assert parse(line) is None


def test_heat_command_clamps():
    assert heat(1, 140) == frame("HEAT 1 100") and heat(0, -5) == frame("HEAT 0 0")


def test_dew_point_magnus():
    assert dew_point_c(25, 100) == pytest.approx(25, abs=0.01)
    assert dew_point_c(25, 50) == pytest.approx(13.9, abs=0.2)  # standard table value
    assert dew_point_c(24, 85) == pytest.approx(21.3, abs=0.3)  # muggy Florida night


def test_heater_steps_and_humid_floor():
    dry = Environment(20, 40, math.nan)  # dew point ~6 C: far away
    assert heater_percent(dry) == 0
    muggy = Environment(24, 90, math.nan)  # margin ~1.6 C
    assert heater_percent(muggy) == 85
    cold_optic = Environment(24, 80, 19.0)  # optic radiated below the dew point (~20.3 C)
    assert heater_percent(cold_optic) == 100
    humid_but_warm = Environment(30, 86, 40.0)
    assert heater_percent(humid_but_warm) == 20


class FakeSerial:
    """Feeds queued lines to readline() and records writes."""

    def __init__(self, lines):
        self.lines, self.written = list(lines), []

    def readline(self):
        import time

        if self.lines:
            return self.lines.pop(0)
        time.sleep(0.01)
        return b""

    def write(self, data):
        self.written.append(data)

    def close(self):
        pass


def make_mcu(lines):
    from astro.devices.mcu import Mcu

    fake = FakeSerial(lines)
    return Mcu("/dev/null", serial_factory=lambda *a, **k: fake), fake


def test_mcu_tracks_position_and_drives_heaters_from_env():
    import time

    mcu, fake = make_mcu([frame("VER astro-mcu 0.1"), b"garbage\n", frame("POS 100 -20"),
                          frame("ENV 24 90 nan")])
    mcu.start()
    time.sleep(0.2)
    assert mcu.counts() == (100, -20) and mcu.version == Version("astro-mcu", "0.1")
    assert mcu.heat == 85
    assert heat(0, 85) in fake.written and heat(1, 85) in fake.written
    assert fake.written[0] == frame("VER?")
    mcu.close()
    assert fake.written[-2:] == [heat(0, 0), heat(1, 0)]  # heaters off on shutdown


def test_failed_sensor_read_turns_heaters_off():
    assert heater_percent(Environment(math.nan, math.nan, math.nan)) == 0


def test_empty_valid_line_dropped():
    assert parse(b"*00\n") is None


def test_close_waits_for_reader_before_heaters_off():
    import time

    mcu, fake = make_mcu([frame("ENV 24 90 nan")] * 50)
    mcu.start()
    time.sleep(0.05)
    mcu.close()
    assert fake.written[-2:] == [heat(0, 0), heat(1, 0)]
    assert not mcu._reader.is_alive()


class FlakySerial(FakeSerial):
    """First readline after the lines run out raises, like an unplugged cable."""

    def __init__(self, lines, fail_once=True):
        super().__init__(lines)
        self.fail_once = fail_once

    def readline(self):
        if not self.lines and self.fail_once:
            self.fail_once = False
            import serial

            raise serial.SerialException("device disconnected")
        return super().readline()


def test_unplug_reopens_the_port():
    import time

    from astro.devices.mcu import Mcu

    opened = []

    def factory(*a, **k):
        port = FlakySerial([frame("POS 1 2")] if not opened else [frame("POS 3 4")],
                           fail_once=not opened)
        opened.append(port)
        return port

    mcu = Mcu("/dev/null", serial_factory=factory).start()
    time.sleep(0.3)
    mcu.close()
    assert len(opened) == 2 and mcu.counts() == (3, 4)  # reopened and reading again


def test_heaters_off_when_readings_stop(monkeypatch):
    import time

    from astro.devices import mcu as mcu_module

    monkeypatch.setattr(mcu_module, "PING_EVERY_S", 0.05)
    monkeypatch.setattr(mcu_module, "ENV_STALE_S", 0.1)
    m, fake = make_mcu([frame("ENV 24 90 nan")])
    m.start()
    time.sleep(0.4)  # one reading (85%), then silence
    assert m.heat == 0 and heat(0, 0) in fake.written
    m.close()


def test_boot_after_positions_means_reboot():
    from astro.devices.mcu_protocol import Boot

    m, _ = make_mcu([])
    reboots = []
    m.on_reboot = lambda: reboots.append(1)
    m.handle(Boot("astro-mcu", "0.1"))  # start-up: fine
    m.handle(Position(5, 5))
    m.handle(Boot("astro-mcu", "0.1"))  # mid-session: counts were reset
    assert reboots == [1]
    assert parse(frame("BOOT astro-mcu 0.1")) == Boot("astro-mcu", "0.1")


class WriteFailsSerial(FakeSerial):
    def write(self, data):
        import serial

        raise serial.SerialException("unplugged")


def test_close_still_closes_the_port_when_heater_writes_fail():
    from astro.devices.mcu import Mcu

    port = WriteFailsSerial([])
    closed = []
    port.close = lambda: closed.append(1)
    m = Mcu("/dev/null", serial_factory=lambda *a, **k: port)
    m.close()
    assert closed == [1]


def test_write_error_while_handling_env_reopens_instead_of_dying():
    import time

    from astro.devices.mcu import Mcu

    opened = []

    def factory(*a, **k):
        port = WriteFailsSerial([frame("ENV 24 90 nan")]) if not opened else FakeSerial([frame("POS 7 8")])
        opened.append(port)
        return port

    m = Mcu("/dev/null", serial_factory=factory)
    m._reader.start()  # skip start()'s VER? write on the failing port
    time.sleep(0.3)
    m._stop.set()
    assert len(opened) == 2 and m.counts() == (7, 8)
