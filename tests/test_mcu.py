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
