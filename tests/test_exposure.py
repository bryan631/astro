from datetime import UTC, datetime
from types import SimpleNamespace

import numpy as np

from astro.capture.exposure import FINDER_DAY_RANGE, next_settings
from astro.devices.stream import ThreadStream
from astro.pointing.coords import Site
from astro.session import MAIN_DAY, MAIN_TWILIGHT_RANGE, Session
from tests.fake_camera import FakeCamera


def settle(light: float, exposure: float = 0.1, gain: int = 400, steps: int = 30):
    """Run auto-exposure on a scene whose pixels are light * exposure * gain / 400 (8-bit)."""
    for _ in range(steps):
        frame = np.full((64, 64), min(255, light * exposure * gain / 400), np.uint8)
        new = next_settings(frame, exposure, gain)
        if new is None:
            break
        exposure, gain = new
    return exposure, gain


def test_brightens_a_dim_scene_and_dims_a_bright_one():
    exposure, gain = settle(200)
    assert 60 <= 200 * exposure * gain / 400 <= 220 and exposure > 0.1
    assert settle(20000)[0] < 0.05


def test_a_capped_lens_stops_at_the_longest_exposure():
    assert settle(0) == (0.5, 400)


def test_daylight_lowers_the_gain_once_the_exposure_is_shortest():
    exposure, gain = settle(3_000_000)
    assert exposure == 0.0001 and gain < 400
    assert 60 <= 3_000_000 * exposure * gain / 400 <= 220



def test_session_switches_the_streamed_main_camera_between_day_and_night(tmp_path):
    now = [datetime(2026, 10, 10, 3, 0, tzinfo=UTC)]  # 11 PM in Florida
    main = ThreadStream(FakeCamera())
    s = Session(Site(26.6, -80.1), lambda: (45.0, 180.0), clock=lambda: now[0], main_camera=main,
                main_sensor=(64, 48), data_dir=tmp_path)
    s.tick(0.0)
    assert (main.exposure_s, main.gain) == (0.01, 0)  # the stream's own (night) settings
    now[0] = datetime(2026, 10, 10, 17, 0, tzinfo=UTC)  # 1 PM
    s.tick(2.0)
    assert (main.exposure_s, main.gain) == MAIN_DAY
    now[0] = datetime(2026, 10, 11, 3, 0, tzinfo=UTC)
    s.tick(4.0)
    assert (main.exposure_s, main.gain) == (0.01, 0)
    main.close()


class GlaringStream:
    """A streamed camera whose every frame is clipped white (dawn, or a sunny day)."""

    def __init__(self):
        self.exposure_s, self.gain, self.seq = 0.01, 100, 0

    def latest(self):
        self.seq += 1
        return np.full((48, 64), 255, np.uint8), self.seq, 0.0

    def set_exposure(self, s):
        self.exposure_s = s

    def set_gain(self, g):
        self.gain = g


def test_dawn_auto_exposes_the_main_view_but_never_during_a_capture(tmp_path):
    """2026-10-10: both views went white at 6:50 AM on their night settings."""
    main = GlaringStream()
    s = Session(Site(26.6, -80.1), lambda: (45.0, 180.0), data_dir=tmp_path, main_camera=main,
                clock=lambda: datetime(2026, 10, 10, 3, 0, tzinfo=UTC))  # night by the Sun
    s._camera_busy = lambda: True
    s.tick(0.0)
    assert (main.exposure_s, main.gain) == (0.01, 100)  # a capture owns the settings
    s._camera_busy = lambda: False
    for t in range(1, 6):
        s.tick(float(t))
    assert MAIN_TWILIGHT_RANGE[0] <= main.exposure_s < 0.01


def test_finder_view_auto_exposes_by_day(tmp_path):
    cam = GlaringStream()
    finder = SimpleNamespace(camera=cam, position=lambda: (45.0, 180.0), synced=False)
    s = Session(Site(26.6, -80.1), finder=finder, data_dir=tmp_path,
                clock=lambda: datetime(2026, 10, 10, 17, 0, tzinfo=UTC))  # 1 PM
    for t in range(6):
        s.tick(float(t))
    assert FINDER_DAY_RANGE[0] <= cam.exposure_s < 0.01
