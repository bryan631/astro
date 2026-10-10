from datetime import UTC, datetime

import numpy as np

from astro.capture.exposure import next_settings
from astro.devices.stream import ThreadStream
from astro.pointing.coords import Site
from astro.session import MAIN_DAY, Session
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
