import numpy as np

from astro.capture.recorder import Recorder
from astro.capture.roi import companions, roi_around


def jupiter_and_moons():
    frame = np.full((2180, 3856), 10, np.uint8)
    y, x = np.mgrid[0:2180, 0:3856]
    frame[(x - 1900) ** 2 + (y - 1000) ** 2 <= 40**2] = 250  # the disk
    moons = [(1300, 1010), (2300, 990), (2900, 1020)]
    for mx, my in moons:
        frame[my - 2:my + 3, mx - 2:mx + 3] = 160
    return frame, moons


def test_companions_are_the_moons_brightest_first():
    frame, moons = jupiter_and_moons()
    found = companions(frame, (1900, 1000), reach=1400)
    assert len(found) == 3 and {round(f[0]) for f in found} == {m[0] for m in moons}


def test_rectangular_roi_stays_on_the_sensor_on_even_pixels():
    roi = roi_around((3800, 50), 1000, (3856, 2180), height=400)
    assert (roi.width, roi.height) == (1000, 400) and roi.x + roi.width <= 3856 and roi.y == 0
    assert roi.x % 2 == 0


def test_recording_roi_takes_in_the_moons_and_keeps_the_planet_in_place():
    frame, moons = jupiter_and_moons()
    rec = Recorder(camera=None, sensor_size=(3856, 2180), out_dir=None)
    roi = rec._roi_with_moons((1900, 1000), companions(frame, (1900, 1000), 1400))
    for mx, my in moons[:2]:  # the farthest may not fit in MAX_ROI_PIXELS
        assert roi.x <= mx < roi.x + roi.width and roi.y <= my < roi.y + roi.height
    rec.roi = roi
    moved = rec._roi_for((1950, 1010))  # the planet drifted: the ROI follows, same size
    assert (moved.width, moved.height) == (roi.width, roi.height)
    assert (moved.x - roi.x, moved.y - roi.y) == (50, 10)
