from datetime import UTC, datetime

import numpy as np
import pytest
from scipy import ndimage

from astro.capture.focus import FocusCoach, half_flux_radius, laplacian_variance
from astro.capture.roi import brightest_blob, roi_around
from astro.capture.ser import SerWriter, read_ser


def star(sigma, size=41):
    img = np.zeros((size, size))
    img[size // 2, size // 2] = 1
    return (ndimage.gaussian_filter(img, sigma) * 200 / ndimage.gaussian_filter(img, sigma).max()).astype(np.uint8)


def test_ser_roundtrip(tmp_path):
    frames = np.random.default_rng(0).integers(0, 255, (3, 4, 6), dtype=np.uint8)
    with SerWriter(tmp_path / "a.ser", 6, 4) as w:
        for f in frames:
            w.write(f, datetime(2026, 10, 3, tzinfo=UTC))
    meta, back = read_ser(tmp_path / "a.ser")
    assert meta == {"color_id": 9, "width": 6, "height": 4, "frames": 3, "instrument": "SV705C"}
    assert (back == frames).all()
    assert (tmp_path / "a.ser").stat().st_size == 178 + 3 * 24 + 3 * 8


def test_ser_rejects_wrong_shape(tmp_path):
    with SerWriter(tmp_path / "b.ser", 6, 4) as w, pytest.raises(ValueError):
        w.write(np.zeros((5, 6), np.uint8))


def test_brightest_blob_finds_planet():
    frame = np.full((200, 300), 5, np.uint8)
    frame[120:140, 60:80] = 200
    x, y = brightest_blob(frame)
    assert (x, y) == (pytest.approx(69.5, abs=1), pytest.approx(129.5, abs=1))
    assert brightest_blob(np.full((50, 50), 5, np.uint8)) is None


def test_roi_clamped_and_even():
    roi = roi_around((5, 1000), 256, (3840, 2160))
    assert roi.x == 0 and roi.y % 2 == 0 and roi.y + 256 <= 2160
    assert roi_around((3839, 2159), 256, (3840, 2160)).x == 3840 - 256


def test_sharper_star_has_smaller_hfr():
    assert half_flux_radius(star(1.0)) < half_flux_radius(star(3.0))


def test_laplacian_prefers_sharp():
    sharp = star(1.0)
    assert laplacian_variance(sharp) > laplacian_variance(ndimage.gaussian_filter(sharp, 2))


def test_focus_coach_sequence():
    coach = FocusCoach()
    said = [coach.update(s) for s in [10, 12, 15, 15.1, 13, 11]]
    assert said[:3] == ["keep turning the focus knob slowly", "sharper", "sharper"]
    assert said[3] == "that's the sharpest so far"
    assert said[4] == "passed it, go back slowly"


def test_file_names_are_safe():
    from astro.capture.recorder import safe_name

    assert safe_name("Barnard's Star") == "Barnards_Star"
    assert safe_name("M31/Andromeda") == "M31Andromeda"
    assert safe_name("///") == "target"


def test_roi_never_negative_on_a_small_sensor():
    from astro.capture.roi import roi_around

    roi = roi_around((100, 100), 512, (400, 300))
    assert (roi.x, roi.y, roi.width, roi.height) == (0, 0, 300, 300)


def test_sharpest_is_said_once_per_plateau():
    coach = FocusCoach()
    said = [coach.update(s) for s in [10, 15, 15.1, 15.0, 15.1, 11, 15, 15.1]]
    assert said.count("that's the sharpest so far") == 2  # once, then again after a change
    assert said[2] == "that's the sharpest so far" and said[3] is None and said[4] is None
