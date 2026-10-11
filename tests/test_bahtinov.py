"""Bahtinov mask focus: found automatically, read as a signed miss, never seen where it isn't.

The frames come from a Fourier-optics model of the XT8 with the mask (tests/bahtinov_model.py),
at the telescope view's binned 1"/px. Sweeping seeing 1-4", the mask turned any way, slit
angles 15-25 degrees and stars of magnitude ~0-2.5 found 284 of 288 patterns, the miss within
0.08 px rms at focus (the in-focus zone is +-0.75 px), and nothing in 57 frames without a mask.
"""

import numpy as np
import pytest

from astro.capture.bahtinov import focus_number, measure_bahtinov
from astro.process.planet import superpixel_rgb
from astro.process.view import remove_hot_pixels, render
from tests.bahtinov_model import frame

PX_PER_UM = 0.0159  # the middle spike's slide per micron of focus error, binned 1"/px (model)


def test_the_miss_is_zero_at_focus_and_signed_either_side():
    misses = {d: measure_bahtinov(frame(d, 2.0, flux=1e6)).offset_px for d in (-60.0, 0.0, 60.0)}
    assert abs(misses[0.0]) < 0.15
    assert misses[-60.0] == pytest.approx(60 * PX_PER_UM, abs=0.15)
    assert misses[60.0] == pytest.approx(-60 * PX_PER_UM, abs=0.15)


@pytest.mark.parametrize("mask_angle,theta,seeing,trail", [(45.0, 15.0, 3.0, 2.5), (73.0, 25.0, 1.0, 0.0),
                                                         (17.0, 20.0, 4.0, 4.0)])
def test_found_however_the_mask_is_turned_in_any_seeing_and_drift(mask_angle, theta, seeing, trail):
    found = measure_bahtinov(frame(30.0, seeing, theta_deg=theta, mask_angle_deg=mask_angle, flux=1e6,
                                   trail_px=trail, trail_deg=30.0))
    assert found is not None and found.offset_px == pytest.approx(-30 * PX_PER_UM, abs=0.25)
    middle = found.angles_deg[1] % 180
    assert min(abs(middle - mask_angle % 180), 180 - abs(middle - mask_angle % 180)) < 1.0


def test_no_mask_no_pattern():
    """The spider's four spikes, a bright star alone, real frames from 2026-10-10 (a saturated
    star in Align, star fields): never read as a mask."""
    assert measure_bahtinov(frame(0.0, 2.0, mask=False, flux=3e6)) is None
    assert measure_bahtinov(frame(0.0, 1.0, mask=False, flux=1e7, trail_px=2.5)) is None
    d = np.load("tests/data/align/bright-star-2026-10-10.npz")
    for raw in d["main"][::4]:
        lum = remove_hot_pixels(superpixel_rgb(raw, str(d["main_bayer"])).astype(np.float32)).mean(axis=2)
        assert measure_bahtinov(lum) is None


def test_too_bright_a_star_is_refused_not_misread():
    """Brighter than magnitude ~-1 at the night view's settings the spikes clip wider than
    they can be read: no number, rather than a wrong one."""
    assert measure_bahtinov(frame(60.0, 1.0, mask_angle_deg=17.0, flux=3e7)) is None


def test_the_telescope_view_reads_the_mask_as_its_focus_number():
    lum = frame(0.0, 2.0, flux=1e6)
    raw = np.kron(lum, np.ones((2, 2))).astype(np.uint8)  # a gray "Bayer" frame: superpixels = lum
    _, metrics = render(raw, "GRBG", rotate=0, night=True, mask=True)
    assert metrics["focus_mode"] == "bahtinov" and metrics["focus"] > 90 and abs(metrics["mask_offset"]) < 0.25
    _, finder = render(raw, "GRBG", rotate=0, night=True)  # not looked for: the stars' number
    assert finder["focus_mode"] != "bahtinov"
    assert focus_number(measure_bahtinov(frame(-60.0, 2.0, flux=1e6))) < 70  # off focus: lower
