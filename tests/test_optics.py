import pytest

from astro.optics import drift_px_per_s, fov_arcmin, plate_scale

MAIN_PX_UM, MAIN_FL = 2.9, 1200
FINDER_PX_UM, FINDER_FL = 3.75, 25


def test_main_plate_scale():
    assert plate_scale(MAIN_PX_UM, MAIN_FL) == pytest.approx(0.50, abs=0.01)


def test_barlow_halves_scale():
    assert plate_scale(MAIN_PX_UM, 2 * MAIN_FL) == pytest.approx(0.25, abs=0.01)


def test_main_fov():
    w, h = fov_arcmin(3840, 2160, plate_scale(MAIN_PX_UM, MAIN_FL))
    assert (w, h) == (pytest.approx(32, abs=1), pytest.approx(18, abs=1))


def test_finder_scale_and_fov():
    scale = plate_scale(FINDER_PX_UM, FINDER_FL)
    assert scale == pytest.approx(31, abs=0.5)
    w, h = fov_arcmin(1280, 960, scale)
    assert (w / 60, h / 60) == (pytest.approx(11, abs=0.3), pytest.approx(8, abs=0.3))


def test_drift_at_prime_focus():
    assert drift_px_per_s(0, plate_scale(MAIN_PX_UM, MAIN_FL)) == pytest.approx(30, abs=1)


def test_drift_shrinks_toward_pole():
    scale = plate_scale(MAIN_PX_UM, MAIN_FL)
    assert drift_px_per_s(80, scale) < drift_px_per_s(0, scale) / 5
