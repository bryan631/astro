"""Align: the main camera's place on the finder, from one bright star seen by both."""

import time
from datetime import UTC, datetime

import numpy as np
import pytest
from scipy import ndimage

from astro import session as session_module
from astro.devices.stream import ThreadStream
from astro.pointing.align import (
    box_on_finder_view,
    finder_offset_to_sky,
    fit_main_in_finder,
    pick_star,
    sky_offset_to_altaz,
)
from astro.pointing.coords import Site
from astro.pointing.platesolve import FinderSolver, Solution, finder_gray
from astro.session import Session

FIXTURE = "tests/data/align/bright-star-2026-10-10.npz"  # both cameras, time-stamped


def rot(deg):
    r = np.radians(deg)
    return np.array([[np.cos(r), -np.sin(r)], [np.sin(r), np.cos(r)]])


def test_fit_recovers_scale_rotation_and_center_from_a_drift():
    a, c = 58 * rot(15), np.array([700.0, 430.0])
    finder_t = np.arange(0, 10, 0.8)
    finder_xy = c + np.c_[-0.5 * finder_t, 0.17 * finder_t] + [3, -2]  # the star drifts past c
    main_t = np.arange(0.1, 10, 0.5)
    f_at = c + np.c_[-0.5 * main_t, 0.17 * main_t] + [3, -2]
    main_xy = (f_at - c) @ a.T + np.random.default_rng(0).normal(0, 1, (len(main_t), 2))
    fit = fit_main_in_finder(finder_t, finder_xy, main_t, main_xy)
    assert fit.scale == pytest.approx(58, rel=0.02) and fit.rotation_deg == pytest.approx(15, abs=1)
    assert np.allclose(fit.center, c, atol=0.3) and fit.residual_px < 3
    assert fit_main_in_finder(finder_t, finder_xy, main_t[:3], main_xy[:3]) is None  # too short


def test_the_box_sits_on_the_center_and_turns_with_the_finder_view():
    fit = fit_main_in_finder([0, 10], [[640, 480], [635, 482]], np.arange(6) * 2.0,
                             np.c_[np.arange(6) * 58.0 * -1, np.arange(6) * 23.2] - [0, 0])
    box = np.array(box_on_finder_view(fit, (3856, 2180), (1280, 960), 0))
    assert np.allclose(box.mean(axis=0), 0, atol=0.01)  # main center on the finder center
    assert np.ptp(box[:, 0]) == pytest.approx(3856 / 58 / 1280, rel=0.1)  # ~5% of the finder width
    turned = np.array(box_on_finder_view(fit, (3856, 2180), (1280, 960), 180))
    assert np.allclose(turned, -box)


def star_field(stars, size=(480, 640), sky=40.0):
    img = np.zeros(size)
    for x, y, flux in stars:
        img[int(y), int(x)] += flux
    rng = np.random.default_rng(1)
    return ndimage.gaussian_filter(img, 1.2) + sky + rng.normal(0, 2, size)


def test_pick_star_wants_one_clear_winner_near_the_expected_place():
    near = (320, 240)
    clear = star_field([(330, 250, 5000), (300, 200, 600), (100, 100, 20000)])  # far one ignored
    assert pick_star(clear, near) == pytest.approx((330, 250), abs=0.5)
    ambiguous = star_field([(330, 250, 900), (300, 200, 700)])
    assert pick_star(ambiguous, near) is None


def test_pick_star_sees_through_a_cloud_glow_gradient():
    """2026-10-10: cloud glow made the finder's bottom ~2.5x brighter than its top; one sky
    level for the frame put the threshold above the stars."""
    yy = np.mgrid[:480, :640][0]
    glow = star_field([(330, 150, 3000)]) + 80.0 * yy / 480
    assert pick_star(glow, (320, 240)) == pytest.approx((330, 150), abs=0.5)


def test_real_field_without_a_bright_star_is_ambiguous():
    """2026-10-10: the main camera's star was one of several equally faint finder stars, so
    Align must ask for a brighter star rather than guess."""
    gray = finder_gray(np.load(FIXTURE)["finder"][0])
    assert pick_star(gray, (gray.shape[1] / 2, gray.shape[0] / 2)) is None


def test_finder_pixels_to_sky_follow_the_solver():
    """Catalog stars land on detected finder stars with this convention (36 of 37 on
    2026-10-10; the other three roll/flip combinations matched 0-2)."""
    raw = np.load(FIXTURE)["finder"][0]
    solver = FinderSolver()
    sol = solver.solve(raw, timeout_ms=5000)
    gray = finder_gray(raw)
    bg = np.median(gray)
    labels, n = ndimage.label(gray > bg + 6 * 1.4826 * np.median(np.abs(gray - bg)))
    found = np.array(ndimage.center_of_mass(gray, labels, range(1, n + 1)))[:, ::-1] * 2  # raw px
    t = solver.star_table
    r0, d0 = np.radians(sol.ra_deg), np.radians(sol.dec_deg)
    ra, de = t[:, 0], t[:, 1]
    near = np.degrees(np.arccos(np.clip(np.sin(d0) * np.sin(de) + np.cos(d0) * np.cos(de)
                                        * np.cos(ra - r0), -1, 1))) < 4
    hits = 0
    for ra_s, de_s in zip(np.degrees(ra[near]), np.degrees(de[near]), strict=True):
        # the pixel whose sky offset is this star's: invert the (linear) pixel -> sky map
        east = (ra_s - sol.ra_deg) * np.cos(d0)
        north = de_s - sol.dec_deg
        jac = np.array([finder_offset_to_sky(1, 0, sol, 1280), finder_offset_to_sky(0, 1, sol, 1280)]).T
        dx, dy = np.linalg.solve(jac, [east, north])
        hits += np.hypot(*(found - [640 + dx, 480 + dy]).T).min() < 3
    assert hits >= 30


def test_sky_offset_becomes_the_scopes_own_offset():
    sol = Solution(ra_deg=300.0, dec_deg=20.0, roll_deg=0.0, fov_deg=10.38, rmse_arcsec=20,
                   matches=40, false_prob=1e-9, ms=300)
    when = datetime(2026, 10, 10, 2, 0, tzinfo=UTC)
    off = sky_offset_to_altaz(sol, 0.0, 0.0, Site(26.6, -80.1), when)
    assert abs(off.d_az_sky_deg) < 1e-6 and abs(off.d_alt_deg) < 1e-6
    off = sky_offset_to_altaz(sol, 0.05, 0.0, Site(26.6, -80.1), when)
    assert np.hypot(off.d_az_sky_deg, off.d_alt_deg) == pytest.approx(0.05, rel=0.02)


class DriftingSky:
    """A camera seeing a bright star drift (finder) or that star through the main camera
    (main px = A (finder px - C)). Frames depend on the time since the test began."""

    bayer, gain = "RGGB", 0

    def __init__(self, main: bool, start: float):
        self.main, self.start, self.exposure_s, self.roi = main, start, 0.05, None
        self.sensor_size = (1000, 600) if main else (1280, 960)

    def star_in_finder(self):
        t = time.monotonic() - self.start
        return np.array([700 - 5 * t, 430 + 2 * t])

    def capture(self):
        w, h = self.sensor_size
        img = np.zeros((h, w))
        if self.main:
            x, y = A @ (self.star_in_finder() - C) + [w / 2, h / 2]
            spots = [(x, y, 4000)]
        else:
            x, y = self.star_in_finder()
            spots = [(x, y, 6000), (x + 150, y - 90, 900), (x - 120, y + 60, 700)]  # fainter neighbors
        for sx, sy, flux in spots:
            if 0 <= sx < w and 0 <= sy < h:
                img[int(sy), int(sx)] = flux
        return np.clip(ndimage.gaussian_filter(img, 2.0) + 20, 0, 255).astype(np.uint8)

    def set_exposure(self, s):
        self.exposure_s = s

    def set_gain(self, g):
        self.gain = g

    def set_roi(self, roi):
        self.roi = roi

    def close(self):
        pass


A, C = 20 * rot(10), np.array([705.0, 425.0])


class FakeFinder:
    synced, on_change, last_solution = True, None, Solution(300.0, 20.0, 15.0, 10.38, 20, 40, 1e-9, 300)

    def __init__(self, camera):
        self.camera = camera

    def sync(self, fresh=False):
        return True, "Got it."

    def position(self):
        return 45.0, 180.0

    def alignment(self):
        return 1, None


def test_align_button_fits_the_main_camera_onto_the_finder(monkeypatch, tmp_path):
    monkeypatch.setattr(session_module, "ALIGN_S", 2.5)
    start = time.monotonic()
    finder_cam, main_cam = ThreadStream(DriftingSky(False, start)), ThreadStream(DriftingSky(True, start))
    s = Session(Site(26.6, -80.1), clock=lambda: datetime(2026, 10, 10, 2, 0, tzinfo=UTC),
                finder=FakeFinder(finder_cam), main_camera=main_cam, main_sensor=(1000, 600),
                data_dir=tmp_path)
    finder_cam.capture(), main_cam.capture()  # both streaming
    assert "Aligning" in s.action("align")[0]["text"]
    said = []
    while not said and time.monotonic() - start < 10:
        said = [m["text"] for m in s.tick(0.0) if m["type"] == "say"]
        time.sleep(0.05)
    assert said and said[0].startswith("Aligned"), said
    fit = s.main_in_finder
    assert np.allclose(fit.center, C, atol=1.5) and fit.rotation_deg == pytest.approx(10, abs=1.5)
    assert fit.scale == pytest.approx(20, rel=0.05)
    assert s.main_box and s.centerer.offset.observations == 1
    assert s.calibration()["main_in_finder"]["center"] == pytest.approx(list(fit.center))
    finder_cam.close(), main_cam.close()
