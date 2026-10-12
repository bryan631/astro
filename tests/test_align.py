"""Align: the main camera's place on the finder, from plate solves of both cameras."""

import time
from datetime import UTC, datetime

import numpy as np
import pytest
from scipy import ndimage

from astro.devices.stream import ThreadStream
from astro.pointing import align, main_solve
from astro.pointing.align import (
    AlignRun,
    MainInFinder,
    box_on_finder_view,
    finder_offset_to_sky,
    sky_offset_to_altaz,
)
from astro.pointing.coords import Site
from astro.pointing.labels import gnomonic, sky_to_pixels
from astro.pointing.main_solve import _untangent
from astro.pointing.platesolve import FinderSolver, Solution, finder_gray
from astro.session import Session
from tests.test_main_solve import seen_by_main, sky_field

FIXTURE = "tests/data/align/bright-star-2026-10-10.npz"  # both cameras, time-stamped


def rot(deg):
    r = np.radians(deg)
    return np.array([[np.cos(r), -np.sin(r)], [np.sin(r), np.cos(r)]])


def test_the_box_sits_on_the_center_and_turns_with_the_finder_view():
    fit = MainInFinder(58 * rot(5), np.array([640.0, 480.0]), 0.0)
    box = np.array(box_on_finder_view(fit, (3856, 2180), (1280, 960), 0))
    assert np.allclose(box.mean(axis=0), 0, atol=0.01)  # main center on the finder center
    assert np.ptp(box[:, 0]) == pytest.approx(3856 / 58 / 1280, rel=0.1)  # ~5% of the finder width
    turned = np.array(box_on_finder_view(fit, (3856, 2180), (1280, 960), 180))
    assert np.allclose(turned, -box)


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


class DarkMain(DriftingSky):
    def capture(self):
        w, h = self.sensor_size
        return np.full((h, w), 20, np.uint8)  # no star in the telescope view


class NoSolver:
    def solve(self, frame):
        return None


@pytest.mark.parametrize("solver,cached,expected", [
    (NoSolver(), False, "The finder couldn't plate-solve"),
    ("sky", False, "I don't have the star catalog"),
    ("sky", True, "I couldn't plate-solve the telescope view"),
])
def test_align_failures_say_why_and_keep_the_old_box(monkeypatch, tmp_path, solver, cached, expected):
    """Plate solves only: no star needs to be in both cameras, and a failure says which part
    failed instead of asking for "the finder star in the telescope"."""
    monkeypatch.setattr(align, "ALIGN_S", 3.0)
    if cached:
        (tmp_path / "gaia").mkdir()
        np.save(main_solve._tile_path(SKY_FINDER.ra_deg, SKY_FINDER.dec_deg, tmp_path / "gaia"),
                sky_field(np.random.default_rng(8)))
    start = time.monotonic()
    finder_cam, main_cam = ThreadStream(DriftingSky(False, start)), ThreadStream(DarkMain(True, start))
    finder = SolvingFinder(finder_cam)
    if solver != "sky":
        finder.solver = solver
    old_box = [[1, 2], [3, 4], [5, 6], [7, 8]]
    s = Session(Site(26.6, -80.1), clock=lambda: datetime(2026, 10, 10, 2, 0, tzinfo=UTC),
                finder=finder, main_camera=main_cam, main_sensor=(1000, 600),
                data_dir=tmp_path, calibration={"main_box": old_box})
    finder_cam.capture(), main_cam.capture()
    assert "Aligning" in s.action("align")[0]["text"]
    said = []
    while not said and time.monotonic() - start < 10:
        said = [m["text"] for m in s.tick(0.0) if m["type"] == "say"]
        time.sleep(0.05)
    assert said and said[0].startswith(expected), said
    assert s.main_box == old_box and s.main_in_finder is None and s._align is None
    finder_cam.close(), main_cam.close()


def test_align_run_saves_its_frames(tmp_path):
    run = AlignRun((1280, 960))
    for i in range(4):
        run.add("finder", (np.zeros((96, 128), np.uint8), i, float(i)))
        run.add("main", (np.full((40, 60), i, np.uint8), i, float(i)))
    run.save(tmp_path / "a.npz")
    got = np.load(tmp_path / "a.npz")
    assert got["finder"].shape == (4, 96, 128) and got["main"][:, 0, 0].tolist() == [0, 3]


class StaticMain(DriftingSky):
    """The main camera seeing a fixed synthetic star field (rendered once)."""

    frame: np.ndarray

    def __init__(self, frame):
        super().__init__(True, 0.0)
        self.frame, self.sensor_size = frame, frame.shape[::-1]

    def capture(self):
        return self.frame


SKY_FINDER = Solution(150.0, 30.0, 25.0, 10.38, 20, 40, 1e-9, 300)


class FakeSolver:
    def solve(self, frame):
        return SKY_FINDER


class SolvingFinder(FakeFinder):
    solver = FakeSolver()


def test_align_from_two_plate_solves(monkeypatch, tmp_path):
    """With the field's Gaia stars cached, one frame pair places the main camera, with no
    guessing which finder star is the one in the main camera."""
    monkeypatch.setattr(align, "ALIGN_S", 30.0)
    rng = np.random.default_rng(8)
    catalog = sky_field(rng)
    ra_c, dec_c = _untangent(complex(1.2 * 3600, 0.5 * 3600), SKY_FINDER.ra_deg, SKY_FINDER.dec_deg)
    xy = seen_by_main(catalog, ra_c, dec_c, 60.0)  # brightest first
    img = np.zeros((2180, 3856))
    for (x, y), flux in zip(xy, np.geomspace(60000, 4000, len(xy)), strict=True):
        img[int(y + 1090), int(x + 1928)] = flux
    frame = np.clip(ndimage.gaussian_filter(img, 3.0) + 30 + rng.normal(0, 8, img.shape), 0, 255).astype(np.uint8)
    (tmp_path / "gaia").mkdir()
    np.save(main_solve._tile_path(SKY_FINDER.ra_deg, SKY_FINDER.dec_deg, tmp_path / "gaia"), catalog)
    start = time.monotonic()
    finder_cam, main_cam = ThreadStream(DriftingSky(False, start)), ThreadStream(StaticMain(frame))
    s = Session(Site(26.6, -80.1), clock=lambda: datetime(2026, 10, 10, 2, 0, tzinfo=UTC),
                finder=SolvingFinder(finder_cam), main_camera=main_cam, main_sensor=(3856, 2180),
                data_dir=tmp_path)
    finder_cam.capture(), main_cam.capture()
    s.action("align")
    said = []
    while not said and time.monotonic() - start < 15:
        said = [m["text"] for m in s.tick(0.0) if m["type"] == "say"]
        time.sleep(0.05)
    assert said and said[0].startswith("Aligned from the stars"), said
    east, north = gnomonic(SKY_FINDER.ra_deg, SKY_FINDER.dec_deg, ra_c, dec_c)
    expected = np.add(sky_to_pixels(east, north, SKY_FINDER, 1280), [640, 480])
    assert np.allclose(s.main_in_finder.center, expected, atol=0.1)
    assert s.main_box and s.centerer.offset.observations == 1
    finder_cam.close(), main_cam.close()


def star_frame(xy, rng):
    """A main frame (2180 x 3856) with stars at `xy` (raw px from the center, brightest first)."""
    img = np.zeros((2180, 3856))
    for (x, y), flux in zip(xy, np.geomspace(60000, 4000, len(xy)), strict=True):
        img[int(y + 1090), int(x + 1928)] = flux
    return np.clip(ndimage.gaussian_filter(img, 3.0) + 30 + rng.normal(0, 8, img.shape), 0, 255).astype(np.uint8)


def test_align_with_the_barlow_measures_its_power(monkeypatch, tmp_path):
    """A "2x" Barlow is ~2.2-2.6x at the sensor: Align tries a range, and remembers the power."""
    monkeypatch.setattr(align, "ALIGN_S", 30.0)
    rng = np.random.default_rng(9)
    catalog = sky_field(rng, n=60000)  # the Barlow's 13' x 8' field needs a rich one
    ra_c, dec_c = _untangent(complex(0.6 * 3600, -0.4 * 3600), SKY_FINDER.ra_deg, SKY_FINDER.dec_deg)
    w = main_solve._tangent(catalog[:, 0], catalog[:, 1], ra_c, dec_c) / (main_solve.MAIN_SCALE / 2.45)
    w = w * np.exp(-1j * np.radians(30.0))
    inside = (abs(w.real) < 1900) & (abs(w.imag) < 1060)
    xy = np.c_[w[inside].real, w[inside].imag][np.argsort(catalog[inside, 2])[:25]]
    (tmp_path / "gaia").mkdir()
    np.save(main_solve._tile_path(SKY_FINDER.ra_deg, SKY_FINDER.dec_deg, tmp_path / "gaia"), catalog)
    start = time.monotonic()
    finder_cam, main_cam = ThreadStream(DriftingSky(False, start)), ThreadStream(StaticMain(star_frame(xy, rng)))
    s = Session(Site(26.6, -80.1), clock=lambda: datetime(2026, 10, 10, 2, 0, tzinfo=UTC),
                finder=SolvingFinder(finder_cam), main_camera=main_cam, main_sensor=(3856, 2180),
                data_dir=tmp_path, calibration={"barlow": True})
    finder_cam.capture(), main_cam.capture()
    s.action("align")
    said = []
    while not said and time.monotonic() - start < 60:  # ~19 s on a laptop, slower on CI: each
        # wrong power tried costs a full search (~5 s)
        said = [m["text"] for m in s.tick(0.0) if m["type"] == "say"]
        time.sleep(0.05)
    assert said and "The Barlow magnifies 2.5 times" in said[0], said  # 2.45, said to a tenth
    assert s.barlow_power == pytest.approx(2.45, abs=0.02) and s._main_scales() == (main_solve.MAIN_SCALE / s.barlow_power,)
    assert s.calibration()["barlow_power"] == s.barlow_power
    finder_cam.close(), main_cam.close()


def test_barlow_in_shrinks_the_box_about_its_center_and_out_restores_it(tmp_path):
    fit = {"a": [[58.0, 0.0], [0.0, 58.0]], "center": [700.0, 400.0], "residual_px": 0.5}
    s = Session(Site(26.6, -80.1), data_dir=tmp_path, main_sensor=(3856, 2180),
                calibration={"main_in_finder": fit, "main_solved": True, "barlow_power": 2.5})
    s.main_box = box_on_finder_view(s.main_in_finder, (3856, 2180), (1280, 960), 180)
    before = np.array(s.main_box)
    s._main_fix = (time.monotonic(), None, (45.0, 180.0))  # a telescope-view solve at prime focus
    assert "resized the telescope's box" in s.handle("the barlow is in")[0]["text"]
    assert s._main_fix is None  # its scale no longer holds
    after = np.array(s.main_box)
    assert np.allclose(after.mean(axis=0), before.mean(axis=0), atol=1e-3)  # the box is rounded to 1e-4
    assert np.ptp(after[:, 0]) == pytest.approx(np.ptp(before[:, 0]) / 2.5, rel=0.01)
    assert s.calibration()["barlow"] is True
    s.handle("took out the barlow")
    assert np.allclose(s.main_box, before, atol=1e-3)
