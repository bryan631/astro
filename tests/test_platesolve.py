import os

import numpy as np
import pytest

if not os.environ.get("ASTRO_REQUIRE_SOLVER"):
    pytest.importorskip("tetra3", reason="run scripts/install-solver.sh")

from astro.devices.sim.sky import render
from astro.pointing.geometry import separation_deg
from astro.pointing.platesolve import FinderSolver, bin2x2
from tests import frames


@pytest.fixture(scope="module")
def solver():
    return FinderSolver()


def test_remove_hot_pixels_keeps_stars():
    from astro.pointing.platesolve import remove_hot_pixels

    img = np.full((9, 9), 10, np.uint8)
    img[2, 2] = 200  # hot pixel: neighbors dark
    img[5:8, 5:8] = 60
    img[6, 6] = 200  # star: light spills into neighbors
    out = remove_hot_pixels(img)
    assert out[2, 2] == 10 and out[6, 6] == 200


def test_bin2x2():
    raw = np.arange(16, dtype=np.uint8).reshape(4, 4)
    assert bin2x2(raw).tolist() == [[10, 18], [42, 50]]


@pytest.mark.parametrize("ra,dec,roll", [(83.8, -5.4, 0), (279.2, 38.8, 30), (10.7, 41.3, -70),
                                         (200.0, 55.0, 120)])
def test_solves_synthetic_field(solver, ra, dec, roll):
    img = render(solver._t3.star_table, ra, dec, roll, 11.0)
    sol = solver.solve(img, bayer=False)
    assert sol is not None
    # separation_deg works on any lat/lon pair, so reuse it for (dec, ra).
    assert separation_deg(sol.dec_deg, sol.ra_deg, dec, ra) < 0.05
    assert abs((sol.roll_deg - roll + 180) % 360 - 180) < 1
    assert sol.fov_deg == pytest.approx(11.0, rel=0.03)


def test_suburban_sky_mag6(solver):
    """Bortle 7-8: only stars brighter than ~mag 6 show in a short finder exposure."""
    img = render(solver._t3.star_table, 279.2, 38.8, 15, 11.0, mag_limit=6.0)
    sol = solver.solve(img, bayer=False)
    assert sol is not None and sol.matches >= 8
    assert separation_deg(sol.dec_deg, sol.ra_deg, 38.8, 279.2) < 0.05


def test_blank_frame_returns_none(solver):
    noise = np.random.default_rng(1).normal(20, 3, (480, 640)).clip(0, 255).astype(np.uint8)
    assert solver.solve(noise, bayer=False) is None


def test_solve_rate_on_finder_format_frames(solver):
    """Full-res frames, binned like real finder data: guards solver settings (e.g. binary_open)."""
    rng = np.random.default_rng(5)
    solved = 0
    for i in range(30):
        ra, dec = rng.uniform(0, 360), np.degrees(np.arcsin(rng.uniform(-0.5, 1)))
        img = render(solver._t3.star_table, ra, dec, rng.uniform(0, 360), 11.0, size=(1280, 960),
                     sigma_px=2.0, mag_limit=6.5, seed=i)
        solved += solver.solve(img) is not None
    assert solved >= 26


def test_real_finder_frame_solves(solver):
    """SV905C, 0.8 s gain 100, 2026-10-03 22:36 PDT; Saturn ~0.94 deg from center."""
    raw = frames.load("saturn_field")
    sol = solver.solve(raw)
    assert sol is not None and sol.matches >= 8
    assert separation_deg(sol.dec_deg, sol.ra_deg, 2.112, 10.014) < 0.02
    assert sol.fov_deg == pytest.approx(10.39, abs=0.05)


def test_confidence_is_minus_log10_false_prob():
    from astro.pointing.platesolve import Solution

    assert Solution(0, 0, 0, 10, 0, 10, 1e-12, 5).confidence == pytest.approx(12)


def test_real_frame_reports_pixel_scale(solver):
    raw = frames.load("saturn_field")
    sol = solver.solve(raw)
    assert sol.scale_arcsec_px == pytest.approx(10.39 * 3600 / 1280, rel=0.01)  # ~29.2"/px
