import os

import numpy as np
import pytest

if not os.environ.get("ASTRO_REQUIRE_SOLVER"):
    pytest.importorskip("tetra3", reason="run scripts/install-solver.sh")

from astro.devices.sim.sky import render
from astro.pointing.geometry import separation_deg
from astro.pointing.platesolve import FinderSolver, bin2x2


@pytest.fixture(scope="module")
def solver():
    return FinderSolver()


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
