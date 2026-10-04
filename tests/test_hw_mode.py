import os
import time
from datetime import UTC, datetime

import numpy as np
import pytest

if not os.environ.get("ASTRO_REQUIRE_SOLVER"):
    pytest.importorskip("tetra3", reason="run scripts/install-solver.sh")

from astro.devices import config as devices
from astro.devices.sim.sky import render
from astro.pointing.coords import Site, altaz_to_radec
from astro.pointing.geometry import separation_deg
from astro.pointing.platesolve import FinderSolver
from astro.pointing.solve_tracker import SolveTracker

WPB = Site(26.7, -80.1)
NOW = datetime(2026, 10, 4, 2, 0, tzinfo=UTC)


class SkyFinder:
    """Finder stub that sees the real sky at a settable alt/az (or clouds)."""

    bayer = "GRBG"

    def __init__(self, solver, alt, az):
        self.solver, self.alt, self.az, self.cloudy = solver, alt, az, False

    def capture(self):
        if self.cloudy:
            return np.full((960, 1280), 20, np.uint8)
        ra, dec = altaz_to_radec(self.alt, self.az, WPB, NOW)
        return render(self.solver._t3.star_table, ra, dec, 0, 10.4, size=(1280, 960),
                      sigma_px=2.0, mag_limit=6.5)


@pytest.fixture(scope="module")
def solver():
    return FinderSolver()


def test_tracker_follows_pushes_without_encoders(solver):
    cam = SkyFinder(solver, 50, 120)
    tracker = SolveTracker(cam, solver, WPB, lambda: NOW).start()
    try:
        assert tracker.sync()[0]
        assert separation_deg(*tracker.position(), 50, 120) < 0.1
        cam.alt, cam.az = 62, 140  # user pushes; the background loop re-solves
        deadline = time.monotonic() + 10
        while separation_deg(*tracker.position(), 62, 140) > 0.1 and time.monotonic() < deadline:
            time.sleep(0.1)
        assert separation_deg(*tracker.position(), 62, 140) < 0.1
    finally:
        tracker.stop()


def test_tracker_reports_why_it_cannot_sync(solver):
    cam = SkyFinder(solver, 50, 120)
    cam.cloudy = True
    tracker = SolveTracker(cam, solver, WPB, lambda: NOW)  # not started: sync solves directly
    ok, msg = tracker.sync()
    assert not ok and "can't see any stars" in msg


def test_unknown_drivers_rejected(solver):
    with pytest.raises(ValueError, match="mount driver"):
        devices.build_pointing({"finder": {"driver": "svbony", "model": "X"},
                                "mount": {"driver": "telepathy"}}, solver, WPB, lambda: NOW)
