import os
from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

if not os.environ.get("ASTRO_REQUIRE_SOLVER"):
    pytest.importorskip("tetra3", reason="run scripts/install-solver.sh")

from astro.devices.sim.finder import SimFinderCamera
from astro.devices.sim.scope import SimEncoders, SimScope
from astro.pointing.coords import Site
from astro.pointing.finder_sync import FinderSync, check_focus
from astro.pointing.geometry import separation_deg
from astro.pointing.mount_model import MountModel
from astro.pointing.platesolve import FinderSolver

WPB = Site(26.7, -80.1)
EVENING = datetime(2026, 10, 3, 21, 0, tzinfo=timezone(timedelta(hours=-4)))


@pytest.fixture(scope="module")
def solver():
    return FinderSolver()


def make(solver, alt=60, az=200, **cam):
    scope = SimScope(alt, az)
    clock = lambda: EVENING
    camera = SimFinderCamera(lambda: (scope.alt, scope.az), WPB, clock, solver._t3.star_table, **cam)
    return scope, FinderSync(camera, solver, MountModel(), SimEncoders(scope), WPB, clock)


def test_sync_learns_mount_offsets(solver):
    scope, fs = make(solver)
    ok, msg = fs.sync()
    assert ok, msg
    assert separation_deg(*fs.position(), scope.alt, scope.az) < 0.2


def test_two_syncs_track_anywhere(solver):
    scope, fs = make(solver, 50, 100)
    assert fs.sync()[0]
    scope.alt, scope.az = 70, 250
    assert fs.sync()[0]
    scope.alt, scope.az = 40, 330
    assert separation_deg(*fs.position(), scope.alt, scope.az) < 0.2


def test_defocused_finder_explains(solver):
    _, fs = make(solver, blur_px=9)
    ok, msg = fs.sync()
    assert not ok and "out of focus" in msg


def test_blank_sky_explains():
    dark = np.random.default_rng(0).normal(20, 3, (480, 640))
    assert "can't see any stars" in check_focus(dark).reason


def test_sharp_vs_soft_hfr(solver):
    sharp = make(solver)[1].focus_report()
    soft = make(solver, blur_px=5)[1].focus_report()
    assert sharp.ok and sharp.hfr_px < soft.hfr_px
