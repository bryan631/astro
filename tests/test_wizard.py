import os

import pytest

if not os.environ.get("ASTRO_REQUIRE_SOLVER"):
    pytest.importorskip("tetra3", reason="run scripts/install-solver.sh")

from astro.intents import Intent, parse
from astro.pointing.platesolve import FinderSolver
from tests.test_finder_sync import make_session


@pytest.fixture(scope="module")
def solver():
    return FinderSolver()


def texts(msgs):
    return [m["text"] for m in msgs if m["type"] == "say"]


def test_intents():
    assert parse("set up the telescope") == Intent("setup")
    assert parse("ready") == Intent("ready")
    assert parse("skip") == Intent("skip")


def test_full_setup_walkthrough(solver):
    scope, fs, s = make_session(solver, 60, 200)
    out = s.handle("set up the telescope")
    assert any(m["type"] == "get_location" for m in out)
    assert "waiting" in texts(s.handle("ready"))[0]  # no sync before the location arrives
    out = s.set_location(26.7, -80.1, None, 5.0)  # the tablet's GPS fix
    assert "high in the sky" in texts(out)[-1]
    reports = []
    for alt, az in [(60, 200), (45, 290), (70, 60)]:  # three parts of the sky
        scope.alt, scope.az = alt, az
        said = texts(s.handle("ready"))
        reports.append(said[0])
    assert reports[0] == "Got it."
    assert reports[2].startswith("Got it. The alignment is good, about")
    assert "treeline" in said[-1]
    assert texts(s.handle("skip")) == ["Setup is complete. Say 'what's good tonight' to begin."]
    assert fs.alignment()[0] == 3 and not s.wizard.active


def test_failed_sync_can_retry_and_stop_ends_setup(solver):
    _, _, s = make_session(solver, blur_px=9)
    s.handle("set up the telescope")
    s.location_failed("Permission denied.")  # no GPS: setup goes on with the saved site
    assert texts(s.handle("ready"))[0].endswith("Say ready to try again, or skip.")
    assert "setup stopped" in texts(s.handle("stop"))[0]
    assert "didn't catch that" in texts(s.handle("ready"))[0]  # no setup running: not a command


def test_far_gps_fix_cannot_wipe_a_sync(solver):
    """Review H7: the first sync only happens after the location, so a move can't erase it."""
    _, fs, s = make_session(solver, 60, 200)
    s.handle("set up the telescope")
    s.handle("ready")  # too early: still waiting
    assert fs.alignment()[0] == 0
    s.set_location(26.9, -80.3, None, None)  # ~30 km away: resets the model, before any sync
    assert "Got it" in texts(s.handle("ready"))[0] and fs.alignment()[0] == 1
