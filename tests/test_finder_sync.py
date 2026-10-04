import os
from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

if not os.environ.get("ASTRO_REQUIRE_SOLVER"):
    pytest.importorskip("tetra3", reason="run scripts/install-solver.sh")

from astro.devices.sim.finder import SimFinderCamera
from astro.devices.sim.scope import SimEncoders, SimScope
from astro.pointing.coords import Site, radec_to_altaz
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


def test_bright_scene_is_not_called_out_of_focus():
    rng = np.random.default_rng(0)
    scene = rng.normal(20, 3, (480, 640))  # lit room / twilight: thousands of bright details
    ys, xs = rng.integers(0, 480, 3000), rng.integers(0, 639, 3000)
    scene[ys, xs] = scene[ys, xs + 1] = 200  # 2-pixel details, like a textured lit scene
    assert "doesn't look like a starry sky" in check_focus(scene).reason


def test_sharp_vs_soft_hfr(solver):
    sharp = make(solver)[1].focus_report()
    soft = make(solver, blur_px=5)[1].focus_report()
    assert sharp.ok and sharp.hfr_px < soft.hfr_px


def make_session(solver, alt=60, az=200, **cam):
    from astro.session import Session

    scope, fs = make(solver, alt, az, **cam)
    return scope, fs, Session(WPB, clock=lambda: EVENING, finder=fs)


def test_goto_syncs_first_then_guides_true_scope_onto_target(solver):
    from astro.devices.sim.scope import SimUser

    scope, _, s = make_session(solver)
    said = [m["text"] for m in s.handle("go to albireo")]
    assert said == ["Got it, I know where we're pointing.", "Let's find Albireo."]
    user, t, state = SimUser(), 0.0, None
    for _ in range(1500):
        for m in s.tick(t):
            if m["type"] == "say":
                user.hear(m["text"], t)
            else:
                state = m
        scope.step(*user.act(t), 0.1)
        t += 0.1
    assert state["on_target"]
    # The *true* scope (not just the model) ended up on Albireo.
    assert separation_deg(scope.alt, scope.az, *radec_to_altaz(292.68, 27.96, WPB, EVENING)) < 0.15


def test_goto_refuses_with_reason_when_finder_soft(solver):
    _, _, s = make_session(solver, blur_px=9)
    assert "out of focus" in s.handle("go to albireo")[0]["text"]


def test_voice_guided_finder_focus(solver):
    _, fs, s = make_session(solver, blur_px=6)
    assert "focus ring" in s.handle("focus the finder")[0]["text"]
    said = []
    for i, blur in enumerate([6, 4, 2.5, 2.0, 3.5]):  # user turns the ring past best focus
        fs.camera.blur_px = blur
        said += [m["text"] for m in s.tick(float(i * 2))]
    assert "sharper" in said and said[-1].startswith("passed it")
    assert s.handle("stop")[0]["text"] == "OK, focus is set."


def test_where_syncs_before_answering(solver):
    _, fs, s = make_session(solver, 70, 300)
    out = s.handle("what am I looking at")[0]["text"]
    assert fs.synced and "pointing yet" not in out


def test_where_explains_when_it_cannot_sync(solver):
    _, _, s = make_session(solver, blur_px=9)
    assert s.handle("where am i")[0]["text"].startswith("I don't know where we're pointing yet.")


def test_goto_ends_finder_focus_mode(solver):
    _, _, s = make_session(solver)
    s.handle("focus the finder")
    s.handle("go to albireo")
    assert any(m["type"] == "state" for m in s.tick(0.0))


# Real SV905C frames, 0.2 s at gain 1000 (2026-10-03): one bright star, and the lens cap on.
DATA = __import__("pathlib").Path(__file__).parent / "data"


def test_real_capped_frame_has_no_stars():
    from astro.pointing.platesolve import finder_gray

    report = check_focus(finder_gray(np.load(DATA / "finder_cap_on.npy")))
    assert report.stars == 0 and "can't see any stars" in report.reason


def test_real_one_star_frame_is_not_called_out_of_focus():
    from astro.pointing.platesolve import finder_gray

    report = check_focus(finder_gray(np.load(DATA / "finder_one_star.npy")))
    assert report.stars == 1 and "a star or two" in report.reason
