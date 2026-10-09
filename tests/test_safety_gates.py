"""Review H2/H3/H8: every finder exposure is gated; goto respects the treeline; moving resets it."""

from datetime import UTC, datetime

import numpy as np

from astro.guidance.engine import Guide
from astro.planner.horizon import HorizonMask
from astro.pointing.coords import Site, body_altaz
from astro.pointing.finder_sync import FinderSync
from astro.pointing.mount_model import MountModel
from astro.session import Session

WPB = Site(26.7, -80.1)
NOON = datetime(2026, 6, 21, 17, 0, tzinfo=UTC)
NIGHT = datetime(2026, 10, 4, 2, 0, tzinfo=UTC)


class CountingCamera:
    bayer = "GRBG"

    def __init__(self):
        self.captures = 0

    def capture(self):
        self.captures += 1
        return np.zeros((960, 1280), np.uint8)


def daytime_session():
    cam = CountingCamera()
    finder = FinderSync(cam, solver=None, model=MountModel(), encoders=lambda: (45.0, 180.0),
                        site=WPB, clock=lambda: NOON)
    return Session(WPB, clock=lambda: NOON, finder=finder), cam


def texts(msgs):
    return [m["text"] for m in msgs if m["type"] == "say"]


def night_session(mask):
    pos = {"alt": 45.0, "az": 0.0}
    s = Session(WPB, lambda: (pos["alt"], pos["az"]), clock=lambda: NIGHT, horizon=mask)
    return s, pos


def test_goto_refuses_targets_behind_the_trees():
    s, _ = night_session(HorizonMask(((0.0, 60.0),)))  # trees up to 60 deg all around
    alt, _ = body_altaz("saturn", WPB, NIGHT)
    assert 0 < alt < 60
    assert texts(s.handle("go to saturn")) == ["I can't go to Saturn: it's behind the trees right now."]


def test_guidance_stops_when_target_sinks_behind_trees():
    s, _ = night_session(HorizonMask())
    s.handle("go to saturn")
    assert s.guide is not None
    s.horizon = HorizonMask(((0.0, 60.0),))
    assert "no longer safe" in texts(s.tick(0.0))[0] and s.guide is None


def test_moving_resets_the_treeline():
    saved = []
    s = Session(WPB, lambda: (45.0, 0.0), clock=lambda: NIGHT,
                horizon=HorizonMask(((0.0, 40.0),)), on_horizon_change=saved.append)
    s.handle("start the horizon walk")
    s.set_location(40.0, -105.0, None, None)
    assert s.horizon == HorizonMask() and saved == [HorizonMask()] and s._horizon is None


def test_guidance_stops_on_frozen_encoders_and_lost_alignment():
    for broken in ("stale", "reset"):
        cam = CountingCamera()
        finder = FinderSync(cam, solver=None, model=MountModel(), encoders=lambda: (45.0, 0.0),
                            site=WPB, clock=lambda: NIGHT)
        finder.synced = True
        s = Session(WPB, clock=lambda: NIGHT, finder=finder)
        s.target, s.guide = "Saturn", Guide(40, 120)
        if broken == "stale":
            finder.encoder_age = lambda: 5.0
        else:
            finder.reset(WPB)
        said = texts(s.tick(0.0))[0]
        assert s.guide is None and s.target == "Saturn"
        assert ("position sensors" if broken == "stale" else "lost track") in said


def test_centering_stops_too_when_pointing_is_lost():
    cam = CountingCamera()
    finder = FinderSync(cam, solver=None, model=MountModel(), encoders=lambda: (45.0, 0.0),
                        site=WPB, clock=lambda: NIGHT)
    finder.synced = True
    s = Session(WPB, clock=lambda: NIGHT, finder=finder)
    s.target, s._centering = "Saturn", True
    finder.encoder_age = lambda: 5.0
    assert "position sensors" in texts(s.tick(0.0))[0] and not s._centering


def test_daytime_toggle_skips_all_checks_but_the_sun():
    session = Session(WPB, clock=lambda: NOON)
    assert not session.target_safety(10.0, 0.0).ok  # daytime lockout
    session.override = True
    assert session.target_safety(-5.0, 0.0).ok  # below horizon and behind trees: allowed
    sun_alt, sun_az = body_altaz("sun", WPB, NOON)
    assert not session.target_safety(sun_alt, sun_az).ok


def test_toggle_does_nothing_at_night():
    session = Session(WPB, clock=lambda: NIGHT)
    session.override = True
    assert not session.target_safety(-5.0, 0.0).ok
