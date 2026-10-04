"""Review H2/H3/H8: every finder exposure is gated; goto respects the treeline; moving resets it."""

from datetime import UTC, datetime

import numpy as np

from astro.planner.horizon import HorizonMask
from astro.pointing.coords import Site, body_altaz
from astro.pointing.finder_sync import FinderSync
from astro.pointing.mount_model import MountModel
from astro.pointing.solve_tracker import SolveTracker
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


def test_daytime_commands_never_expose_the_finder():
    s, cam = daytime_session()
    for command in ("go to mizar", "where am i", "sync", "start the horizon walk"):
        said = " ".join(texts(s.handle(command)))
        assert "daytime" in said, (command, said)
    s.handle("set up the telescope")
    s.handle("ready")
    assert cam.captures == 0


def test_solve_tracker_idles_while_unsafe():
    cam = CountingCamera()
    tracker = SolveTracker(cam, solver=None, site=WPB, clock=lambda: NOON)
    tracker.safety = lambda: "it's daytime"
    tracker.start()
    try:
        assert not tracker.sync()[0]
        __import__("time").sleep(0.2)
    finally:
        tracker.stop()
    assert cam.captures == 0


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


def test_tracker_starts_only_once_the_session_gates_it():
    cam = CountingCamera()
    tracker = SolveTracker(cam, solver=None, site=WPB, clock=lambda: NOON)
    assert not tracker._thread.is_alive()  # devices.build_pointing no longer starts it
    Session(WPB, clock=lambda: NOON, finder=tracker)  # installs the gate, then starts it
    try:
        assert tracker._thread.is_alive() and tracker.safety is not None
        __import__("time").sleep(0.2)
    finally:
        tracker.stop()
    assert cam.captures == 0  # daytime: the gate held from the first loop


def test_fresh_solve_does_not_bypass_the_gate():
    tracker = SolveTracker(CountingCamera(), solver=None, site=WPB, clock=lambda: NOON)
    tracker.synced, tracker._solved_at = True, __import__("time").monotonic()
    tracker.safety = lambda: "it's daytime"
    ok, msg = tracker.sync()
    assert not ok and "daytime" in msg
