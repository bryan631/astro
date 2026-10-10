import re
from datetime import UTC, datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

from astro.devices.sim.scope import SimScope, SimUser
from astro.guidance.engine import Guide
from astro.pointing import coords
from astro.pointing.coords import Site
from astro.session import Session

WPB = Site(lat_deg=26.7, lon_deg=-80.1)
EVENING = datetime(2026, 10, 3, 20, 0, tzinfo=timezone(timedelta(hours=-4)))
NOON = datetime(2026, 10, 3, 12, 0, tzinfo=timezone(timedelta(hours=-4)))


def make(when=EVENING, scope=None):
    scope = scope or SimScope(45, 180)
    return Session(WPB, lambda: (scope.alt, scope.az), clock=lambda: when), scope


def texts(msgs):
    return [m["text"] for m in msgs if m["type"] == "say"]


def test_tonight_then_next():
    s, _ = make()
    assert "is the best" in texts(s.handle("what's good tonight?"))[0]
    assert texts(s.handle("next"))[0].startswith("Let's find")


def test_goto_unknown_and_known():
    s, _ = make()
    assert texts(s.handle("go to pizza")) == ["I don't know pizza."]
    assert texts(s.handle("go to albireo")) == ["Let's find Albireo."]


def test_daytime_refused():
    s, _ = make(NOON)
    assert "daytime lockout" in texts(s.handle("go to mizar"))[0]


def test_below_horizon_refused():
    s, _ = make()
    assert "below the horizon" in texts(s.handle("go to M41"))[0]


def test_full_guided_session_reaches_target():
    s, scope = make()
    s.handle("go to Albireo")
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


def test_where_reports_nearest():
    s, scope = make()
    scope.alt, scope.az = s.altaz_of("Albireo")
    assert texts(s.where()) == ["You're on Albireo."]


def test_tonight_asked_in_the_afternoon_plans_the_night_in_local_time():
    """Review H9: asked at 4 PM it used to say "Nothing good is up right now"."""
    afternoon = datetime(2026, 10, 3, 16, 0, tzinfo=timezone(timedelta(hours=-4)))
    s, _ = make(afternoon)
    said = texts(s.handle("what's good tonight?"))[0]
    assert re.match(r"It's still light out\. Once it's dark, around [78]:\d\d PM: ", said)
    assert "is the best" in said


def test_agent_list_uses_local_times():
    s, _ = make(datetime(2026, 10, 4, 0, 30, tzinfo=UTC))  # 8:30 PM EDT
    lines = s.tonight_by_category().splitlines()
    assert all("PM" in line or "AM" in line for line in lines if "best around" in line)


def test_timezone_lookup_uses_exact_coordinates(monkeypatch):
    """Rounding could move a site near a time-zone border into the neighboring zone."""

    seen = []

    class Finder:
        def timezone_at(self, lat, lng):
            seen.append((lat, lng))
            return "America/New_York"

    monkeypatch.setattr(coords, "_timezone_finder", lambda: Finder())
    coords._zone_at.cache_clear()
    assert str(Site(26.712345, -80.054321).timezone) == "America/New_York"
    assert seen == [(26.712345, -80.054321)]
    coords._zone_at.cache_clear()


def test_describe_target():
    s, _ = make()
    said = texts(s.handle("tell me about Albireo"))[0]
    assert said.startswith("Albireo is a double star (Beta Cygni). Gold and blue pair")
    assert "degrees up, toward the" in said
    assert "below the horizon" in texts(s.handle("what is M41"))[0]  # October evening
    assert texts(s.handle("tell me about pizza")) == ["I don't know pizza."]


def test_learns_the_users_left_and_right():
    """Review G3: a user whose 'right' turns the scope toward smaller azimuth still arrives."""
    s, scope = make(scope=SimScope(45, 300))
    s.handle("go to Albireo")
    user, t, state, said = SimUser(right_is_plus_az=False), 0.0, None, []
    for _ in range(2000):
        for m in s.tick(t):
            if m["type"] == "say":
                user.hear(m["text"], t)
                said.append(m["text"])
            else:
                state = m
        scope.step(*user.act(t), 0.1)
        t += 0.1
        if state and state["on_target"] and user.v == (0.0, 0.0):
            break
    assert "Got it, I'll use your left and right from now on." in said
    assert s.right_is_plus_az is False and state["on_target"]
    assert state["right_is_plus_az"] is False  # the tablet arrow follows the user's sense


def _guiding_session(az):
    pos = {"alt": 45.0, "az": az}
    s = Session(WPB, lambda: (pos["alt"], pos["az"]), clock=lambda: EVENING)
    s.target = "Albireo"
    s.guide = Guide(45.0, az + 10.0)
    s._resolved_at = 1e9  # keep the test's target (no catalog refresh)
    return s, pos


def test_direction_flip_is_heard_without_a_stale_cue():
    s, pos = _guiding_session(100.0)
    s._direction_probe = ("right", 100.0, 0.0)  # we said "right"...
    pos["az"] = 99.0  # ...and the user pushed toward smaller azimuth
    said = texts(s.tick(0.2))
    assert said == ["Got it, I'll use your left and right from now on."]
    assert s.right_is_plus_az is False and s.guide.right_is_plus_az is False


def test_direction_settles_on_the_first_clear_move_and_expires():
    s, pos = _guiding_session(100.0)
    s._direction_probe = ("right", 100.0, 0.0)
    pos["az"] = 100.6  # clear move well inside the window
    s.tick(0.3)
    assert s._direction_known and s.right_is_plus_az is True
    s2, _ = _guiding_session(100.0)
    s2._direction_probe = ("right", 100.0, 0.0)
    s2.tick(2.0)  # no movement within the window: discarded, not learned
    assert not s2._direction_known
    assert s2._direction_probe is None or s2._direction_probe[2] == 2.0  # only a fresh probe


def test_probe_is_dropped_when_guidance_stops():
    s, _ = _guiding_session(100.0)
    s._direction_probe = ("right", 100.0, 0.0)
    s.handle("stop")
    s.tick(0.5)
    assert s._direction_probe is None


def test_stale_solve_fix_holds_cues_until_a_fresh_one():
    s, pos = _guiding_session(100.0)
    age = [5.0]
    s.finder = SimpleNamespace(synced=True, fix_age=lambda: age[0],
                               position=lambda: (pos["alt"], pos["az"]))
    s._direction_probe = ("right", 100.0, 0.0)
    held = s.tick(0.1)
    assert held[0] == {"type": "hold"} and s._direction_probe is None
    assert texts(held) == ["Hold still for a second so I can see where we are."]
    assert s.tick(0.2) == []  # said once per stale stretch, no cues from the old fix
    age[0] = 0.5
    assert any(m["type"] == "state" for m in s.tick(0.3))


def test_stale_fix_holds_centering_too():
    s, pos = _guiding_session(100.0)
    s.guide, s._centering = None, True
    s.finder = SimpleNamespace(synced=True, fix_age=lambda: 5.0,
                               position=lambda: (pos["alt"], pos["az"]))
    s._center_step = lambda t: pytest.fail("centered from a stale fix")
    assert texts(s.tick(0.1)) == ["Hold still for a second so I can see where we are."]


def test_describe_planets_and_the_moon_and_time_a_star():
    s, _ = make()
    assert texts(s.describe("Jupiter"))[0].startswith("Jupiter is a planet.")
    assert texts(s.describe("Moon"))[0].startswith("Moon is our Moon.")
    assert "Albireo" in texts(s.timing("Albireo"))[0]
    assert texts(s.timing("pizza")) == ["I don't know pizza."]


class BlindFinder:
    """Encoders and a camera, but no plate solve (clouds)."""
    synced, last_solution, on_change, camera = False, None, None, None

    def __init__(self, age=0.1):
        self.encoder_age = lambda: age

    def sync(self, fresh=False):
        return False, "I can't see any stars."

    def position(self):
        return 45.0, 180.0

    def alignment(self):
        return 0, None

    def raw_counts(self):
        raise OSError("port gone")


def test_commands_that_need_the_pointing_say_why_they_cant():
    s = Session(WPB, finder=BlindFinder(), clock=lambda: EVENING)
    assert texts(s.goto("Albireo")) == ["Before we go, I need to see the stars. I can't see any stars."]
    assert texts(s.where()) == ["I don't know where we're pointing yet. I can't see any stars."]
    assert texts(s.start_horizon()) == ["Before the horizon walk, I need to see the stars. I can't see any stars."]
    s.recorder = SimpleNamespace(busy=True, current=SimpleNamespace(paused=SimpleNamespace(is_set=lambda: False)))
    assert texts(s._recenter()) == ["I can't guide back yet: I can't see any stars."]


@pytest.mark.parametrize("age,ok", [(0.1, True), (5.0, False)])
def test_connections_report_frozen_encoders(age, ok):
    s = Session(WPB, finder=BlindFinder(age), clock=lambda: EVENING)
    assert s.connections() == {"finder": None, "main": None, "encoders": ok}


def test_debug_info_survives_a_dead_serial_port():
    info = Session(WPB, finder=BlindFinder(), clock=lambda: EVENING).debug_info()
    assert info["encoders"] == {"error": "port gone"} and info["pointing"]["synced"] is False
    assert info["main_camera"] == {"connected": False}


def test_a_crashing_background_solve_is_logged_and_the_next_one_runs(caplog):
    class Broken(BlindFinder):
        def sync(self, fresh=False):
            raise ZeroDivisionError

    s = Session(WPB, finder=Broken(), clock=lambda: EVENING)
    s._auto_solve(was_synced=False)
    assert not s._solving and "auto solve failed" in caplog.text
