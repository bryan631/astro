import re
from datetime import UTC, datetime, timedelta, timezone

from astro.devices.sim.scope import SimScope, SimUser
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
    from astro.pointing import coords

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
