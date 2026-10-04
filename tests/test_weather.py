from datetime import UTC, datetime

from astro.pointing.coords import Site
from astro.session import Session

NIGHT = datetime(2026, 10, 4, 2, 0, tzinfo=UTC)


def session(clouds):
    calls = []

    def weather(lat, lon, when):
        calls.append(when)
        return clouds

    s = Session(Site(26.7, -80.1), lambda: (45, 180), clock=lambda: NIGHT, weather=weather)
    return s, calls


def test_cloudy_night_is_mentioned_and_cached():
    s, calls = session(80.0)
    said = s.handle("what's good tonight")[0]["text"]
    assert said.startswith("It looks about 80 percent cloudy")
    s.handle("what's good tonight")
    assert len(calls) == 1  # cached, not asked again


def test_clear_or_offline_says_nothing_about_clouds():
    for clouds in (10.0, None):
        s, _ = session(clouds)
        assert "cloudy" not in s.handle("what's good tonight")[0]["text"]


def test_agent_list_includes_cloud_cover():
    s, _ = session(35.0)
    assert "cloud cover: about 35%" in s.tonight_by_category()
