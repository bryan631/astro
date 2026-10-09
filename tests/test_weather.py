import io
import json
from datetime import UTC, datetime

from astro.planner import weather
from astro.planner.weather import forecast_text
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


def test_fifty_percent_counts_as_cloudy():
    s, _ = session(50.0)
    assert s.handle("what's good tonight")[0]["text"].startswith("It looks about 50 percent cloudy")


def test_moving_refetches_the_forecast():
    s, calls = session(20.0)
    s.handle("what's good tonight")
    s.set_location(40.0, -105.0, None, None)
    s.handle("what's good tonight")
    assert len(calls) == 2


def test_clear_or_offline_says_nothing_about_clouds():
    for clouds in (10.0, None):
        s, _ = session(clouds)
        assert "cloudy" not in s.handle("what's good tonight")[0]["text"]


def test_agent_list_includes_cloud_cover():
    s, _ = session(35.0)
    assert "cloud cover: about 35%" in s.tonight_by_category()


def hourly(lat, lon):
    times = [f"2026-10-08T{h:02d}:00" for h in range(18, 24)]
    n = len(times)
    return {"current": {"time": "2026-10-08T20:15", "cloud_cover": 90},
            "hourly": {"time": times, "cloud_cover": [100, 100, 90, 50, 20, 10],
                       "cloud_cover_low": [0] * n, "cloud_cover_mid": [0] * n,
                       "cloud_cover_high": [100, 100, 90, 50, 20, 10],
                       "precipitation_probability": [5] * n, "temperature_2m": [77.4] * n,
                       "dew_point_2m": [74.0] * n, "wind_speed_10m": [3.2] * n}}


def test_forecast_text_starts_at_the_current_hour():
    lines = forecast_text(26.6, -80.1, hours=3, get=hourly).splitlines()
    assert lines[0] == "Now 20:15 local; cloud cover 90%."
    assert lines[1].startswith("10-08 20:00: clouds 90%")
    assert lines[-2].startswith("10-08 22:00: clouds 20% (low 0, mid 0, high 20), rain 5%, 77F")
    assert len(lines) == 5
    assert lines[-1] == "Dew unlikely in this period."  # 77F air, 74F dew point
    assert forecast_text(26.6, -80.1, get=lambda lat, lon: {}) is None  # unexpected reply


def test_forecast_text_warns_of_dew():
    def dewy(lat, lon):
        data = hourly(lat, lon)
        data["hourly"]["dew_point_2m"] = [70, 71, 72, 75, 76, 77]  # air stays 77.4F
        return data

    assert forecast_text(26.6, -80.1, get=dewy).splitlines()[-1].startswith(
        "Dew likely from 10-08 21:00")


def test_null_forecast_hours_are_skipped(monkeypatch):
    body = {"hourly": {"time": ["2026-10-08T21:00", "2026-10-08T22:00"], "cloud_cover": [40, None]}}
    monkeypatch.setattr(weather.urllib.request, "urlopen",
                        lambda url, timeout: io.BytesIO(json.dumps(body).encode()))
    assert weather.hourly_cloud_cover(1, 2) == {"2026-10-08T21:00": 40.0}

    def gappy(lat, lon):
        data = hourly(lat, lon)
        data["hourly"]["temperature_2m"][3] = None  # 21:00
        return data

    lines = weather.forecast_text(26.6, -80.1, hours=3, get=gappy).splitlines()
    assert [ln[:11] for ln in lines[1:-1]] == ["10-08 20:00", "10-08 22:00"]
