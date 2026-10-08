from datetime import UTC, datetime

from astro.planner.almanac import events_text, moon_text, phase_name, sun_events
from astro.pointing.coords import Site

PALM_BEACH = Site(26.62, -80.14)
OCT8_4PM = datetime(2026, 10, 8, 20, 0, tzinfo=UTC)  # 4 PM EDT


def test_sun_events_in_order_with_known_sunset_and_sunrise():
    events = sun_events(PALM_BEACH, OCT8_4PM)
    names = [name for _, name in events]
    assert names[0] == "sunset" and names[-1] == "sunrise" and len(names) == 8
    sunset, sunrise = events[0][0], events[-1][0]
    # West Palm Beach in early October: sunset about 6:58 PM, sunrise 7:16 AM EDT (UTC-4).
    assert abs((sunset - datetime(2026, 10, 8, 22, 58, tzinfo=UTC)).total_seconds()) < 180
    assert abs((sunrise - datetime(2026, 10, 9, 11, 16, tzinfo=UTC)).total_seconds()) < 180


def test_phase_names():
    assert phase_name(0.01, True) == "new" and phase_name(0.99, False) == "full"
    assert phase_name(0.5, True) == "first quarter" and phase_name(0.5, False) == "last quarter"
    assert phase_name(0.2, False) == "waning crescent" and phase_name(0.8, True) == "waxing gibbous"


def test_moon_two_days_before_new():
    text = moon_text(PALM_BEACH, OCT8_4PM)
    assert "waning crescent" in text and "moonrise" in text
    assert "Next new Moon: Sat Oct 10" in text


def test_events_include_the_orionids():
    assert "Wed Oct 21: Orionids meteor shower peaks" in events_text(PALM_BEACH, OCT8_4PM)


def test_each_moon_phase_is_listed_once():
    text = moon_text(PALM_BEACH, datetime(2026, 10, 9, 12, 0, tzinfo=UTC))  # a day before new
    assert text.count("Next new Moon") == 1 and text.count("Next full Moon") == 1
