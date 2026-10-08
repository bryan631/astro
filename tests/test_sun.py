from datetime import UTC, datetime

from astro.planner.sun import sun_events
from astro.pointing.coords import Site

PALM_BEACH = Site(26.62, -80.14)


def test_sun_events_in_order_with_published_sunset_and_sunrise():
    events = sun_events(PALM_BEACH, datetime(2026, 10, 8, 20, 0, tzinfo=UTC))  # 4 PM EDT
    names = [name for _, name in events]
    assert names[0] == "sunset" and names[-1] == "sunrise" and len(names) == 8
    sunset, sunrise = events[0][0], events[-1][0]
    # West Palm Beach in early October: sunset about 6:58 PM, sunrise 7:16 AM EDT (UTC-4).
    assert abs((sunset - datetime(2026, 10, 8, 22, 58, tzinfo=UTC)).total_seconds()) < 180
    assert abs((sunrise - datetime(2026, 10, 9, 11, 16, tzinfo=UTC)).total_seconds()) < 180
