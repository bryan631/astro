from datetime import UTC, datetime

from astro.pointing.coords import Site, body_altaz
from astro.safety import check_target

WPB = Site(lat_deg=26.7, lon_deg=-80.1)
NOON = datetime(2026, 6, 21, 17, 0, tzinfo=UTC)
NIGHT = datetime(2026, 6, 21, 4, 0, tzinfo=UTC)


def test_sun_refused_even_with_override():
    sun = body_altaz("sun", WPB, NOON)
    r = check_target(*sun, WPB, NOON, developer_override=True)
    assert not r.ok and "Sun" in r.reason


def test_just_inside_exclusion_refused():
    sun_alt, sun_az = body_altaz("sun", WPB, NOON)
    assert not check_target(sun_alt - 19, sun_az, WPB, NOON, developer_override=True).ok


def test_exclusion_cannot_be_narrowed():
    sun_alt, sun_az = body_altaz("sun", WPB, NOON)
    assert not check_target(sun_alt - 10, sun_az, WPB, NOON, True, exclusion_deg=5).ok


def test_daytime_lockout_and_override():
    assert check_target(30, 0, WPB, NOON).reason == "daytime lockout"
    assert check_target(30, 0, WPB, NOON, developer_override=True).ok


def test_below_horizon_refused():
    assert check_target(-5, 0, WPB, NIGHT).reason == "below the horizon"


def test_night_target_ok():
    assert check_target(45, 0, WPB, NIGHT).ok
