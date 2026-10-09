from datetime import datetime, timedelta, timezone

from astro.planner.catalog import load_targets
from astro.planner.horizon import HorizonMask
from astro.planner.tonight import moonlight_factor, next_dark, night_times, plan
from astro.pointing.coords import Site

WPB = Site(lat_deg=26.7, lon_deg=-80.1)
EDT = timezone(timedelta(hours=-4))
EVENING = datetime(2026, 10, 3, 19, 0, tzinfo=EDT)


def test_catalog_loads():
    targets = load_targets()
    assert len(targets) >= 20
    assert {t.category for t in targets} >= {"nebula", "cluster", "double star"}


def test_night_window_excludes_daylight():
    times = night_times(WPB, EVENING)
    assert len(times) > 20
    assert all(t.to_datetime(timezone=EDT).hour not in range(8, 18) for t in times)


def test_plan_ranks_categories():
    choices = plan(WPB, EVENING)
    assert "planet" in choices and "cluster" in choices
    for items in choices.values():
        assert items == sorted(items, key=lambda c: -c.score)
        assert all(c.best_alt_deg > 20 for c in items)


def test_high_horizon_hides_everything():
    assert plan(WPB, EVENING, mask=HorizonMask(((0, 89),))) == {}


def test_horizon_mask_interpolates_and_wraps():
    m = HorizonMask(((0, 10), (90, 30), (270, 10)))
    assert m.min_alt(45) == 20
    assert m.min_alt(315) == 10
    assert m.to_stellarium().splitlines()[1] == "90.0 30.0"


def test_next_dark_plans_the_coming_night_when_asked_by_day():
    afternoon = datetime(2026, 10, 3, 16, 0, tzinfo=EDT)
    start = next_dark(WPB, afternoon)
    assert start.date() == afternoon.date() and 19 <= start.hour <= 20  # nautical dusk
    assert next_dark(WPB, EVENING.replace(hour=22)) == EVENING.replace(hour=22)  # already dark


def test_moon_phase_scales_the_moonlight_penalty():
    assert moonlight_factor("galaxy", 20, 1.0) == 0.5  # full Moon nearby: halved
    assert moonlight_factor("galaxy", 90, 1.0) == 0.75  # full Moon far away: still hurts
    assert moonlight_factor("galaxy", 20, 0.0) == 1.0  # new Moon: no harm
    assert moonlight_factor("double", 20, 1.0) > moonlight_factor("galaxy", 20, 1.0)
