from datetime import datetime, timedelta, timezone

from astro.planner.catalog import load_targets
from astro.planner.horizon import HorizonMask
from astro.planner.tonight import night_times, plan, start_of_evening
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


def test_start_of_evening():
    noon = datetime(2026, 10, 3, 12, tzinfo=EDT)
    assert start_of_evening(noon).hour == 18
    assert start_of_evening(EVENING) == EVENING
