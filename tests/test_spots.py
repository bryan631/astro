from datetime import UTC, datetime

from astro.planner.horizon import HorizonMask
from astro.planner.report import build, render_html
from astro.pointing.coords import Site
from astro.session import Session
from astro.spots import Spot, declination_deg, dumps, parse, upsert

DECK = Spot("Back deck", Site(26.62, -80.14), HorizonMask(((0.0, 40.0), (180.0, 25.0))))
DRIVE = Spot("Driveway", Site(26.62, -80.14), HorizonMask(((0.0, 80.0), (180.0, 80.0))))
EVENING = datetime(2026, 10, 8, 23, 0, tzinfo=UTC)  # 7 PM EDT


def test_spots_round_trip_and_upsert_replaces_by_name():
    spots, current = parse(dumps([DECK, DRIVE], "Driveway"))
    assert spots == [DECK, DRIVE] and current == "Driveway"
    moved = Spot("back DECK", Site(1.0, 2.0), DECK.mask)
    assert upsert(spots, moved) == [DRIVE, moved]
    odd = Spot('Bob\'s "north" \\ lawn', DECK.site, DECK.mask)  # quotes and a backslash
    assert parse(dumps([odd], odd.name)) == ([odd], odd.name)


def session(spots=(), spot=None):
    changes = []
    s = Session(Site(26.7, -80.1), lambda: (45, 180), clock=lambda: EVENING, spots=list(spots),
                spot=spot, on_spots_change=lambda sp, cur: changes.append((sp, cur)))
    return s, changes


def test_choosing_a_spot_uses_its_place_and_treeline():
    s, changes = session([DECK, DRIVE])
    assert s.use_spot("the driveway")[0]["text"].startswith("OK, we're at Driveway")
    assert s.horizon == DRIVE.mask and s.site.lat_deg == 26.62 and changes[-1][1] == "Driveway"
    assert "Known spots: Back deck, Driveway" in s.use_spot("the moon")[0]["text"]


def test_telescope_treeline_walk_replaces_the_current_spots_treeline():
    s, changes = session([DECK], spot="Back deck")
    s.handle("start the horizon walk")
    for _ in range(3):
        s.handle("mark")
    s.handle("done")
    assert changes[-1][0][0].mask == s.horizon != DECK.mask


def test_report_ranks_the_open_spot_first_and_counts_clouds():
    night = build([DRIVE, DECK], EVENING, clouds=lambda lat, lon: {}, targets=[])
    assert [n.spot.name for n in night.spots] == ["Back deck", "Driveway"]
    assert night.spots[0].picks and night.spots[0].picks[0].cloud_pct is None  # no forecast
    cloudy = build([DECK], EVENING, targets=[],
                   clouds=lambda lat, lon: {k: 100.0 for k in _hours()})
    assert cloudy.spots[0].score < night.spots[0].score
    page = render_html(night, [DRIVE, DECK])
    assert "Best spot: <b>Back deck</b>" in page and "Driveway" in page


def _hours():
    return [f"2026-10-{d:02d}T{h:02d}:00" for d in (8, 9) for h in range(24)]


def test_declination_from_the_world_magnetic_model():
    when = datetime(2026, 10, 8, tzinfo=UTC)
    assert -16 < declination_deg(42.4, -71.1, when) < -12  # Boston: about 14 W
    assert 13 < declination_deg(47.6, -122.3, when) < 17  # Seattle: about 15 E
