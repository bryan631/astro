from astro.planner.moon_features import best_features, lit_side


def names(d):
    return [f.name for f, _ in best_features(d)]


def test_terminator_sweeps_east_to_west():
    assert lit_side(0) == (90, True)  # new Moon: sunrise at the east limb
    assert lit_side(90) == (0, True)  # first quarter: line down the middle, east lit
    assert lit_side(270) == (0, False)  # last quarter: west lit


def test_first_quarter_shows_the_central_line():
    shown = names(90)  # terminator at 0: lit east of it
    assert "Theophilus" in shown and "Posidonius" in shown
    assert "Copernicus" not in shown  # still dark


def test_waxing_crescent_shows_the_east_and_waning_the_west():
    assert names(45)[0] in ("Mare Crisium", "Petavius")  # terminator at +45
    assert "Aristarchus" in names(305)  # waning: sunset line at -35, just west of it


def test_plan_lists_moon_features_and_goto_aims_at_the_moon():
    from datetime import datetime, timedelta, timezone

    from astropy.time import Time

    from astro.planner.tonight import moon_minus_sun_deg, plan
    from astro.pointing.coords import Site
    from astro.session import Session

    site = Site(26.7, -80.1)
    evening = datetime(2026, 10, 19, 20, 0, tzinfo=timezone(timedelta(hours=-4)))  # ~1st qtr
    assert 60 < moon_minus_sun_deg(Time(evening)) < 120
    features = plan(site, evening).get("moon feature", [])
    assert features and all(c.category == "moon feature" for c in features)
    s = Session(site, lambda: (45, 180), clock=lambda: evening)
    assert s.altaz_of(features[0].name) == s.altaz_of("Moon")
    assert "feature on the Moon" in s.describe(features[0].name)[0]["text"]
