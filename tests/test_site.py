from datetime import UTC, datetime

from astro import site_store
from astro.intents import Intent, parse
from astro.pointing.coords import Site
from astro.session import Session


def test_saved_site_overrides_default(tmp_path):
    (tmp_path / "config").mkdir()
    (tmp_path / "config" / "site.toml").write_text("lat_deg = 26.7\nlon_deg = -80.1\n")
    assert not site_store.has_saved(tmp_path) and site_store.load(tmp_path).lat_deg == 26.7
    site_store.save(tmp_path, Site(37.12345, -122.5, 30))
    assert site_store.has_saved(tmp_path)
    assert site_store.load(tmp_path) == Site(37.12345, -122.5, 30)


def test_location_intent():
    assert parse("use my location") == Intent("location")
    assert parse("update the gps") == Intent("location")


class FakeFinder:
    def __init__(self):
        self.reset_to = None

    def reset(self, site):
        self.reset_to = site

    def position(self):
        return (45.0, 180.0)


def session(finder=None, saved=None):
    return Session(Site(26.7, -80.1), clock=lambda: datetime(2026, 10, 4, 3, tzinfo=UTC),
                   finder=finder, on_site_change=(saved.append if saved is not None else None))


def test_set_location_moves_site_resets_pointing_and_saves():
    finder, saved = FakeFinder(), []
    s = session(finder, saved)
    out = s.set_location(37.1, -122.5, None, 8.0)
    assert out[0]["text"] == "Got it, I know where we are, accurate to about 8 meters."
    assert s.site.lat_deg == 37.1 and s.site.elevation_m == 0
    assert finder.reset_to == s.site and saved == [s.site]


def test_tiny_gps_jitter_keeps_pointing():
    finder = FakeFinder()
    s = session(finder)
    s.set_location(26.7001, -80.1001, 3.0, None)
    assert finder.reset_to is None and s.site.elevation_m == 3.0


def test_bad_coordinates_rejected():
    s = session()
    assert "doesn't look right" in s.set_location(123, 0, None, None)[0]["text"]
    assert s.site.lat_deg == 26.7


def test_move_is_measured_as_distance_from_model_site():
    finder = FakeFinder()
    s = session(finder)
    for i in range(1, 6):  # five 0.4 km steps north: each small, together 2 km
        s.set_location(26.7 + i * 0.0036, -80.1, None, None)
    assert finder.reset_to is not None and abs(finder.reset_to.lat_deg - 26.7 - 3 * 0.0036) < 1e-9


def test_dateline_crossing_is_a_small_move():
    finder = FakeFinder()
    s = Session(Site(0.0, 179.9995), clock=lambda: datetime(2026, 10, 4, 3, tzinfo=UTC), finder=finder)
    s.set_location(0.0, -179.9995, None, None)  # ~110 m east across the dateline
    assert finder.reset_to is None
