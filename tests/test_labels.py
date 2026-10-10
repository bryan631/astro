from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import numpy as np
import pytest
from scipy import ndimage

from astro.pointing.coords import Site, radec_to_altaz
from astro.pointing.labels import altaz_now, load_stars, place, tube_map
from astro.pointing.platesolve import FinderSolver, finder_gray
from astro.process.view import ROTATE
from astro.session import Session

FIXTURE = "tests/data/align/bright-star-2026-10-10.npz"


@pytest.fixture(scope="module")
def solved():
    frame = np.load(FIXTURE)["finder"][0]
    return frame, FinderSolver().solve(frame, binned=2)


def test_star_list_is_bright_first_with_names_and_designations():
    names, ra, dec = load_stars()
    assert names[:2] == ["Sirius", "Canopus"] and len(names) > 3000
    assert {"Vega", "Albireo", "ζ Cyg", "72 Psc"} <= set(names)
    assert ((ra >= 0) & (ra < 360)).all() and (abs(dec) <= 90).all()


def field_at(sol, altitude):
    """A site and time where the solved field is at `altitude` (any will do: the map and the
    stars are computed for the same ones). At latitude = the field's dec, it passes overhead."""
    site, t = Site(sol.dec_deg, 0.0), datetime(2026, 10, 10, tzinfo=UTC)
    alt, az = radec_to_altaz(sol.ra_deg, sol.dec_deg, site, t)
    while abs(alt - altitude) > 4:  # wait for the field to reach that altitude
        t += timedelta(minutes=10)
        alt, az = radec_to_altaz(sol.ra_deg, sol.dec_deg, site, t)
    return site, t, (alt, az)


def detected_stars(frame):
    """Star centers in sensor (x, y) pixels."""
    gray = finder_gray(frame).astype(float)
    bg = np.median(gray)
    blobs, n = ndimage.label(ndimage.uniform_filter(gray, 3) > bg + 6 * 1.4826 * np.median(abs(gray - bg)))
    return np.array(ndimage.center_of_mass(gray - bg, blobs, range(1, n + 1)))[:, ::-1] * 2


def misses(stars, x, y):
    """Each label's distance (finder pixels) to the nearest detected star."""
    return [np.hypot(*(stars - [xi, yi]).T).min() for xi, yi in zip(x, y, strict=True)]


@pytest.mark.parametrize("altitude", [25, 50, 80])
def test_names_land_on_the_stars_in_a_real_finder_frame(solved, altitude):
    """The solve's map, then the stars' alt-az from the same pointing: every named star in the
    2026-10-10 frame sits on a detected star, low or high in the sky."""
    frame, sol = solved
    h, w = frame.shape
    site, t, center = field_at(sol, altitude)
    _, ra, dec = load_stars()
    x, y, on = place(*altaz_now(ra, dec, site, t), center, tube_map(sol, w, site, t), (w, h))
    d = misses(detected_stars(frame), x[on], y[on])
    assert len(d) >= 5 and max(d) < 6  # finder pixels (~1.5 arcmin)


def test_session_names_sit_on_the_stars_and_come_back_after_a_restart(solved, tmp_path):
    """Through the session: the page's fractions (turned like the view) land on the stars, and
    the saved map brings the same names back after a restart, before any new solve."""
    frame, sol = solved
    h, w = frame.shape
    site, t, center = field_at(sol, 50)
    saved = []

    def session(last_solution, calibration=None):
        cam = SimpleNamespace(capture=lambda: frame, bayer=None, exposure_s=1.0, gain=0)
        finder = SimpleNamespace(camera=cam, last_solution=last_solution, synced=True,
                                 position=lambda: center)
        s = Session(site, clock=lambda: t, finder=finder, data_dir=tmp_path,
                    calibration=calibration, on_calibration_change=saved.append)
        s.camera("finder").capture()
        return s

    labels = session(sol).finder_labels()
    stars = [lb for lb in labels if lb[1] == "star"]
    sign = -1 if ROTATE["finder"] == 180 else 1
    x = [(sign * fx + 0.5) * w for *_, fx, _ in stars]
    y = [(sign * fy + 0.5) * h for *_, fy in stars]
    d = misses(detected_stars(frame), x, y)
    assert len(d) >= 5 and max(d) < 6
    assert saved[-1]["finder_map"] is not None
    assert session(None, saved[-1]).finder_labels() == labels
