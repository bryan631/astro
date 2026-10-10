from datetime import UTC, datetime, timedelta

import numpy as np
import pytest
from scipy import ndimage

from astro.pointing.coords import Site, radec_to_altaz
from astro.pointing.labels import altaz_now, load_stars, place, tube_map
from astro.pointing.platesolve import FinderSolver, finder_gray

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


@pytest.mark.parametrize("altitude", [25, 50, 80])
def test_names_land_on_the_stars_in_a_real_finder_frame(solved, altitude):
    """The solve's map, then the stars' alt-az from the same pointing: every named star in the
    2026-10-10 frame sits on a detected star, low or high in the sky (any site and time will
    do: the map and the stars are computed for the same ones)."""
    frame, sol = solved
    h, w = frame.shape
    site, t = Site(sol.dec_deg, 0.0), datetime(2026, 10, 10, tzinfo=UTC)  # it passes overhead
    alt, az = radec_to_altaz(sol.ra_deg, sol.dec_deg, site, t)
    while abs(alt - altitude) > 4:  # wait for the field to reach that altitude
        t += timedelta(minutes=10)
        alt, az = radec_to_altaz(sol.ra_deg, sol.dec_deg, site, t)
    _, ra, dec = load_stars()
    x, y, on = place(*altaz_now(ra, dec, site, t), (alt, az), tube_map(sol, w, site, t), (w, h))
    gray = finder_gray(frame).astype(float)
    bg = np.median(gray)
    blobs, n = ndimage.label(ndimage.uniform_filter(gray, 3) > bg + 6 * 1.4826 * np.median(abs(gray - bg)))
    stars = np.array(ndimage.center_of_mass(gray - bg, blobs, range(1, n + 1)))[:, ::-1] * 2  # sensor (x, y)
    misses = [np.hypot(*(stars - [x[i], y[i]]).T).min() for i in np.flatnonzero(on)]
    assert len(misses) >= 5 and max(misses) < 6  # finder pixels (~1.5 arcmin)
