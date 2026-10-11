"""Main camera plate solve: its stars matched to catalog stars near the finder's solve."""

import urllib.request

import numpy as np
import pytest
from scipy import ndimage

from astro.pointing import main_solve
from astro.pointing.align import box_on_finder_view, finder_offset_to_sky, fit_from_solves
from astro.pointing.labels import gnomonic, sky_to_pixels
from astro.pointing.main_solve import (
    MAIN_SCALE,
    _tangent,
    _untangent,
    cached_catalog,
    detect_stars,
    field_catalog,
    parse_rows,
    solve_main,
    tile,
)
from astro.pointing.platesolve import Solution

FIXTURE = "tests/data/main_solve/cygnus-2026-10-10.npz"  # detected stars + Gaia G < 12.5 nearby
FINDER = Solution(150.0, 30.0, 25.0, 10.38, 20, 40, 1e-9, 300)  # a made-up finder solve


def sky_field(rng, ra0=FINDER.ra_deg, dec0=FINDER.dec_deg, n=12000):
    """Random catalog stars (ra, dec, G) within ~5 degrees, as dense as the Milky Way at G < 12.5."""
    r, a = 5 * np.sqrt(rng.random(n)), rng.random(n) * 2 * np.pi
    dec = dec0 + r * np.sin(a)
    ra = ra0 + r * np.cos(a) / np.cos(np.radians(dec0))
    return np.c_[ra, dec, 12.5 - rng.exponential(1.0, n)]


def seen_by_main(catalog, ra_c, dec_c, roll_deg, mirrored=False, rng=None):
    """Main pixels from the center of the catalog stars in a 3856 x 2180 frame at MAIN_SCALE,
    brightest first (up to 25), with 0.3" of centroid noise."""
    w = _tangent(catalog[:, 0], catalog[:, 1], ra_c, dec_c) / MAIN_SCALE
    z = w * np.exp(-1j * np.radians(roll_deg))
    z = z.conj() if mirrored else z
    inside = (abs(z.real) < 1928) & (abs(z.imag) < 1090)
    order = np.argsort(catalog[inside, 2])[:25]
    xy = np.c_[z[inside].real, z[inside].imag][order]
    return xy + (rng.normal(0, 0.6, xy.shape) if rng is not None else 0)


@pytest.mark.parametrize("mirrored", [False, True])
def test_solve_finds_the_main_camera_a_few_degrees_from_the_finder(mirrored):
    rng = np.random.default_rng(3)
    catalog = sky_field(rng)
    ra_c, dec_c = _untangent(complex(2.1 * 3600, -1.4 * 3600), FINDER.ra_deg, FINDER.dec_deg)
    xy = seen_by_main(catalog, ra_c, dec_c, 123.0, mirrored, rng)
    xy = np.r_[xy[:3], [[100.0, -50.0]], xy[3:]]  # a hot pixel or satellite among the bright ones
    sol = solve_main(xy, catalog, FINDER.ra_deg, FINDER.dec_deg)
    assert sol is not None and sol.mirrored == mirrored
    off = abs(_tangent(sol.ra_deg, sol.dec_deg, ra_c, dec_c))
    assert off < 1.0 and sol.matches == len(xy) - 1 and sol.rms_arcsec < 1.0  # all but the fake
    assert sol.roll_deg % 360 == pytest.approx(123.0, abs=0.05)
    assert sol.scale_arcsec_px == pytest.approx(MAIN_SCALE, rel=0.002)


def test_no_solve_from_stars_that_are_not_in_the_catalog():
    rng = np.random.default_rng(4)
    xy = rng.uniform([-1900, -1000], [1900, 1000], (20, 2))
    assert solve_main(xy, sky_field(rng), FINDER.ra_deg, FINDER.dec_deg) is None
    assert solve_main(xy[:4], sky_field(rng), FINDER.ra_deg, FINDER.dec_deg) is None  # too few


def test_detect_stars_centroids_in_raw_pixels_from_the_center():
    rng = np.random.default_rng(5)
    h, w = 600, 800
    stars = [(500.0, 200.0, 40000), (130.0, 450.0, 15000), (700.0, 520.0, 6000)]
    img = np.zeros((h, w))
    for x, y, flux in stars:
        img[int(y), int(x)] = flux
    raw = np.clip(ndimage.gaussian_filter(img, 3.0) + 30 + rng.normal(0, 10, (h, w)), 0, 255).astype(np.uint8)
    xy, flux = detect_stars(raw)
    assert len(xy) == 3 and np.all(np.diff(flux) < 0)
    assert np.allclose(xy, [(x - w / 2, y - h / 2) for x, y, _ in stars], atol=1.0)


def test_real_night_frame_solves():
    """2026-10-10, Cygnus: 30 of 30 stars on Gaia stars, 0.4" rms, not mirrored, 0.500"/px."""
    got = np.load(FIXTURE)
    ra, dec, roll, fov = got["finder"]
    sol = solve_main(got["main_xy"].astype(float), got["catalog"], ra, dec)
    assert sol.matches == 30 and sol.rms_arcsec < 0.6 and not sol.mirrored
    assert (sol.ra_deg, sol.dec_deg) == pytest.approx((318.1679, 44.5914), abs=2e-4)
    assert sol.scale_arcsec_px == pytest.approx(0.5002, abs=0.0005)
    fit = fit_from_solves(Solution(ra, dec, roll, fov, 0, 0, 0, 0), (1280, 960), sol)
    # The same place on every pair that night: (644, 495) on the finder, 58.4x its scale.
    assert np.allclose(fit.center, [643.8, 495.1], atol=1.0) and fit.scale == pytest.approx(58.37, abs=0.1)


def test_two_solves_place_the_main_camera_on_the_finder():
    """The main center lands where the finder's solve puts its RA/Dec, and finder_offset_to_sky
    takes it back to the same sky offset."""
    rng = np.random.default_rng(6)
    catalog = sky_field(rng)
    ra_c, dec_c = _untangent(complex(-1.5 * 3600, 0.8 * 3600), FINDER.ra_deg, FINDER.dec_deg)
    xy = seen_by_main(catalog, ra_c, dec_c, 40.0)
    main = solve_main(xy, catalog, FINDER.ra_deg, FINDER.dec_deg)
    fit = fit_from_solves(FINDER, (1280, 960), main)
    east, north = gnomonic(FINDER.ra_deg, FINDER.dec_deg, ra_c, dec_c)
    assert np.allclose(fit.center, np.add(sky_to_pixels(east, north, FINDER, 1280), [640, 480]), atol=0.05)
    assert finder_offset_to_sky(fit.center[0] - 640, fit.center[1] - 480, FINDER, 1280) == pytest.approx(
        (east, north), abs=1e-4)
    assert fit.scale == pytest.approx(FINDER.fov_deg * 3600 / 1280 / MAIN_SCALE, rel=0.01)
    box = np.array(box_on_finder_view(fit, (3856, 2180), (1280, 960), 0))
    assert np.allclose(box.mean(axis=0), (fit.center - [640, 480]) / [1280, 960], atol=1e-3)


def test_catalog_is_fetched_once_per_tile_then_read_from_disk(tmp_path, monkeypatch):
    tsv = "#comment\nRA_ICRS\tDE_ICRS\tGmag\ndeg\tdeg\tmag\n-----\t-----\t-----\n150.1\t+30.2\t9.5\n150.3\t+29.9\t11.0\n\n"
    calls = []

    class Reply:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def read(self):
            return tsv.encode()

    def urlopen(url, timeout):
        calls.append(url)
        return Reply()

    monkeypatch.setattr(urllib.request, "urlopen", urlopen)
    stars = field_catalog(150.0, 30.0, tmp_path)
    assert stars.tolist() == [[150.1, 30.2, 9.5], [150.3, 29.9, 11.0]]
    assert np.array_equal(field_catalog(150.4, 30.3, tmp_path), stars) and len(calls) == 1  # same tile

    def offline(url, timeout):
        raise OSError("no network")

    monkeypatch.setattr(urllib.request, "urlopen", offline)
    assert field_catalog(10.0, -40.0, tmp_path) is None


def test_a_neighbor_tile_serves_a_field_just_across_the_edge(tmp_path):
    """2026-10-10: the scope drifted 0.1 degrees into the next tile and Align had no stars."""
    stars = np.array([[150.1, 30.2, 9.5]])
    np.save(main_solve._tile_path(150.0, 30.0, tmp_path), stars)
    ra_c, dec_c = main_solve.tile(150.0, 30.0)
    assert np.array_equal(cached_catalog(ra_c + 1.3, dec_c, tmp_path), stars)  # the next tile over
    assert cached_catalog(ra_c + 4.0, dec_c, tmp_path) is None  # too far: its cone falls short


def test_tiles_cover_the_search_radius():
    """A tile's cone reaches SEARCH_DEG past any finder center that falls in the tile."""
    rng = np.random.default_rng(7)
    for ra, dec in zip(rng.uniform(0, 360, 2000), rng.uniform(-85, 85, 2000), strict=True):
        ra_c, dec_c = tile(ra, dec)
        x, y = gnomonic(ra_c, dec_c, ra, dec)
        assert np.hypot(x, y) + main_solve.SEARCH_DEG + 0.3 < main_solve.TILE_RADIUS_DEG
    assert parse_rows("").shape == (0, 3)
