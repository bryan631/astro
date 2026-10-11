import time
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import numpy as np
import pytest
from scipy import ndimage

from astro.planner.catalog import Target
from astro.pointing.coords import Site, radec_to_altaz
from astro.pointing.labels import FinderLabels, altaz_now, load_stars, place, tube_map
from astro.pointing.main_solve import MainSolution
from astro.pointing.platesolve import FinderSolver, finder_gray
from astro.process.view import ROTATE
from astro.session import Session

FIXTURE = "tests/data/align/bright-star-2026-10-10.npz"
# Stars in that frame (HYG): ε and δ Psc lost their names with the Greek letters (2026-10-11).
FIELD_STARS = (["ε Psc", "δ Psc"], np.array([15.7359, 12.1706]), np.array([7.8901, 7.5851]), np.array([4.27, 4.44]))


@pytest.fixture(scope="module")
def solved():
    frame = np.load(FIXTURE)["finder"][0]
    return frame, FinderSolver().solve(frame, binned=2)


def test_star_list_is_bright_first_with_proper_names_only():
    names, ra, dec, mag = load_stars()
    assert names[:2] == ["Sirius", "Canopus"] and len(names) > 300 and (np.diff(mag) >= 0).all()
    assert {"Vega", "Albireo", "Deneb"} <= set(names)
    # no "72 Psc" or "δ Cyg": numbers and letters say nothing on the view (2026-10-10, -11)
    assert not any(n[0].isdigit() or n[0] in "αβγδεζηθικλμνξοπρστυφχψω" for n in names)
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
    _, ra, dec, _ = FIELD_STARS
    x, y, on = place(*altaz_now(ra, dec, site, t), center, tube_map(sol, w, site, t), (w, h))
    d = misses(detected_stars(frame), x[on], y[on])
    assert len(d) == 2 and max(d) < 6  # finder pixels (~1.5 arcmin)


def test_session_names_sit_on_the_stars_and_come_back_after_a_restart(solved, tmp_path, monkeypatch):
    """Through the session: the page's fractions (turned like the view) land on the stars, and
    the saved map brings the same names back after a restart, before any new solve."""
    frame, sol = solved
    h, w = frame.shape
    site, t, center = field_at(sol, 50)
    saved = []
    monkeypatch.setattr("astro.pointing.labels.load_stars", lambda: FIELD_STARS)

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
    assert len(d) == 2 and max(d) < 6
    assert saved[-1]["finder_map"] is not None
    assert session(None, saved[-1]).finder_labels() == labels


def test_level_one_names_only_bright_stars_and_targets_get_their_outline(solved, monkeypatch):
    """Names level 1: stars to BRIGHT_MAG; level 2: all. A target with a size carries its
    ellipse: the semi-major axis in view fractions, as long as the sky size at the finder's scale."""
    frame, sol = solved
    h, w = frame.shape
    site, t, center = field_at(sol, 50)
    names, ra, dec, _ = FIELD_STARS
    monkeypatch.setattr("astro.pointing.labels.load_stars", lambda: (names, ra, dec, np.array([2.5, 4.4])))
    blob = Target("Blob", "M99", "galaxy", sol.ra_deg, sol.dec_deg, "", (60.0, 30.0), 0.0)
    labels = FinderLabels([blob])
    one, _ = labels.labels(sol, (w, h), center, site, t, 0.0, 0, level=1)
    two, _ = labels.labels(sol, (w, h), center, site, t, 0.0, 0, level=2)
    assert [lb[0] for lb in one] == ["ε Psc", "M99"] and [lb[0] for lb in two] == ["ε Psc", "δ Psc", "M99"]
    *_, ex, ey, ratio = one[-1]
    semi_px = np.hypot(ex * w, ey * h)
    assert semi_px == pytest.approx(0.5 / (sol.fov_deg / w), rel=0.03) and ratio == 0.5  # 30' at the scale


def test_main_view_names_come_through_the_align_box(tmp_path):
    """A name at the box's center is the main view's center; at its top-left corner, the main
    view's top-left; outside the box, not on the main view. Outlines turn with the box."""
    s = Session(Site(40.0, -75.0), data_dir=tmp_path)
    s.main_box = [[0.1, 0.0], [0.2, 0.0], [0.2, 0.05], [0.1, 0.05]]  # main frame corners on the finder view
    s.finder_labels = lambda level=2: [["A", "star", 0.15, 0.025], ["B", "star", 0.1, 0.0],
                                       ["C", "star", -0.3, 0.2], ["M1", "target", 0.15, 0.025, 0.01, 0.0, 0.5]]
    got = {lb[0]: lb[2:] for lb in s.main_labels()}
    assert got["A"] == pytest.approx([0, 0], abs=1e-3) and got["B"] == pytest.approx([-0.5, -0.5], abs=1e-3)
    assert "C" not in got and got["M1"] == pytest.approx([0, 0, 0.1, 0, 0.5], abs=1e-3)


def test_main_view_names_from_its_own_plate_solve_are_exact(tmp_path):
    """Once the telescope view plate-solves, its names come from that solve: a target at the
    solve's center is the view's center, one 5' east lies 600 px along the solve's east."""
    s = Session(Site(26.6, -80.1), lambda: (45.0, 180.0), data_dir=tmp_path, main_sensor=(3856, 2180))
    msol = MainSolution(150.0, 30.0, 90.0, 0.5, 20, 0.4, 10, False, np.zeros((0, 2)), np.zeros((0, 2)))
    east = 150.0 + 5 / 60 / np.cos(np.radians(30.0))
    s.labels = FinderLabels([Target("Here", "T1", "galaxy", 150.0, 30.0, "", (4.0, 2.0), 0.0),
                             Target("East", "T2", "galaxy", east, 30.0, "")])
    s._main_fix = (time.monotonic(), msol, (45.0, 180.0))
    s._main_solve_at = time.monotonic()  # no new solve started by the test
    got = {lb[0]: lb[2:] for lb in s.main_labels() if lb[1] == "target"}
    assert got["T1"][:2] == pytest.approx([0, 0], abs=1e-4)
    assert got["T2"] == pytest.approx([0, -600 / 2180], abs=2e-3)  # roll 90: east is up the view (y down)
    assert np.hypot(got["T1"][2] * 3856, got["T1"][3] * 2180) == pytest.approx(240, rel=0.01)  # 2' semi-axis
    s._main_fix = (time.monotonic() - 100, msol, (45.0, 180.0))  # stale: back to Align's box (none yet)
    assert s.main_labels() == []
