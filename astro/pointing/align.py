"""Align: where the main camera's view sits on the finder's, from one bright star in both.

The scope is assumed nearly aligned. The user puts a bright star in the main view and taps
Align; for ~10 s both cameras stream (astro/devices/stream.py) and the star drifts. In the
finder it is the brightest star near where the main camera pointed last time; in the main
camera, the brightest object. Its finder positions (fitted over time) and main positions at
the same moments give, by least squares,

    main px from center = A (finder px - c)

with A a scale-and-rotation (neither camera mirrors the sky: a refractor finder and a two-mirror
Newtonian) and c the main camera's center on the finder's sensor. From that: the main camera's
box on the finder view, and with one finder plate solve, its offset on the sky for "go to".
"""

from dataclasses import dataclass
from datetime import datetime

import numpy as np
from scipy import ndimage

from astro.capture.focus import flatten_sky
from astro.pointing.coords import Site, radec_to_altaz
from astro.pointing.main_offset import MainOffset, local_delta
from astro.pointing.platesolve import Solution

MIN_DRIFT_PX = 50.0  # main-camera pixels the star must move for a rotation (~2 s of sidereal drift)
MIN_SAMPLES = 5
BRIGHTER_BY = 3.0  # the chosen finder star must outshine every other candidate by this much
SEARCH_RADIUS_PX = 250  # finder pixels (~2 degrees): "nearly aligned"
DETECT_SIGMA = 8.0


@dataclass(frozen=True)
class MainInFinder:
    a: np.ndarray  # 2x2: finder pixels (from c) -> main pixels (from the main camera's center)
    center: np.ndarray  # the main camera's center, in finder sensor pixels (x, y)
    residual_px: float  # RMS misfit in main-camera pixels

    @property
    def scale(self) -> float:
        """Main pixels per finder pixel (~58 for this pair)."""
        return float(np.sqrt(abs(np.linalg.det(self.a))))

    @property
    def rotation_deg(self) -> float:
        return float(np.degrees(np.arctan2(self.a[1, 0], self.a[0, 0])))

    def to_finder(self, main_px: np.ndarray) -> np.ndarray:
        """Main pixels (from its center) -> finder sensor pixels."""
        return self.center + np.linalg.solve(self.a, np.asarray(main_px, float).T).T


def fit_main_in_finder(finder_t, finder_xy, main_t, main_xy) -> MainInFinder | None:
    """Fit from the star's finder positions (sensor px) and main positions (px from center),
    each with its mid-exposure time. None until the star has drifted enough to show the
    rotation. Finder frames come slower, so the finder track is a straight-line fit in time."""
    finder_t, main_t = np.asarray(finder_t, float), np.asarray(main_t, float)
    finder_xy, main_xy = np.asarray(finder_xy, float), np.asarray(main_xy, float)
    if len(finder_t) < 2 or len(main_t) < MIN_SAMPLES:
        return None
    if np.ptp(main_xy, axis=0).max() < MIN_DRIFT_PX:
        return None
    t0 = main_t[0]
    fx = np.polyval(np.polyfit(finder_t - t0, finder_xy[:, 0], 1), main_t - t0)
    fy = np.polyval(np.polyfit(finder_t - t0, finder_xy[:, 1], 1), main_t - t0)
    # main = [[p, -q], [q, p]] finder + d  ->  linear in (p, q, dx, dy)
    n = len(main_t)
    rows = np.zeros((2 * n, 4))
    rows[0::2] = np.c_[fx, -fy, np.ones(n), np.zeros(n)]
    rows[1::2] = np.c_[fy, fx, np.zeros(n), np.ones(n)]
    sol, *_ = np.linalg.lstsq(rows, main_xy.ravel(), rcond=None)
    p, q, dx, dy = sol
    a = np.array([[p, -q], [q, p]])
    center = -np.linalg.solve(a, [dx, dy])
    residual = float(np.sqrt(np.mean((rows @ sol - main_xy.ravel()) ** 2)))
    return MainInFinder(a, center, residual)


def pick_star(gray: np.ndarray, near: tuple[float, float], radius: float = SEARCH_RADIUS_PX
              ) -> tuple[float, float] | None:
    """The bright star near `near` (finder gray-frame pixels): the brightest within `radius`,
    if it outshines every other candidate by BRIGHTER_BY; None if none or ambiguous."""
    img = flatten_sky(gray.astype(np.float32))  # cloud glow's gradient would hide the star
    bg = float(np.median(img))
    noise = max(1.4826 * float(np.median(np.abs(img - bg))), 0.5)
    labels, n = ndimage.label(ndimage.uniform_filter(img, 3) > bg + DETECT_SIGMA * noise)
    if n == 0:
        return None
    ids = np.arange(1, n + 1)
    flux = ndimage.sum_labels(img - bg, labels, ids)
    centers = np.array(ndimage.center_of_mass(img - bg, labels, ids))[:, ::-1]  # (x, y)
    close = np.hypot(*(centers - near).T) <= radius
    if not close.any():
        return None
    order = np.argsort(-flux[close])
    if len(order) > 1 and flux[close][order[0]] < BRIGHTER_BY * flux[close][order[1]]:
        return None
    x, y = centers[close][order[0]]
    return float(x), float(y)


def finder_offset_to_sky(dx: float, dy: float, sol: Solution, width_px: int) -> tuple[float, float]:
    """Finder sensor pixels from the image center -> (east, north) degrees on the sky, in the
    plate solver's convention (checked against catalog stars: tests/test_align.py)."""
    scale = sol.fov_deg / width_px  # degrees per pixel (the solver's FOV is the image width)
    u, v = -dx * scale, -dy * scale
    r = np.radians(sol.roll_deg)
    return float(np.cos(r) * u - np.sin(r) * v), float(np.sin(r) * u + np.cos(r) * v)


def sky_offset_to_altaz(sol: Solution, east: float, north: float, site: Site,
                        when: datetime) -> MainOffset:
    """An (east, north) offset from the finder's solved center, as the scope's own offset
    (d_az on the sky, d_alt): fixed to the tube, so it holds wherever the scope points."""
    d0 = np.radians(sol.dec_deg)
    dec = sol.dec_deg + north
    ra = sol.ra_deg + east / max(np.cos(d0), 1e-6)
    alt0, az0 = radec_to_altaz(sol.ra_deg, sol.dec_deg, site, when)
    alt1, az1 = radec_to_altaz(ra, dec, site, when)
    d_az, d_alt = local_delta(alt0, az0, alt1, az1)
    return MainOffset(float(d_az), float(d_alt), 1)


def box_on_finder_view(fit: MainInFinder, main_size: tuple[int, int], finder_size: tuple[int, int],
                       finder_rotate: int) -> list[list[float]]:
    """The main camera's corners on the finder view, as fractions from the view's center
    (what the page draws), for a finder view turned by `finder_rotate` (0 or 180)."""
    w, h = main_size
    corners = fit.to_finder(np.array([[-w / 2, -h / 2], [w / 2, -h / 2], [w / 2, h / 2], [-w / 2, h / 2]]))
    fw, fh = finder_size
    frac = corners / [fw, fh] - 0.5
    if finder_rotate == 180:
        frac = -frac
    return [[round(float(x), 4), round(float(y), 4)] for x, y in frac]
