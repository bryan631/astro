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

Better, when the Gaia stars for the field are on disk: plate-solve one main frame and the finder
frame taken with it (astro/pointing/main_solve.py). The main camera's stars, placed on the finder
through the finder's solve, give the same fit with no guess at which finder star is which: a
very bright finder star beat the dim one actually in the main view (2026-10-10).
"""

import logging
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
from scipy import ndimage

from astro.capture.focus import flatten_sky
from astro.capture.roi import brightest_blob
from astro.pointing.coords import Site, radec_to_altaz
from astro.pointing.labels import gnomonic, sky_to_pixels
from astro.pointing.main_offset import MainOffset, local_delta
from astro.pointing.main_solve import MainSolution, detect_stars, field_catalog, solve_main
from astro.pointing.platesolve import FinderSolver, Solution, finder_gray

ALIGN_S = 10.0  # Align watches the bright star drift this long in both cameras
ALIGN_SCALE = (25.0, 130.0)  # main pixels per finder pixel: ~58 for this pair (29.2" vs 0.50")
log = logging.getLogger(__name__)

MIN_DRIFT_PX = 50.0  # main-camera pixels the star must move for a rotation (~2 s of sidereal drift)
MIN_SAMPLES = 5
BRIGHTER_BY = 3.0  # the chosen finder star must outshine every other candidate by this much
# Binned finder pixels (~60 per degree): the nearest ring with one clear star decides. One wide
# circle took a brighter star 3.5 degrees out over the right one near the center (2026-10-10).
SEARCH_RADII_PX = (60, 120, 250)
DETECT_SIGMA = 8.0
KEEP_RUNS = 20  # saved Align runs (~20 MB each)
PAIR_S = 1.0  # a main and a finder frame this close in time make a pair to plate-solve
MAX_SOLVES = 3  # frame pairs tried per Align before the drift decides
CATALOG_WAIT_S = 60.0  # the field's Gaia download, when Align started it
SOLVE_WAIT_S = 60.0  # Align waits this long past ALIGN_S for a plate solve still running


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
    return fit_similarity(np.c_[fx, fy], main_xy)


def fit_similarity(finder_xy: np.ndarray, main_xy: np.ndarray) -> MainInFinder:
    """Least squares main = A (finder - c), A a scale and rotation, from matched positions:
    finder sensor px and main px from the main camera's center."""
    fx, fy = finder_xy[:, 0], finder_xy[:, 1]
    # main = [[p, -q], [q, p]] finder + d  ->  linear in (p, q, dx, dy)
    n = len(fx)
    rows = np.zeros((2 * n, 4))
    rows[0::2] = np.c_[fx, -fy, np.ones(n), np.zeros(n)]
    rows[1::2] = np.c_[fy, fx, np.zeros(n), np.ones(n)]
    sol, *_ = np.linalg.lstsq(rows, main_xy.ravel(), rcond=None)
    p, q, dx, dy = sol
    a = np.array([[p, -q], [q, p]])
    center = -np.linalg.solve(a, [dx, dy])
    residual = float(np.sqrt(np.mean((rows @ sol - main_xy.ravel()) ** 2)))
    return MainInFinder(a, center, residual)


def pick_star(gray: np.ndarray, near: tuple[float, float], radii: tuple = SEARCH_RADII_PX
              ) -> tuple[float, float] | None:
    """The bright star near `near` (finder gray-frame pixels): within the smallest of `radii`
    that has candidates, the brightest, if it outshines the others there by BRIGHTER_BY; None if
    there are none or the nearest ring with any is ambiguous."""
    img = flatten_sky(gray.astype(np.float32))  # cloud glow's gradient would hide the star
    bg = float(np.median(img))
    noise = max(1.4826 * float(np.median(np.abs(img - bg))), 0.5)
    labels, n = ndimage.label(ndimage.uniform_filter(img, 3) > bg + DETECT_SIGMA * noise)
    if n == 0:
        return None
    ids = np.arange(1, n + 1)
    flux = ndimage.sum_labels(img - bg, labels, ids)
    centers = np.array(ndimage.center_of_mass(img - bg, labels, ids))[:, ::-1]  # (x, y)
    dist = np.hypot(*(centers - near).T)
    radius = next((r for r in radii if (dist <= r).any()), None)
    if radius is None:
        return None
    close = dist <= radius
    order = np.argsort(-flux[close])
    if len(order) > 1 and flux[close][order[0]] < BRIGHTER_BY * flux[close][order[1]]:
        return None
    x, y = centers[close][order[0]]
    return float(x), float(y)


def fit_from_solves(sol: Solution, finder_size: tuple[int, int], main: MainSolution) -> MainInFinder:
    """The fit from a finder solve and a main solve of frames taken together: the main camera's
    matched stars, placed on the finder's sensor through the finder's solve."""
    w, h = finder_size
    east, north = gnomonic(sol.ra_deg, sol.dec_deg, main.star_radec[:, 0], main.star_radec[:, 1])
    dx, dy = sky_to_pixels(east, north, sol, w)
    return fit_similarity(np.c_[dx + w / 2, dy + h / 2], main.main_xy)


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
    return radec_offset_to_altaz(sol, sol.ra_deg + east / max(np.cos(d0), 1e-6), sol.dec_deg + north,
                                 site, when)


def radec_offset_to_altaz(sol: Solution, ra: float, dec: float, site: Site, when: datetime) -> MainOffset:
    """Where (ra, dec) sits from the finder's solved center, as the scope's own offset."""
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


class AlignRun:
    """One Align in progress: the bright star's track in both cameras for ALIGN_S."""

    def __init__(self, near: np.ndarray, finder_size: tuple[int, int], sol: Solution | None):
        """`near`: where the main camera's center sat on the finder last time (sensor px);
        `sol`: a fresh finder solve, or None (clouds: the box moves, the sky offset doesn't)."""
        self.until = time.monotonic() + ALIGN_S
        self.near, self.finder_size, self.sol = near, finder_size, sol
        self.finder: list[tuple[float, float, float]] = []  # (t, x, y) finder sensor px
        self.main: list[tuple[float, float, float]] = []  # (t, x, y) main px from its center
        self.main_size: tuple[int, int] | None = None
        self.seen: dict[str, int | None] = {"finder": None, "main": None}  # newest frame used
        self.no_star = 0  # finder frames without one clear star near `near`
        self.frames: dict[str, list] = {"finder": [], "main": []}  # (t, frame), for save()
        self.solving: threading.Thread | None = None
        self.attempts, self._solved_upto = 0, 0.0  # plate solves tried; the newest main frame tried
        self.why = ("I couldn't get a picture from both cameras at once, so I kept the alignment. "
                    "Tap Align again.")  # said when no plate solve worked (the last attempt's reason)
        self.solved: tuple[MainInFinder, Solution, MainSolution] | None = None  # plate-solve path

    def add(self, name: str, got: tuple[np.ndarray, int, float] | None) -> None:
        """A camera's newest (frame, frame number, mid-exposure time), once per frame."""
        if got is None or got[1] == self.seen[name]:
            return
        frame, self.seen[name], t_mid = got
        kept = self.frames[name]
        if name == "main" and len(kept) > 1:  # main frames are 8 MB: the first and the latest
            kept.pop()
        kept.append((t_mid, frame.copy()))
        if name == "finder":
            star = pick_star(finder_gray(frame), tuple(self.near / 2))  # binned 2x2
            if star is None:
                self.no_star += 1
            else:
                self.finder.append((t_mid, star[0] * 2, star[1] * 2))
            return
        h, w = frame.shape
        self.main_size = (w, h)
        if (blob := brightest_blob(frame)) is not None:
            self.main.append((t_mid, blob[0] - w / 2, blob[1] - h / 2))

    def start_solve(self, solver: FinderSolver, cache_dir: Path) -> None:
        """Once the newest main and finder frames were taken together, plate-solve them in a
        thread (~1 s); `solved` is set if both solve. The drift goes on meanwhile as the fallback."""
        busy = self.solving is not None and self.solving.is_alive()
        if busy or self.solved is not None or self.attempts >= MAX_SOLVES or not self.frames["main"] \
                or not self.frames["finder"]:
            return
        (t_main, main), (t_finder, finder) = self.frames["main"][-1], self.frames["finder"][-1]
        if abs(t_main - t_finder) > PAIR_S or t_main <= self._solved_upto:
            return  # not a pair, or one already tried: a failed solve tries the next pair
        self.attempts, self._solved_upto = self.attempts + 1, t_main
        self.solving = threading.Thread(target=self._solve, args=(solver, cache_dir, finder, main),
                                        daemon=True)
        self.solving.start()

    def _solve(self, solver: FinderSolver, cache_dir: Path, finder: np.ndarray, main: np.ndarray) -> None:
        sol = solver.solve(finder)
        # waits for the field's download (~30 s) if Align just started it: solving at once found
        # no stars on disk and the drift fallback took a wrong star, four times (2026-10-10)
        stars = None if sol is None else field_catalog(sol.ra_deg, sol.dec_deg, cache_dir, CATALOG_WAIT_S)
        msol = None if stars is None else solve_main(detect_stars(main)[0], stars, sol.ra_deg, sol.dec_deg)
        fit = None if msol is None else fit_from_solves(sol, self.finder_size, msol)
        log.info("align solved", extra={"data": {
            "finder": sol and [sol.ra_deg, sol.dec_deg, sol.roll_deg], "catalog": stars is not None and len(stars),
            "main": msol and [msol.ra_deg, msol.dec_deg, msol.roll_deg, msol.scale_arcsec_px, msol.matches,
                              round(msol.rms_arcsec, 2), round(msol.ms)],
            "center": fit and fit.center.round(1).tolist(), "scale": fit and fit.scale,
            "rotation": fit and fit.rotation_deg, "residual": fit and fit.residual_px}})
        if fit is not None and ALIGN_SCALE[0] <= fit.scale <= ALIGN_SCALE[1]:
            self.solved = (fit, sol, msol)
        elif sol is None:
            self.why = ("The finder couldn't plate-solve (clouds, dew, or the finder out of focus?), "
                        "so I kept the alignment. Tap Align again when the finder shows stars.")
        elif stars is None:
            self.why = ("I don't have the star catalog for this part of the sky, and couldn't download "
                        "it (no internet?), so I kept the alignment.")
        else:
            self.why = ("I couldn't plate-solve the telescope view (too few stars, or out of focus?), "
                        "so I kept the alignment. Try a part of the sky with more stars and tap Align.")

    def save(self, path: Path) -> None:
        """The run's frames and tracks, to look at a failed Align later (np.load)."""
        path.parent.mkdir(parents=True, exist_ok=True)
        f, m = self.frames["finder"], self.frames["main"]
        np.savez(path, finder=np.array([fr for _, fr in f]), finder_t=[t for t, _ in f],
                 main=np.array([fr for _, fr in m]), main_t=[t for t, _ in m], near=self.near,
                 finder_track=np.array(self.finder).reshape(-1, 3), main_track=np.array(self.main).reshape(-1, 3))
        for old in sorted(path.parent.glob("*.npz"))[:-KEEP_RUNS]:
            old.unlink(missing_ok=True)

    def result(self) -> MainInFinder | str:
        """The fit, or what went wrong, said to the user (the old box stays)."""
        if len(self.main) < MIN_SAMPLES:
            return ("I don't see a bright star in the telescope view. Put one in the middle of "
                    "it, then tap Align.")
        if len(self.finder) < 2:
            return ("I couldn't tell which finder star is the one in the telescope view. Pick a "
                    "brighter star, one you can see by eye, and tap Align.")
        f, m = np.array(self.finder), np.array(self.main)
        fit = fit_main_in_finder(f[:, 0], f[:, 1:], m[:, 0], m[:, 1:])
        log.info("align tracks", extra={"data": {"finder": f.round(2).tolist(), "main": m.round(2).tolist(),
                                                  "no_star": self.no_star, "scale": fit and fit.scale,
                                                  "rotation": fit and fit.rotation_deg}})
        if fit is None:
            return ("The star didn't drift enough to measure. Tap Align and wait the full "
                    f"{ALIGN_S:.0f} seconds without touching the scope.")
        if not ALIGN_SCALE[0] <= fit.scale <= ALIGN_SCALE[1]:
            return (f"That didn't fit (the star moved {fit.scale:.1f} times as far in the "
                    "telescope as in the finder; it should be about 58). The scope may have moved, "
                    "or I followed the wrong star. Tap Align again; the box is unchanged.")
        return fit
