"""Align: where the main camera's view sits on the finder's, from plate solves of both.

A main frame and a finder frame taken together are plate-solved: the finder by tetra3, the main
camera against the Gaia stars near the finder's solve (astro/pointing/main_solve.py). The main
camera's stars, placed on the finder's sensor through the finder's solve, give by least squares

    main px from center = A (finder px - c)

with A a scale-and-rotation (neither camera mirrors the sky: a refractor finder and a two-mirror
Newtonian) and c the main camera's center on the finder's sensor. From that: the main camera's
box on the finder view, and its offset on the sky for "go to". No star needs to be in both: an
older drift method followed one bright star in both cameras and picked wrong ones (2026-10-10).
"""

import logging
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np

from astro.pointing.coords import Site, radec_to_altaz
from astro.pointing.labels import gnomonic, sky_to_pixels
from astro.pointing.main_offset import MainOffset, local_delta
from astro.pointing.main_solve import (
    MAIN_SCALE,
    MainSolution,
    detect_stars,
    field_catalog,
    solve_main,
)
from astro.pointing.platesolve import FinderSolver, Solution

ALIGN_S = 10.0  # Align tries frame pairs for this long
# Main pixels per finder pixel, at prime focus: ~58 for this pair (29.2" vs 0.50"). With a
# Barlow, the same bounds apply to the fit scaled back to prime focus.
ALIGN_SCALE = (25.0, 130.0)
log = logging.getLogger(__name__)

KEEP_RUNS = 20  # saved Align runs (~20 MB each)
PAIR_S = 1.0  # a main and a finder frame this close in time make a pair to plate-solve
MAX_SOLVES = 3  # frame pairs tried per Align
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
    """One Align in progress: frame pairs from both cameras, plate-solved for up to ALIGN_S."""

    def __init__(self, finder_size: tuple[int, int], scales: tuple[float, ...] = (MAIN_SCALE,)):
        """`scales`: the main camera's arcsec per pixel to try (several with a Barlow whose
        power isn't measured yet)."""
        self.until = time.monotonic() + ALIGN_S
        self.finder_size, self.scales = finder_size, scales
        self.main_size: tuple[int, int] | None = None
        self.seen: dict[str, int | None] = {"finder": None, "main": None}  # newest frame used
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
        if name == "main":
            self.main_size = frame.shape[::-1]

    def start_solve(self, solver: FinderSolver, cache_dir: Path) -> None:
        """Once the newest main and finder frames were taken together, plate-solve them in a
        thread (~1 s); `solved` is set if both solve."""
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
        # no stars on disk (2026-10-10)
        stars = None if sol is None else field_catalog(sol.ra_deg, sol.dec_deg, cache_dir, CATALOG_WAIT_S)
        msol = None
        if stars is not None:
            xy = detect_stars(main)[0]
            msol = next((m for scale in self.scales
                         if (m := solve_main(xy, stars, sol.ra_deg, sol.dec_deg, scale)) is not None), None)
        fit = None if msol is None else fit_from_solves(sol, self.finder_size, msol)
        log.info("align solved", extra={"data": {
            "finder": sol and [sol.ra_deg, sol.dec_deg, sol.roll_deg], "catalog": stars is not None and len(stars),
            "main": msol and [msol.ra_deg, msol.dec_deg, msol.roll_deg, msol.scale_arcsec_px, msol.matches,
                              round(msol.rms_arcsec, 2), round(msol.ms)],
            "center": fit and fit.center.round(1).tolist(), "scale": fit and fit.scale,
            "rotation": fit and fit.rotation_deg, "residual": fit and fit.residual_px}})
        if fit is not None and ALIGN_SCALE[0] <= fit.scale * msol.scale_arcsec_px / MAIN_SCALE <= ALIGN_SCALE[1]:
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
        """The run's frames, to look at a failed Align later (np.load)."""
        path.parent.mkdir(parents=True, exist_ok=True)
        f, m = self.frames["finder"], self.frames["main"]
        np.savez(path, finder=np.array([fr for _, fr in f]), finder_t=[t for t, _ in f],
                 main=np.array([fr for _, fr in m]), main_t=[t for t, _ in m])
        for old in sorted(path.parent.glob("*.npz"))[:-KEEP_RUNS]:
            old.unlink(missing_ok=True)
