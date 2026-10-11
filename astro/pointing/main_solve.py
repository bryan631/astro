"""Plate-solve the main camera: its few stars, matched to Gaia stars near the finder's solve.

The main camera sees only ~32' x 18', too little for a blind solver, but Align already has the
finder's solve and the main camera points within a few degrees of it, at a known plate scale.
So: project the Gaia stars around the finder's center onto a tangent plane; for a pair of bright
detected stars, every catalog pair at the same separation gives a guess of the main camera's
position and rotation; the guess that lands most other detected stars on catalog stars wins,
then a least-squares fit on the tangent plane at the main camera's own center refines it.

Sky positions on the tangent plane are complex numbers (east + i north, arcsec) and main pixels
too (x + i y from the frame's center), so a guess is one complex multiply-add: sky = m z + t.

The Gaia stars come from VizieR, cached on disk per coarse sky tile (the scope often has
internet, not always): one cone covers every finder solve whose center falls in the tile.
"""

import logging
import math
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy import ndimage
from scipy.spatial import cKDTree

from astro.capture.focus import flatten_sky
from astro.pointing.labels import gnomonic
from astro.pointing.platesolve import bin2x2, remove_hot_pixels

log = logging.getLogger(__name__)

MAIN_SCALE = 0.50  # arcsec per raw main pixel: prime focus, 2.9 um at ~1200 mm (solved 2026-10-10)
SCALE_TOL = 0.02  # focus and the solve wander well under this (0.500-0.50x on 2026-10-10)
SEARCH_DEG = 4.0  # the main camera points within this far of the finder's center
DETECT_SIGMA = 5.0
MAX_STARS = 30  # the brightest detected stars are plenty for a match
ANCHORS = 6  # pairs among this many brightest detected stars seed the guesses
# Night frames (0.25 s, gain 480) show stars to G ~12; the six brightest were all G < 12
# (2026-10-10). A deeper catalog only adds time and chance matches.
CATALOG_G = 12.5
ANCHOR_G = 12.0  # pairs among the brightest detected stars are matched to pairs this bright
CHECK_FIRST = 10  # a guess must land 4 of this many brightest stars before a full count
MATCH_ARCSEC = 3.0  # a detected star counts as matched this close to a catalog star
MIN_MATCHES = 6
TILE_DEG = 2.0  # catalog cache tiles; a cone of TILE_RADIUS_DEG covers SEARCH_DEG from any center in one
TILE_RADIUS_DEG = 6.0
VIZIER = ("https://vizier.cds.unistra.fr/viz-bin/asu-tsv?-source=I/355/gaiadr3&-c={ra:.2f}%20{dec:+.2f}"
          "&-c.rd={r:.1f}&-out=RA_ICRS,DE_ICRS,Gmag&Gmag=%3C{g}&-out.max=200000")


@dataclass(frozen=True)
class MainSolution:
    ra_deg: float  # the main frame's center
    dec_deg: float
    roll_deg: float  # angle from the sensor's +x axis (y flipped if mirrored) to east, toward north
    scale_arcsec_px: float  # per raw main pixel
    matches: int
    rms_arcsec: float
    ms: float
    mirrored: bool  # the image is the sky's mirror image
    main_xy: np.ndarray  # (n, 2) matched stars, raw main pixels from the frame's center
    star_radec: np.ndarray  # (n, 2) their catalog (ra, dec) degrees


def detect_stars(raw: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Stars in a raw main frame, brightest first: (n, 2) raw pixels from the frame's center and
    their fluxes. Binned 2x2 (stars are ~10 raw px wide) and smoothed: the night view's 0.25 s at
    gain 480 is noisy (sky MAD ~10 ADU)."""
    h, w = raw.shape
    img = flatten_sky(remove_hot_pixels(bin2x2(raw)))
    smooth = ndimage.gaussian_filter(img, 1.5)
    bg = float(np.median(smooth[::4, ::4]))
    noise = max(1.4826 * float(np.median(np.abs(smooth[::4, ::4] - bg))), 0.5)
    labels, n = ndimage.label(smooth > bg + DETECT_SIGMA * noise)
    if n == 0:
        return np.zeros((0, 2)), np.zeros(0)
    ids = np.arange(1, n + 1)
    flux = ndimage.sum_labels(img - bg, labels, ids)
    yx = np.array(ndimage.center_of_mass(smooth - bg, labels, ids))
    xy = yx[:, ::-1] * 2 + 0.5 - [w / 2, h / 2]  # binned -> raw pixels from the center
    order = np.argsort(-flux)[:MAX_STARS]
    return xy[order], flux[order]


def _tangent(ra, dec, ra0, dec0) -> np.ndarray:
    """(ra, dec) degrees -> east + i north arcsec on the tangent plane at (ra0, dec0)."""
    x, y = gnomonic(ra0, dec0, ra, dec)
    return (np.asarray(x) + 1j * np.asarray(y)) * 3600


def _untangent(w: complex, ra0: float, dec0: float) -> tuple[float, float]:
    """The inverse of _tangent for one point."""
    x, y = np.radians(w.real / 3600), np.radians(w.imag / 3600)
    d0 = np.radians(dec0)
    rho = np.hypot(x, y)
    c = np.arctan(rho)
    dec = np.arcsin(np.cos(c) * np.sin(d0) + (y * np.sin(c) * np.cos(d0) / rho if rho else 0))
    ra = ra0 + np.degrees(np.arctan2(x * np.sin(c), rho * np.cos(d0) * np.cos(c) - y * np.sin(d0) * np.sin(c)))
    return float(ra % 360), float(np.degrees(dec))


def _count(tree: cKDTree, m: np.ndarray, t: np.ndarray, z: np.ndarray) -> np.ndarray:
    """For each guess sky = m z + t: how many of the stars z land on a catalog star."""
    w = m[:, None] * z[None, :] + t[:, None]
    d, _ = tree.query(np.c_[w.real.ravel(), w.imag.ravel()], distance_upper_bound=MATCH_ARCSEC)
    return (d < MATCH_ARCSEC).reshape(w.shape).sum(axis=1)


def _guess(z: np.ndarray, sky: np.ndarray, bright: np.ndarray, scale: float
           ) -> tuple[complex, complex, int] | None:
    """The best sky = m z + t from pairs of the brightest detected stars and pairs of `bright`
    catalog stars at the same separation (within the scale tolerance): (m, t, stars matched)."""
    tree = cKDTree(np.c_[sky.real, sky.imag])
    anchors = sky[bright]
    longest = scale * (1 + SCALE_TOL) * np.abs(z[:ANCHORS, None] - z[None, :ANCHORS]).max()
    pairs = cKDTree(np.c_[anchors.real, anchors.imag]).query_pairs(longest, output_type="ndarray")
    pairs = np.r_[pairs, pairs[:, ::-1]]  # either star of a catalog pair may be star i
    sep = np.abs(anchors[pairs[:, 0]] - anchors[pairs[:, 1]])
    order = np.argsort(sep)
    pairs, sep = pairs[order], sep[order]
    best = None
    k = min(ANCHORS, len(z))
    for i, j in sorted(((a, b) for a in range(k) for b in range(a + 1, k)), key=sum):
        d = scale * abs(z[j] - z[i])
        lo, hi = np.searchsorted(sep, [d * (1 - SCALE_TOL) - MATCH_ARCSEC, d * (1 + SCALE_TOL) + MATCH_ARCSEC])
        a, b = anchors[pairs[lo:hi, 0]], anchors[pairs[lo:hi, 1]]
        m = (b - a) / (z[j] - z[i])
        t = a - m * z[i]
        # The next few bright stars first: most guesses miss them all, and that is cheap to see.
        few = _count(tree, m, t, z[:CHECK_FIRST]) >= 4
        if not few.any():
            continue
        m, t = m[few], t[few]
        hits = _count(tree, m, t, z)
        top = int(np.argmax(hits))
        if best is None or hits[top] > best[2]:
            best = (complex(m[top]), complex(t[top]), int(hits[top]))
        if best[2] >= max(MIN_MATCHES, len(z) // 2):
            break  # half the stars on catalog stars: no coincidence
    return best


def _refine(z: np.ndarray, ra: np.ndarray, dec: np.ndarray, m: complex, t: complex, ra0: float,
            dec0: float) -> tuple[float, float, complex, np.ndarray, np.ndarray, float] | None:
    """Least squares on the tangent plane at the main camera's own center (the finder center's
    plane is ~0.5% anisotropic 4 degrees out): (ra, dec, m, matched z index, catalog index, rms)."""
    for _ in range(3):
        ra0, dec0 = _untangent(t, ra0, dec0)
        sky = _tangent(ra, dec, ra0, dec0)
        near = np.abs(sky) < 3600  # well past the frame's corners
        tree = cKDTree(np.c_[sky[near].real, sky[near].imag])
        w = m * z  # the center is now the tangent point
        dist, idx = tree.query(np.c_[w.real, w.imag], distance_upper_bound=MATCH_ARCSEC)
        ok = dist < MATCH_ARCSEC
        zi, ci = np.nonzero(ok)[0], np.nonzero(near)[0][idx[ok]]
        a = np.c_[z[zi], np.ones(len(zi))]
        if len(zi) < MIN_MATCHES:
            return None
        (m, t), *_ = np.linalg.lstsq(a, sky[ci], rcond=None)
    rms = float(np.sqrt(np.mean(np.abs(a @ [m, t] - sky[ci]) ** 2)))
    ra0, dec0 = _untangent(t, ra0, dec0)
    return ra0, dec0, complex(m), zi, ci, rms


def solve_main(xy: np.ndarray, catalog: np.ndarray, ra0: float, dec0: float,
               scale: float = MAIN_SCALE) -> MainSolution | None:
    """Match detected stars (raw px from the frame's center, brightest first) to `catalog`
    ((n, 3): ra, dec, G) stars within SEARCH_DEG of (ra0, dec0), the finder's solved center.
    Tries both mirror images; None without MIN_MATCHES stars on catalog stars."""
    t0 = time.perf_counter()
    if len(xy) < MIN_MATCHES:
        return None
    sky = _tangent(catalog[:, 0], catalog[:, 1], ra0, dec0)
    keep = np.abs(sky) < (SEARCH_DEG + 0.3) * 3600
    ra, dec, sky, bright = catalog[keep, 0], catalog[keep, 1], sky[keep], catalog[keep, 2] < ANCHOR_G
    # Not mirrored: 12 of 12 night frames (2026-10-10). The mirror image is a fallback in case
    # a diagonal or a flip ever joins the light path.
    for parity in (1, -1):
        z = xy[:, 0] + 1j * parity * xy[:, 1]
        got = _guess(z, sky, bright, scale)
        if got and got[2] >= MIN_MATCHES:
            break
    else:
        return None
    m, t, _ = got
    fit = _refine(z, ra, dec, m, t, ra0, dec0)
    if fit is None:
        return None
    ra_c, dec_c, m, zi, ci, rms = fit
    if abs(abs(m) / scale - 1) > SCALE_TOL:
        return None
    return MainSolution(ra_c, dec_c, float(np.degrees(np.angle(m))), abs(m), len(zi), rms,
                        (time.perf_counter() - t0) * 1000, parity < 0, xy[zi], np.c_[ra[ci], dec[ci]])


def tile(ra: float, dec: float) -> tuple[float, float]:
    """The center of the cache tile holding (ra, dec): TILE_DEG rings of declination, each cut
    into RA steps about TILE_DEG wide on the sky."""
    dec_c = max(-90.0, min(90.0, round(dec / TILE_DEG) * TILE_DEG))
    n = max(1, int(360 * math.cos(math.radians(dec_c)) / TILE_DEG))
    step = 360 / n
    return round(ra / step) % n * step, dec_c


def parse_tsv(text: str) -> np.ndarray:
    """VizieR's tab-separated (ra, dec, G) rows -> (n, 3); comments, headers and blanks skipped."""
    rows = []
    for line in text.splitlines():
        parts = line.split("\t")
        if len(parts) == 3 and not line.startswith("#"):
            try:
                rows.append([float(p) for p in parts])
            except ValueError:  # the column names, units and dashes under them
                continue
    return np.array(rows).reshape(-1, 3)


def _tile_path(ra: float, dec: float, cache_dir: Path) -> Path:
    ra_c, dec_c = tile(ra, dec)
    return cache_dir / f"gaia_g{CATALOG_G}_{ra_c:05.1f}_{dec_c:+05.1f}.npy"


def cached_catalog(ra: float, dec: float, cache_dir: Path) -> np.ndarray | None:
    """The tile's Gaia stars (n, 3: ra, dec, G) if already on disk."""
    path = _tile_path(ra, dec, cache_dir)
    return np.load(path) if path.exists() else None


_fetching: set[Path] = set()  # tiles being downloaded (a second Align tap waits for the first)


def field_catalog(ra: float, dec: float, cache_dir: Path, timeout_s: float = 120) -> np.ndarray | None:
    """Gaia stars (n, 3: ra, dec, G) within SEARCH_DEG of (ra, dec), from the tile's cache or
    VizieR. None when offline and not cached. VizieR can take minutes for one cone (~13k rows
    near the Milky Way: 4 min on 2026-10-10), so call this off the main loop."""
    if (stars := cached_catalog(ra, dec, cache_dir)) is not None:
        return stars
    ra_c, dec_c = tile(ra, dec)
    path = _tile_path(ra, dec, cache_dir)
    if path in _fetching:
        return None
    _fetching.add(path)
    url = VIZIER.format(ra=ra_c, dec=dec_c, r=TILE_RADIUS_DEG, g=CATALOG_G)
    try:
        with urllib.request.urlopen(url, timeout=timeout_s) as r:
            stars = parse_tsv(r.read().decode("utf-8", "replace"))
    except OSError as e:  # offline, or VizieR slow or down: Align falls back to the drift
        log.warning("gaia fetch failed", extra={"data": {"tile": [ra_c, dec_c], "error": str(e)}})
        return None
    finally:
        _fetching.discard(path)
    if len(stars) == 0:
        return None
    cache_dir.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp.npy")
    np.save(tmp, stars)
    tmp.replace(path)  # a half-written file never passes for the cache
    log.info("gaia fetched", extra={"data": {"tile": [ra_c, dec_c], "stars": len(stars)}})
    return stars
