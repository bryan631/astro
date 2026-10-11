"""Plate-solve the main camera: its few stars, matched to Gaia stars near the finder's solve.

The main camera sees only ~32' x 18', too little for a blind solver, but Align already has the
finder's solve and the main camera points within a few degrees of it, at a known plate scale.
So: project the Gaia stars around the finder's center onto a tangent plane; for a pair of bright
detected stars, every catalog pair at the same separation gives a guess of the main camera's
position and rotation; the guess that lands most other detected stars on catalog stars wins,
then a least-squares fit on the tangent plane at the main camera's own center refines it.

Sky positions on the tangent plane are complex numbers (east + i north, arcsec) and main pixels
too (x + i y from the frame's center), so a guess is one complex multiply-add: sky = m z + t.

The Gaia stars come from ESA's Gaia archive (VizieR as a fallback), cached on disk per coarse
sky tile (the scope often has internet, not always): one cone covers every finder solve whose
center falls in the tile.
"""

import logging
import math
import re
import time
import urllib.parse
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
# One cone (G < 12.5, 6 degrees, ~33k stars in Cygnus): ESA's archive took 26 s, VizieR over
# 15 min, since it scans every faint star in the cone (2026-10-10).
ESA_TAP = "https://gea.esac.esa.int/tap-server/tap/sync?REQUEST=doQuery&LANG=ADQL&FORMAT=csv&QUERY="
ESA_ADQL = ("SELECT ra, dec, phot_g_mean_mag FROM gaiadr3.gaia_source WHERE phot_g_mean_mag < {g} AND "
            "1 = CONTAINS(POINT(ra, dec), CIRCLE({ra:.2f}, {dec:.2f}, {r:.1f}))")
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


def _untangent(w, ra0: float, dec0: float):
    """The inverse of _tangent: (ra, dec) degrees."""
    x, y = np.radians(np.real(w) / 3600), np.radians(np.imag(w) / 3600)
    d0 = np.radians(dec0)
    dec = np.arcsin((np.sin(d0) + y * np.cos(d0)) / np.sqrt(1 + x * x + y * y))
    ra = ra0 + np.degrees(np.arctan2(x, np.cos(d0) - y * np.sin(d0)))
    return ra % 360, np.degrees(dec)


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
    """Least squares, moving the tangent plane to the main camera's own center (the finder
    center's plane is skewed ~0.5% and turned up to ~2 degrees 4 degrees out):
    (ra, dec, m, matched z index, catalog index, rms)."""
    for _ in range(3):
        sky = _tangent(ra, dec, ra0, dec0)
        near = np.nonzero(abs(sky - t) < 3600)[0]  # well past the frame's corners
        w = m * z + t
        dist, idx = cKDTree(np.c_[sky[near].real, sky[near].imag]).query(
            np.c_[w.real, w.imag], distance_upper_bound=MATCH_ARCSEC)
        zi = np.nonzero(dist < MATCH_ARCSEC)[0]
        if len(zi) < MIN_MATCHES:
            return None
        ci = near[idx[zi]]
        a = np.c_[z[zi], np.ones(len(zi))]
        (m, t), *_ = np.linalg.lstsq(a, sky[ci], rcond=None)
        rms = float(np.sqrt(np.mean(np.abs(a @ [m, t] - sky[ci]) ** 2)))
        # Carry the fit onto the plane touching the sky at the main camera's center.
        ra1, dec1 = _untangent(t, ra0, dec0)
        w1 = _tangent(*_untangent(m * z + t, ra0, dec0), ra1, dec1)
        (m, t), *_ = np.linalg.lstsq(np.c_[z, np.ones(len(z))], w1, rcond=None)
        ra0, dec0 = float(ra1), float(dec1)
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


def parse_rows(text: str) -> np.ndarray:
    """(ra, dec, G) rows, comma- or tab-separated -> (n, 3); comments, headers, blanks skipped."""
    rows = []
    for line in text.splitlines():
        parts = re.split("[,\t]", line)
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
    """Gaia stars (n, 3: ra, dec, G) on disk whose cone covers SEARCH_DEG around (ra, dec): the
    tile's own, or a neighbor's (a field just across a tile edge downloaded again, 2026-10-10)."""
    path = _tile_path(ra, dec, cache_dir)
    if path.exists():
        return np.load(path)
    for other in sorted(cache_dir.glob(f"gaia_g{CATALOG_G}_*.npy")):
        try:
            ra_c, dec_c = (float(v) for v in other.stem.split("_")[2:4])
        except ValueError:
            continue
        if _separation_deg(ra, dec, ra_c, dec_c) <= TILE_RADIUS_DEG - SEARCH_DEG:
            return np.load(other)
    return None


def _separation_deg(ra1: float, dec1: float, ra2: float, dec2: float) -> float:
    r1, d1, r2, d2 = (math.radians(v) for v in (ra1, dec1, ra2, dec2))
    c = math.sin(d1) * math.sin(d2) + math.cos(d1) * math.cos(d2) * math.cos(r1 - r2)
    return math.degrees(math.acos(max(-1.0, min(1.0, c))))


_fetching: set[Path] = set()  # tiles being downloaded (a second Align tap waits for the first)


def field_catalog(ra: float, dec: float, cache_dir: Path, timeout_s: float = 600) -> np.ndarray | None:
    """Gaia stars (n, 3: ra, dec, G) within SEARCH_DEG of (ra, dec), from the tile's cache or
    the internet. None when offline and not cached. A download takes ~30 s (minutes if only
    VizieR answers), so call this off the main loop."""
    if (stars := cached_catalog(ra, dec, cache_dir)) is not None:
        return stars
    ra_c, dec_c = tile(ra, dec)
    path = _tile_path(ra, dec, cache_dir)
    if path in _fetching:  # Align's own download, started as it began: wait for it
        deadline = time.monotonic() + timeout_s
        while path in _fetching and time.monotonic() < deadline:
            time.sleep(0.5)
        return cached_catalog(ra, dec, cache_dir)
    _fetching.add(path)
    cone = {"ra": ra_c, "dec": dec_c, "r": TILE_RADIUS_DEG, "g": CATALOG_G}
    try:  # waiters resume only once the tile is on disk (or the download failed)
        stars = _download(ESA_TAP + urllib.parse.quote(ESA_ADQL.format(**cone)), timeout_s)
        if stars is None:
            stars = _download(VIZIER.format(**cone), timeout_s)
        if stars is None or len(stars) == 0:
            return None
        cache_dir.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp.npy")
        np.save(tmp, stars)
        tmp.replace(path)  # a half-written file never passes for the cache
    finally:
        _fetching.discard(path)
    log.info("gaia fetched", extra={"data": {"tile": [ra_c, dec_c], "stars": len(stars)}})
    return stars


def _download(url: str, timeout_s: float) -> np.ndarray | None:
    try:
        with urllib.request.urlopen(url, timeout=timeout_s) as r:
            return parse_rows(r.read().decode("utf-8", "replace"))
    except OSError as e:  # offline, or the archive slow or down: Align says so
        log.warning("gaia fetch failed", extra={"data": {"url": url[:60], "error": str(e)}})
        return None
