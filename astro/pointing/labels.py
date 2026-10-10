"""Names on the finder view: stars (HYG, star_names.csv), the planets and Moon, Go to targets.

A plate solve fixes how the finder's picture sits on the sky: the scale and which way is up, as
a 2x2 map from the local alt-az frame to finder pixels. That map is fixed to the tube, so
between solves the encoders say where the finder points and the labels follow the scope.
"""

import csv
from datetime import datetime
from pathlib import Path

import astropy.units as u
import numpy as np
from astropy.coordinates import SkyCoord

from astro.pointing.coords import Site, altaz_to_radec, radec_to_altaz
from astro.pointing.platesolve import Solution

STAR_NAMES = Path(__file__).with_name("star_names.csv")
STEP_DEG = 0.5  # the probe offsets that measure the map: small, but far above the solve's error


def load_stars(path: Path = STAR_NAMES) -> tuple[list[str], np.ndarray, np.ndarray]:
    """Names and J2000 (ra, dec) degrees, brightest first."""
    with open(path, encoding="utf-8") as f:
        rows = list(csv.DictReader(line for line in f if not line.startswith("#")))
    return ([r["name"] for r in rows], np.array([float(r["ra_deg"]) for r in rows]),
            np.array([float(r["dec_deg"]) for r in rows]))


def gnomonic(lon0, lat0, lon, lat):
    """Tangent-plane offsets (degrees) of (lon, lat) from (lon0, lat0), all in degrees: x along
    increasing lon, y along increasing lat. NaN behind the tangent point."""
    lon0, lat0, lon, lat = (np.radians(np.asarray(a, float)) for a in (lon0, lat0, lon, lat))
    cos_c = np.sin(lat0) * np.sin(lat) + np.cos(lat0) * np.cos(lat) * np.cos(lon - lon0)
    cos_c = np.where(cos_c > 0.05, cos_c, np.nan)  # more than ~87 degrees away: not on the view
    x = np.cos(lat) * np.sin(lon - lon0) / cos_c
    y = (np.cos(lat0) * np.sin(lat) - np.sin(lat0) * np.cos(lat) * np.cos(lon - lon0)) / cos_c
    return np.degrees(x), np.degrees(y)


def sky_to_pixels(east, north, sol: Solution, width_px: int):
    """(east, north) degrees from the solved center -> finder pixels from the image center: the
    inverse of align.finder_offset_to_sky (the plate solver's convention)."""
    scale = sol.fov_deg / width_px
    r = np.radians(sol.roll_deg)
    u_, v = np.cos(r) * east + np.sin(r) * north, -np.sin(r) * east + np.cos(r) * north
    return -u_ / scale, -v / scale


def tube_map(sol: Solution, width_px: int, site: Site, when: datetime) -> np.ndarray:
    """2x2: local tangent-plane degrees around the finder's alt-az (x: azimuth, y: altitude)
    -> finder pixels from the image center. Measured with two probe points from the solve."""
    alt0, az0 = radec_to_altaz(sol.ra_deg, sol.dec_deg, site, when)
    up = STEP_DEG if alt0 + STEP_DEG < 89.5 else -STEP_DEG  # near the zenith, probe downward
    probes = [(alt0 + up, az0), (alt0, az0 + STEP_DEG / max(np.cos(np.radians(alt0)), 0.05))]
    local, pixels = [], []
    for alt, az in probes:
        ra, dec = altaz_to_radec(alt, az, site, when)
        local.append(gnomonic(az0, alt0, az, alt))
        pixels.append(sky_to_pixels(*gnomonic(sol.ra_deg, sol.dec_deg, ra, dec), sol, width_px))
    return np.array(pixels).T @ np.linalg.inv(np.array(local).T)


def altaz_now(ra, dec, site: Site, when: datetime) -> tuple[np.ndarray, np.ndarray]:
    """Many (ra, dec) at once -> (alt, az) degrees (one astropy transform)."""
    aa = SkyCoord(ra=np.asarray(ra) * u.deg, dec=np.asarray(dec) * u.deg).transform_to(site.frame(when))
    return aa.alt.deg, aa.az.deg


def place(alt, az, center: tuple[float, float], m: np.ndarray, size: tuple[int, int],
          margin: float = 0.0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Objects' (alt, az) -> finder pixel (x, y) for a finder pointing at center (alt, az), and
    which of them land on the picture (within `margin` of a frame width outside it)."""
    alt0, az0 = center
    lx, ly = gnomonic(az0, alt0, az, alt)
    px, py = m @ np.vstack([lx, ly])
    w, h = size
    x, y = px + w / 2, py + h / 2
    pad = margin * w
    on = (x > -pad) & (x < w + pad) & (y > -pad) & (y < h + pad)  # NaN (behind) compares False
    return x, y, on
