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

from astro.planner.catalog import Target
from astro.planner.tonight import PLANETS
from astro.pointing.coords import Site, altaz_to_radec, body_altaz, radec_to_altaz
from astro.pointing.platesolve import Solution

STAR_NAMES = Path(__file__).with_name("star_names.csv")
STEP_DEG = 0.5  # the probe offsets that measure the map: small, but far above the solve's error
REFRESH_S = 5.0  # the sky turns ~0.02 degrees in 5 s: under 2 finder pixels
BODIES = (*PLANETS, "moon")
BRIGHT_MAG = 3.0  # Names level 1: the prominent stars (~130 in the sky); level 2 names all ~350


def load_stars(path: Path = STAR_NAMES) -> tuple[list[str], np.ndarray, np.ndarray, np.ndarray]:
    """Names, J2000 (ra, dec) degrees and magnitudes, brightest first."""
    with open(path, encoding="utf-8") as f:
        rows = list(csv.DictReader(line for line in f if not line.startswith("#")))
    return ([r["name"] for r in rows], np.array([float(r["ra_deg"]) for r in rows]),
            np.array([float(r["dec_deg"]) for r in rows]), np.array([float(r["mag"]) for r in rows]))


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


class FinderLabels:
    """The finder view's names: stars, Go to targets (by catalog id), the planets and Moon.

    `map` comes from the latest solve and is fixed to the tube, so the owner saves it and a
    restart names stars before any new solve. The objects' alt-az is cached for REFRESH_S."""

    def __init__(self, targets: list[Target]):
        self.map: np.ndarray | None = None
        self._targets = targets
        self._fixed: dict | None = None  # names, kinds, ra, dec, mag, outlines: loaded on first use
        self._cache: dict | None = None

    def labels(self, sol: Solution | None, size: tuple[int, int], pointing: tuple[float, float],
               site: Site, when: datetime, now: float, rotate: int, level: int = 2
               ) -> tuple[list[list], bool]:
        """([[name, kind, x, y], ...] with x, y as fractions from the view's center, for a
        view turned by `rotate`; True if `sol` gave a new map to save). A target with a size
        adds its outline: [..., ex, ey, ratio], the ellipse's semi-major axis (fractions of the
        view's width and height) and its minor/major ratio. Level 1 names only the stars to
        BRIGHT_MAG, level 2 all of them; targets and planets always. Call only once a map
        exists or `sol` is set; `now` is a monotonic clock."""
        w, h = size
        c, new_map = self._cache, False
        if c is None or c["sol"] is not sol or now - c["at"] > REFRESH_S:
            if sol is not None and (c is None or c["sol"] is not sol):
                self.map, new_map = tube_map(sol, w, site, when), True
            f = self._fixed_objects()
            alt, az = altaz_now(np.r_[f["ra"], f["end_ra"]], np.r_[f["dec"], f["end_dec"]], site, when)
            n, body = len(f["ra"]), np.array([body_altaz(b, site, when) for b in BODIES])
            c = self._cache = {"sol": sol, "m": self.map, "at": now,
                               "names": f["names"] + [b.capitalize() for b in BODIES],
                               "kinds": f["kinds"] + ["planet"] * len(BODIES),
                               "alt": np.r_[alt[:n], body[:, 0]], "az": np.r_[az[:n], body[:, 1]],
                               "mag": np.r_[f["mag"], np.full(len(BODIES), -np.inf)],
                               "outlined": f["outlined"], "ratio": f["ratio"],
                               "end_alt": alt[n:], "end_az": az[n:]}
        x, y, on = place(c["alt"], c["az"], pointing, c["m"], size)
        ex, ey, _ = place(c["end_alt"], c["end_az"], pointing, c["m"], size)
        sign = -1 if rotate == 180 else 1  # the view is turned like the picture
        outline = {i: [round(sign * (ex[k] - x[i]) / w, 4), round(sign * (ey[k] - y[i]) / h, 4),
                       c["ratio"][k]] for k, i in enumerate(c["outlined"])}
        return [[c["names"][i], c["kinds"][i], round(sign * (x[i] / w - 0.5), 4),
                 round(sign * (y[i] / h - 0.5), 4), *outline.get(i, [])]
                for i in np.flatnonzero(on & (c["alt"] > 0) & ((c["mag"] <= BRIGHT_MAG) | (level >= 2)))
                ], new_map

    def _fixed_objects(self) -> dict:
        """Stars and targets; each target with a size also gets the sky point at the end of its
        major axis (PA east of north), which the map turns into the outline on the view."""
        if self._fixed is None:
            stars, ra, dec, mag = load_stars()
            t = self._targets
            sized = [k for k, x in enumerate(t) if x.size_arcmin[0] > 0]
            semi = np.array([t[k].size_arcmin[0] / 120 for k in sized])  # degrees
            pa = np.radians([t[k].pa_deg for k in sized])
            t_dec = np.array([t[k].dec for k in sized])
            self._fixed = {
                "names": stars + [x.id for x in t], "kinds": ["star"] * len(stars) + ["target"] * len(t),
                "ra": np.r_[ra, [x.ra for x in t]], "dec": np.r_[dec, [x.dec for x in t]],
                "mag": np.r_[mag, np.full(len(t), -np.inf)],
                "outlined": [len(stars) + k for k in sized],
                "end_ra": np.array([t[k].ra for k in sized]) + semi * np.sin(pa) / np.cos(np.radians(t_dec)),
                "end_dec": t_dec + semi * np.cos(pa),
                "ratio": [round(t[k].size_arcmin[1] / t[k].size_arcmin[0], 3) for k in sized]}
        return self._fixed
