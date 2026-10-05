"""Simulated main camera: shows the planet or Moon the simulated scope points at.

Renders at half the SV705C resolution (same field of view) to keep CPU low. Bodies drift
with the real sky clock, so ROI tracking and pointing error behave like the real thing.
"""

from collections.abc import Callable
from datetime import datetime

import numpy as np
from scipy import ndimage

from astro.devices.base import Roi
from astro.optics import MAIN_SENSOR_PX
from astro.pointing.coords import Site, altaz_to_radec, body_altaz, radec_to_altaz

# Apparent diameter, arcsec (typical). Every body the planner can suggest must be here.
BODIES = {"moon": 1870, "mercury": 7, "venus": 25, "mars": 12, "jupiter": 45, "saturn": 18,
          "uranus": 3.7, "neptune": 2.3}
MIN_RADIUS_PX = 3  # draw tiny disks (Uranus, Neptune) big enough to find and focus on
POSITION_REFRESH_S = 1.0  # bodies move slowly; look up positions once a second, not per frame
SENSOR = (MAIN_SENSOR_PX[0] // 2, MAIN_SENSOR_PX[1] // 2)  # SV705C binned 2x2
NOISE = 4.0
# Synthetic deep-sky stars (the real camera sees mag ~12+, far deeper than any catalog here):
# a random field fixed to the sky (stable 1-degree tiles, the 3x3 around the view combined),
# so frames drift and rotate exactly as the sky does.
STAR_TILE_DEG = 1.0  # stars are generated per sky tile, seeded by the tile, so they never jump
STARS_PER_SQ_ARCMIN = 0.5
STAR_FLUX = (100.0, 1500.0)  # summed ADU before blur; far fainter than any planet
STAR_SEEING_PX = 1.5


class SimMainCamera:
    bayer = "GRBG"

    def __init__(self, true_altaz: Callable[[], tuple[float, float]], site: Site,
                 clock: Callable[[], datetime], arcsec_per_px: float = 1.0, blur_px: float = 1.0,
                 seed: int = 0):
        """`arcsec_per_px` is 1.0 at prime focus (0.5"/px binned 2x2); 0.5 with a 2x Barlow."""
        self.true_altaz, self.site, self.clock = true_altaz, site, clock
        self.scale, self.blur_px = arcsec_per_px, blur_px
        self.sensor_size = SENSOR
        self.exposure_s, self.gain = 0.01, 0
        self._roi: Roi | None = None
        self._rng = np.random.default_rng(seed)
        self._tiles: dict = {}  # star field per sky tile, generated once
        self._diam = np.array(list(BODIES.values()))
        self._positions: tuple[datetime, np.ndarray, np.ndarray] | None = None  # (when, alt, az)

    def set_site(self, site: Site) -> None:
        self.site, self._positions = site, None  # body positions depend on the site

    def connect(self) -> None: ...
    def close(self) -> None: ...
    def set_exposure(self, seconds: float) -> None:
        self.exposure_s = seconds

    def set_gain(self, gain: int) -> None:
        self.gain = gain

    def set_roi(self, roi: Roi | None) -> None:
        self._roi = roi

    def temperature_c(self) -> float | None:
        return None

    def capture(self) -> np.ndarray:
        w, h = SENSOR
        roi = self._roi or Roi(0, 0, w, h)
        img = self._rng.normal(10, NOISE, (roi.height, roi.width))
        alt, az = self.true_altaz()
        b_alt, b_az = self._body_positions()
        # Small-angle offsets of every body from the boresight, in sensor pixels (vectorized).
        dx = ((b_az - az + 180) % 360 - 180) * np.cos(np.radians(alt)) * 3600 / self.scale
        dy = -(b_alt - alt) * 3600 / self.scale
        cx, cy = w / 2 + dx - roi.x, h / 2 + dy - roi.y
        r = np.maximum(self._diam / 2 / self.scale, MIN_RADIUS_PX)
        visible = (-r < cx) & (cx < roi.width + r) & (-r < cy) & (cy < roi.height + r)
        for i in np.flatnonzero(visible):  # usually zero or one body
            img += _disk(roi.width, roi.height, cx[i], cy[i], r[i])
        img += self._stars(alt, az, roi)
        if self.blur_px > 0:
            img = ndimage.gaussian_filter(img, self.blur_px)
        return np.clip(img, 0, 255).astype(np.uint8)

    def _stars(self, alt: float, az: float, roi: Roi) -> np.ndarray:
        """Point stars from the sky-fixed random field, placed in the current alt/az frame."""
        when = self.clock()
        ra0, dec0 = altaz_to_radec(alt, az, self.site, when)
        # Which way is celestial north on the sensor (gives field rotation for free): the
        # alt/az of a point just north of the boresight.
        n_alt, n_az = radec_to_altaz(ra0, min(dec0 + 0.05, 89.99), self.site, when)
        north = np.arctan2(-(n_alt - alt), ((n_az - az + 180) % 360 - 180) * np.cos(np.radians(alt)))
        ra, dec, flux = self._field(ra0, dec0)
        # Offsets east/north in arcsec, then rotate into sensor x (right) / y (down).
        east = ((ra - ra0 + 180) % 360 - 180) * np.cos(np.radians(dec0)) * 3600
        up = (dec - dec0) * 3600
        ang = north - np.pi / 2
        x = (east * np.cos(ang) - up * np.sin(ang)) / self.scale + SENSOR[0] / 2 - roi.x
        y = (east * np.sin(ang) + up * np.cos(ang)) / self.scale + SENSOR[1] / 2 - roi.y
        ok = (x >= 0) & (x < roi.width) & (y >= 0) & (y < roi.height)
        img = np.zeros((roi.height, roi.width))
        np.add.at(img, (y[ok].astype(int), x[ok].astype(int)), flux[ok])
        return ndimage.gaussian_filter(img, STAR_SEEING_PX)

    def _field(self, ra0: float, dec0: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Stars from the 3x3 tiles around the view; each tile is generated once from its key."""
        ti, tj = int(np.floor(ra0 / STAR_TILE_DEG)), int(np.floor(dec0 / STAR_TILE_DEG))
        tiles = [self._tile(ti + di, tj + dj) for di in (-1, 0, 1) for dj in (-1, 0, 1)]
        return tuple(np.concatenate(parts) for parts in zip(*tiles, strict=True))

    def _tile(self, i: int, j: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        cache = self._tiles
        if (i, j) not in cache:
            rng = np.random.default_rng([i % round(360 / STAR_TILE_DEG), j + 900])  # deterministic per tile
            n = rng.poisson(STARS_PER_SQ_ARCMIN * (STAR_TILE_DEG * 60) ** 2
                            * np.cos(np.radians((j + 0.5) * STAR_TILE_DEG)))
            dec = (j + rng.uniform(0, 1, n)) * STAR_TILE_DEG
            ra = (i + rng.uniform(0, 1, n)) * STAR_TILE_DEG
            flux = np.exp(rng.uniform(*np.log(STAR_FLUX), n))  # many faint, few bright
            cache[(i, j)] = (ra, dec, flux)
        return cache[(i, j)]


    def _body_positions(self) -> tuple[np.ndarray, np.ndarray]:
        when = self.clock()
        cached = self._positions
        if cached is None or abs((when - cached[0]).total_seconds()) >= POSITION_REFRESH_S:
            alt_az = np.array([body_altaz(name, self.site, when) for name in BODIES])
            self._positions = cached = (when, alt_az[:, 0], alt_az[:, 1])
        return cached[1], cached[2]


def _disk(w: int, h: int, cx: float, cy: float, r: float) -> np.ndarray:
    """Bright disk with cloud-like bands, so sharpness metrics have detail to measure."""
    y, x = np.ogrid[0:h, 0:w]
    inside = (x - cx) ** 2 + (y - cy) ** 2 <= r**2
    bands = 150 + 40 * np.sin((y - cy) / max(r, 1) * 12)
    return np.where(inside, bands, 0.0)
