"""Simulated main camera: shows the planet or Moon the simulated scope points at.

Renders at half the SV705C resolution (same field of view) to keep CPU low. Bodies drift
with the real sky clock, so ROI tracking and pointing error behave like the real thing.
"""

from collections.abc import Callable
from datetime import datetime

import numpy as np
from scipy import ndimage

from astro.devices.base import Roi
from astro.pointing.coords import Site, body_altaz

# Apparent diameter, arcsec (typical). Every body the planner can suggest must be here.
BODIES = {"moon": 1870, "mercury": 7, "venus": 25, "mars": 12, "jupiter": 45, "saturn": 18,
          "uranus": 3.7, "neptune": 2.3}
MIN_RADIUS_PX = 3  # draw tiny disks (Uranus, Neptune) big enough to find and focus on
POSITION_REFRESH_S = 1.0  # bodies move slowly; look up positions once a second, not per frame
SENSOR = (1928, 1090)  # SV705C 3856x2180 binned 2x2
NOISE = 4.0


class SimMainCamera:
    bayer = "GRBG"

    def __init__(self, true_altaz: Callable[[], tuple[float, float]], site: Site,
                 clock: Callable[[], datetime], arcsec_per_px: float = 1.0, blur_px: float = 1.0,
                 seed: int = 0):
        """`arcsec_per_px` is 1.0 at prime focus (0.5"/px binned 2x2); 0.5 with a 2x Barlow."""
        self.true_altaz, self.site, self.clock = true_altaz, site, clock
        self.scale, self.blur_px = arcsec_per_px, blur_px
        self.sensor_size = SENSOR
        self._roi: Roi | None = None
        self._rng = np.random.default_rng(seed)
        self._diam = np.array(list(BODIES.values()))
        self._positions: tuple[datetime, np.ndarray, np.ndarray] | None = None  # (when, alt, az)

    def connect(self) -> None: ...
    def close(self) -> None: ...
    def set_exposure(self, seconds: float) -> None: ...
    def set_gain(self, gain: int) -> None: ...

    def set_roi(self, roi: Roi | None) -> None:
        self._roi = roi

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
        if self.blur_px > 0:
            img = ndimage.gaussian_filter(img, self.blur_px)
        return np.clip(img, 0, 255).astype(np.uint8)


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
