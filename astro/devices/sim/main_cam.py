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

BODIES = {"moon": 1870, "jupiter": 45, "saturn": 18, "mars": 12, "venus": 25}  # diameter, arcsec
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
        when = self.clock()
        for name, diam in BODIES.items():
            b_alt, b_az = body_altaz(name, self.site, when)
            # Small-angle offset of the body from the boresight, in sensor pixels.
            dx = ((b_az - az + 180) % 360 - 180) * np.cos(np.radians(alt)) * 3600 / self.scale
            dy = -(b_alt - alt) * 3600 / self.scale
            cx, cy, r = w / 2 + dx - roi.x, h / 2 + dy - roi.y, diam / 2 / self.scale
            if -r < cx < roi.width + r and -r < cy < roi.height + r:
                img += _disk(roi.width, roi.height, cx, cy, r)
        if self.blur_px > 0:
            img = ndimage.gaussian_filter(img, self.blur_px)
        return np.clip(img, 0, 255).astype(np.uint8)


def _disk(w: int, h: int, cx: float, cy: float, r: float) -> np.ndarray:
    """Bright disk with cloud-like bands, so sharpness metrics have detail to measure."""
    y, x = np.ogrid[0:h, 0:w]
    inside = (x - cx) ** 2 + (y - cy) ** 2 <= r**2
    bands = 150 + 40 * np.sin((y - cy) / max(r, 1) * 12)
    return np.where(inside, bands, 0.0)
