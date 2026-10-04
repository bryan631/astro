"""Simulated finder camera: renders the real sky wherever the simulated scope points."""

from collections.abc import Callable
from datetime import datetime

import numpy as np

from astro.devices.base import Roi
from astro.devices.sim.sky import render
from astro.pointing.coords import Site, altaz_to_radec


class SimFinderCamera:
    bayer = "GRBG"

    def __init__(self, true_altaz: Callable[[], tuple[float, float]], site: Site,
                 clock: Callable[[], datetime], star_table: np.ndarray, fov_deg: float = 11.0,
                 blur_px: float = 2.0, mag_limit: float = 6.5):
        self.true_altaz, self.site, self.clock = true_altaz, site, clock
        self.star_table, self.fov = star_table, fov_deg
        self.blur_px, self.mag_limit = blur_px, mag_limit  # raise blur_px to simulate defocus
        self._n = 0

    def connect(self) -> None: ...
    def close(self) -> None: ...
    def set_exposure(self, seconds: float) -> None: ...
    def set_gain(self, gain: int) -> None: ...
    def set_roi(self, roi: Roi | None) -> None: ...

    def temperature_c(self) -> float | None:
        return None

    def capture(self) -> np.ndarray:
        ra, dec = altaz_to_radec(*self.true_altaz(), self.site, self.clock())
        self._n += 1
        return render(self.star_table, ra, dec, 0.0, self.fov, size=(1280, 960), seed=self._n,
                      mag_limit=self.mag_limit, sigma_px=self.blur_px)
