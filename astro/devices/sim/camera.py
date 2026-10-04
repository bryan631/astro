"""Simulated camera: noisy black frame, deterministic per seed."""

import numpy as np

from astro.devices.base import Roi


class SimCamera:
    bayer = "GRBG"

    def __init__(self, width: int = 1280, height: int = 960, seed: int = 0):
        self._size = (height, width)
        self._rng = np.random.default_rng(seed)
        self._roi: Roi | None = None
        self.exposure = 0.01
        self.gain = 0
        self.connected = False

    def connect(self) -> None:
        self.connected = True

    def close(self) -> None:
        self.connected = False

    def set_exposure(self, seconds: float) -> None:
        self.exposure = seconds

    def set_gain(self, gain: int) -> None:
        self.gain = gain

    def set_roi(self, roi: Roi | None) -> None:
        self._roi = roi

    def capture(self) -> np.ndarray:
        if not self.connected:
            raise RuntimeError("camera not connected")
        h, w = (self._roi.height, self._roi.width) if self._roi else self._size
        return self._rng.integers(0, 8, size=(h, w), dtype=np.uint8)
