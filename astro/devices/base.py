"""Device interfaces. Real drivers and simulators both implement these."""

from dataclasses import dataclass
from typing import Protocol

import numpy as np


@dataclass(frozen=True)
class Roi:
    x: int
    y: int
    width: int
    height: int


class Camera(Protocol):
    """A camera returning 8-bit RAW (Bayer) frames."""

    bayer: str  # "RGGB", "GRBG", ... for debayering and SER headers
    exposure_s: float  # current settings, so a mode change can be undone
    gain: int
    sensor_size: tuple[int, int]  # full frame (width, height), pixels

    def connect(self) -> None: ...
    def close(self) -> None: ...
    def set_exposure(self, seconds: float) -> None: ...
    def set_gain(self, gain: int) -> None: ...
    def set_roi(self, roi: Roi | None) -> None: ...
    def capture(self) -> np.ndarray:
        """Block until one frame is ready; returns a 2-D uint8 array."""
        ...

    def temperature_c(self) -> float | None:
        """Sensor temperature in Celsius, or None if the camera can't tell."""
        ...


class MountEncoders(Protocol):
    """Raw encoder counts for the two axes."""

    def counts(self) -> tuple[int, int]:
        """Return (azimuth_counts, altitude_counts)."""
        ...
