"""Horizon mask: minimum usable altitude by azimuth (trees, houses)."""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class HorizonMask:
    points: tuple[tuple[float, float], ...] = ((0.0, 20.0),)  # (az, min_alt), sorted by az

    def min_alt(self, az_deg: float | np.ndarray) -> float | np.ndarray:
        az = np.array([p[0] for p in self.points])
        alt = np.array([p[1] for p in self.points])
        return np.interp(np.asarray(az_deg) % 360, az, alt, period=360)

    def to_stellarium(self) -> str:
        """Stellarium polygonal horizon file: one 'az alt' line per point."""
        return "\n".join(f"{a:.1f} {h:.1f}" for a, h in self.points) + "\n"
