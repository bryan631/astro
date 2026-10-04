"""Finder -> main camera offset, learned from the main camera's own view.

The finder solve puts a target "on target", but the main camera (32' field) points a little
differently. Its image orientation is unknown too: any rotation in the focuser, possibly
mirrored. Both are learned without a second plate solver:

1. CameraAxes: while the user pushes, encoder motion (d_az on the sky, d_alt) and the planet's
   motion in the image (dx, dy) are paired; least squares gives the 2x2 matrix mapping one to
   the other (rotation, scale and any flip in one go).
2. MainOffset: with the axes known, a target seen off-center converts to a sky offset (where the
   main camera points relative to the finder model), averaged over observations.
"""

from dataclasses import dataclass, field

import numpy as np

MIN_MOTION_DEG = 0.02  # ignore tiny moves (encoder noise, wobble)
MIN_SPREAD = 0.3  # moves must not all be in one direction (needs both axes to solve)


@dataclass
class CameraAxes:
    """Fits pixels = A @ [d_az_sky_deg, d_alt_deg] from paired moves."""

    _sky: list[np.ndarray] = field(default_factory=list)
    _px: list[np.ndarray] = field(default_factory=list)
    matrix: np.ndarray | None = None  # A, pixels per degree

    def add_move(self, d_az_sky_deg: float, d_alt_deg: float, dx_px: float, dy_px: float) -> bool:
        """Record one move; returns True once the axes are solved."""
        sky = np.array([d_az_sky_deg, d_alt_deg])
        if np.hypot(*sky) < MIN_MOTION_DEG:
            return self.matrix is not None
        self._sky.append(sky)
        self._px.append(np.array([dx_px, dy_px]))
        s, p = np.array(self._sky), np.array(self._px)
        # Both directions present? (smallest singular value relative to largest)
        sv = np.linalg.svd(s / np.linalg.norm(s, axis=1, keepdims=True), compute_uv=False)
        if len(s) >= 2 and sv[-1] / sv[0] > MIN_SPREAD:
            self.matrix = np.linalg.lstsq(s, p, rcond=None)[0].T
        return self.matrix is not None

    def sky_offset(self, dx_px: float, dy_px: float) -> tuple[float, float]:
        """Pixels from image center -> (d_az_sky_deg, d_alt_deg)."""
        if self.matrix is None:
            raise ValueError("camera axes not calibrated yet")
        d_az, d_alt = np.linalg.solve(self.matrix, [dx_px, dy_px])
        return float(d_az), float(d_alt)


@dataclass
class MainOffset:
    """Running mean of where the main camera points relative to the finder's model (deg)."""

    d_az_sky_deg: float = 0.0
    d_alt_deg: float = 0.0
    observations: int = 0

    def observe(self, axes: CameraAxes, target_dx_px: float, target_dy_px: float) -> None:
        """The target sits (dx, dy) px from the main image center while guidance says on target.

        The camera then points the opposite way from the target, by that sky offset.
        """
        t_az, t_alt = axes.sky_offset(target_dx_px, target_dy_px)
        n = self.observations
        self.d_az_sky_deg = (self.d_az_sky_deg * n - t_az) / (n + 1)
        self.d_alt_deg = (self.d_alt_deg * n - t_alt) / (n + 1)
        self.observations = n + 1

    def correct(self, target_alt: float, target_az: float) -> tuple[float, float]:
        """Where to point (through the finder model) so the target lands in the main camera."""
        cos_alt = max(np.cos(np.radians(target_alt)), 1e-6)
        return target_alt - self.d_alt_deg, target_az - self.d_az_sky_deg / cos_alt
