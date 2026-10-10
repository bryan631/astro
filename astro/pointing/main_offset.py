"""Finder -> main camera offset, learned from the main camera's own view.

The finder solve puts a target "on target", but the main camera (32' field) points a little
differently. Its image orientation is unknown too: any rotation in the focuser, possibly
mirrored. Both are learned without a second plate solver:

1. CameraAxes: while the user pushes, encoder motion (d_az on the sky, d_alt) and the planet's
   motion in the image (dx, dy) are paired. A push moves a fixed target the *opposite* way in
   the image, so least squares on (push, -motion) gives the 2x2 matrix A that maps a target's
   sky offset from the camera's aim to its pixel offset from center (rotation, scale, flip).
2. MainOffset: with the axes known, a target seen off-center converts to a sky offset (where the
   main camera points relative to the finder model), averaged over observations.
"""

from dataclasses import dataclass, field

import numpy as np

# IntelliScope encoders: 9216 counts/rev = 0.039 deg per count (~280 main-camera pixels at
# 0.5"/px). A calibration move must span at least 2 counts to mean anything.
MIN_MOTION_DEG = 0.08
MIN_SPREAD = 0.3  # moves must not all be in one direction (needs both axes to solve)


def local_delta(alt0: float, az0: float, alt1: float, az1: float) -> np.ndarray:
    """Small-angle sky offset from (alt0, az0) to (alt1, az1): (wrapped d_az * cos(alt0), d_alt)."""
    d_az = (az1 - az0 + 180) % 360 - 180
    return np.array([d_az * np.cos(np.radians(alt0)), alt1 - alt0])


@dataclass
class CameraAxes:
    """A maps a target's sky offset from the camera aim (d_az_sky, d_alt) to pixels from center."""

    _sky: list[np.ndarray] = field(default_factory=list)
    _px: list[np.ndarray] = field(default_factory=list)
    matrix: np.ndarray | None = None  # A, pixels per degree

    @property
    def moves(self) -> int:
        """Pushes recorded so far."""
        return len(self._sky)

    def add_move(self, d_az_sky_deg: float, d_alt_deg: float, dx_px: float, dy_px: float) -> bool:
        """Record one push (scope motion) and how far the target moved in the image.

        Returns True once the axes are solved."""
        sky = np.array([d_az_sky_deg, d_alt_deg])
        if np.hypot(*sky) < MIN_MOTION_DEG:
            return self.matrix is not None
        self._sky.append(sky)
        self._px.append(-np.array([dx_px, dy_px]))  # target moves opposite to the push
        s, p = np.array(self._sky), np.array(self._px)
        # Both directions present? (smallest singular value relative to largest)
        sv = np.linalg.svd(s / np.linalg.norm(s, axis=1, keepdims=True), compute_uv=False)
        if len(s) >= 2 and sv[-1] / sv[0] > MIN_SPREAD:
            self.matrix = np.linalg.lstsq(s, p, rcond=None)[0].T
        return self.matrix is not None

    def sky_offset(self, dx_px: float, dy_px: float) -> tuple[float, float]:
        """A target's pixels from image center -> its sky offset from the camera aim (deg)."""
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

    def observe(self, axes: CameraAxes, target_dx_px: float, target_dy_px: float,
                target_from_model: tuple[float, float] = (0.0, 0.0)) -> None:
        """The target sits (dx, dy) px from the main image center.

        `target_from_model` is the target's local sky offset (d_az_sky, d_alt) from where the
        finder model says the scope points: the finder's leftover error, plus any correction
        already applied. Camera offset = target_from_model - (target's offset from the camera).
        """
        t_az, t_alt = axes.sky_offset(target_dx_px, target_dy_px)
        c_az, c_alt = target_from_model[0] - t_az, target_from_model[1] - t_alt
        n = self.observations
        self.d_az_sky_deg = (self.d_az_sky_deg * n + c_az) / (n + 1)
        self.d_alt_deg = (self.d_alt_deg * n + c_alt) / (n + 1)
        self.observations = n + 1

    def main_center(self, finder_alt: float, finder_az: float) -> tuple[float, float]:
        """Where the main camera points when the finder model points at (alt, az): the inverse
        of correct."""
        cos_alt = max(np.cos(np.radians(finder_alt)), 1e-6)
        return finder_alt + self.d_alt_deg, finder_az + self.d_az_sky_deg / cos_alt

    def correct(self, target_alt: float, target_az: float) -> tuple[float, float]:
        """Where to point (through the finder model) so the target lands in the main camera."""
        cos_alt = max(np.cos(np.radians(target_alt)), 1e-6)
        return target_alt - self.d_alt_deg, target_az - self.d_az_sky_deg / cos_alt


MIN_SPREAD_DEG = 0.015  # RMS spread (about 100 px) needed in both axes to fit orientation


def fit_axes_and_offset(sky: np.ndarray, px: np.ndarray) -> tuple[np.ndarray, MainOffset] | None:
    """Camera axes and finder->main offset in one fit, from timestamped samples.

    sky: each frame's target offset from where the finder model pointed (d_az_sky, d_alt), at the
    frame's time; px: the target's pixel offset from the main image center in that frame. The
    target's offset from the camera aim is sky - offset, so px = A (sky - offset) = A sky + b, and
    offset = -A^-1 b. None until the samples span both axes."""
    if len(sky) < 4:
        return None
    sv = np.linalg.svd(sky - sky.mean(axis=0), compute_uv=False) / np.sqrt(len(sky))
    if sv[-1] < MIN_SPREAD_DEG:
        return None
    coef = np.linalg.lstsq(np.c_[sky, np.ones(len(sky))], px, rcond=None)[0]  # rows: A^T, b
    a, b = coef[:2].T, coef[2]
    off = -np.linalg.solve(a, b)
    return a, MainOffset(float(off[0]), float(off[1]), len(sky))
