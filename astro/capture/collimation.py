"""Collimation check (F3) from a defocused star in the main camera: the "donut".

A collimated scope shows the secondary's shadow centered in an evenly lit ring. Prototype:
tuned on synthetic donuts; retune on the real sky (seeing makes the ring lumpy)."""

from dataclasses import dataclass

import numpy as np
from scipy import ndimage

CENTERED = 0.05  # shadow offset (fraction of the ring radius) that counts as collimated
MIN_RADIUS_PX = 15  # smaller: not defocused enough to see the shadow
SECTORS = 8  # ring brightness is compared around the circle in this many slices
CHANGE = 0.02  # offset change (fraction of radius) worth calling better or worse


@dataclass(frozen=True)
class Donut:
    offset: tuple[float, float]  # shadow center minus ring center, fraction of radius (x, y)
    radius_px: float
    unevenness: float  # spread of ring brightness around the circle (0 = even)

    @property
    def off(self) -> float:
        return float(np.hypot(*self.offset))

    @property
    def clock(self) -> int:
        """Where the shadow sits, as a clock position on the screen (12 = up)."""
        angle = float(np.degrees(np.arctan2(self.offset[0], -self.offset[1]))) % 360  # y is down
        return round(angle / 30) % 12 or 12


def analyze(gray: np.ndarray) -> Donut | str:
    """Measure the brightest donut, or say (a string) why it can't."""
    smooth = ndimage.gaussian_filter(gray.astype(np.float32), 1.5)
    bg = float(np.median(smooth))
    lit = smooth > bg + 0.4 * (smooth.max() - bg)
    labels, n = ndimage.label(lit)
    if n == 0:
        return "I don't see a star in the main camera."
    biggest = labels == 1 + int(np.argmax(np.bincount(labels.ravel())[1:]))
    disk = ndimage.binary_fill_holes(biggest)
    radius = float(np.sqrt(disk.sum() / np.pi))
    hole = disk & ~biggest
    if radius < MIN_RADIUS_PX or hole.sum() < 0.01 * disk.sum():
        return "Turn the focus knob until the star becomes a big donut with a dark middle."
    ring_y, ring_x = ndimage.center_of_mass(disk)
    hole_y, hole_x = ndimage.center_of_mass(hole)
    ys, xs = np.nonzero(biggest)
    sector = ((np.arctan2(ys - ring_y, xs - ring_x) + np.pi) / (2 * np.pi) * SECTORS).astype(int)
    light = np.bincount(sector % SECTORS, weights=smooth[ys, xs] - bg, minlength=SECTORS)
    return Donut(((hole_x - ring_x) / radius, (hole_y - ring_y) / radius), radius,
                 float(light.std() / max(light.mean(), 1e-9)))


class CollimationCoach:
    """Spoken feedback while the user turns the primary mirror's screws."""

    def __init__(self):
        self.last: float | None = None

    def update(self, donut: Donut) -> str | None:
        prev, self.last = self.last, donut.off
        if donut.off <= CENTERED:
            return "The shadow is centered. Collimation looks good."
        where = (f"The shadow is toward {donut.clock} o'clock, "
                 f"{donut.off * 100:.0f} percent off center.")
        if prev is None:
            return (f"{where} Turn one of the primary mirror's screws a little and I'll tell you "
                    "if it gets better.")
        if prev - donut.off > CHANGE:
            return "Better, keep going."
        if donut.off - prev > CHANGE:
            return "Worse. Turn that screw back, or try another one."
        return None
