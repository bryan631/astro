"""Focus metrics and a spoken 'sharper / passed it' coach."""

import numpy as np
from scipy import ndimage


def laplacian_variance(frame: np.ndarray) -> float:
    """Sharpness of an extended object (planet, Moon). Higher is sharper."""
    return float(ndimage.laplace(frame.astype(np.float32)).var())


def half_flux_radius(cutout: np.ndarray) -> float:
    """HFR in pixels of a single star cutout. Lower is sharper."""
    img = cutout.astype(np.float64) - np.median(cutout)
    img[img < 0] = 0
    total = img.sum()
    if total == 0:
        return float("inf")
    y, x = np.indices(img.shape)
    cy, cx = (img * y).sum() / total, (img * x).sum() / total
    r = np.hypot(y - cy, x - cx)
    order = np.argsort(r, axis=None)
    cumulative = np.cumsum(img.flat[order])
    return float(r.flat[order][np.searchsorted(cumulative, total / 2)])


class FocusCoach:
    """Feed a sharpness score (higher = better); get short spoken feedback."""

    def __init__(self, tolerance: float = 0.03):
        self.tol = tolerance
        self.best: float | None = None
        self.last: float | None = None
        self.samples = 0  # readings so far ("done" too early means focus was never checked)

    def update(self, score: float) -> str | None:
        self.samples += 1
        prev, self.last = self.last, score
        if self.best is None or score > self.best:
            self.best = score
        if prev is None:
            return "keep turning the focus knob slowly"
        change = (score - prev) / max(abs(prev), 1e-9)
        if change > self.tol:
            return "sharper"
        if change < -self.tol:
            return "passed it, go back slowly" if self.best > score * (1 + self.tol) else "softer"
        if abs(score - self.best) <= self.tol * self.best:
            return "that's the sharpest so far"
        return None
