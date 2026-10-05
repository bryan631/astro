"""Keep a planet centered: find the brightest blob and place an ROI around it."""

import numpy as np
from scipy import ndimage

from astro.devices.base import Roi


def brightest_blob(frame: np.ndarray) -> tuple[float, float] | None:
    """(x, y) centroid of the brightest connected region, or None for an empty frame."""
    smooth = ndimage.uniform_filter(frame.astype(np.float32), 5)
    peak = smooth.max()
    if peak - np.median(smooth) < 10:
        return None
    labels, _ = ndimage.label(smooth > (peak + np.median(smooth)) / 2)
    region = labels == labels[np.unravel_index(np.argmax(smooth), smooth.shape)]
    y, x = ndimage.center_of_mass(smooth * region)
    return float(x), float(y)


def roi_around(center: tuple[float, float], size: int, sensor: tuple[int, int]) -> Roi:
    """Square ROI centered on `center`, clamped to the sensor (w, h), on even pixels for Bayer."""
    w, h = sensor
    x = int(max(min(max(center[0] - size / 2, 0), w - size), 0)) & ~1  # 0 if size > sensor
    y = int(max(min(max(center[1] - size / 2, 0), h - size), 0)) & ~1
    return Roi(x, y, size, size)
