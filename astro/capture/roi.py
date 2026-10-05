"""Keep a planet centered: find the brightest blob and place an ROI around it."""

import numpy as np
from scipy import ndimage

from astro.devices.base import Roi

MIN_CONTRAST_ADU = 10  # 8-bit frames: a peak this far over the median is "something there"

def brightest_blob(frame: np.ndarray) -> tuple[float, float] | None:
    """(x, y) centroid of the brightest connected region, or None for an empty frame."""
    smooth = ndimage.uniform_filter(frame.astype(np.float32), 5)
    peak = smooth.max()
    if peak - np.median(smooth) < MIN_CONTRAST_ADU:
        return None
    labels, _ = ndimage.label(smooth > (peak + np.median(smooth)) / 2)
    region = labels == labels[np.unravel_index(np.argmax(smooth), smooth.shape)]
    y, x = ndimage.center_of_mass(smooth * region)
    return float(x), float(y)


def roi_around(center: tuple[float, float], size: int, sensor: tuple[int, int]) -> Roi:
    """Square ROI centered on `center`, clamped to the sensor (w, h), on even pixels for Bayer."""
    w, h = sensor
    size = min(size, w, h) & ~1  # a small sensor gets a smaller ROI, never one past its edge
    x = int(min(max(center[0] - size / 2, 0), w - size)) & ~1
    y = int(min(max(center[1] - size / 2, 0), h - size)) & ~1
    return Roi(x, y, size, size)
