"""Keep a planet centered: find the brightest blob and place an ROI around it."""

import numpy as np
from scipy import ndimage

from astro.devices.base import Roi

MIN_CONTRAST_ADU = 10  # 8-bit frames: a peak this far over the median is "something there"
COMPANION_SIGMA = 10.0  # a moon is this far above the sky's noise
MAX_COMPANION_AREA_PX = 400  # a moon is a point; anything bigger is another disk or glare


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


def roi_around(center: tuple[float, float], size: int, sensor: tuple[int, int],
               height: int | None = None) -> Roi:
    """ROI centered on `center` (square unless `height`), clamped to the sensor (w, h), on even
    pixels for Bayer, width a multiple of 8 (the SVBony SDK refuses others)."""
    w, h = sensor
    if height is None:
        rw = rh = min(size, w, h) & ~7  # a small sensor gets a smaller ROI, never one past its edge
    else:
        rw, rh = min(size, w) & ~7, min(height, h) & ~1
    x = int(min(max(center[0] - rw / 2, 0), w - rw)) & ~1
    y = int(min(max(center[1] - rh / 2, 0), h - rh)) & ~1
    return Roi(x, y, rw, rh)


def companions(frame: np.ndarray, planet: tuple[float, float], reach: float) -> list[tuple[float, float]]:
    """Point sources within `reach` px of a planet, brightest first: Jupiter's moons, Titan.
    Small blobs well above the sky, not the planet's own blob."""
    img = ndimage.uniform_filter(ndimage.median_filter(frame, 3).astype(np.float32), 3)  # no hot pixels
    bg = float(np.median(img[::4, ::4]))
    noise = max(1.4826 * float(np.median(np.abs(img[::4, ::4] - bg))), 0.5)
    labels, n = ndimage.label(img > bg + COMPANION_SIGMA * noise)
    if n == 0:
        return []
    ids = np.arange(1, n + 1)
    own = labels[int(planet[1]), int(planet[0])]
    areas = ndimage.sum_labels(np.ones_like(img), labels, ids)
    peaks = ndimage.maximum(img, labels, ids)
    centers = np.array(ndimage.center_of_mass(img, labels, ids))[:, ::-1]  # (x, y)
    near = np.hypot(*(centers - planet).T) <= reach
    keep = near & (ids != own) & (areas <= MAX_COMPANION_AREA_PX)
    return [tuple(map(float, centers[i])) for i in np.flatnonzero(keep)[np.argsort(-peaks[keep])]]
