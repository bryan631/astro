"""Focus metrics: the number on each camera view (higher is sharper), and star/planet helpers."""

from dataclasses import dataclass

import numpy as np
from scipy import ndimage

# The focus number (Phase 3b). Stars: 100 / median half-flux radius (px) over every usable
# star, so the many faint stars decide it; saturated stars look wider than they are and are
# skipped. Planets and the Moon (few stars, one big bright object): edge sharpness.
MAD_SIGMA = 1.4826  # median absolute deviation -> standard deviation, for Gaussian noise
DETECT_SIGMA = 5.0  # a star is a blob above sky + 5 sigma (after a light smoothing)
MEASURE_SIGMA = 8.0  # ... and is measured only if its peak is this far above: noise inflates HFR
# Inside a cutout, pixels below this fraction of the star's own peak count as sky. Relative,
# so every star is cut at the same place on its profile and brightness doesn't change its
# radius; 25% is above 2 sigma for every measured (>= 8 sigma) star.
REL_CLIP = 0.25
SATURATED = 245.0  # 8-bit: a star this bright at its peak is clipped
MIN_AREA_PX, MAX_AREA_PX = 3, 3000  # a lone hot pixel isn't a star; a planet or donut isn't either
MIN_STARS = 3  # fewer: judge by the brightest object's edges instead
MAX_STARS = 300
MIN_HFR_PX = 0.5  # a perfectly sharp star can't make the number blow up
PLANET_AREA_PX = 80  # a bright blob at least this big is a disk (Saturn: ~20 px across, binned)
SKY_BLOCK_PX = 64  # the sky's own gradient (cloud glow, the horizon) is smooth over this


@dataclass(frozen=True)
class FocusMeasure:
    score: float | None  # higher is sharper; None when there's nothing to judge
    stars: int  # stars measured
    hfr_px: float | None  # median half-flux radius (stars mode)
    mode: str  # "stars", "planet" or "none"


def _sky(img: np.ndarray) -> tuple[float, float]:
    sample = img[::4, ::4]
    bg = float(np.median(sample))
    noise = MAD_SIGMA * float(np.median(np.abs(sample - bg)))
    return bg, max(noise, 0.5)


def sky_map(img: np.ndarray) -> np.ndarray:
    """The frame's smooth sky: a median per block (every other pixel is plenty), interpolated.
    Cloud glow brightens one side of the finder ~2.5x; one sky level for the frame hid every
    star (2026-10-10). Used by the focus number, Align's star pick and the night view."""
    b = SKY_BLOCK_PX
    h, w = img.shape[0] // b * b, img.shape[1] // b * b
    if h == 0 or w == 0:
        return np.full(img.shape, np.median(img), np.float32)
    s = b // 2
    sample = img[:h:2, :w:2]
    blocks = np.median(sample.reshape(h // b, s, w // b, s).swapaxes(1, 2).reshape(h // b, w // b, -1), axis=2)
    return ndimage.zoom(blocks, (img.shape[0] / blocks.shape[0], img.shape[1] / blocks.shape[1]),
                        order=1, mode="nearest", grid_mode=True)


def flatten_sky(img: np.ndarray) -> np.ndarray:
    """The frame minus its smooth sky (sky_map)."""
    return img - sky_map(img)


def measure_focus(lum: np.ndarray) -> FocusMeasure:
    """The focus number for a frame (8-bit scale gray, hot pixels already removed)."""
    raw = lum.astype(np.float32)
    img = flatten_sky(raw)
    bg, noise = _sky(img)
    above = bg + DETECT_SIGMA * noise
    labels, n = ndimage.label(ndimage.gaussian_filter(img, 1.0) > above)
    if n == 0:
        return FocusMeasure(None, 0, None, "none")
    ids = np.arange(1, n + 1)
    on = labels > 0  # the per-star sums below look only at star pixels (a full frame is slow)
    lab = labels[on]
    areas = ndimage.sum_labels(img[on] > above, lab, ids)  # unsmoothed: a hot pixel stays 1 px
    peaks = ndimage.maximum(img[on], lab, ids)
    clipped = ndimage.maximum(raw[on], lab, ids) >= SATURATED
    usable = ids[(areas >= MIN_AREA_PX) & (areas <= MAX_AREA_PX) & ~clipped
                 & (peaks > bg + MEASURE_SIGMA * noise)]
    if len(usable) >= MIN_STARS:
        if len(usable) > MAX_STARS:  # an even spread over brightness, not just the brightest
            usable = usable[np.argsort(peaks[usable - 1])][:: len(usable) // MAX_STARS + 1]
        centers = ndimage.center_of_mass(img - bg, labels, usable)
        hfrs = [_hfr(img, bg, c, areas[i - 1], peaks[i - 1] - bg)
                for c, i in zip(centers, usable, strict=True)]
        hfr = max(float(np.median(hfrs)), MIN_HFR_PX)
        return FocusMeasure(100 / hfr, len(usable), hfr, "stars")
    big = ids[areas >= PLANET_AREA_PX]
    if len(big):
        return FocusMeasure(_edge_sharpness(img, labels, int(big[np.argmax(areas[big - 1])]), bg),
                            0, None, "planet")
    return FocusMeasure(None, len(usable), None, "none")


def _hfr(img, bg, center, area, peak) -> float:
    """Half-flux radius of one star, in a cutout sized to the star."""
    r = int(np.clip(3 * np.sqrt(area / np.pi), 4, 20))
    cy, cx = (round(v) for v in center)
    cut = img[max(cy - r, 0):cy + r + 1, max(cx - r, 0):cx + r + 1] - bg
    cut = np.where(cut > REL_CLIP * peak, cut, 0)
    return half_flux_radius(cut, subtract_median=False)


def _edge_sharpness(img, labels, label: int, bg: float) -> float:
    """Mean gradient at the disk's edge, relative to the disk's brightness (so exposure
    doesn't matter): a soft edge spreads the same step over more pixels."""
    ys, xs = ndimage.find_objects((labels == label).astype(np.int32))[0]
    pad = 10
    crop = img[max(ys.start - pad, 0):ys.stop + pad, max(xs.start - pad, 0):xs.stop + pad] - bg
    grad = np.hypot(ndimage.sobel(crop, 0), ndimage.sobel(crop, 1))
    level = float(np.percentile(crop, 99))
    edge = grad > 0.1 * grad.max()
    return 100 * float(grad[edge].mean()) / max(level, 1e-6)


def laplacian_variance(frame: np.ndarray) -> float:
    """Sharpness of an extended object (planet, Moon). Higher is sharper."""
    return float(ndimage.laplace(frame.astype(np.float32)).var())


def half_flux_radius(cutout: np.ndarray, subtract_median: bool = True) -> float:
    """HFR in pixels of a single star cutout. Lower is sharper."""
    img = cutout.astype(np.float64) - (np.median(cutout) if subtract_median else 0)
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
