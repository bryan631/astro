"""Focus metrics: the number on each camera view (higher is sharper), and star/planet helpers."""

from dataclasses import dataclass

import numpy as np
from scipy import ndimage, optimize

# The focus number (Phase 3b). Stars: 100 / median half-flux radius (px) over the usable stars,
# so the many faint stars decide it; saturated stars look wider than they are and are skipped.
# Each star's width is a fitted Gaussian's (2026-10-11): the pixel-counted radius stepped
# through 1, 1.41, 2 px, and its size test dropped sharp stars, so near best focus the number
# vanished. Planets and the Moon (few stars, one big bright object): edge sharpness.
MAD_SIGMA = 1.4826  # median absolute deviation -> standard deviation, for Gaussian noise
DETECT_SIGMA = 5.0  # a star is a blob above sky + 5 sigma (after a light smoothing)
MEASURE_SIGMA = 8.0  # ... and is measured only if its peak is this far above: noise inflates HFR
SATURATED = 245.0  # 8-bit: a pixel this bright is clipped (left out of a star's fit)
MAX_CLIPPED_PX = 30  # a star with more clipped pixels than this is too burnt out to fit
MIN_AREA_PX, MAX_AREA_PX = 3, 3000  # detected (smoothed) pixels: a planet or donut isn't a star
MIN_STARS = 3  # fewer: judge by the brightest object's edges instead
MAX_STARS = 40  # fitted per frame, spread over brightness
MIN_SIGMA_PX, MAX_SIGMA_PX = 0.4, 10.0  # a fitted width narrower than this is a hot pixel
GAUSS_HFR = 1.1774  # a Gaussian's half-flux radius in sigmas (sqrt(2 ln 2))
MIN_HFR_PX = 0.5  # a perfectly sharp star can't make the number blow up
PLANET_AREA_PX = 80  # a bright blob at least this big is a disk (Saturn: ~20 px across, binned)
SKY_BLOCK_PX = 64  # the sky's own gradient (cloud glow, the horizon) is smooth over this
GRADIENT_DOMINATES = 4.0  # take the gradient off only when the spread drops this much without it


@dataclass(frozen=True)
class FocusMeasure:
    score: float | None  # higher is sharper; None when there's nothing to judge
    stars: int  # stars measured
    hfr_px: float | None  # median half-flux radius of the fitted stars' narrow axis (stars mode)
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
    """The frame minus its sky: the smooth sky_map when a gradient dominates the frame's spread
    (cloud glow: 30x less spread without it), else one level. Treetops have sharp edges a smooth
    sky can't follow: the gaps between leaves would pass for stars (only 2x less spread)."""
    flat = img - sky_map(img)
    if _spread(img) > GRADIENT_DOMINATES * _spread(flat):
        return flat
    return img - np.median(img)


def _spread(img: np.ndarray) -> float:
    sample = img[::2, ::2]
    return float(np.median(np.abs(sample - np.median(sample))))


def measure_focus(lum: np.ndarray) -> FocusMeasure:
    """The focus number for a frame (8-bit scale gray, hot pixels already removed)."""
    raw = lum.astype(np.float32)
    img = flatten_sky(raw)
    smooth = ndimage.gaussian_filter(img, 1.0)
    # The thresholds use the smoothed frame's own noise: the unsmoothed noise is ~1.6x higher on
    # the main camera at night and hid half its few stars (2026-10-10).
    bg, noise = _sky(smooth)
    above = bg + DETECT_SIGMA * noise
    labels, n = ndimage.label(smooth > above)
    if n == 0:
        return FocusMeasure(None, 0, None, "none")
    ids = np.arange(1, n + 1)
    on = labels > 0  # the per-star sums below look only at star pixels (a full frame is slow)
    lab = labels[on]
    areas = np.bincount(lab, minlength=n + 1)[1:]
    # Smoothed peaks: a noise spike on a faint star's top moved where its profile is cut, and
    # with the main camera's ~8 stars the number jumped 66-83 on a still scope (2026-10-10).
    peaks = ndimage.maximum(smooth[on], lab, ids)
    clipped = np.bincount(lab, weights=raw[on] >= SATURATED, minlength=n + 1)[1:]
    ok = (areas >= MIN_AREA_PX) & (areas <= MAX_AREA_PX) & (peaks > bg + MEASURE_SIGMA * noise)
    usable = ids[ok & (clipped == 0)]
    # Too few unclipped stars: clipped ones too, fitted on their unclipped pixels. Sharp stars
    # saturate first, and without them the number vanished near best focus. Only then: a burnt
    # out star's unclipped pixels are its wide halo, which read 4x too wide (2026-10-10 frames).
    if len(usable) < MIN_STARS:
        usable = ids[ok & (clipped <= MAX_CLIPPED_PX)]
    if len(usable) > MAX_STARS:  # an even spread over brightness, not just the brightest
        usable = usable[np.argsort(peaks[usable - 1])][np.linspace(0, len(usable) - 1, MAX_STARS).astype(int)]
    centers = ndimage.center_of_mass(img - bg, labels, usable) if len(usable) else []
    sigmas = [w for c, i in zip(centers, usable, strict=True)
              if (w := star_sigma(img - bg, c, areas[i - 1], raw < SATURATED)) is not None]  # hot pixels: None
    if len(sigmas) >= MIN_STARS:
        hfr = max(GAUSS_HFR * float(np.median(sigmas)), MIN_HFR_PX)
        return FocusMeasure(100 / hfr, len(sigmas), hfr, "stars")
    big = ids[areas >= PLANET_AREA_PX]
    if len(big):
        return FocusMeasure(_edge_sharpness(img, labels, int(big[np.argmax(areas[big - 1])]), bg),
                            0, None, "planet")
    return FocusMeasure(None, len(usable), None, "none")


def star_sigma(img: np.ndarray, center: tuple[float, float], area: float,
               valid: np.ndarray | None = None) -> float | None:
    """A star's width (px) along its narrow axis: a fitted elliptical Gaussian, in a cutout
    sized to the star (`img` has the sky off), on the `valid` (unclipped) pixels. Smooth at any
    size, and the drift trail of a 0.25 s frame from an untracked scope, along the other axis,
    doesn't count. None if the fit fails or is too narrow for a star (a hot pixel) or too wide."""
    r = int(np.clip(2.5 * np.sqrt(area / np.pi), 4, 15))
    cy, cx = (round(v) for v in center)
    if cy - r < 0 or cx - r < 0 or cy + r >= img.shape[0] or cx + r >= img.shape[1]:
        return None
    window = (slice(cy - r, cy + r + 1), slice(cx - r, cx + r + 1))
    cut = img[window].astype(np.float64)
    keep = np.ones(cut.shape, bool) if valid is None else valid[window]
    yy, xx = np.mgrid[-r:r + 1, -r:r + 1]

    def residual(p):
        a, b, y0, x0, sa, sb, th = p
        c, s = np.cos(th), np.sin(th)
        u, v = (xx - x0) * c + (yy - y0) * s, (yy - y0) * c - (xx - x0) * s
        return (b + a * np.exp(-u * u / (2 * sa * sa) - v * v / (2 * sb * sb)) - cut)[keep]

    w0 = max(np.sqrt(area / np.pi) / 2, 0.8)
    fit = optimize.least_squares(residual, [cut.max(), 0.0, center[0] - cy, center[1] - cx, w0, w0, 0.0],
                                 method="lm", max_nfev=200)
    a, _, y0, x0, sa, sb, _ = fit.x
    narrow = min(abs(sa), abs(sb))
    if not fit.success or a <= 0 or max(abs(y0), abs(x0)) > r or not MIN_SIGMA_PX <= narrow <= MAX_SIGMA_PX:
        return None
    return float(narrow)


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
