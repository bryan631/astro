"""Finishing a deep-sky stack (C4): remove the light-pollution gradient, balance the color,
stretch. In-process numpy, standing in for GraXpert (background) and Siril (color, stretch);
the linear stack is also saved as FITS so either can be run on it later."""

from pathlib import Path

import numpy as np
from astropy.io import fits
from scipy import ndimage

GRID = 16  # background sampled in GRID x GRID tiles
CLIP_SIGMA = 2.0  # tiles this much brighter than the rest hold a target or stars: not sky
STAR_PCT = 99.9  # star peaks standing out this much (percentile) set the color balance
MIN_STARS = 10
DENOISE_SIGMA_PX = 1.5  # blur for the faint sky (the stretch amplifies its noise most)
DENOISE_KNEE = 3.0  # this many noise sigmas above the sky counts as real detail, kept sharp
STAR_WINDOW_PX = 9  # stars are smaller than this; nebulae much bigger
SKY_LEVEL = 0.1  # the stretched sky's gray level
BLACK_SIGMA = 2.0  # the black point, in sky noise below the sky


def remove_gradient(rgb: np.ndarray) -> np.ndarray:
    """Subtract a smooth (quadratic) sky background fitted per channel to the faint tiles."""
    h, w, _ = rgb.shape
    th, tw = h // GRID, w // GRID
    tiles = rgb[:th * GRID, :tw * GRID].reshape(GRID, th, GRID, tw, 3)
    medians = np.median(tiles, axis=(1, 3)).reshape(-1, 3)  # one sky sample per tile
    gy, gx = np.mgrid[:GRID, :GRID]
    terms = _quadratic((gx.ravel() + 0.5) / GRID, (gy.ravel() + 0.5) / GRID)
    sky = np.ones(len(medians), bool)
    for _ in range(3):  # fit, drop tiles well above the fit (the target, bright stars), refit
        coef, *_ = np.linalg.lstsq(terms[sky], medians[sky], rcond=None)
        resid = (medians - terms @ coef).mean(axis=-1)
        spread = 1.4826 * np.median(np.abs(resid[sky] - np.median(resid[sky])))
        sky = resid <= np.median(resid[sky]) + CLIP_SIGMA * max(spread, 1e-6)
    y, x = np.mgrid[:h, :w]
    out = rgb - (_quadratic(x / w, y / h) @ coef).astype(np.float32)
    return out - np.percentile(out, 5, axis=(0, 1))  # sky near zero, nothing hugely negative


def _quadratic(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    return np.stack([np.ones_like(x), x, y, x * y, x**2, y**2], axis=-1)


def balance_color(rgb: np.ndarray) -> np.ndarray:
    """Scale channels so the stars come out white on average (most stars are near-white): the
    median color of clear star peaks above their local background. Picking "stars" as the
    brightest excess pixels picked the main camera's noisier red and blue and turned
    Andromeda green (2026-10-10). A bright nebula (extended) keeps its own color."""
    smooth = ndimage.gaussian_filter(rgb, (1.5, 1.5, 0))
    excess = smooth - ndimage.median_filter(smooth, size=(STAR_WINDOW_PX, STAR_WINDOW_PX, 1))
    lum = excess.mean(axis=-1)
    peaks = (lum == ndimage.maximum_filter(lum, STAR_WINDOW_PX)) & (lum >= np.percentile(lum, STAR_PCT))
    stars = excess[peaks & (excess.min(axis=-1) > 0)]
    if len(stars) < MIN_STARS:  # featureless: nothing to calibrate on
        return rgb
    means = np.median(stars, axis=0)
    return rgb * (means.mean() / means)


def denoise(rgb: np.ndarray, sigma: float = DENOISE_SIGMA_PX) -> np.ndarray:
    """Smooth the faint sky, where noise is all there is, and leave stars and bright detail
    sharp: blend toward a blurred copy by how close each pixel is to the background."""
    blurred = ndimage.gaussian_filter(rgb, (sigma, sigma, 0))
    lum = rgb.mean(axis=-1, keepdims=True)  # unblurred: a faint star keeps its full peak
    sky = np.median(lum)
    noise = 1.4826 * np.median(np.abs(lum - sky))  # robust: stars and nebulae don't inflate it
    signal = np.clip((lum - sky) / max(DENOISE_KNEE * noise, 1e-6), 0, 1)  # 0 sky, 1 detail
    return signal * rgb + (1 - signal) * blurred


def stretch(rgb: np.ndarray, sky_level: float = SKY_LEVEL) -> np.ndarray:
    """Lift the faint target without blowing out the stars, one scale per pixel (keeps colors):
    the sky (at 0, see finish) lands at `sky_level` gray, the black point BLACK_SIGMA noise
    below it. The old stretch put the sky at ~45% gray: washed out (2026-10-10)."""
    lum = rgb.mean(axis=-1)
    off = BLACK_SIGMA * 1.4826 * float(np.median(np.abs(lum - np.median(lum))))
    top = float(np.percentile(lum, 99.95)) + off
    x = np.clip((lum + off) / max(top, 1e-6), 1e-6, None)
    k = _asinh_strength(off / max(top, 1e-6), sky_level)
    scale = np.arcsinh(k * x) / np.arcsinh(k) / x
    return (np.clip((rgb + off) / max(top, 1e-6) * scale[..., None], 0, 1) * 255).astype(np.uint8)


def _asinh_strength(x: float, level: float) -> float:
    """k with asinh(k x) / asinh(k) = level (bisection on a log scale)."""
    lo, hi = 1e-3, 1e5
    for _ in range(60):
        k = float(np.sqrt(lo * hi))
        lo, hi = (k, hi) if np.arcsinh(k * x) / np.arcsinh(k) < level else (lo, k)
    return k


def finish(rgb: np.ndarray) -> np.ndarray:
    """Linear stacked RGB -> display-ready 8-bit RGB. The sky goes to 0 in each color (gray: the
    main camera's sky is ~1.7x more red and blue than green)."""
    flat = remove_gradient(rgb)
    flat = flat - np.median(flat.reshape(-1, 3), axis=0)
    return stretch(denoise(balance_color(flat)))


def save_fits(rgb: np.ndarray, path: Path) -> None:
    """The linear stack, 16-bit, channels first (as Siril and GraXpert expect)."""
    scaled = rgb / max(float(rgb.max()), 1e-6) * 65535
    fits.PrimaryHDU(np.moveaxis(scaled, -1, 0).astype(np.uint16)[:, ::-1]).writeto(
        path, overwrite=True)  # FITS rows run bottom-up
