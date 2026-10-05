"""Finishing a deep-sky stack (C4): remove the light-pollution gradient, balance the color,
stretch. In-process numpy, standing in for GraXpert (background) and Siril (color, stretch);
the linear stack is also saved as FITS so either can be run on it later."""

from pathlib import Path

import numpy as np
from astropy.io import fits
from scipy import ndimage

GRID = 16  # background sampled in GRID x GRID tiles
CLIP_SIGMA = 2.0  # tiles this much brighter than the rest hold a target or stars: not sky
STAR_PCT = 99.5  # pixels standing out this much (percentile) are stars, for color balance
DENOISE_SIGMA_PX = 1.5  # blur for the faint sky (the stretch amplifies its noise most)
DENOISE_KNEE = 3.0  # this many noise sigmas above the sky counts as real detail, kept sharp
STAR_WINDOW_PX = 9  # stars are smaller than this; nebulae much bigger


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
    """Scale channels so the stars come out white on average (most stars are near-white).
    Stars are picked by how much they stand out from their surroundings, so a bright nebula
    (extended) keeps its own color."""
    excess = rgb - ndimage.median_filter(rgb, size=(STAR_WINDOW_PX, STAR_WINDOW_PX, 1))
    compact = excess.mean(axis=-1)  # starlight above whatever lies behind it
    stars = compact >= np.percentile(compact, STAR_PCT)
    if np.ptp(compact) == 0:  # featureless: nothing to calibrate on
        return rgb
    means = excess[stars].mean(axis=0)
    return rgb * (means.mean() / np.maximum(means, 1e-6))


def denoise(rgb: np.ndarray, sigma: float = DENOISE_SIGMA_PX) -> np.ndarray:
    """Smooth the faint sky, where noise is all there is, and leave stars and bright detail
    sharp: blend toward a blurred copy by how close each pixel is to the background."""
    blurred = ndimage.gaussian_filter(rgb, (sigma, sigma, 0))
    lum = blurred.mean(axis=-1, keepdims=True)
    sky = np.median(lum)
    noise = 1.4826 * np.median(np.abs(rgb.mean(axis=-1) - rgb.mean(axis=-1).mean()))
    signal = np.clip((lum - sky) / max(DENOISE_KNEE * noise, 1e-6), 0, 1)  # 0 sky, 1 detail
    return signal * rgb + (1 - signal) * blurred


def asinh_stretch(rgb: np.ndarray, strength: float = 50) -> np.ndarray:
    """Lift the faint nebula without blowing out the stars; keeps colors (one scale per pixel)."""
    lum = rgb.mean(axis=-1, keepdims=True)
    top = np.percentile(lum, 99.9)
    norm = np.clip(lum / max(top, 1e-6), 0, None)
    scale = np.arcsinh(strength * norm) / np.arcsinh(strength) / np.maximum(norm, 1e-6)
    return (np.clip(rgb / max(top, 1e-6) * scale, 0, 1) * 255).astype(np.uint8)


def finish(rgb: np.ndarray) -> np.ndarray:
    """Linear stacked RGB -> display-ready 8-bit RGB."""
    return asinh_stretch(denoise(balance_color(np.clip(remove_gradient(rgb), 0, None))))


def save_fits(rgb: np.ndarray, path: Path) -> None:
    """The linear stack, 16-bit, channels first (as Siril and GraXpert expect)."""
    scaled = rgb / max(float(rgb.max()), 1e-6) * 65535
    fits.PrimaryHDU(np.moveaxis(scaled, -1, 0).astype(np.uint16)[:, ::-1]).writeto(
        path, overwrite=True)  # FITS rows run bottom-up
