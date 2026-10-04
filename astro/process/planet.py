"""Planetary 'lucky imaging' prototype: SER video -> sharpest frames -> aligned stack -> PNG.

Pure numpy/scipy so it runs anywhere. PlanetarySystemStacker can replace it later if the
results fall short (plan Phase 1 step 8).
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage

from astro.capture.focus import laplacian_variance
from astro.capture.ser import read_ser

KEEP_FRACTION = 0.25  # stack the sharpest quarter of the frames
MIN_FRAMES = 3
SHARPEN_SIGMA_PX = 1.5  # unsharp mask radius on the half-resolution color image
SHARPEN_AMOUNT = 1.0
CROP_MARGIN = 1.6  # crop to this many planet radii around the center
DISK_THRESHOLD = 0.3  # fraction of the brightest level that counts as "planet" for centering

# Position of R, G1, G2, B within each 2x2 Bayer cell, as (row, col).
_LAYOUT = {
    "RGGB": ((0, 0), (0, 1), (1, 0), (1, 1)),
    "BGGR": ((1, 1), (0, 1), (1, 0), (0, 0)),
    "GRBG": ((0, 1), (0, 0), (1, 1), (1, 0)),
    "GBRG": ((1, 0), (0, 0), (1, 1), (0, 1)),
}
_COLOR_IDS = {8: "RGGB", 9: "GRBG", 10: "GBRG", 11: "BGGR"}


@dataclass(frozen=True)
class StackResult:
    path: Path
    frames_total: int
    frames_used: int


def superpixel_rgb(raw: np.ndarray, bayer: str) -> np.ndarray:
    """Half-resolution float RGB from one Bayer frame: one pixel per 2x2 cell."""
    (ry, rx), (g1y, g1x), (g2y, g2x), (by, bx) = _LAYOUT[bayer]
    h, w = raw.shape[0] // 2 * 2, raw.shape[1] // 2 * 2
    r = raw[ry:h:2, rx:w:2].astype(np.float32)
    g = (raw[g1y:h:2, g1x:w:2].astype(np.float32) + raw[g2y:h:2, g2x:w:2]) / 2
    b = raw[by:h:2, bx:w:2].astype(np.float32)
    return np.dstack([r, g, b])


def planet_center(lum: np.ndarray) -> tuple[float, float]:
    """(y, x) center of mass of the bright disk."""
    level = lum.min() + DISK_THRESHOLD * (lum.max() - lum.min())
    return ndimage.center_of_mass(np.where(lum > level, lum, 0))


def stack(frames: np.ndarray, bayer: str) -> np.ndarray:
    """Pick the sharpest frames, align them on the planet, and average. Returns float RGB."""
    rgb = [superpixel_rgb(f, bayer) for f in frames]
    lum = [im.mean(axis=2) for im in rgb]
    quality = np.array([laplacian_variance(im) for im in lum])
    keep = max(MIN_FRAMES, int(len(frames) * KEEP_FRACTION))
    best = np.argsort(quality)[::-1][:keep]
    ref_y, ref_x = planet_center(lum[best[0]])
    out = np.zeros_like(rgb[0])
    for i in best:
        y, x = planet_center(lum[i])
        out += ndimage.shift(rgb[i], (ref_y - y, ref_x - x, 0), order=1, mode="nearest")
    return out / len(best)


def align_channels(img: np.ndarray) -> np.ndarray:
    """Center red and blue on green: removes atmospheric dispersion and Bayer offsets."""
    gy, gx = planet_center(img[..., 1])
    out = img.copy()
    for c in (0, 2):
        y, x = planet_center(img[..., c])
        out[..., c] = ndimage.shift(img[..., c], (gy - y, gx - x), order=1, mode="nearest")
    return out


def finish(img: np.ndarray) -> np.ndarray:
    """Align colors, sharpen, crop around the planet, stretch to 8-bit."""
    img = align_channels(img)
    blurred = ndimage.gaussian_filter(img, (SHARPEN_SIGMA_PX, SHARPEN_SIGMA_PX, 0))
    img = img + SHARPEN_AMOUNT * (img - blurred)
    lum = img.mean(axis=2)
    cy, cx = planet_center(lum)
    radius = np.sqrt((lum > lum.min() + DISK_THRESHOLD * np.ptp(lum)).sum() / np.pi)
    half = int(max(radius * CROP_MARGIN, 16))
    y0, x0 = max(int(cy) - half, 0), max(int(cx) - half, 0)
    img = img[y0:y0 + 2 * half, x0:x0 + 2 * half]
    lo, hi = np.percentile(img, 0.5), np.percentile(img, 99.9)
    return ((img - lo) / max(hi - lo, 1e-6) * 255).clip(0, 255).astype(np.uint8)


def process_ser(ser: Path, out_dir: Path) -> StackResult:
    """SER recording -> PNG in `out_dir` (same name)."""
    meta, frames = read_ser(ser)
    bayer = _COLOR_IDS.get(meta["color_id"])
    if bayer is None:
        raise ValueError(f"unsupported SER color id {meta['color_id']}")
    if len(frames) < MIN_FRAMES:
        raise ValueError(f"only {len(frames)} frames; need at least {MIN_FRAMES}")
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / (ser.stem + ".png")
    Image.fromarray(finish(stack(frames, bayer))).save(path)
    used = max(MIN_FRAMES, int(len(frames) * KEEP_FRACTION))
    return StackResult(path, len(frames), used)
