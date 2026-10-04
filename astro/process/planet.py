"""Planetary 'lucky imaging' prototype: SER video -> sharpest frames -> aligned stack -> PNG.

Pure numpy/scipy so it runs anywhere; PlanetarySystemStacker can replace it later if the
results fall short (plan Phase 1 step 8). Works in batches on a memory-mapped SER, so a
60 s recording (~8,000 frames) never has to fit in memory: one pass scores every frame,
a second pass decodes, aligns and sums only the sharpest ones.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import fft, ndimage

from astro.capture.ser import read_ser

KEEP_FRACTION = 0.25  # stack the sharpest quarter of the frames
MIN_FRAMES = 3
BATCH = 16  # frames per vectorized batch: bounds memory (~0.3 GB peak at 512 px ROI)
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
    """Half-resolution float RGB, one pixel per 2x2 Bayer cell. Works on (h, w) or (n, h, w)."""
    (ry, rx), (g1y, g1x), (g2y, g2x), (by, bx) = _LAYOUT[bayer]
    h, w = raw.shape[-2] // 2 * 2, raw.shape[-1] // 2 * 2
    r = raw[..., ry:h:2, rx:w:2].astype(np.float32)
    g = (raw[..., g1y:h:2, g1x:w:2].astype(np.float32) + raw[..., g2y:h:2, g2x:w:2]) / 2
    b = raw[..., by:h:2, bx:w:2].astype(np.float32)
    return np.stack([r, g, b], axis=-1)


def sharpness(lum: np.ndarray) -> np.ndarray:
    """Laplacian variance of each frame in an (n, h, w) batch (higher = sharper)."""
    lap = (lum[:, :-2, 1:-1] + lum[:, 2:, 1:-1] + lum[:, 1:-1, :-2] + lum[:, 1:-1, 2:]
           - 4 * lum[:, 1:-1, 1:-1])
    return lap.var(axis=(1, 2))


def planet_centers(lum: np.ndarray) -> np.ndarray:
    """(n, 2) center of mass (y, x) of the bright disk in each frame of an (n, h, w) batch."""
    lo = lum.min(axis=(1, 2), keepdims=True)
    hi = lum.max(axis=(1, 2), keepdims=True)
    weight = np.where(lum > lo + DISK_THRESHOLD * (hi - lo), lum, 0)
    total = weight.sum(axis=(1, 2))
    ys, xs = np.arange(lum.shape[1]), np.arange(lum.shape[2])
    return np.stack([(weight.sum(axis=2) * ys).sum(axis=1) / total,
                     (weight.sum(axis=1) * xs).sum(axis=1) / total], axis=1)


def planet_center(lum: np.ndarray) -> tuple[float, float]:
    y, x = planet_centers(lum[None])[0]
    return float(y), float(x)


def fourier_shift(images: np.ndarray, shifts: np.ndarray) -> np.ndarray:
    """Shift each image in an (n, h, w, c) batch by its (dy, dx), subpixel, all at once."""
    h, w = images.shape[1:3]
    ky = fft.fftfreq(h).astype(np.float32)[None, :, None, None]
    kx = fft.fftfreq(w).astype(np.float32)[None, None, :, None]
    dy = shifts[:, 0].astype(np.float32)[:, None, None, None]
    dx = shifts[:, 1].astype(np.float32)[:, None, None, None]
    phase = np.exp(np.complex64(-2j * np.pi) * (ky * dy + kx * dx))
    # scipy.fft keeps float32/complex64 (numpy.fft would double the memory).
    return fft.ifft2(fft.fft2(images.astype(np.float32), axes=(1, 2)) * phase, axes=(1, 2)).real


def stack(frames: np.ndarray, bayer: str) -> tuple[np.ndarray, int]:
    """Pick the sharpest frames, align them on the planet, and average.

    `frames` may be a memmap; it is read in batches. Returns (float RGB, frames used).
    """
    n = len(frames)
    quality = np.concatenate([sharpness(superpixel_rgb(frames[i:i + BATCH], bayer).mean(axis=-1))
                              for i in range(0, n, BATCH)])
    keep = np.sort(np.argsort(quality)[::-1][:max(MIN_FRAMES, int(n * KEEP_FRACTION))])
    ref = planet_centers(superpixel_rgb(frames[[keep[np.argmax(quality[keep])]]], bayer)
                         .mean(axis=-1))[0]
    total = None
    for i in range(0, len(keep), BATCH):
        rgb = superpixel_rgb(frames[keep[i:i + BATCH]], bayer)
        aligned = fourier_shift(rgb, ref - planet_centers(rgb.mean(axis=-1)))
        total = aligned.sum(axis=0) if total is None else total + aligned.sum(axis=0)
    return total / len(keep), len(keep)


def align_channels(img: np.ndarray) -> np.ndarray:
    """Center red and blue on green: removes atmospheric dispersion and Bayer offsets."""
    centers = planet_centers(np.moveaxis(img, -1, 0))  # one "frame" per color channel
    shifts = centers[1] - centers  # move every channel onto green
    return fourier_shift(np.moveaxis(img, -1, 0)[..., None], shifts)[..., 0].transpose(1, 2, 0)


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
    img, used = stack(frames, bayer)
    Image.fromarray(finish(img)).save(path)
    return StackResult(path, len(frames), used)
