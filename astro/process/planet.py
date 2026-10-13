"""Planetary 'lucky imaging': SER video -> sharpest frames -> registered Bayer drizzle -> PNG.

Pure numpy/scipy so it runs on the mini PC. Streams the memory-mapped SER, so a 60 s
recording (~3,000 frames, ~1 GB) never has to fit in memory:
1. measure: one pass finds the planet in every frame and scores its sharpness;
2. register: the sharpest frames' planet crops are aligned to a reference, subpixel;
3. drizzle: those frames' raw Bayer pixels are dropped onto a 2x output grid at their
   registered positions (no demosaic blur; the drift dithers the samples);
4. finish: colors aligned and balanced, wavelet-sharpened, cropped, stretched.

2026-10-10 field notes (Saturn, 8" Dob at prime focus, 0.50"/px, non-tracking): the old
stacker worked on half-resolution superpixels, cropped the rings' tips off, clipped the
disk to white and left the picture pink; its PNG was a 36 px thumbnail.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import fft, ndimage

from astro.capture.roi import companions
from astro.capture.ser import read_ser

KEEP_FRACTION = 0.2  # the sharpest fifth: 2026-10-10 Saturn frames varied only ~8% in sharpness,
# so keeping 5% gained ~5-8% detail for twice the noise
MIN_FRAMES = 3
REF_FRAMES = 20  # the reference is the average of this many of the very best frames
SATURATED = 254  # raw level treated as clipped
CLIP_TOLERANCE = 0.05  # a channel saturated in this fraction of frames is rebuilt
SCALE = 2  # drizzle output pixels per raw pixel
BIN = 4  # binning for the quick planet search in pass 1
BATCH = 32  # frames read from the memmap at once in pass 1
DISK_THRESHOLD = 0.3  # fraction of the brightest level that counts as "planet" for centering
EXTENT_THRESHOLD = 0.08  # fraction of the brightest level that still counts (dim ring tips)
CROP_PAD = 1.25  # crop the picture to this many times the planet's (ring) extent
MOON_PAD = 0.3  # ... grown to keep each moon, with this many planet extents around it
SKY_TOLERANCE = 1e-3  # frames whose sky (off the planet) is brighter than usual by this fraction of
# the planet's peak are garbage: 2026-10-12, ~90 of 840 Barlow frames had horizontal stripes right
# of the planet (sky 0.01 -> 8 DN) and, the stripes passing for detail, ranked sharpest
QUALITY_SIGMA = 1.5  # smoothing (raw px) before scoring sharpness: seeing detail, not noise
# Wavelet sharpening on the 2x output: per-layer gains (finest first) and noise thresholds
# in units of each layer's noise. Layer 1 is mostly noise at 2x; layers 2-4 hold the bands,
# ring edges and shadows (tuned by eye on the 2026-10-10 Saturn videos).
B3_KERNEL = np.array([1, 4, 6, 4, 1], np.float32) / 16
WAVELET_GAINS = (1.0, 4.0, 3.0, 1.5)
WAVELET_NOISE = (3.0, 2.0, 1.0, 0.0)

# Position of R, G1, G2, B within each 2x2 Bayer cell, as (row, col).
_LAYOUT = {
    "RGGB": ((0, 0), (0, 1), (1, 0), (1, 1)),
    "BGGR": ((1, 1), (0, 1), (1, 0), (0, 0)),
    "GRBG": ((0, 1), (0, 0), (1, 1), (1, 0)),
    "GBRG": ((1, 0), (0, 0), (1, 1), (0, 1)),
}
_CHANNELS = (0, 1, 1, 2)  # color channel of each _LAYOUT entry
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


def bayer_lum(raw: np.ndarray) -> np.ndarray:
    """Full-resolution luminance of a Bayer mosaic: every 2x2 window holds one R, two G and
    one B whatever its phase, so a sliding 2x2 sum has no Bayer pattern (half a pixel off)."""
    r = raw.astype(np.float32)
    return r[..., :-1, :-1] + r[..., 1:, :-1] + r[..., :-1, 1:] + r[..., 1:, 1:]


def planet_centers(lum: np.ndarray) -> np.ndarray:
    """(n, 2) center of mass (y, x) of the bright disk in each frame of an (n, h, w) batch."""
    lo = lum.min(axis=(1, 2), keepdims=True)
    hi = lum.max(axis=(1, 2), keepdims=True)
    weight = np.where(lum > lo + DISK_THRESHOLD * (hi - lo), lum, 0)
    total = np.maximum(weight.sum(axis=(1, 2)), 1e-9)
    ys, xs = np.arange(lum.shape[1]), np.arange(lum.shape[2])
    return np.stack([(weight.sum(axis=2) * ys).sum(axis=1) / total,
                     (weight.sum(axis=1) * xs).sum(axis=1) / total], axis=1)


def planet_center(lum: np.ndarray) -> tuple[float, float]:
    y, x = planet_centers(lum[None])[0]
    return float(y), float(x)


def sharpness(lum: np.ndarray) -> np.ndarray:
    """Gradient energy of each smoothed frame of an (n, h, w) batch, normalized by its
    brightness so thin cloud or scintillation doesn't pass for sharpness. Smoothing first
    makes it rank seeing, not shot noise (raw Laplacian variance mostly measured noise)."""
    smooth = ndimage.gaussian_filter(lum, (0, QUALITY_SIGMA, QUALITY_SIGMA))
    gy, gx = np.gradient(smooth, axis=(1, 2))
    return (gy**2 + gx**2).mean(axis=(1, 2)) / np.maximum(smooth.mean(axis=(1, 2)), 1e-6) ** 2


def binned(frames: np.ndarray, start: int, stop: int) -> np.ndarray:
    """Frames start..stop summed over BIN x BIN blocks: cheap to search for the planet."""
    raw = np.asarray(frames[start:stop])
    n, h, w = raw.shape
    h, w = h // BIN * BIN, w // BIN * BIN
    return raw[:, :h, :w].reshape(n, h // BIN, BIN, w // BIN, BIN).sum(axis=(2, 4), dtype=np.float32)


def planet_extent(frames: np.ndarray) -> int:
    """Half-size (raw px) of a box holding the planet with its rings, from a few frames."""
    idx = np.linspace(0, len(frames) - 1, min(len(frames), 16)).astype(int)
    radii = []
    for img in (binned(frames, i, i + 1)[0] for i in idx):
        img = ndimage.gaussian_filter(img, 1)
        labels, n = ndimage.label(img > EXTENT_THRESHOLD * img.max())
        if n == 0 or img.max() <= 0:
            continue
        blob = labels == np.bincount(labels.ravel())[1:].argmax() + 1  # the planet
        ys, xs = np.nonzero(blob)
        radii.append(np.hypot(ys - ys.mean(), xs - xs.mean()).max() * BIN + BIN)
    return int(np.median(radii)) + 16 if radii else 32  # room for centering error


def crop(frame: np.ndarray, center: np.ndarray, half: int) -> tuple[np.ndarray, int, int] | None:
    """Raw square of side 2*half + 2 around center, starting on an even pixel so the Bayer
    phase is kept, and its corner (y0, x0); None if it doesn't fit inside the frame (planet
    at the ROI's edge)."""
    y0, x0 = ((round(c) - half) // 2 * 2 for c in center)
    if y0 < 0 or x0 < 0 or y0 + 2 * half + 2 > frame.shape[0] or x0 + 2 * half + 2 > frame.shape[1]:
        return None
    return frame[y0:y0 + 2 * half + 2, x0:x0 + 2 * half + 2], y0, x0


def sky_levels(small: np.ndarray, centers: np.ndarray, half: float) -> np.ndarray:
    """Mean of each (n, h, w) frame outside the box of half-size `half` around its center."""
    ys, xs = np.arange(small.shape[1]), np.arange(small.shape[2])
    off_y = np.abs(ys[None, :] - centers[:, :1]) > half
    off_x = np.abs(xs[None, :] - centers[:, 1:]) > half
    sky = off_y[:, :, None] | off_x[:, None, :]
    return (small * sky).sum(axis=(1, 2)) / np.maximum(sky.sum(axis=(1, 2)), 1)


def measure(frames: np.ndarray, half: int) -> tuple[np.ndarray, np.ndarray]:
    """Pass 1: per-frame planet center (raw px) and sharpness; -inf for unusable frames
    (planet missing, cut by the ROI edge as it drifts out of a non-tracking scope, or
    garbage off the planet)."""
    n = len(frames)
    centers, quality = np.zeros((n, 2)), np.full(n, -np.inf)
    peak, sky = np.zeros(n), np.zeros(n)
    for i in range(0, n, BATCH):
        small = ndimage.gaussian_filter(binned(frames, i, i + BATCH), (0, 1, 1))
        at = planet_centers(small)
        centers[i:i + len(small)] = at * BIN + (BIN - 1) / 2
        peak[i:i + len(small)] = small.max(axis=(1, 2))
        sky[i:i + len(small)] = sky_levels(small, at, half / BIN)
    present = peak > 0.3 * np.median(peak)
    present &= sky - np.median(sky[present]) < SKY_TOLERANCE * np.median(peak[present])
    for i in np.flatnonzero(present):
        cut = crop(frames[i], centers[i], half)
        if cut is not None:
            quality[i] = sharpness(bayer_lum(cut[0])[None])[0]
    return centers, quality


def register(images: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """(n, 2) subpixel shifts (dy, dx) that move each of (n, h, w) images onto ref: FFT
    cross-correlation peak, refined with a parabola through its neighbors. No window: the
    planet sits on black sky, and a fixed window would bias the shifts toward zero."""
    h, w = ref.shape

    def prep(a):
        return a - a.mean(axis=(-2, -1), keepdims=True)

    xc = fft.irfft2(fft.rfft2(prep(ref)) * np.conj(fft.rfft2(prep(images))), s=(h, w))
    n = len(images)
    peak = xc.reshape(n, -1).argmax(axis=1)
    py, px = peak // w, peak % w
    rows = np.arange(n)

    def vertex(m, c, p):  # subpixel offset of the parabola through (-1, 0, +1)
        d = m - 2 * c + p
        return np.where(d < 0, 0.5 * (m - p) / np.where(d < 0, d, -1), 0)

    c = xc[rows, py, px]
    dy = vertex(xc[rows, (py - 1) % h, px], c, xc[rows, (py + 1) % h, px])
    dx = vertex(xc[rows, py, (px - 1) % w], c, xc[rows, py, (px + 1) % w])
    sy = (py + h // 2) % h - h // 2 + dy  # wrap to signed shifts
    sx = (px + w // 2) % w - w // 2 + dx
    return np.stack([sy, sx], axis=1)


def shift_images(images: np.ndarray, shifts: np.ndarray) -> np.ndarray:
    """Shift each of (n, h, w) images by its (dy, dx), subpixel, in Fourier space."""
    h, w = images.shape[1:]
    ky = fft.fftfreq(h).astype(np.float32)[None, :, None]
    kx = fft.fftfreq(w).astype(np.float32)[None, None, :]
    dy = shifts[:, 0].astype(np.float32)[:, None, None]
    dx = shifts[:, 1].astype(np.float32)[:, None, None]
    phase = np.exp(np.complex64(-2j * np.pi) * (ky * dy + kx * dx))
    return fft.ifft2(fft.fft2(images.astype(np.float32)) * phase).real


def positions(frames: np.ndarray, keep: np.ndarray, centers: np.ndarray, half: int
              ) -> tuple[np.ndarray, np.ndarray]:
    """Pass 2: where the reference planet sits in each kept frame (raw px, subpixel), and
    the reference image. Registers on the whole planet against an average of the best
    frames, so clipped pixels or a moon can't drag it the way they drag a centroid."""
    cuts = [crop(frames[i], centers[i], half) for i in keep]
    lums = np.stack([bayer_lum(c[0]) for c in cuts])
    origins = np.array([(c[1], c[2]) for c in cuts], float)
    best = lums[:REF_FRAMES]  # keep is sorted best first
    ref = shift_images(best, register(best, best[0])).mean(axis=0)
    shifts = register(lums, ref)
    ref = shift_images(lums, shifts).mean(axis=0)  # sharper reference from all kept frames
    shifts = register(lums, ref)
    return origins - shifts, ref


def drizzle(frames: np.ndarray, keep: np.ndarray, origins: np.ndarray, bayer: str,
            shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Pass 3, Bayer drizzle: each raw pixel of each kept frame is splatted (bilinear) into
    its color channel of a SCALE x finer output grid. origins[k] is the raw position in
    frame keep[k] that lands on output pixel (0, 0). Every Bayer sub-grid of one frame
    shares one fractional offset, so the splat is 16 strided adds per frame.
    Returns (sum, clipped, weight), each (3, h, w); `clipped` sums the weight of saturated
    samples. Pixels off the frame simply add nothing."""
    oh, ow = shape
    pad = 3 * SCALE + 2  # room for the splats of samples just outside the box
    acc = np.zeros((3, oh + 2 * pad, ow + 2 * pad), np.float32)
    wgt, clipped = np.zeros_like(acc), np.zeros_like(acc)
    h, w = frames.shape[1:]
    for i, (oy, ox) in zip(keep, origins, strict=True):
        y0, x0 = (max(int(np.floor(o)) - 1, 0) // 2 * 2 for o in (oy, ox))  # even: Bayer phase
        y1, x1 = min(int(oy + oh / SCALE) + 2, h), min(int(ox + ow / SCALE) + 2, w)
        if y1 <= y0 + 1 or x1 <= x0 + 1:
            continue
        raw = frames[i, y0:y1, x0:x1].astype(np.float32)
        for c, (py, px) in zip(_CHANNELS, _LAYOUT[bayer], strict=True):
            sub = raw[py::2, px::2]
            sat = (sub >= SATURATED).astype(np.float32)
            fy, fx = (y0 + py - oy) * SCALE + pad, (x0 + px - ox) * SCALE + pad
            iy, ix = int(np.floor(fy)), int(np.floor(fx))
            for ay, wy in ((0, 1 - (fy - iy)), (1, fy - iy)):
                for ax, wx in ((0, 1 - (fx - ix)), (1, fx - ix)):
                    rows = slice(iy + ay, iy + ay + 2 * SCALE * sub.shape[0], 2 * SCALE)
                    cols = slice(ix + ax, ix + ax + 2 * SCALE * sub.shape[1], 2 * SCALE)
                    acc[c, rows, cols] += wy * wx * sub
                    clipped[c, rows, cols] += wy * wx * sat
                    wgt[c, rows, cols] += wy * wx
    inner = (slice(None), slice(pad, -pad), slice(pad, -pad))
    return acc[inner], clipped[inner], wgt[inner]


def normalize(acc: np.ndarray, wgt: np.ndarray) -> np.ndarray:
    """Drizzle sum / weight as (h, w, 3), through a small Gaussian drop kernel: point drops
    left a cross-hatch where the dither covered pixels unevenly (2026-10-10), and the
    seeing blur is far wider than one output pixel, so this costs no real detail. Thinly
    covered pixels (few frames, frame edges) get a wider kernel instead of holes."""
    def kernel(sigma):
        blur = (0, sigma, sigma)
        return ndimage.gaussian_filter(acc, blur), ndimage.gaussian_filter(wgt, blur)

    (a, w), (wide_a, wide_w) = kernel(SCALE / 2), kernel(2 * SCALE)
    thin = w < 0.3 * w.max(axis=(1, 2), keepdims=True)
    out = np.where(thin, wide_a / np.maximum(wide_w, 1e-6), a / np.maximum(w, 1e-6))
    return out.transpose(1, 2, 0)


def stack(frames: np.ndarray, bayer: str) -> tuple[np.ndarray, np.ndarray, int]:
    """Pick the sharpest frames, register them and drizzle them. `frames` may be a memmap.
    Returns float RGB at SCALE x the raw resolution cropped around the planet, the fraction
    of each pixel's samples that were saturated, and the number of frames used."""
    half = planet_extent(frames)
    centers, quality = measure(frames, half)
    usable = np.flatnonzero(np.isfinite(quality))
    if len(usable) < MIN_FRAMES:
        raise ValueError(f"the planet is clear in only {len(usable)} frames")
    count = min(len(usable), max(MIN_FRAMES, int(len(frames) * KEEP_FRACTION)))
    keep = usable[np.argsort(quality[usable])[::-1][:count]]  # best first
    origins, ref = positions(frames, keep, centers, half)
    # The picture: the planet's box, grown for moons; corner in reference-crop raw px.
    # (+0.5: the 2x2-sum luminance sits half a pixel off the raw grid.)
    center = np.array(planet_center(ref)) + 0.5
    reach = (half - 8) * CROP_PAD
    top, left, bottom, right = moon_margin(frames, keep[0], centers[keep[0]], reach)
    corner = center - reach - (top, left)
    shape = (int((2 * reach + top + bottom) * SCALE), int((2 * reach + left + right) * SCALE))
    acc, clipped, wgt = drizzle(frames, keep, origins + corner, bayer, shape)
    return normalize(acc, wgt), normalize(clipped, wgt), len(keep)


def moon_margin(frames: np.ndarray, best: int, center: np.ndarray, reach: float
                ) -> tuple[float, float, float, float]:
    """(top, left, bottom, right) raw px to grow the planet's box by so every moon stays in
    the picture (2026-10-10: Jupiter's moons belong in the picture)."""
    lum = superpixel_rgb(np.asarray(frames[best]), "RGGB").mean(axis=-1)  # phase is irrelevant
    cy, cx = center / 2
    grow = np.zeros(4)
    for mx, my in companions(lum, (cx, cy), max(lum.shape)):
        dy, dx = (my - cy) * 2, (mx - cx) * 2  # raw px from the planet
        pad = MOON_PAD * reach
        grow = np.maximum(grow, (-dy + pad - reach, -dx + pad - reach,
                                 dy + pad - reach, dx + pad - reach))
    return tuple(float(g) for g in grow)


def align_channels(img: np.ndarray, clipped: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Register red and blue on green (atmospheric dispersion: the planet was low); the
    clipped-fraction maps move with their channels."""
    chans = np.moveaxis(img, -1, 0)
    shifts = register(chans, chans[1])
    return tuple(np.moveaxis(shift_images(np.moveaxis(a, -1, 0), shifts), 0, -1)
                 for a in (img, clipped))


def repair_clipping(img: np.ndarray, clipped: np.ndarray) -> np.ndarray:
    """Rebuild channels that were saturated in many frames from the least clipped channel
    times their usual ratio to it. 2026-10-10: red clipped on most of Saturn's disk in
    80-100% of frames (the metering's 99.9th percentile mixes all Bayer channels), which
    turned the stacked disk flat and cyan."""
    ref = int(np.argmin(clipped.sum(axis=(0, 1))))
    lum = img[..., ref]
    bright = lum > DISK_THRESHOLD * lum.max()
    clean = bright & (clipped.max(axis=2) < 0.01)
    if clean.sum() < 10:
        return img
    out = img.copy()
    for c in range(3):
        ratio = np.median(img[clean, c] / np.maximum(lum[clean], 1e-6))
        # Grown, then feathered: the drizzle kernel spread clipping a little past the mask.
        grown = ndimage.maximum_filter((clipped[..., c] / CLIP_TOLERANCE).clip(0, 1), 2 * SCALE + 1)
        blend = ndimage.gaussian_filter(grown, SCALE)
        out[..., c] = (1 - blend) * img[..., c] + blend * ratio * lum
    return out


def white_balance(img: np.ndarray) -> np.ndarray:
    """Scale R and B so the planet's median color is neutral: the raw sensor colors (no
    color correction) made Saturn pink."""
    lum = img.mean(axis=2)
    disk = lum > DISK_THRESHOLD * lum.max()
    med = np.median(img[disk], axis=0)
    return img * (med[1] / np.maximum(med, 1e-6))


def wavelet_sharpen(img: np.ndarray) -> np.ndarray:
    """A trous (B3 spline) wavelet sharpening, like Registax: boost the layers that carry
    detail, soft-thresholding each by its noise so the boost doesn't amplify grain."""
    smooth, out = img, np.zeros_like(img)
    for level, (gain, k) in enumerate(zip(WAVELET_GAINS, WAVELET_NOISE, strict=True)):
        coarser = _atrous(_atrous(smooth, 2**level, 0), 2**level, 1)
        layer = smooth - coarser
        noise = 1.4826 * np.median(np.abs(layer), axis=(0, 1))  # robust, per channel
        out += gain * np.sign(layer) * np.maximum(np.abs(layer) - k * noise, 0)
        smooth = coarser
    return out + smooth


def _atrous(a: np.ndarray, step: int, axis: int) -> np.ndarray:
    """The B3 spline kernel with `step - 1` holes between its taps, along one axis: only the 5
    taps are added (convolving the dilated kernel, mostly zeros, took 37 s on a big crop)."""
    pad = min(2 * step, a.shape[axis] - 1)
    p = np.pad(a, [(pad, pad) if ax == axis else (0, 0) for ax in range(a.ndim)], mode="symmetric")
    n = a.shape[axis]
    return sum(w * np.take(p, np.clip(np.arange(n) + pad + (i - 2) * step, 0, n + 2 * pad - 1), axis=axis)
               for i, w in enumerate(B3_KERNEL))


def finish(img: np.ndarray, clipped: np.ndarray) -> np.ndarray:
    """Align colors, repair clipping, balance, sharpen, stretch to 8-bit without clipping
    the disk."""
    img = wavelet_sharpen(white_balance(repair_clipping(*align_channels(img, clipped))))
    lum = ndimage.gaussian_filter(img.mean(axis=2), SCALE)
    edge = np.concatenate([lum[:4].ravel(), lum[-4:].ravel(), lum[:, :4].ravel(), lum[:, -4:].ravel()])
    lo = np.median(edge)  # the sky at the picture's edge is black (sharpening halos go below)
    hi = lum.max()  # the smoothed disk's brightest level stays just below white
    return ((img - lo) / max(hi - lo, 1e-6) * 240).clip(0, 255).astype(np.uint8)


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
    img, clipped, used = stack(frames, bayer)
    Image.fromarray(finish(img, clipped)).save(path)
    return StackResult(path, len(frames), used)
