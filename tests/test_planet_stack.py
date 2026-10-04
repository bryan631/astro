import numpy as np
from PIL import Image
from scipy import ndimage

from astro.capture.ser import SerWriter
from astro.process.planet import (
    _LAYOUT,
    align_channels,
    planet_center,
    process_ser,
    sharpness,
    stack,
    superpixel_rgb,
)

SIZE = 96


def planet_scenes(cx, cy, r=20):
    """(n, h, w, 3) banded planets at per-frame centers (more red than blue, like Saturn)."""
    y, x = np.mgrid[0:SIZE, 0:SIZE]
    cx, cy = np.asarray(cx, float)[:, None, None], np.asarray(cy, float)[:, None, None]
    disk = (x - cx) ** 2 + (y - cy) ** 2 <= r**2
    lum = disk * (160 + 50 * np.sin((y - cy) / r * 9))
    return lum[..., None] * np.array([1.0, 0.85, 0.6])


def mosaic(rgb, bayer="GRBG"):
    """(..., h, w, 3) RGB -> (..., h, w) Bayer raw, as the camera records it."""
    raw = np.zeros(rgb.shape[:-1])
    (ry, rx), (g1y, g1x), (g2y, g2x), (by, bx) = _LAYOUT[bayer]
    raw[..., ry::2, rx::2] = rgb[..., ry::2, rx::2, 0]
    raw[..., g1y::2, g1x::2] = rgb[..., g1y::2, g1x::2, 1]
    raw[..., g2y::2, g2x::2] = rgb[..., g2y::2, g2x::2, 1]
    raw[..., by::2, bx::2] = rgb[..., by::2, bx::2, 2]
    return raw


def seeing_frames(n=40, seed=0):
    """Planet jittering ~3 px with per-frame blur and noise, like seeing; built vectorized."""
    rng = np.random.default_rng(seed)
    dy, dx = rng.normal(0, 3, (2, n))
    scenes = planet_scenes(48 + dx, 48 + dy)
    sigma = rng.uniform(0.5, 3.5, n)[:, None, None, None]
    k2 = np.fft.fftfreq(SIZE)[:, None] ** 2 + np.fft.fftfreq(SIZE)[None, :] ** 2
    blur = np.exp(-2 * np.pi**2 * sigma**2 * k2[None, :, :, None])  # Gaussian, per frame
    scenes = np.fft.ifft2(np.fft.fft2(scenes, axes=(1, 2)) * blur, axes=(1, 2)).real
    return (mosaic(scenes) + rng.normal(8, 6, (n, SIZE, SIZE))).clip(0, 255).astype(np.uint8)


def test_superpixel_puts_colors_in_place():
    rgb = superpixel_rgb(mosaic(planet_scenes([48], [48])[0]).astype(np.uint8), "GRBG")
    r, g, b = (rgb[..., i].sum() for i in range(3))
    assert r > g > b


def test_batch_sharpness_matches_scipy_laplacian():
    lum = superpixel_rgb(seeing_frames(4), "GRBG").mean(axis=-1)
    inner = [ndimage.laplace(f)[1:-1, 1:-1].var() for f in lum]  # reference, frame by frame
    assert np.allclose(sharpness(lum), inner, rtol=1e-4)


def test_stack_is_sharper_and_cleaner_than_typical_frame():
    frames = seeing_frames()
    result, used = stack(frames, "GRBG")
    result = result.mean(axis=2)
    singles = superpixel_rgb(frames, "GRBG").mean(axis=-1)
    typical = singles[np.argsort(sharpness(singles))[len(singles) // 2]]
    sky = (slice(0, 8), slice(0, 8))
    assert used == 10
    assert result[sky].std() < typical[sky].std() / 1.5  # averaging cuts noise
    edge = (np.abs(np.diff(result, axis=1)).max(), np.abs(np.diff(typical, axis=1)).max())
    assert edge[0] > edge[1] * 0.8  # alignment kept the disk edge crisp (no smear)


def test_align_channels_removes_color_offset():
    img = planet_scenes([48], [48])[0]
    img[..., 0] = ndimage.shift(img[..., 0], (0, 3))  # red displaced 3 px (dispersion)
    out = align_channels(img)
    assert abs(planet_center(out[..., 0])[1] - planet_center(out[..., 1])[1]) < 0.2


def test_process_ser_writes_png(tmp_path):
    with SerWriter(tmp_path / "saturn.ser", SIZE, SIZE, bayer="GRBG") as w:
        for f in seeing_frames(12):
            w.write(f)
    result = process_ser(tmp_path / "saturn.ser", tmp_path / "gallery")
    img = np.asarray(Image.open(result.path))
    assert result.frames_total == 12 and result.frames_used == 3
    assert img.ndim == 3 and img.max() == 255 and img.shape[0] >= 32
