import numpy as np
from PIL import Image
from scipy import ndimage

from astro.capture.focus import laplacian_variance
from astro.capture.ser import SerWriter
from astro.process.planet import _LAYOUT, process_ser, stack, superpixel_rgb


def planet_scene(size=96, cx=48.0, cy=48.0, r=20):
    """RGB planet with bands (more red than blue, like Saturn/Jupiter)."""
    y, x = np.mgrid[0:size, 0:size]
    disk = ((x - cx) ** 2 + (y - cy) ** 2 <= r**2).astype(float)
    bands = 160 + 50 * np.sin((y - cy) / r * 9)
    return np.dstack([disk * bands, disk * bands * 0.85, disk * bands * 0.6])


def mosaic(rgb, bayer="GRBG"):
    """RGB -> Bayer raw (what the camera records)."""
    raw = np.zeros(rgb.shape[:2])
    (ry, rx), (g1y, g1x), (g2y, g2x), (by, bx) = _LAYOUT[bayer]
    raw[ry::2, rx::2] = rgb[ry::2, rx::2, 0]
    raw[g1y::2, g1x::2] = rgb[g1y::2, g1x::2, 1]
    raw[g2y::2, g2x::2] = rgb[g2y::2, g2x::2, 1]
    raw[by::2, bx::2] = rgb[by::2, bx::2, 2]
    return raw


def seeing_frames(n=40, seed=0):
    """Frames jittering a few px with random blur and noise, like a planet through air."""
    rng = np.random.default_rng(seed)
    frames = []
    for _ in range(n):
        dy, dx = rng.normal(0, 3, 2)
        scene = planet_scene(cx=48 + dx, cy=48 + dy)
        scene = ndimage.gaussian_filter(scene, (rng.uniform(0.5, 3.5),) * 2 + (0,))
        frames.append((mosaic(scene) + rng.normal(8, 6, scene.shape[:2])).clip(0, 255))
    return np.array(frames, np.uint8)


def test_superpixel_puts_colors_in_place():
    raw = mosaic(planet_scene()).astype(np.uint8)
    rgb = superpixel_rgb(raw, "GRBG")
    r, g, b = (rgb[..., i].sum() for i in range(3))
    assert r > g > b


def test_stack_is_sharper_and_cleaner_than_typical_frame():
    frames = seeing_frames()
    result = stack(frames, "GRBG").mean(axis=2)
    singles = [superpixel_rgb(f, "GRBG").mean(axis=2) for f in frames]
    typical = sorted(singles, key=laplacian_variance)[len(singles) // 2]
    sky = (slice(0, 8), slice(0, 8))
    assert result[sky].std() < typical[sky].std() / 1.5  # averaging cuts noise
    edge = (np.abs(np.diff(result, axis=1)).max(), np.abs(np.diff(typical, axis=1)).max())
    assert edge[0] > edge[1] * 0.8  # alignment kept the disk edge crisp (no smear)


def test_process_ser_writes_png(tmp_path):
    frames = seeing_frames(12)
    with SerWriter(tmp_path / "saturn.ser", 96, 96, bayer="GRBG") as w:
        for f in frames:
            w.write(f)
    result = process_ser(tmp_path / "saturn.ser", tmp_path / "gallery")
    img = np.asarray(Image.open(result.path))
    assert result.frames_total == 12 and result.frames_used == 3
    assert img.ndim == 3 and img.max() == 255 and img.shape[0] >= 32


def test_align_channels_removes_color_offset():
    from astro.process.planet import align_channels, planet_center

    img = planet_scene()
    img[..., 0] = ndimage.shift(img[..., 0], (0, 3))  # red displaced 3 px (dispersion)
    out = align_channels(img)
    assert abs(planet_center(out[..., 0])[1] - planet_center(out[..., 1])[1]) < 0.2
