import numpy as np
from PIL import Image
from scipy import ndimage

from astro.capture.ser import SerWriter
from astro.process.planet import (
    _LAYOUT,
    SCALE,
    align_channels,
    bayer_lum,
    blur_sigma,
    drizzle,
    measure,
    moon_margin,
    normalize,
    planet_center,
    planet_extent,
    process_ser,
    register,
    repair_clipping,
    sharpen,
    sharpness,
    shift_images,
    stack,
    superpixel_rgb,
)

SIZE = 128
COLOR = np.array([1.0, 0.85, 0.6])  # more red than blue, like Saturn


def planet_scenes(cx, cy, r=20, size=SIZE):
    """(n, h, w, 3) banded planets at per-frame centers."""
    y, x = np.mgrid[0:size, 0:size]
    cx, cy = np.asarray(cx, float)[:, None, None], np.asarray(cy, float)[:, None, None]
    disk = (x - cx) ** 2 + (y - cy) ** 2 <= r**2
    lum = disk * (160 + 50 * np.sin((y - cy) / r * 9))
    return lum[..., None] * COLOR


def mosaic(rgb, bayer="GRBG"):
    """(..., h, w, 3) RGB -> (..., h, w) Bayer raw, as the camera records it."""
    raw = np.zeros(rgb.shape[:-1])
    (ry, rx), (g1y, g1x), (g2y, g2x), (by, bx) = _LAYOUT[bayer]
    raw[..., ry::2, rx::2] = rgb[..., ry::2, rx::2, 0]
    raw[..., g1y::2, g1x::2] = rgb[..., g1y::2, g1x::2, 1]
    raw[..., g2y::2, g2x::2] = rgb[..., g2y::2, g2x::2, 1]
    raw[..., by::2, bx::2] = rgb[..., by::2, bx::2, 2]
    return raw


def blur(scenes, sigma):
    """Gaussian blur of (n, h, w, 3) scenes, a different sigma per frame (seeing)."""
    size = scenes.shape[1]
    k2 = np.fft.fftfreq(size)[:, None] ** 2 + np.fft.fftfreq(size)[None, :] ** 2
    otf = np.exp(-2 * np.pi**2 * np.asarray(sigma)[:, None, None, None] ** 2 * k2[None, :, :, None])
    return np.fft.ifft2(np.fft.fft2(scenes, axes=(1, 2)) * otf, axes=(1, 2)).real


def seeing_frames(n=40, seed=0):
    """Planet jittering ~3 px with per-frame blur and noise, like seeing."""
    rng = np.random.default_rng(seed)
    dy, dx = rng.normal(0, 3, (2, n))
    scenes = blur(planet_scenes(64 + dx, 64 + dy), rng.uniform(0.5, 3.5, n))
    return (mosaic(scenes) + rng.normal(8, 6, (n, SIZE, SIZE))).clip(0, 255).astype(np.uint8)


def test_superpixel_puts_colors_in_place():
    rgb = superpixel_rgb(mosaic(planet_scenes([48], [48])[0]).astype(np.uint8), "GRBG")
    r, g, b = (rgb[..., i].sum() for i in range(3))
    assert r > g > b


def test_bayer_lum_has_no_bayer_pattern():
    flat = mosaic(np.full((1, 16, 16, 3), 100.0) * COLOR)[0]  # uniform colored field
    assert np.ptp(bayer_lum(flat)) < 1e-3


def test_sharpness_prefers_sharp_frames_whatever_their_brightness():
    scenes = blur(planet_scenes([48, 48], [48, 48]), [0.5, 3.0]).mean(axis=-1)
    sharp, soft = sharpness(scenes)
    assert sharp > 1.5 * soft
    assert np.isclose(sharpness(scenes[:1] * 0.5)[0], sharp, rtol=1e-3)  # thin cloud


def test_register_recovers_subpixel_shift():
    ref = blur(planet_scenes([48], [48]), [2.0])[0].mean(axis=-1)
    true = np.array([[1.3, -2.6], [-0.4, 0.25]])
    moved = shift_images(np.stack([ref, ref]), -true)  # shifted away; register undoes it
    assert np.abs(register(moved, ref) - true).max() < 0.05


def test_drizzle_rebuilds_the_scene_at_twice_the_resolution():
    """Frames dithered by known subpixel offsets drizzle back onto the true scene."""
    rng = np.random.default_rng(1)
    offsets = rng.uniform(-6, 6, (60, 2))
    scene = blur(planet_scenes([0.0], [0.0], size=SIZE * SCALE), [6.0])[0]  # planet at 0, 0

    def look(y, x):  # the scene at continuous raw-pixel coordinates (planet centered)
        iy = np.round(y * SCALE).astype(int) % (SIZE * SCALE)
        ix = np.round(x * SCALE).astype(int) % (SIZE * SCALE)
        return scene[iy[:, None], ix[None, :]]

    grid = np.arange(SIZE, dtype=float)
    frames = np.stack([mosaic(look(grid - 48 - dy, grid - 48 - dx)[None])[0] for dy, dx in offsets])
    corner = -24.0  # output (0, 0) sits 24 raw px up-left of the planet
    acc, clipped, wgt = drizzle(frames, np.arange(60), 48 + offsets + corner, "GRBG", (96, 96))
    out = normalize(acc, wgt)
    truth = look(corner + np.arange(96) / SCALE, corner + np.arange(96) / SCALE)
    inner = (slice(8, -8), slice(8, -8))
    assert np.abs(out[inner] - truth[inner]).max() < 0.06 * truth.max()
    assert clipped.max() == 0


def test_stack_is_sharp_and_cleaner_than_a_single_frame():
    frames = seeing_frames()
    result, _, noise, used = stack(frames, "GRBG")
    g = result[..., 1]
    assert np.abs(noise[..., 1]).mean() < 0.02 * g.max()  # noise only: the planet cancels out
    assert used == 8 and result.shape[0] > 2 * 40  # 20% of the frames; 2x the planet's size
    single = superpixel_rgb(frames[0], "GRBG")[..., 1]
    assert g[4:24, 4:24].std() < single[2:12, 2:12].std() / 2  # averaging cuts the sky's noise
    assert abs(planet_center(g)[0] - g.shape[0] / 2) < 3  # planet centered in the crop
    assert np.abs(np.diff(g, axis=0)).max() > 0.2 * g.max()  # bands and limb stay crisp


def test_streaked_frames_are_dropped():
    """2026-10-12: the sensor streaked the planet's rows to its right in some frames, and the
    streaks, passing for detail, ranked sharpest."""
    rng = np.random.default_rng(3)
    scenes = blur(planet_scenes(np.full(10, 128.0), np.full(10, 128.0), size=256), np.full(10, 2.0))
    frames = mosaic(scenes) + rng.normal(8, 3, (10, 256, 256))
    frames[:3, 120:136, 160:] += (np.arange(96) % 16 < 8) * 10.0  # blocks along the planet's rows
    frames = frames.clip(0, 255).astype(np.uint8)
    _, quality = measure(frames, planet_extent(frames))
    assert np.isinf(quality[:3]).all() and np.isfinite(quality[3:]).all()


def test_blur_sigma_measures_the_blur():
    for sigma in (1.5, 4.0):
        lum = blur(planet_scenes([64], [64]), [sigma])[0].mean(axis=-1)
        assert abs(blur_sigma(lum) - sigma) < 0.2 * sigma


def test_sharpen_restores_detail_without_halos_or_grain():
    """Two half-stacks of a blurred, noisy planet: sharpening brings it closer to the truth,
    the sky stays black (no halo below it) and its noise isn't amplified."""
    rng = np.random.default_rng(4)
    truth = planet_scenes([64], [64])[0].mean(axis=-1)
    blurred = blur(truth[None, ..., None], [3.0])[0, ..., 0]
    a, b = (blurred + rng.normal(0, 2, truth.shape) for _ in range(2))
    lum, noise = (a + b) / 2, (a - b) / 2
    out = sharpen(lum, noise)
    disk = ndimage.binary_dilation(truth > 0, iterations=4)
    assert np.abs(out - truth)[disk].mean() < 0.8 * np.abs(lum - truth)[disk].mean()
    assert out.min() >= 0
    sky = ~ndimage.binary_dilation(truth > 0, iterations=12)
    assert out[sky].std() < lum[sky].std()


def test_align_channels_removes_color_offset():
    img = blur(planet_scenes([48], [48]), [1.0])[0]
    img[..., 0] = ndimage.shift(img[..., 0], (0, 3))  # red displaced 3 px (dispersion)
    out, _ = align_channels(img, np.zeros_like(img))
    assert abs(planet_center(out[..., 0])[1] - planet_center(out[..., 1])[1]) < 0.2


def test_repair_clipping_rebuilds_red_from_green():
    """2026-10-10: red clipped on Saturn's disk; the stacked disk came out cyan."""
    img = planet_scenes([64], [64])[0]
    truth = img[..., 0].copy()
    clipped = np.zeros_like(img)
    core = (img[..., 1] > 0) & (np.hypot(*np.mgrid[-64:64, -64:64]) < 14)
    img[core, 0] = 150.0  # flat: saturated
    clipped[core, 0] = 1.0
    out = repair_clipping(img, clipped)
    assert np.abs(out[core, 0] - truth[core]).mean() < 0.1 * np.abs(img[core, 0] - truth[core]).mean()


def test_moon_margin_grows_the_box_toward_the_moon():
    """2026-10-10: Jupiter's moons belong in the picture."""
    frame = np.zeros((200, 400), np.uint8)
    y, x = np.mgrid[0:200, 0:400]
    frame[(x - 200) ** 2 + (y - 100) ** 2 <= 20**2] = 180  # the disk
    frame[98:104, 337:343] = 120  # a moon 140 px right of it
    top, left, _, right = moon_margin(frame[None], 0, np.array([100.0, 200.0]), 40)
    assert right > 100 and top == left == 0


def test_process_ser_writes_png(tmp_path):
    with SerWriter(tmp_path / "saturn.ser", SIZE, SIZE, bayer="GRBG") as w:
        for f in seeing_frames(12):
            w.write(f)
    result = process_ser(tmp_path / "saturn.ser", tmp_path / "gallery")
    img = np.asarray(Image.open(result.path))
    assert result.frames_total == 12 and result.frames_used == 3
    assert img.ndim == 3 and img.shape[0] > 2 * 40 and 200 < img.max() <= 255
