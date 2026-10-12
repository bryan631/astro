import io

import numpy as np
import pytest
from PIL import Image

from astro.process.view import remove_hot_pixels, render


def decode(raw):
    return np.asarray(Image.open(io.BytesIO(render(raw, "GRBG")[0])))


def test_dark_noise_stays_dark_and_stars_show():
    rng = np.random.default_rng(0)
    dark = rng.normal(20, 3, (960, 1280)).clip(0, 255).astype(np.uint8)
    assert decode(dark).mean() < 60  # noise isn't stretched to full scale
    sky = dark.copy()
    sky[400:404, 600:604] = 250  # a star
    assert decode(sky).max() > 200 and decode(sky).shape == (480, 640, 3)


def test_saturated_frame_is_white_not_black():
    img = Image.open(io.BytesIO(render(np.full((64, 64), 255, np.uint8), "GRBG")[0]))
    assert np.median(np.asarray(img)) > 200  # (the corner has the sharpness text)


def test_zoom_crops_the_center():
    raw = np.zeros((400, 800), np.uint8)
    raw[150:250, 300:500] = 200  # a bright center fills the 4x view
    zoomed = Image.open(io.BytesIO(render(raw, "GRBG", zoom=4)[0]))
    assert zoomed.size == (400, 200) and np.asarray(zoomed)[100:, :].mean() > 150


def test_hot_pixel_removed_star_kept():
    rgb = np.random.default_rng(0).normal(20, 2, (40, 40, 3))
    rgb[10, 10] = 200  # hot pixel: alone
    rgb[29:32, 29:32] = 150  # star: several pixels
    out = remove_hot_pixels(rgb)
    assert out[10, 10].max() < 40 and out[30, 30].min() == 150


def test_zoom_crops_around_a_center_kept_inside_the_frame():
    raw = np.zeros((96, 128), np.uint8)
    raw[:48, 64:] = 200  # the sensor's top-right quarter is bright
    _, m = render(raw, "RGGB", zoom=2, rotate=0, night=False, center=(0.3, -0.1))
    assert m["center"] == pytest.approx((0.25, -0.1), abs=0.02)  # x clamped: the crop stays inside
    data, m = render(raw, "RGGB", zoom=2, rotate=180, night=False, center=(-0.25, 0.25))
    assert m["center"] == pytest.approx((-0.25, 0.25), abs=0.02)
    img = np.asarray(Image.open(io.BytesIO(data)))
    assert img.mean() > 150  # turned 180, the view's top-left is the sensor's bright top-right


def mosaic(rgb):
    """An RGB image as a GRBG raw frame, one superpixel per pixel."""
    h, w, _ = rgb.shape
    raw = np.empty((2 * h, 2 * w), np.uint8)
    raw[0::2, 0::2] = raw[1::2, 1::2] = rgb[..., 1]
    raw[0::2, 1::2], raw[1::2, 0::2] = rgb[..., 0], rgb[..., 2]
    return raw


def star_field(noise, stars, size=(240, 320)):
    """A sky with the given noise and stars: (y, x, (r, g, b) peak above the sky), Gaussian.
    Red and blue ~1.7x green, as the main camera's sky."""
    rng = np.random.default_rng(1)
    rgb = rng.normal((75, 45, 75), noise, (*size, 3))
    yy, xx = np.mgrid[:size[0], :size[1]]
    for y, x, peak in stars:
        rgb += np.exp(-((yy - y) ** 2 + (xx - x) ** 2) / 2)[..., None] * np.array(peak)
    return mosaic(rgb.clip(0, 255).astype(np.uint8))


def night_view(raw):
    return np.asarray(Image.open(io.BytesIO(render(raw, "GRBG", rotate=0, night=True)[0]))).astype(float)


def test_a_noisy_sky_with_few_stars_stays_dark():
    """The main camera at high gain: few stars, so its noise used to be stretched gray."""
    view = night_view(star_field(14, [(60, 80, (150,) * 3), (180, 250, (120,) * 3)]))
    assert view.mean() < 32 and view.max() > 200  # was 42: a gray speckle


def test_stars_keep_their_colors_and_a_clipped_one_is_white():
    """Stars differ by R/B ~0.6-1.5 (2026-10-10 frames): shown boosted. A clipped core's color
    is lost (it reads the sky's backwards, green here): white."""
    rng = np.random.default_rng(2)
    white = [(y, x, (90, 90, 90)) for y, x in rng.integers(20, 220, (30, 2))]
    orange, blue, burnt = (100, 100, (90, 80, 70)), (100, 200, (70, 80, 90)), (180, 160, (400, 400, 400))
    view = night_view(star_field(2, [*white, orange, blue, burnt]))
    r, _, b = view[orange[:2]]
    assert r > 1.5 * b  # R/B 1.3: brought out
    r, _, b = view[blue[:2]]
    assert b > 1.5 * r
    assert view[burnt[:2]].min() > 0.9 * view[burnt[:2]].max()
