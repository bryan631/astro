import io

import numpy as np
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
