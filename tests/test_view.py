import io

import numpy as np
from PIL import Image

from astro.process.view import jpeg


def decode(raw):
    return np.asarray(Image.open(io.BytesIO(jpeg(raw, "GRBG"))))


def test_dark_noise_stays_dark_and_stars_show():
    rng = np.random.default_rng(0)
    dark = rng.normal(20, 3, (960, 1280)).clip(0, 255).astype(np.uint8)
    assert decode(dark).mean() < 60  # noise isn't stretched to full scale
    sky = dark.copy()
    sky[400:404, 600:604] = 250  # a star
    assert decode(sky).max() > 200 and decode(sky).shape == (480, 640, 3)


def test_saturated_frame_is_white_not_black():
    img = Image.open(io.BytesIO(jpeg(np.full((64, 64), 255, np.uint8), "GRBG")))
    assert np.asarray(img).min() > 200
