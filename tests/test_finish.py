import numpy as np
from astropy.io import fits

from astro.process.finish import balance_color, finish, remove_gradient, save_fits


def field(rng, size=256):
    """Faint nebula and stars on a sky with a light-pollution gradient and an orange cast."""
    y, x = np.mgrid[:size, :size] / size
    sky = 40 + 60 * x + 20 * y  # brighter toward a city
    rgb = np.stack([sky * 1.3, sky * 1.0, sky * 0.7], axis=-1)  # sodium-orange glow
    nebula = 15 * np.exp(-((x - 0.5) ** 2 + (y - 0.5) ** 2) / 0.01)
    rgb += nebula[..., None] * [0.8, 1.0, 1.2]
    for sy, sx in rng.integers(10, size - 10, (40, 2)):
        rgb[sy - 1:sy + 2, sx - 1:sx + 2] += 400 * np.array([1.3, 1.0, 0.7])  # same cast
    return rgb + rng.normal(0, 1, rgb.shape)


def test_gradient_is_flattened():
    flat = remove_gradient(field(np.random.default_rng(0)))
    corners = [np.median(flat[5:30, 5:30]), np.median(flat[5:30, -30:-5]),
               np.median(flat[-30:-5, -30:-5])]  # medians: a star may land in a corner
    assert np.ptp(corners) < 5  # was ~70 across the frame


def test_color_cast_is_removed():
    balanced = balance_color(np.clip(remove_gradient(field(np.random.default_rng(1))), 0, None))
    lum = balanced.mean(axis=-1)
    stars = balanced[lum > np.percentile(lum, 99.5)].mean(axis=0)
    assert np.ptp(stars) / stars.mean() < 0.05  # white stars


def test_finish_gives_8bit_with_the_nebula_above_the_sky():
    out = finish(field(np.random.default_rng(2)))
    assert out.dtype == np.uint8 and out.shape == (256, 256, 3)
    assert out[128, 128].mean() > out[20, 128].mean() + 10


def test_fits_is_16bit_channels_first(tmp_path):
    save_fits(np.random.default_rng(3).random((20, 30, 3)), tmp_path / "s.fits")
    data = fits.getdata(tmp_path / "s.fits")
    assert data.shape == (3, 20, 30) and data.dtype.kind in "iu"
