import numpy as np
from astropy.io import fits

from astro.process.finish import balance_color, denoise, finish, remove_gradient, save_fits


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
    rgb = np.zeros((20, 30, 3))
    rgb[0, 0] = [1.0, 0.5, 0.25]  # top-left pixel: brightest, so it scales to 65535
    save_fits(rgb, tmp_path / "s.fits")
    data = fits.getdata(tmp_path / "s.fits")
    assert data.shape == (3, 20, 30)
    assert list(data[:, -1, 0]) == [65535, 32767, 16383]  # top row stored last (bottom-up)
    assert data[:, :-1].max() == 0 and data[:, -1, 1:].max() == 0


def test_a_bright_nebula_keeps_its_color():
    rng = np.random.default_rng(5)
    size = 128
    y, x = np.mgrid[:size, :size] / size
    nebula = 200 * np.exp(-((x - 0.5) ** 2 + (y - 0.5) ** 2) / 0.02)
    rgb = np.stack([nebula * 1.0, nebula * 0.3, nebula * 0.3], axis=-1) + 10  # red emission
    for sy, sx in rng.integers(5, size - 5, (30, 2)):
        rgb[sy, sx] += 300  # white stars
    out = balance_color(rgb)
    center = out[60:68, 60:68].mean(axis=(0, 1))
    assert center[0] > 2 * center[1]  # still red


def test_flat_frame_is_left_alone():
    flat = np.full((32, 32, 3), 7.0)
    assert np.array_equal(balance_color(flat), flat)


def test_denoise_smooths_sky_and_keeps_stars():
    rng = np.random.default_rng(6)
    rgb = 10 + rng.normal(0, 2, (64, 64, 3))
    rgb[32, 32] += 200  # a star
    out = denoise(rgb)
    assert out[:20, :20].std() < rgb[:20, :20].std() / 2
    assert out[32, 32].mean() > 0.9 * rgb[32, 32].mean()


def test_denoise_keeps_a_faint_star_next_to_a_bright_nebula():
    rng = np.random.default_rng(7)
    rgb = 10 + rng.normal(0, 2, (64, 64, 3))
    rgb[5:25, 5:25] += 150  # extended bright target in one corner
    rgb[48, 48] += 20  # faint star: 10 sigma
    out = denoise(rgb)
    assert out[48, 48].mean() - 10 > 0.8 * 20
    assert out[40:60, 30:45].std() < rgb[40:60, 30:45].std() / 1.5  # sky still smoothed
