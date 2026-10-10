"""The focus number: higher is sharper, decided by the many faint stars, not one bright one."""

import itertools

import numpy as np
from scipy import ndimage

from astro.capture.focus import measure_focus
from astro.process.planet import superpixel_rgb

RNG = np.random.default_rng(3)


def field(n: int, sigma: float, peak: float, size=(400, 600), sky=20.0, noise=2.0, seed=0) -> np.ndarray:
    """A night-sky frame (8-bit scale) with n Gaussian stars of the given width and peak."""
    rng = np.random.default_rng(seed)
    img = np.zeros(size)
    ys, xs = rng.uniform(20, size[0] - 20, n), rng.uniform(20, size[1] - 20, n)
    for y, x in zip(ys.astype(int), xs.astype(int), strict=True):
        img[y, x] += 1.0
    img = ndimage.gaussian_filter(img, sigma) * (2 * np.pi * sigma**2) * peak
    return np.clip(img + sky + rng.normal(0, noise, size), 0, 255)


def add_star(img, y, x, sigma, peak):
    one = np.zeros_like(img)
    one[y, x] = 1.0
    return np.clip(img + ndimage.gaussian_filter(one, sigma) * 2 * np.pi * sigma**2 * peak, 0, 255)


def test_score_rises_as_stars_get_sharper():
    scores = [measure_focus(field(60, s, 60)).score for s in (4.0, 2.5, 1.5, 0.9)]
    assert all(a < b for a, b in itertools.pairwise(scores)), scores


def test_faint_stars_decide_not_one_bright_star():
    sharp = measure_focus(field(60, 1.0, 40)).score
    soft = measure_focus(field(60, 3.0, 40)).score
    with_bright_soft = measure_focus(add_star(field(60, 1.0, 40), 200, 300, 6.0, 2000)).score
    with_bright_sharp = measure_focus(add_star(field(60, 3.0, 40), 200, 300, 0.8, 2000)).score
    assert abs(with_bright_soft - sharp) < 0.1 * sharp  # a big saturated star doesn't drag it down
    assert abs(with_bright_sharp - soft) < 0.1 * soft  # nor does one sharp bright star lift it


def test_exposure_does_not_change_the_score():
    dim, bright = measure_focus(field(60, 1.5, 30)), measure_focus(field(60, 1.5, 90))
    assert abs(dim.score - bright.score) < 0.1 * dim.score and dim.stars > 30


def test_noise_and_lone_hot_pixels_are_not_stars():
    img = 20 + RNG.normal(0, 2, (400, 600))
    img[RNG.integers(0, 400, 40), RNG.integers(0, 600, 40)] = 200  # lone hot pixels
    m = measure_focus(np.clip(img, 0, 255))
    assert m.score is None and m.mode == "none"


def test_planet_mode_rises_as_the_disk_sharpens():
    y, x = np.indices((300, 300))
    disk = np.where(np.hypot(y - 150, x - 150) < 40, 180.0, 15.0)
    scores = [measure_focus(ndimage.gaussian_filter(disk, b) + RNG.normal(0, 1, disk.shape)) for b in (6, 3, 1)]
    assert all(m.mode == "planet" for m in scores)
    assert scores[0].score < scores[1].score < scores[2].score


def test_real_saturated_star_alone_gets_an_edge_number():
    """A bright star saturated in the main camera with few others around (2026-10-10,
    tests/data/align): judged by its edges, not 'nothing to judge'."""
    d = np.load("tests/data/align/bright-star-2026-10-10.npz")
    lum = superpixel_rgb(d["main"][10], str(d["main_bayer"])).astype(np.float32).mean(axis=2)
    m = measure_focus(lum)
    assert m.mode == "planet" and m.score > 0
