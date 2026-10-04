import numpy as np
import pytest

pytest.importorskip("tetra3")  # focus_numbers uses the finder focus check

from astro.capture.preview import focus_numbers, superpixel_rgb


def test_superpixel_grbg_puts_red_and_blue_in_place():
    raw = np.zeros((4, 4), np.uint8)
    raw[0::2, 1::2] = 200  # R in GRBG
    raw[1::2, 0::2] = 50  # B
    rgb = superpixel_rgb(raw, "GRBG")
    assert rgb.shape == (2, 2, 3) and (rgb[..., 0] == 200).all() and (rgb[..., 2] == 50).all()


def test_focus_numbers_on_blank_frame():
    f = focus_numbers(np.full((96, 128), 20, np.uint8))
    assert f.stars == 0 and f.hfr_px == float("inf")
