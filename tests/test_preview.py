import numpy as np
import pytest

pytest.importorskip("tetra3")  # focus_numbers uses the finder focus check

from astro.capture.preview import focus_numbers


def test_focus_numbers_on_blank_frame():
    f = focus_numbers(np.full((96, 128), 20, np.uint8))
    assert f.stars == 0 and f.hfr_px == float("inf")
