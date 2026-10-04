import numpy as np
import pytest

from astro.pointing.main_offset import CameraAxes, MainOffset

SCALE = 7200  # px per degree (0.5"/px)


def camera(rot_deg, mirrored=False):
    """True mapping sky (d_az_sky, d_alt) -> pixels for a camera at some angle in the focuser."""
    c, s = np.cos(np.radians(rot_deg)), np.sin(np.radians(rot_deg))
    m = SCALE * np.array([[c, -s], [s, c]]) @ np.array([[1, 0], [0, -1]])  # y grows downward
    return m @ np.diag([-1, 1]) if mirrored else m


@pytest.mark.parametrize("rot,mirrored", [(0, False), (37, False), (200, True)])
def test_axes_learned_from_pushes(rot, mirrored):
    true = camera(rot, mirrored)
    axes = CameraAxes()
    rng = np.random.default_rng(0)
    for sky in ([0.05, 0], [0.04, 0.01], [0, -0.06], [0.03, 0.03]):
        px = true @ sky + rng.normal(0, 3, 2)  # 3 px centroid noise
        axes.add_move(*sky, *px)
    assert axes.matrix is not None
    d_az, d_alt = axes.sky_offset(*(true @ [0.01, -0.02]))
    assert (d_az, d_alt) == (pytest.approx(0.01, abs=5e-4), pytest.approx(-0.02, abs=5e-4))


def test_one_direction_is_not_enough():
    axes = CameraAxes()
    axes.add_move(0.05, 0, 360, 0)
    axes.add_move(0.08, 0.001, 576, 7)
    assert axes.matrix is None
    assert not axes.add_move(0.001, 0.0, 7, 0)  # tiny move ignored


def test_offset_correction_centers_target():
    true = camera(30)
    axes = CameraAxes()
    for sky in ([0.05, 0], [0, 0.05]):
        axes.add_move(*sky, *(true @ sky))
    # The main camera points 0.1 deg right and 0.05 deg low of the finder model, so when the
    # model says "on target" the target sits up-left of center in the main image.
    cam_offset = np.array([0.1, -0.05])
    offset = MainOffset()
    offset.observe(axes, *(true @ -cam_offset))
    assert (offset.d_az_sky_deg, offset.d_alt_deg) == (pytest.approx(0.1), pytest.approx(-0.05))
    alt, az = offset.correct(40.0, 100.0)
    assert alt == pytest.approx(40.05) and az == pytest.approx(100 - 0.1 / np.cos(np.radians(40)))
