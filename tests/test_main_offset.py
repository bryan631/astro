import numpy as np
import pytest

from astro.pointing.main_offset import CameraAxes, MainOffset

SCALE = 7200  # px per degree (0.5"/px)


def camera(rot_deg, mirrored=False):
    """True map: a target's sky offset from the camera aim -> its pixel offset from center."""
    c, s = np.cos(np.radians(rot_deg)), np.sin(np.radians(rot_deg))
    m = SCALE * np.array([[c, -s], [s, c]]) @ np.array([[1, 0], [0, -1]])  # y grows downward
    return m @ np.diag([-1, 1]) if mirrored else m


@pytest.mark.parametrize("rot,mirrored", [(0, False), (37, False), (200, True)])
def test_axes_learned_from_pushes(rot, mirrored):
    true = camera(rot, mirrored)
    axes = CameraAxes()
    rng = np.random.default_rng(0)
    for sky in ([0.15, 0], [0.12, 0.03], [0, -0.18], [0.09, 0.09]):  # >= 2 encoder counts
        px = true @ -np.array(sky) + rng.normal(0, 3, 2)  # pushing moves the target opposite
        axes.add_move(*sky, *px)
    assert axes.matrix is not None
    d_az, d_alt = axes.sky_offset(*(true @ [0.01, -0.02]))
    assert (d_az, d_alt) == (pytest.approx(0.01, abs=5e-4), pytest.approx(-0.02, abs=5e-4))


def test_one_direction_is_not_enough():
    axes = CameraAxes()
    axes.add_move(0.15, 0, 1080, 0)
    axes.add_move(0.24, 0.003, 1728, 21)
    assert axes.matrix is None
    assert not axes.add_move(0.001, 0.0, 7, 0)  # tiny move ignored


def test_offset_correction_centers_target():
    true = camera(30)
    axes = CameraAxes()
    for sky in ([0.15, 0], [0, 0.15]):
        axes.add_move(*sky, *(true @ -np.array(sky)))  # physical: target moves opposite
    # The main camera aims 0.1 deg right and 0.05 deg low of the finder model, so when the
    # model says "on target" the target sits at sky offset -cam_offset from the camera's aim.
    cam_offset = np.array([0.1, -0.05])
    offset = MainOffset()
    offset.observe(axes, *(true @ -cam_offset))
    assert (offset.d_az_sky_deg, offset.d_alt_deg) == (pytest.approx(0.1), pytest.approx(-0.05))
    alt, az = offset.correct(40.0, 100.0)
    assert alt == pytest.approx(40.05) and az == pytest.approx(100 - 0.1 / np.cos(np.radians(40)))


def test_correction_moves_target_toward_center():
    """End to end with physical signs: after correcting, the target lands at the center."""
    true = camera(110, mirrored=True)
    axes = CameraAxes()
    for sky in ([0.12, 0.0], [0.0, 0.12], [0.06, -0.09]):
        axes.add_move(*sky, *(true @ -np.array(sky)))
    cam_offset = np.array([-0.07, 0.12])  # where the main camera aims vs the finder model
    offset = MainOffset()
    offset.observe(axes, *(true @ -cam_offset))
    t_alt, t_az = 40.0, 100.0
    aim_alt, aim_az = offset.correct(t_alt, t_az)  # finder model now points here
    cos = np.cos(np.radians(t_alt))
    cam_alt, cam_az_sky = aim_alt + cam_offset[1], aim_az * cos + cam_offset[0]
    assert cam_alt == pytest.approx(t_alt, abs=1e-6)
    assert cam_az_sky == pytest.approx(t_az * cos, abs=1e-6)
