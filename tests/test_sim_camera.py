import pytest

from astro.devices.base import Roi
from astro.devices.sim.camera import SimCamera


def test_capture_requires_connect():
    with pytest.raises(RuntimeError):
        SimCamera().capture()


def test_full_frame_and_roi_shapes():
    cam = SimCamera(width=64, height=48)
    cam.connect()
    assert cam.capture().shape == (48, 64)
    cam.set_roi(Roi(0, 0, 16, 8))
    assert cam.capture().shape == (8, 16)
