import time

import numpy as np

from astro.capture.live_view import LiveView
from astro.devices.base import Roi


class Cam:
    exposure_s, gain, bayer = 0.8, 200, "GRBG"

    def __init__(self):
        self.frames, self.roi = 0, "unset"

    def set_exposure(self, s): self.exposure_s = s
    def set_gain(self, g): self.gain = g
    def set_roi(self, roi): self.roi = roi

    def capture(self):
        self.frames += 1
        time.sleep(0.005)


def wait_for(cond, s=2.0):
    end = time.monotonic() + s
    while not cond() and time.monotonic() < end:
        time.sleep(0.01)
    return cond()


def test_streams_and_restores_the_camera():
    cam = Cam()
    view = LiveView(cam, 0.05, 600, Roi(0, 0, 640, 360)).start()
    assert (cam.exposure_s, cam.gain) == (0.05, 600) and cam.roi == Roi(0, 0, 640, 360)
    assert wait_for(lambda: cam.frames > 5)
    view.stop()
    assert (cam.exposure_s, cam.gain, cam.roi) == (0.8, 200, None)
    frames = cam.frames
    time.sleep(0.05)
    assert cam.frames == frames and not view.running


class SceneCam(Cam):
    """A scene `light` bright: pixel value = light * exposure, so exposure decides the picture."""

    def __init__(self, light):
        super().__init__()
        self.light = light

    def capture(self):
        super().capture()
        return np.full((64, 64), min(255, self.light * self.exposure_s), np.uint8)


def test_auto_exposure_brightens_a_dim_scene_and_dims_a_bright_one():
    dim, bright = SceneCam(light=200), SceneCam(light=20000)  # 0.1 s: 20 vs 255
    views = [LiveView(c, 0.1, 400, exposure_range=(0.01, 0.5)).start() for c in (dim, bright)]
    assert wait_for(lambda: 60 <= dim.exposure_s * dim.light <= 255 and dim.exposure_s > 0.1)
    assert wait_for(lambda: bright.exposure_s < 0.05)
    for v in views:
        v.stop()


def test_auto_exposure_stops_at_the_limit_for_a_capped_lens():
    capped = SceneCam(light=0)
    view = LiveView(capped, 0.1, 400, exposure_range=(0.01, 0.5)).start()
    assert wait_for(lambda: capped.exposure_s == 0.5)
    view.stop()
    assert capped.exposure_s == 0.8  # the camera's own setting is back


class GainSceneCam(SceneCam):
    def capture(self):
        Cam.capture(self)
        return np.full((64, 64), min(255, self.light * self.exposure_s * self.gain / 400), np.uint8)


def test_auto_exposure_lowers_the_gain_in_daylight():
    room = GainSceneCam(light=100000)  # saturated even at 0.01 s and gain 400
    view = LiveView(room, 0.1, 400, exposure_range=(0.01, 0.5)).start()
    assert wait_for(lambda: 60 <= room.light * room.exposure_s * room.gain / 400 <= 220)
    assert room.exposure_s == 0.01 and room.gain < 400
    view.stop()


def test_manual_exposure_turns_auto_off():
    cam = SceneCam(light=200)
    view = LiveView(cam, 0.1, 400, exposure_range=(0.01, 0.5)).start()
    view.nudge(0.5)
    before = cam.exposure_s
    time.sleep(0.1)
    assert view.exposure_range is None and cam.exposure_s == before
    view.stop()
