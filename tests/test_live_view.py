import time

from astro.capture import live_view
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


def test_stops_with_the_reason_when_the_gate_says_no():
    cam, reasons = Cam(), [None, None, "it's daytime"]
    view = LiveView(cam, 0.05, 600, allowed=lambda: reasons.pop(0) if reasons else "it's daytime").start()
    assert wait_for(lambda: not view.running)
    assert view.stopped_because == "it's daytime"
    view.stop()
    assert cam.exposure_s == 0.8  # restored even after a gate stop


def test_stops_when_nobody_is_watching(monkeypatch):
    monkeypatch.setattr(live_view, "IDLE_STOP_S", 0.05)
    view = LiveView(Cam(), 0.05, 600).start()
    assert wait_for(lambda: not view.running)
    assert view.stopped_because is None
    view.stop()
