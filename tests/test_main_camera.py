import time
from datetime import datetime, timedelta, timezone

import numpy as np

from astro.capture.recorder import Recorder
from astro.capture.ser import read_ser
from astro.devices.base import Roi
from astro.devices.sim.main_cam import SimMainCamera
from astro.pointing.coords import Site, body_altaz
from astro.session import Session

WPB = Site(26.7, -80.1)
EVENING = datetime(2026, 10, 3, 22, 0, tzinfo=timezone(timedelta(hours=-4)))


def on_saturn(**kw):
    alt, az = body_altaz("saturn", WPB, EVENING)
    cam = SimMainCamera(lambda: (alt, az), WPB, lambda: EVENING, **kw)
    return cam, (alt, az)


def make_session(tmp_path, **kw):
    cam, pos = on_saturn(**kw)
    s = Session(WPB, lambda: pos, clock=lambda: EVENING, main_camera=cam,
                main_sensor=cam.sensor_size, data_dir=tmp_path)
    s.target = "Saturn"
    s.record_seconds = 0.5
    return s, cam


def texts(msgs):
    return [m["text"] for m in msgs if m["type"] == "say"]


def test_sim_main_camera_sees_saturn_centered():
    cam, _ = on_saturn()
    x, y = __import__("astro.capture.roi", fromlist=["x"]).brightest_blob(cam.capture())
    assert abs(x - 964) < 3 and abs(y - 545) < 3


def test_capture_is_gated_on_focus_then_records(tmp_path):
    s, cam = make_session(tmp_path, blur_px=5)
    said = texts(s.handle("take a picture"))
    assert said[0] == "Let's make sure it's sharp first." and "focus knob" in said[1]
    cues = []
    for i, blur in enumerate([5, 3, 1.5, 0.8, 2.0]):
        cam.blur_px = blur
        cues += texts(s.tick(float(i * 2)))
    assert "sharper" in cues and cues[-1].startswith("passed it")
    assert texts(s.handle("done")) == ["OK, focus is set."]
    assert texts(s.handle("take a picture"))[0].startswith("Recording for 0.5 seconds")
    s.recorder.current.done.wait(5)
    assert texts(s.tick(100.0))[0].startswith("Done. I saved")
    meta, frames = read_ser(next(tmp_path.glob("captures/*Saturn.ser")))
    assert meta["frames"] > 3 and frames.shape[1:] == (512, 512) and frames.max() > 100


def test_barlow_change_requires_refocus(tmp_path):
    s, _ = make_session(tmp_path)
    s.main_focus_ok = True
    assert "refocus" in texts(s.handle("I put in the barlow"))[0]
    assert texts(s.handle("take a picture"))[0] == "Let's make sure it's sharp first."
    assert s.barlow


def test_capture_without_planet_explains(tmp_path):
    s, cam = make_session(tmp_path)
    cam.true_altaz = lambda: (80.0, 10.0)  # empty sky
    s.main_focus_ok = True
    assert "don't see anything bright" in texts(s.handle("take a picture"))[0]


class DriftingPlanet:
    """Camera stub: a bright square drifting right 1 px per frame (~3x real); honours ROI."""

    bayer = "GRBG"

    def __init__(self):
        self.n, self.roi = 0, None

    def set_roi(self, roi):
        self.roi = roi

    def capture(self):
        self.n += 1
        full = np.full((600, 800), 5, np.uint8)
        x = min(200 + self.n, 760)
        full[290:310, x:x + 20] = 220
        r = self.roi or Roi(0, 0, 800, 600)
        return full[r.y:r.y + r.height, r.x:r.x + r.width]


def test_recorder_recenters_on_drift(tmp_path, monkeypatch):
    monkeypatch.setattr("astro.capture.recorder.ROI_PX", 128)
    cam = DriftingPlanet()
    rec = Recorder(cam, (800, 600), tmp_path).start("Jupiter", 0.3)
    rec.done.wait(5)
    time.sleep(0.05)
    assert rec.frames > 60 and not rec.lost
    _, frames = read_ser(rec.path)
    assert cam.roi is None  # recorder restores full frame when done
    assert frames[-1].max() > 200  # planet still in the last frame after drifting past the ROI
