import time
from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

from astro.capture.recorder import CaptureRefused, Recorder
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
    rec = Recorder(cam, (800, 600), tmp_path, lambda: None).start("Jupiter", 0.3)
    rec.done.wait(5)
    time.sleep(0.05)
    assert rec.frames > 60 and not rec.lost
    _, frames = read_ser(rec.path)
    assert cam.roi is None  # recorder restores full frame when done
    assert frames[-1].max() > 200  # planet still in the last frame after drifting past the ROI


def test_second_capture_is_refused_while_recording(tmp_path):
    s, _ = make_session(tmp_path)
    s.main_focus_ok, s.record_seconds = True, 2
    s.handle("take a picture")
    assert texts(s.handle("take a picture")) == ["I'm already recording."]
    s.recorder.stop()
    s.recorder.current.done.wait(5)


def test_recording_refused_and_stopped_when_unsafe(tmp_path, monkeypatch):
    monkeypatch.setattr("astro.capture.recorder.ROI_PX", 128)
    unsafe = {"reason": "too close to the Sun"}
    rec = Recorder(DriftingPlanet(), (800, 600), tmp_path, lambda: unsafe["reason"])
    with pytest.raises(CaptureRefused, match="too close to the Sun"):
        rec.start("Jupiter", 1)
    unsafe["reason"] = None
    cam = DriftingPlanet()
    rec = Recorder(cam, (800, 600), tmp_path, lambda: unsafe["reason"])
    run = rec.start("Jupiter", 5)
    time.sleep(0.2)
    unsafe["reason"] = "daytime lockout"
    assert run.done.wait(3) and "daytime lockout" in run.error


class FailingCamera(DriftingPlanet):
    def capture(self):
        if self.n > 3:
            raise RuntimeError("SVB error 11")
        return super().capture()


def test_camera_failure_is_reported_not_done(tmp_path, monkeypatch):
    monkeypatch.setattr("astro.capture.recorder.ROI_PX", 128)
    s, _ = make_session(tmp_path)
    s.recorder.camera = FailingCamera()
    s.recorder.sensor = (800, 600)
    s.main_focus_ok = True
    s.handle("take a picture")
    s.recorder.current.done.wait(5)
    said = texts(s.tick(100.0))[0]
    assert said.startswith("Recording failed: SVB error 11") and "Done" not in said


def test_focus_stops_when_pointing_becomes_unsafe(tmp_path):
    s, _ = make_session(tmp_path)
    s.handle("focus")
    s.exposure_safety = lambda: "daytime lockout"
    assert texts(s.tick(0.0)) == ["I stopped focusing: daytime lockout."]
    assert s._focus_coach is None


def test_barlow_retunes_active_guide(tmp_path):
    from astro.guidance.engine import Guide

    s, _ = make_session(tmp_path)
    s.guide = Guide(45, 100)
    s.handle("barlow in")
    assert s.guide.tol_deg == pytest.approx(2 / 60)


@pytest.mark.parametrize("planet", ["mercury", "uranus", "neptune"])
def test_sim_main_camera_shows_every_planet(planet):
    from astro.capture.roi import brightest_blob

    alt, az = body_altaz(planet, WPB, EVENING)
    cam = SimMainCamera(lambda: (alt, az), WPB, lambda: EVENING)
    assert brightest_blob(cam.capture()) is not None


def test_focus_refused_while_recording(tmp_path):
    s, _ = make_session(tmp_path)
    s.main_focus_ok, s.record_seconds = True, 2
    s.handle("take a picture")
    assert "recording right now" in texts(s.handle("focus"))[0]
    s.recorder.stop()
    s.recorder.current.done.wait(5)


class UnresettableCamera(DriftingPlanet):
    def set_roi(self, roi):
        if roi is None and self.n:
            raise RuntimeError("camera unplugged")
        super().set_roi(roi)


def test_recording_finishes_even_if_camera_reset_fails(tmp_path, monkeypatch):
    monkeypatch.setattr("astro.capture.recorder.ROI_PX", 128)
    rec = Recorder(UnresettableCamera(), (800, 600), tmp_path, lambda: None).start("Mars", 0.1)
    assert rec.done.wait(3) and "did not reset" in rec.error


def test_quick_recordings_get_distinct_files(tmp_path, monkeypatch):
    monkeypatch.setattr("astro.capture.recorder.ROI_PX", 128)
    recorder = Recorder(DriftingPlanet(), (800, 600), tmp_path, lambda: None)
    first = recorder.start("Mars", 0.05)
    first.done.wait(3)
    second = recorder.start("Mars", 0.05)
    second.done.wait(3)
    assert first.path != second.path


class DeadCamera(DriftingPlanet):
    def capture(self):
        raise RuntimeError("SVB error 11")


def test_capture_start_failure_is_spoken(tmp_path):
    s, _ = make_session(tmp_path)
    s.recorder.camera = DeadCamera()
    s.main_focus_ok = True
    assert texts(s.handle("take a picture"))[0].startswith("The main camera isn't responding")


def test_focus_camera_failure_stops_focus_without_crashing(tmp_path):
    s, _ = make_session(tmp_path)
    s.handle("focus")
    s.main_camera = DeadCamera()
    assert "stopped responding" in texts(s.tick(0.0))[0] and s._focus_coach is None


def test_barlow_restarts_active_focus_coach(tmp_path):
    s, _ = make_session(tmp_path)
    s.handle("focus")
    old = s._focus_coach
    s.handle("barlow in")
    assert s._focus_coach is not old


def test_stop_recording_after_it_finished(tmp_path):
    s, _ = make_session(tmp_path)
    s.main_focus_ok, s.record_seconds = True, 0.1
    s.handle("take a picture")
    s.recorder.current.done.wait(5)
    assert texts(s.handle("stop recording")) == ["We're not recording."]


def test_refocus_clears_gate_and_blocks_capture_until_done(tmp_path):
    s, _ = make_session(tmp_path)
    s.main_focus_ok = True
    s.handle("focus")
    assert not s.main_focus_ok
    assert texts(s.handle("take a picture")) == ["Let's finish focusing first. Say done when it's sharpest."]
    assert s.recorder.current is None


def test_recording_becomes_a_gallery_picture(tmp_path):
    s, _ = make_session(tmp_path)
    s.main_focus_ok, s.record_seconds = True, 0.5
    s.handle("take a picture")
    s.recorder.current.done.wait(5)
    assert "making your picture" in texts(s.tick(100.0))[0]
    out = []
    for i in range(100):
        out = s.tick(101.0 + i)
        if out:
            break
        time.sleep(0.05)
    assert texts(out) == ["Your picture of Saturn is ready. Tap Pictures to see it."]
    assert (tmp_path / "gallery" / out[1]["file"]).exists()


def test_picture_keeps_capture_time_name_and_every_job_is_announced(tmp_path):
    s, _ = make_session(tmp_path)
    s.main_focus_ok, s.record_seconds = True, 0.3
    said = []
    for target in ("Saturn", "Jupiter"):
        s.target = target
        s.handle("take a picture")
        s.target = "Mars"  # user moved on before the recording finished
        s.recorder.current.done.wait(5)
        said += texts(s.tick(0.0))
    for _ in range(200):
        said += texts(s.tick(1.0))
        if sum("is ready" in x for x in said) == 2:
            break
        time.sleep(0.05)
    ready = [x for x in said if "is ready" in x]
    assert ready == ["Your picture of Saturn is ready. Tap Pictures to see it.",
                     "Your picture of Jupiter is ready. Tap Pictures to see it."]
