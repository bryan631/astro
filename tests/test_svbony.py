"""SvbonyCamera logic against a fake SDK (the real one only exists on hardware machines)."""

import pytest

from astro.devices.base import Roi
from astro.devices.svbony import (
    EXPOSURE,
    GAIN,
    RAW8,
    STARTUP_FRAMES,
    TIMEOUT,
    SvbError,
    SvbonyCamera,
)


class FakeSdk:
    def __init__(self, timeouts=0):
        self.calls, self.open, self.timeouts = [], set(), timeouts
        self.roi = None
        self.controls = {}

    def SVBGetNumOfConnectedCameras(self):
        return 2

    def SVBGetCameraInfo(self, ref, i):
        info = ref._obj
        info.name, info.camera_id = [b"SVBONY SV905C2", b"SVBONY SV705C"][i], 10 + i
        return 0

    def SVBOpenCamera(self, cid):
        self.calls.append("open")
        self.open.add(cid)
        return 0

    def SVBCloseCamera(self, cid):
        self.calls.append("close")
        self.open.discard(cid)
        return 0

    def SVBGetCameraProperty(self, cid, ref):
        ref._obj.max_w, ref._obj.max_h, ref._obj.bayer = 64, 32, 2
        return 0

    def SVBSetROIFormat(self, cid, x, y, w, h, b):
        self.roi = (x, y, w, h)
        return 0

    def SVBStartVideoCapture(self, cid):
        self.calls.append("start")
        return 0

    def SVBStopVideoCapture(self, cid):
        self.calls.append("stop")
        return 0

    def SVBGetVideoData(self, cid, buf, size, wait):
        self.reads = getattr(self, "reads", 0) + 1
        if self.timeouts:
            self.timeouts -= 1
            return TIMEOUT
        for i in range(size):
            buf[i] = 7
        return 0

    def SVBSetControlValue(self, cid, ctrl, value, auto):
        self.controls[ctrl] = value.value  # passed as ctypes c_long
        return 0

    def SVBSetOutputImageType(self, cid, image_type):
        self.image_type = image_type
        return 0


def connected(sdk, model="SV705C"):
    cam = SvbonyCamera(model, lib=sdk)
    cam.connect()
    return cam


def test_finds_model_and_reports_bayer():
    sdk = FakeSdk()
    cam = connected(sdk)
    assert sdk.open == {11} and cam.bayer == "GRBG" and cam.sensor_size == (64, 32)


def test_missing_model_raises():
    with pytest.raises(SvbError, match="connected"):
        connected(FakeSdk(), "ASI585")


def test_capture_full_frame_then_roi_reopens():
    sdk = FakeSdk()
    cam = connected(sdk)
    assert cam.capture().shape == (32, 64)
    cam.set_roi(Roi(8, 4, 16, 8))
    frame = cam.capture()
    assert frame.shape == (8, 16) and (frame == 7).all()
    assert sdk.roi == (8, 4, 16, 8)
    assert sdk.calls == ["open", "start", "stop", "close", "open", "start"]


def test_exposure_and_gain_reach_the_sdk():
    sdk = FakeSdk()
    cam = connected(sdk)
    cam.set_exposure(0.25)
    cam.set_gain(120)
    cam.capture()
    assert sdk.controls == {EXPOSURE: 250_000, GAIN: 120}  # exposure in microseconds
    assert sdk.image_type == RAW8


def test_blank_startup_frames_are_discarded():
    sdk = FakeSdk()
    cam = connected(sdk)
    cam.capture()
    assert sdk.reads == 1 + STARTUP_FRAMES
    cam.capture()
    assert sdk.reads == 2 + STARTUP_FRAMES  # only after a (re)start


def test_unchanged_setting_does_not_reopen():
    sdk = FakeSdk()
    cam = connected(sdk)
    cam.capture()
    cam.set_exposure(cam.exposure_s)
    assert sdk.calls.count("open") == 1


def test_timeout_reopens_once_and_recovers():
    sdk = FakeSdk(timeouts=1)
    cam = connected(sdk)
    assert cam.capture().shape == (32, 64)
    assert sdk.calls.count("open") == 2


def test_close_releases_handle():
    sdk = FakeSdk()
    cam = connected(sdk)
    cam.capture()
    cam.close()
    assert sdk.open == set() and sdk.calls[-2:] == ["stop", "close"]


def test_rejects_misaligned_roi():
    with pytest.raises(ValueError):
        connected(FakeSdk()).set_roi(Roi(1, 0, 16, 8))


def test_temperature_in_tenths_of_a_degree():
    sdk = FakeSdk()

    def get(cid, ctrl, value, auto):
        value._obj.value = 235
        return 0
    sdk.SVBGetControlValue = get
    assert connected(sdk).temperature_c() == 23.5
