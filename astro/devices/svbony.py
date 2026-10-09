"""SVBony camera driver (SV705C main, SV905C finder) over the vendor SDK via ctypes.

SDK quirks (see docs/hardware-results.md):
- exposure/ROI must be set before video capture starts; changing them mid-stream times out,
- stop/start on one open handle is flaky, so any settings change closes and reopens,
- the SDK's bundled libusb is empty; load the system one first,
- the first 2 frames after starting are blank; the driver discards them.
The SDK path comes from SVB_LIB. Everything above this file only sees the `Camera` interface.
"""

import ctypes as C
import os
import time

import numpy as np

from astro.devices.base import Roi

DEFAULT_LIB = "~/sdk/SVBCameraSDK/lib/x64/libSVBCameraSDK.so"
RAW8, GAIN, EXPOSURE = 0, 0, 1  # SVB_IMG_RAW8, SVB_GAIN, SVB_EXPOSURE (microseconds)
CURRENT_TEMPERATURE = 16  # SVB_CURRENT_TEMPERATURE, 0.1 C
TIMEOUT = 11  # SVB_ERROR_TIMEOUT
RECONNECT_S, RECONNECT_MAX_S = 1.0, 30.0  # backoff between reopen attempts
# After video capture starts, the SDK returns 2 blank (bias-only) frames before real exposures
# (seen on the SV905C at 0.4-1.6 s: 0.5 s, then ~0 s, then frames at the exposure time).
STARTUP_FRAMES = 2
BAYER = ["RGGB", "BGGR", "GRBG", "GBRG"]


class Info(C.Structure):
    _fields_ = [("name", C.c_char * 32), ("sn", C.c_char * 32), ("port", C.c_char * 32),
                ("device_id", C.c_uint), ("camera_id", C.c_int)]


class Prop(C.Structure):
    _fields_ = [("max_h", C.c_long), ("max_w", C.c_long), ("color", C.c_int), ("bayer", C.c_int),
                ("bins", C.c_int * 16), ("formats", C.c_int * 8), ("bits", C.c_int),
                ("trigger", C.c_int)]


class SvbError(RuntimeError):
    def __init__(self, what: str, code: int):
        super().__init__(f"{what} failed: SVB error {code}")
        self.code = code


def load_sdk(path: str | None = None):
    C.CDLL("libusb-1.0.so.0", mode=C.RTLD_GLOBAL)
    return C.CDLL(os.path.expanduser(path or os.environ.get("SVB_LIB", DEFAULT_LIB)))


def _check(rc: int, what: str) -> None:
    if rc != 0:
        raise SvbError(what, rc)


def find_camera(lib, model: str) -> Info:
    """First connected camera whose name contains `model`, e.g. 'SV705C' or 'SV905C'."""
    names = []
    for i in range(lib.SVBGetNumOfConnectedCameras()):
        info = Info()
        _check(lib.SVBGetCameraInfo(C.byref(info), i), "SVBGetCameraInfo")
        names.append(info.name.decode())
        if model in names[-1]:
            return info
    raise SvbError(f"find {model} (connected: {names})", -1)


class SvbonyCamera:
    """Implements `astro.devices.base.Camera`."""

    def __init__(self, model: str, lib=None):
        self.model = model
        self._lib = lib
        self._id: int | None = None
        self._lost = False  # a reopen failed: keep trying to reconnect from capture()
        self._retry_at, self._backoff_s = 0.0, RECONNECT_S
        self._streaming = False
        self.prop = Prop()
        self.exposure_s, self.gain, self.roi = 0.01, 0, None

    @property
    def bayer(self) -> str:
        return BAYER[self.prop.bayer]

    @property
    def connected(self) -> bool:
        return self._id is not None

    @property
    def sensor_size(self) -> tuple[int, int]:
        return int(self.prop.max_w), int(self.prop.max_h)

    def connect(self) -> None:
        self._lib = self._lib or load_sdk()
        info = find_camera(self._lib, self.model)
        _check(self._lib.SVBOpenCamera(info.camera_id), "SVBOpenCamera")
        self._id = info.camera_id
        try:
            _check(self._lib.SVBGetCameraProperty(self._id, C.byref(self.prop)), "SVBGetCameraProperty")
        except SvbError:
            self.close()
            raise

    def close(self) -> None:
        self._lost = False  # an explicit close cancels reconnecting; _reopen re-arms it
        if self._id is None:
            return
        if self._streaming:
            self._lib.SVBStopVideoCapture(self._id)
            self._streaming = False
        self._lib.SVBCloseCamera(self._id)
        self._id = None

    def set_exposure(self, seconds: float) -> None:
        self._change(exposure_s=seconds)

    def set_gain(self, gain: int) -> None:
        self._change(gain=gain)

    def set_roi(self, roi: Roi | None) -> None:
        if roi is not None and (roi.x % 2 or roi.y % 2 or roi.width % 8 or roi.height % 2):
            raise ValueError("ROI needs even x/y/height and width divisible by 8")
        self._change(roi=roi)

    def temperature_c(self) -> float | None:
        """Sensor temperature (noise and dark frames depend on it), or None if unavailable."""
        if self._id is None:
            return None
        value, auto = C.c_long(), C.c_int()
        if self._lib.SVBGetControlValue(self._id, CURRENT_TEMPERATURE, C.byref(value),
                                        C.byref(auto)) != 0:
            return None
        return value.value / 10

    def capture(self) -> np.ndarray:
        """Next RAW8 frame. Reopens once on an SDK timeout."""
        if self._id is None:
            if not self._lost:
                raise RuntimeError("camera not connected")
            self._reconnect()
        try:
            return self._grab()
        except SvbError as e:
            if e.code != TIMEOUT:
                raise
            self._reopen()
            return self._grab()

    # --- internals -------------------------------------------------------------------------
    def _change(self, **settings) -> None:
        changed = any(getattr(self, k) != v for k, v in settings.items())
        for k, v in settings.items():
            setattr(self, k, v)
        if changed and self._streaming:
            self._reopen()  # restart on the same handle is flaky; a fresh open is reliable

    def _reopen(self) -> None:
        self.close()
        self._lost, self._retry_at = True, 0.0
        self._reconnect()

    def _reconnect(self) -> None:
        """After USB re-enumeration the camera may need a few seconds: back off between tries."""
        if time.monotonic() < self._retry_at:
            raise RuntimeError("camera reconnecting")
        try:
            self.connect()
        except (SvbError, RuntimeError, OSError):
            self._retry_at = time.monotonic() + self._backoff_s
            self._backoff_s = min(2 * self._backoff_s, RECONNECT_MAX_S)
            raise
        self._lost, self._backoff_s = False, RECONNECT_S

    def _start(self) -> None:
        lib, cid = self._lib, self._id
        roi = self.roi or Roi(0, 0, *self.sensor_size)
        _check(lib.SVBSetOutputImageType(cid, RAW8), "SVBSetOutputImageType")
        _check(lib.SVBSetROIFormat(cid, roi.x, roi.y, roi.width, roi.height, 1), "SVBSetROIFormat")
        _check(lib.SVBSetControlValue(cid, EXPOSURE, C.c_long(int(self.exposure_s * 1e6)), 0),
               "set exposure")
        _check(lib.SVBSetControlValue(cid, GAIN, C.c_long(self.gain), 0), "set gain")
        _check(lib.SVBStartVideoCapture(cid), "SVBStartVideoCapture")
        self._shape = (roi.height, roi.width)
        self._streaming = True

    def _grab(self) -> np.ndarray:
        if not self._streaming:
            self._start()
            for _ in range(STARTUP_FRAMES):
                self._read()
        return self._read()

    def _read(self) -> np.ndarray:
        h, w = self._shape
        buf = (C.c_ubyte * (w * h))()
        wait_ms = int(self.exposure_s * 3000) + 2000
        _check(self._lib.SVBGetVideoData(self._id, buf, len(buf), wait_ms), "SVBGetVideoData")
        return np.frombuffer(buf, np.uint8).reshape(h, w).copy()
