"""Minimal ctypes wrapper for the SVBony SDK (hardware checkout only)."""
import ctypes as C
import os

LIB = os.environ.get("SVB_LIB", os.path.expanduser("~/sdk/SVBCameraSDK/lib/x64/libSVBCameraSDK.so"))
RAW8, GAIN, EXPOSURE = 0, 0, 1  # SVB_IMG_RAW8, SVB_GAIN, SVB_EXPOSURE (exposure in us)
BAYER = ["RG", "BG", "GR", "GB"]


class Info(C.Structure):
    _fields_ = [("name", C.c_char * 32), ("sn", C.c_char * 32), ("port", C.c_char * 32),
                ("device_id", C.c_uint), ("camera_id", C.c_int)]


class Prop(C.Structure):
    _fields_ = [("max_h", C.c_long), ("max_w", C.c_long), ("color", C.c_int), ("bayer", C.c_int),
                ("bins", C.c_int * 16), ("formats", C.c_int * 8), ("bits", C.c_int), ("trigger", C.c_int)]


C.CDLL("libusb-1.0.so.0", mode=C.RTLD_GLOBAL)  # the SDK's own libusb copy is an empty file
lib = C.CDLL(LIB)


def check(rc, what):
    if rc != 0:
        raise RuntimeError(f"{what} failed: SVB_ERROR_CODE {rc}")


def enumerate_cameras():
    out = []
    for i in range(lib.SVBGetNumOfConnectedCameras()):
        info = Info()
        check(lib.SVBGetCameraInfo(C.byref(info), i), "SVBGetCameraInfo")
        out.append(info)
    return out


class Camera:
    def __init__(self, info):
        self.id = info.camera_id
        check(lib.SVBOpenCamera(self.id), "SVBOpenCamera")
        self.prop = Prop()
        check(lib.SVBGetCameraProperty(self.id, C.byref(self.prop)), "SVBGetCameraProperty")

    def set_control(self, ctrl, value):
        check(lib.SVBSetControlValue(self.id, ctrl, C.c_long(value), 0), f"SVBSetControlValue({ctrl})")

    def start(self, x=0, y=0, w=None, h=None):
        w, h = w or self.prop.max_w, h or self.prop.max_h
        check(lib.SVBSetOutputImageType(self.id, RAW8), "SVBSetOutputImageType")
        check(lib.SVBSetROIFormat(self.id, x, y, w, h, 1), "SVBSetROIFormat")
        self.w, self.h = w, h
        check(lib.SVBStartVideoCapture(self.id), "SVBStartVideoCapture")

    def frame(self, wait_ms=5000):
        import numpy as np
        buf = (C.c_ubyte * (self.w * self.h))()
        check(lib.SVBGetVideoData(self.id, buf, len(buf), wait_ms), "SVBGetVideoData")
        return np.frombuffer(buf, np.uint8).reshape(self.h, self.w).copy()

    def stop(self):
        lib.SVBStopVideoCapture(self.id)

    def close(self):
        lib.SVBCloseCamera(self.id)
