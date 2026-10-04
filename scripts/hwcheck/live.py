"""Shared live-camera code for the hwcheck viewers (viewer.py, viewer_plot.py).

A background thread captures, debayers, resizes and overlays stats; viewers only display
`LiveCamera.latest` (BGR). Shutdown waits for the current exposure before closing the SDK,
since closing mid-capture segfaults.
"""

import argparse
import threading
import time

import cv2
import numpy as np

from astro.capture.preview import focus_numbers
from astro.devices.base import Roi
from astro.devices.svbony import SvbonyCamera

MODELS = {"finder": "SV905C", "main": "SV705C"}
FOCUS_EVERY_S = 1.0  # focus numbers are slower than display; update them once a second
MAX_EXPOSURE_S = 10


def parse_args(description: str, extra=None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=description)
    ap.add_argument("cam", choices=MODELS, help="finder (SV905C) or main (SV705C)")
    ap.add_argument("--exp", type=float, default=0.05, help="exposure, seconds")
    ap.add_argument("--gain", type=int, default=100)
    ap.add_argument("--roi", default=None, help="WxH, centered, e.g. 1280x720")
    ap.add_argument("--width", type=int, default=960, help="max display width")
    if extra:
        extra(ap)
    args = ap.parse_args()
    if args.exp > MAX_EXPOSURE_S:
        ap.error(f"--exp is in seconds; {args.exp:g} s is very long. "
                 f"Did you mean {args.exp / 1000:g}?")
    return args


class LiveCamera:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.cam = SvbonyCamera(MODELS[args.cam])
        self.cam.connect()
        self.cam.set_exposure(args.exp)
        self.cam.set_gain(args.gain)
        if args.roi:
            sw, sh = self.cam.sensor_size
            w, h = (int(v) for v in args.roi.split("x"))
            w, h = min(w, sw) // 8 * 8, min(h, sh) // 2 * 2
            self.cam.set_roi(Roi((sw - w) // 4 * 2, (sh - h) // 4 * 2, w, h))
        # OpenCV names Bayer codes by row 2, columns 2-3 (e.g. sensor RGGB -> BayerBG).
        self._debayer = getattr(cv2, f"COLOR_Bayer{self.cam.bayer[3]}{self.cam.bayer[2]}2BGR")
        self.latest: np.ndarray | None = None  # BGR, display-sized, with overlay
        self.raw: np.ndarray | None = None
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._grab)

    def start(self) -> "LiveCamera":
        self._thread.start()
        return self

    def stop(self) -> None:
        self._stop.set()
        self._thread.join()  # never close the SDK while a capture is in flight
        self.cam.close()

    @property
    def stopped(self) -> bool:
        return self._stop.is_set()

    def _grab(self) -> None:
        fps, t, focus, focus_t = 0.0, time.time(), "", 0.0
        while not self._stop.is_set():
            raw = self.cam.capture()  # the driver reopens the camera on SDK timeouts
            now = time.time()
            fps, t = 0.9 * fps + 0.1 / max(now - t, 1e-6), now
            if now - focus_t >= FOCUS_EVERY_S:
                f = focus_numbers(raw)
                focus, focus_t = f"sharp {f.sharpness:.0f}  stars {f.stars} hfr {f.hfr_px:.2f}", now
            img = cv2.cvtColor(raw, self._debayer)
            h, w = img.shape[:2]
            if w > self.args.width:
                img = cv2.resize(img, (self.args.width, h * self.args.width // w),
                                 interpolation=cv2.INTER_AREA)
            text = f"exp {self.cam.exposure_s:g}s gain {self.cam.gain}  {fps:.1f} fps  max {raw.max()}  {focus}"
            cv2.putText(img, text, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)
            self.raw, self.latest = raw, img
