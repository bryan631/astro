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
RETRY_S = 1.0  # pause after a failed capture before trying again
FIRST_FRAME_TIMEOUT_S = 30


def parse_args(description: str, extra=None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=description)
    ap.add_argument("cam", choices=MODELS, help="finder (SV905C) or main (SV705C)")
    ap.add_argument("--exp", type=float, default=0.05, help="exposure, seconds")
    ap.add_argument("--gain", type=int, default=100)
    ap.add_argument("--roi", default=None, help="WxH, centered, e.g. 1280x720")
    ap.add_argument("--width", type=int, default=960, help="max display width")
    ap.add_argument("--raw", action="store_true",
                    help="show the sensor as-is (default rotates 180 deg: a lens inverts the image)")
    if extra:
        extra(ap)
    args = ap.parse_args()
    if args.exp <= 0:
        ap.error("--exp must be more than 0 seconds")
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
        self._pending: dict[str, float] = {}  # setting changes, applied by the capture thread
        self._thread = threading.Thread(target=self._grab)

    def start(self) -> "LiveCamera":
        self._thread.start()
        return self

    def wait_first_frame(self) -> None:
        """Block until a frame exists; exit with a message if the camera never delivers one."""
        deadline = time.time() + FIRST_FRAME_TIMEOUT_S + 3 * self.args.exp
        while self.latest is None:
            if time.time() > deadline:
                self.stop()
                raise SystemExit("No frames from the camera. Is another program using it?")
            time.sleep(0.05)

    def stop(self) -> None:
        self._stop.set()
        self._thread.join()  # never close the SDK while a capture is in flight
        self.cam.close()

    def request(self, exposure_s: float | None = None, gain: int | None = None) -> None:
        """Change settings from any thread. The capture thread applies them between frames,
        because a change reopens the SDK handle and must never race an in-flight capture."""
        if exposure_s is not None:
            self._pending["exposure_s"] = exposure_s
        if gain is not None:
            self._pending["gain"] = gain

    @property
    def stopped(self) -> bool:
        return self._stop.is_set()

    def _grab(self) -> None:
        fps, t, focus, focus_t = 0.0, time.time(), "", 0.0
        while not self._stop.is_set():
            if "exposure_s" in self._pending:
                self.cam.set_exposure(self._pending.pop("exposure_s"))
            if "gain" in self._pending:
                self.cam.set_gain(int(self._pending.pop("gain")))
            try:
                raw = self.cam.capture()  # the driver reopens the camera once on SDK timeouts
            except RuntimeError as e:  # still failing: keep the viewer alive and retry
                print(f"capture failed ({e}); retrying in {RETRY_S:g} s", flush=True)
                self._stop.wait(RETRY_S)
                continue
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
            if not self.args.raw:
                img = cv2.rotate(img, cv2.ROTATE_180)  # display only; frames/solves are unchanged
            text = f"exp {self.cam.exposure_s:g}s gain {self.cam.gain}  {fps:.1f} fps  max {raw.max()}  {focus}"
            cv2.putText(img, text, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)
            self.raw, self.latest = raw, img
