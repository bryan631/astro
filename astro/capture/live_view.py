"""Live video of one camera, for checking that everything is right: cap off, focus close, framing.

Runs only while someone is watching (the page's stream calls `touch`), uses a short exposure
for speed, asks the exposure gate before every frame, and puts the camera's settings back.
"""
import threading
import time
from collections.abc import Callable
from contextlib import AbstractContextManager

import numpy as np

from astro.devices.base import Camera, Roi

LOW_PEAK, HIGH_PEAK = 60, 220  # 8-bit brightest 0.5% of pixels: outside this, change the exposure
STEP = 1.6  # exposure change per frame
IDLE_STOP_S = 15.0  # nobody watching for this long: stop, so the camera isn't left in preview


class LiveView:
    def __init__(self, camera: Camera, exposure_s: float, gain: int, roi: Roi | None = None,
                 lock: AbstractContextManager | None = None,
                 allowed: Callable[[], str | None] = lambda: None,
                 exposure_range: tuple[float, float] | None = None):
        """`allowed()` is the exposure gate: a spoken reason to stop, or None. `lock` is the
        camera's lock when another thread (the plate-solve tracker) also captures.
        `exposure_range` (min, max seconds) turns on auto-exposure: a dim room or sky gets a
        longer exposure, a bright one a shorter one. A capped lens stays black at the maximum."""
        self.camera, self.exposure_s, self.gain, self.roi = camera, exposure_s, gain, roi
        self.exposure_range = exposure_range
        self.lock = lock or threading.Lock()
        self.allowed = allowed
        self.stopped_because: str | None = None  # a spoken reason, when the gate ended it
        self._saved = (camera.exposure_s, camera.gain)
        self._viewed = time.monotonic()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self) -> "LiveView":
        self.camera.set_exposure(self.exposure_s)
        self.camera.set_gain(self.gain)
        if self.roi is not None:
            self.camera.set_roi(self.roi)
        self._thread.start()
        return self

    def touch(self) -> None:
        """Someone is watching."""
        self._viewed = time.monotonic()

    @property
    def running(self) -> bool:
        return self._thread.is_alive()

    def stop(self) -> None:
        """Stop, wait for the last frame, and restore the camera's settings."""
        self._stop.set()
        if self._thread.is_alive() and threading.current_thread() is not self._thread:
            self._thread.join()
        self.camera.set_exposure(self._saved[0])
        self.camera.set_gain(self._saved[1])
        if self.roi is not None:
            self.camera.set_roi(None)

    def _run(self) -> None:
        while not self._stop.is_set():
            if (reason := self.allowed()) is not None:
                self.stopped_because = reason
                return
            if time.monotonic() - self._viewed > IDLE_STOP_S:
                return
            try:
                with self.lock:
                    frame = self.camera.capture()  # the camera's tap keeps it for the stream
                if self.exposure_range is not None:
                    self._auto_expose(frame)
            except (RuntimeError, OSError):  # a hiccup: try again, the page shows the last frame
                self._stop.wait(0.5)

    def _auto_expose(self, frame: np.ndarray) -> None:
        low, high = self.exposure_range
        peak = np.percentile(frame[::4, ::4], 99.5)
        if peak < LOW_PEAK:
            new = min(self.exposure_s * STEP, high)
        elif peak > HIGH_PEAK:
            new = max(self.exposure_s / STEP, low)
        else:
            return
        if new != self.exposure_s:
            self.exposure_s = new
            self.camera.set_exposure(new)
