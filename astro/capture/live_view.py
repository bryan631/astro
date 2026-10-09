"""Live video of one camera, for checking that everything is right: cap off, focus close, framing.

Runs until stopped (or the exposure gate says no), uses a short exposure for speed, asks the
gate before every frame, and puts the camera's settings back.
"""
import logging
import threading
from contextlib import AbstractContextManager

import numpy as np

from astro.devices.base import Camera, Roi

LOW_PEAK, HIGH_PEAK = 60, 220  # 8-bit brightest 0.5% of pixels: outside this, change the exposure
STEP = 2.0  # exposure change per frame
MIN_EXPOSURE_S, MAX_EXPOSURE_S = 0.0001, 2.0  # manual exposure limits
log = logging.getLogger(__name__)
MAX_FAILURES = 3  # captures in a row before the video gives up


class LiveView:
    def __init__(self, camera: Camera, exposure_s: float, gain: int, roi: Roi | None = None,
                 lock: AbstractContextManager | None = None,
                 exposure_range: tuple[float, float] | None = None, max_gain: int | None = None):
        """`lock` is the
        camera's lock when another thread (the plate-solve tracker) also captures.
        `exposure_range` (min, max seconds) turns on auto-exposure: a dim room or sky gets a
        longer exposure, a bright one a shorter one. A capped lens stays black at the maximum.
        In daylight it lowers the gain too, and raises it again (up to `max_gain`) when dim."""
        self.camera, self.exposure_s, self.gain, self.roi = camera, exposure_s, gain, roi
        self.exposure_range, self._base_gain = exposure_range, max_gain or gain
        self.lock = lock or threading.Lock()
        self.stopped_because: str | None = None  # a spoken reason, when it ended by itself
        self._saved = (camera.exposure_s, camera.gain)
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self) -> "LiveView":
        """Apply the video settings (under the lock: the tracker may be mid-capture) and run.
        If that fails, the camera's settings are put back before the error goes on."""
        try:
            with self.lock:
                self.camera.set_exposure(self.exposure_s)
                self.camera.set_gain(self.gain)
                if self.roi is not None:
                    self.camera.set_roi(self.roi)
            self._thread.start()
        except Exception:
            self.stop()
            raise
        return self

    def nudge(self, factor: float) -> None:
        """Manual exposure (the page's Brighter/Darker): turns auto-exposure off. Past the
        shortest exposure, Darker lowers the gain; Brighter brings the gain back first."""
        with self.lock:
            self.exposure_range = None
            if factor < 1 and self.exposure_s <= MIN_EXPOSURE_S and self.gain > 0:
                self.gain = int(self.gain * factor)
                self.camera.set_gain(self.gain)
            elif factor > 1 and self.gain < self._base_gain:
                self.gain = min(round(self.gain * factor) + 1, self._base_gain)
                self.camera.set_gain(self.gain)
            else:
                self.exposure_s = min(max(self.exposure_s * factor, MIN_EXPOSURE_S), MAX_EXPOSURE_S)
                self.camera.set_exposure(self.exposure_s)

    @property
    def running(self) -> bool:
        return self._thread.is_alive()

    def stop(self) -> None:
        """Stop, wait for the last frame, and restore the camera's settings."""
        self._stop.set()
        if self._thread.is_alive() and threading.current_thread() is not self._thread:
            self._thread.join()
        try:
            with self.lock:
                self.camera.set_exposure(self._saved[0])
                self.camera.set_gain(self._saved[1])
                if self.roi is not None:
                    self.camera.set_roi(None)
        except (RuntimeError, OSError) as e:  # unplugged: nothing to put back
            log.warning("couldn't restore camera settings: %r", e)

    def _run(self) -> None:
        failures = 0
        while not self._stop.is_set():
            try:
                with self.lock:
                    frame = self.camera.capture()  # the camera's tap keeps it for the stream
                    if self.exposure_range is not None:
                        self._auto_expose(frame)
            except (RuntimeError, OSError) as e:  # a hiccup: try again, the page shows the last frame
                log.warning("live video capture failed: %r", e)
                failures += 1
                if failures >= MAX_FAILURES:  # unplugged: say so instead of a frozen picture
                    self.stopped_because = "the camera isn't responding. Is it unplugged?"
                    return
                self._stop.wait(0.5)
                continue
            failures = 0

    def _auto_expose(self, frame: np.ndarray) -> None:
        low, high = self.exposure_range
        peak = np.percentile(frame[::4, ::4], 99.5)
        step = STEP * STEP if peak >= 250 or peak < LOW_PEAK / 4 else STEP  # far off: hurry
        if peak > HIGH_PEAK and self.exposure_s <= low and self.gain > 0:  # daylight: lower the gain too
            self.gain = int(self.gain / step)
            self.camera.set_gain(self.gain)
            return
        if peak < LOW_PEAK and self.gain < self._base_gain:  # dim again: gain back first
            self.gain = min(round(self.gain * step) + 1, self._base_gain)
            self.camera.set_gain(self.gain)
            return
        if peak < LOW_PEAK:
            new = min(self.exposure_s * step, high)
        elif peak > HIGH_PEAK:
            new = max(self.exposure_s / step, low)
        else:
            return
        if new != self.exposure_s:
            self.exposure_s = new
            self.camera.set_exposure(new)
