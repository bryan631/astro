"""Live video of one camera, for checking that everything is right: cap off, focus close, framing.

Runs only while someone is watching (the page's stream calls `touch`), uses a short exposure
for speed, asks the exposure gate before every frame, and puts the camera's settings back.
"""
import threading
import time
from collections.abc import Callable
from contextlib import AbstractContextManager

from astro.devices.base import Camera, Roi

IDLE_STOP_S = 15.0  # nobody watching for this long: stop, so the camera isn't left in preview


class LiveView:
    def __init__(self, camera: Camera, exposure_s: float, gain: int, roi: Roi | None = None,
                 lock: AbstractContextManager | None = None,
                 allowed: Callable[[], str | None] = lambda: None):
        """`allowed()` is the exposure gate: a spoken reason to stop, or None. `lock` is the
        camera's lock when another thread (the plate-solve tracker) also captures."""
        self.camera, self.exposure_s, self.gain, self.roi = camera, exposure_s, gain, roi
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
                    self.camera.capture()  # the camera's tap keeps the frame for the stream
            except (RuntimeError, OSError):  # a hiccup: try again, the page shows the last frame
                self._stop.wait(0.5)
