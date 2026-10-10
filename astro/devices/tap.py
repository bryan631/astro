"""Keep each camera's last frame, so the tablet can show what the telescope sees."""
import time

import numpy as np

from astro.devices.stream import CameraStream


class TappedCamera:
    """Wraps a camera. Whoever captures (tracker, focus coach, stacker, recorder), the last frame
    is kept; everything else passes through, so nothing is captured just to be displayed."""

    def __init__(self, camera):
        self._camera, self.last, self.last_at = camera, None, 0.0

    def capture(self):
        frame = self._camera.capture()
        self.last, self.last_at = frame, time.monotonic()
        return frame

    def __getattr__(self, name):  # exposure_s, gain, bayer, set_roi, close, ...
        return getattr(self._camera, name)


def tap(camera):
    """Streams keep their own latest frame (astro/devices/stream.py)."""
    if camera is None or isinstance(camera, (TappedCamera, CameraStream)):
        return camera
    return TappedCamera(camera)


def frame_with_time(cam) -> tuple[np.ndarray, int, float] | None:
    """(frame, frame number, mid-exposure Unix time) of a camera's newest frame, or None."""
    if cam is None:
        return None
    if hasattr(cam, "latest"):  # a stream (astro/devices/stream.py)
        frame, seq, t_mid = cam.latest()
        return None if frame is None else (frame, seq, t_mid)
    if getattr(cam, "last", None) is None:  # a plain tapped camera (tests)
        return None
    t_end = time.time() - (time.monotonic() - cam.last_at)
    return cam.last, round(cam.last_at * 1e6), t_end - cam.exposure_s / 2
