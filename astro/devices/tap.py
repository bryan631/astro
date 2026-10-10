"""Keep each camera's last frame, so the tablet can show what the telescope sees."""
import time

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
