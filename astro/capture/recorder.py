"""Background planetary recording: SER video with an ROI that follows the planet."""

import threading
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from astro.capture.roi import brightest_blob, roi_around
from astro.capture.ser import SerWriter
from astro.devices.base import Camera, Roi

ROI_PX = 512  # square planet ROI, sensor pixels
RECENTER_EVERY = 50  # frames between drift checks
RECENTER_FRACTION = 0.25  # re-centre when the planet drifts this far from the ROI centre


@dataclass
class Recording:
    path: Path
    frames: int = 0
    lost: bool = False  # planet left the frame
    done: threading.Event = field(default_factory=threading.Event)


class Recorder:
    def __init__(self, camera: Camera, sensor_size: tuple[int, int], out_dir: Path):
        self.camera, self.sensor, self.out_dir = camera, sensor_size, out_dir
        self._stop = threading.Event()
        self.current: Recording | None = None

    def start(self, name: str, seconds: float) -> Recording | None:
        """Find the planet, set the ROI and record in a background thread. None if no planet."""
        self.camera.set_roi(None)
        center = brightest_blob(self.camera.capture())
        if center is None:
            return None
        roi = roi_around(center, ROI_PX, self.sensor)
        self.camera.set_roi(roi)
        stamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
        self.out_dir.mkdir(parents=True, exist_ok=True)
        rec = Recording(self.out_dir / f"{stamp}_{name.replace(' ', '_')}.ser")
        self._stop.clear()
        self.current = rec
        threading.Thread(target=self._run, args=(rec, roi, seconds), daemon=True).start()
        return rec

    def stop(self) -> None:
        self._stop.set()

    def _run(self, rec: Recording, roi: Roi, seconds: float) -> None:
        end = time.monotonic() + seconds
        try:
            with SerWriter(rec.path, roi.width, roi.height, bayer=self.camera.bayer) as ser:
                while time.monotonic() < end and not self._stop.is_set():
                    frame = self.camera.capture()
                    ser.write(frame)
                    rec.frames += 1
                    if rec.frames % RECENTER_EVERY == 0:
                        roi = self._recenter(frame, roi, rec)
                        if rec.lost:
                            break
        finally:
            self.camera.set_roi(None)
            rec.done.set()

    def _recenter(self, frame, roi: Roi, rec: Recording) -> Roi:
        blob = brightest_blob(frame)
        if blob is None:
            rec.lost = True
            return roi
        dx, dy = blob[0] - roi.width / 2, blob[1] - roi.height / 2
        if max(abs(dx), abs(dy)) < RECENTER_FRACTION * roi.width:
            return roi
        new = roi_around((roi.x + blob[0], roi.y + blob[1]), roi.width, self.sensor)
        self.camera.set_roi(new)  # real camera reopens here; a dropped frame is fine
        return new
