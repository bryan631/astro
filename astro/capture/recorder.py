"""Background planetary recording: SER video with an ROI that follows the planet."""

import shutil
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from astro.capture.roi import brightest_blob, roi_around
from astro.capture.ser import SerWriter
from astro.devices.base import Camera, Roi

ROI_PX = 512  # square planet ROI, sensor pixels
RECENTER_EVERY = 50  # frames between drift checks
RECENTER_FRACTION = 0.25  # re-center when the planet drifts this far from the ROI center
SAFETY_CHECK_S = 1.0  # pointing safety (Sun, daytime) is re-checked this often while recording
KEEP_RECORDINGS = 3  # raw SER videos kept after processing (names sort by time)
MIN_FREE_BYTES = 2 * 1024**3  # a 60 s recording is ~1.5 GB

# Returns a spoken reason when exposing now is unsafe, else None (see astro/safety.py).
SafetyCheck = Callable[[], str | None]


def prune(out_dir: Path, keep: int = KEEP_RECORDINGS) -> None:
    """Delete all but the newest `keep` recordings (each is ~1.5 GB; pictures are kept)."""
    for old in sorted(out_dir.glob("*.ser"))[:-keep or None]:
        old.unlink(missing_ok=True)


class CaptureRefused(Exception):
    """Recording could not start; the message is a spoken reason."""


@dataclass
class Recording:
    path: Path
    name: str  # target at capture time (the session's target may change while recording)
    frames: int = 0
    lost: bool = False  # planet left the frame
    error: str = ""  # why recording failed or stopped early (empty on success)
    done: threading.Event = field(default_factory=threading.Event)


class Recorder:
    def __init__(self, camera: Camera, sensor_size: tuple[int, int], out_dir: Path,
                 safety: SafetyCheck):
        self.camera, self.sensor, self.out_dir, self.safety = camera, sensor_size, out_dir, safety
        self._stop = threading.Event()
        self.current: Recording | None = None

    @property
    def busy(self) -> bool:
        return self.current is not None and not self.current.done.is_set()

    def start(self, name: str, seconds: float) -> Recording:
        """Find the planet, set the ROI and record in a background thread.

        Raises CaptureRefused (with a spoken reason) if busy, unsafe, or no planet is in view.
        """
        if self.busy:
            raise CaptureRefused("I'm already recording.")
        if reason := self.safety():
            raise CaptureRefused(f"I can't take pictures now: {reason}.")
        self.out_dir.mkdir(parents=True, exist_ok=True)
        if shutil.disk_usage(self.out_dir).free < MIN_FREE_BYTES:
            raise CaptureRefused("The disk is nearly full, so I can't record. "
                                 "Old recordings need to be cleared.")
        try:
            self.camera.set_roi(None)
            center = brightest_blob(self.camera.capture())
        except (RuntimeError, OSError) as e:  # SDK gave up after its retry
            raise CaptureRefused(f"The main camera isn't responding: {e}") from e
        if center is None:
            raise CaptureRefused("I don't see anything bright in the main camera. "
                                 "Let's center it first.")
        roi = roi_around(center, ROI_PX, self.sensor)
        try:
            self.camera.set_roi(roi)
        except (RuntimeError, OSError) as e:
            raise CaptureRefused(f"The main camera isn't responding: {e}") from e
        stamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S_%f")  # microseconds: never reuse a name
        self.out_dir.mkdir(parents=True, exist_ok=True)
        rec = Recording(self.out_dir / f"{stamp}_{name.replace(' ', '_')}.ser", name)
        self._stop.clear()
        self.current = rec
        threading.Thread(target=self._run, args=(rec, roi, seconds), daemon=True).start()
        return rec

    def stop(self) -> None:
        self._stop.set()

    def _run(self, rec: Recording, roi: Roi, seconds: float) -> None:
        end = time.monotonic() + seconds
        checked = -SAFETY_CHECK_S
        try:
            with SerWriter(rec.path, roi.width, roi.height, bayer=self.camera.bayer) as ser:
                while time.monotonic() < end and not self._stop.is_set():
                    if time.monotonic() - checked >= SAFETY_CHECK_S:
                        checked = time.monotonic()
                        if reason := self.safety():
                            rec.error = f"I stopped recording: {reason}."
                            break
                    frame = self.camera.capture()
                    ser.write(frame)
                    rec.frames += 1
                    if rec.frames % RECENTER_EVERY == 0:
                        roi = self._recenter(frame, roi, rec)
                        if rec.lost:
                            rec.error = "The planet drifted out of view, so I stopped early."
                            break
        except (RuntimeError, OSError, ValueError) as e:  # SDK error, disk full, bad frame
            rec.error = f"Recording failed: {e}"
        finally:
            try:
                self.camera.set_roi(None)
            except (RuntimeError, OSError) as e:  # e.g. camera unplugged: still report and finish
                rec.error = rec.error or f"Recording stopped, and the camera did not reset: {e}"
            finally:
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
