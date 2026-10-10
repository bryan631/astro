"""Background planetary recording: SER video with an ROI that follows the planet."""

import logging
import re
import shutil
import threading
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from astro.capture.edge import EdgeClock
from astro.capture.roi import brightest_blob, roi_around
from astro.capture.ser import SerWriter
from astro.devices.base import Camera, Roi

log = logging.getLogger(__name__)


def safe_name(name: str) -> str:
    """A target name as a file name part: "Barnard's Star" -> "Barnards_Star"."""
    return re.sub(r"[^A-Za-z0-9_-]", "", name.replace(" ", "_")) or "target"


ROI_PX = 512  # square planet ROI, sensor pixels
RECENTER_EVERY = 50  # frames between drift checks
RECENTER_FRACTION = 0.25  # re-center when the planet drifts this far from the ROI center
EDGE_EVERY = 10  # frames between planet positions for the edge clock (a blob search is cheap)
PAUSED_WAIT_S = 0.1
KEEP_RECORDINGS = 3  # raw SER videos kept after processing (names sort by time)
MIN_FREE_BYTES = 2 * 1024**3  # a 60 s recording is ~1.5 GB

def prune(out_dir: Path, keep: int = KEEP_RECORDINGS) -> None:
    """Delete all but the newest `keep` recordings (each is ~1.5 GB; pictures are kept)."""
    for old in sorted(out_dir.glob("*.ser"))[:-keep or None]:
        old.unlink(missing_ok=True)


PLANET_EXPOSURE_S, PLANET_GAIN = 0.02, 250


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
    paused: threading.Event = field(default_factory=threading.Event)  # Recenter: nothing written
    edge: EdgeClock | None = None  # where the planet is on the sensor, and when it reaches the edge


class Recorder:
    def __init__(self, camera: Camera, sensor_size: tuple[int, int], out_dir: Path):
        self.camera, self.sensor, self.out_dir = camera, sensor_size, out_dir
        self._stop = threading.Event()
        self.current: Recording | None = None

    @property
    def busy(self) -> bool:
        return self.current is not None and not self.current.done.is_set()

    def start(self, name: str, seconds: float) -> Recording:
        """Find the planet, set the ROI and record in a background thread.

        Raises CaptureRefused (with a spoken reason) if busy or no planet is in view.
        """
        if self.busy:
            raise CaptureRefused("I'm already recording.")
        self.out_dir.mkdir(parents=True, exist_ok=True)
        if shutil.disk_usage(self.out_dir).free < MIN_FREE_BYTES:
            raise CaptureRefused("The disk is nearly full, so I can't record. "
                                 "Old recordings need to be cleared.")
        self._restore = (self.camera.exposure_s, self.camera.gain)  # the live view's, for after
        try:
            roi = self._find_planet()
        except CaptureRefused:
            self._restore_view()
            raise
        stamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S_%f")  # microseconds: never reuse a name
        self.out_dir.mkdir(parents=True, exist_ok=True)
        rec = Recording(self.out_dir / f"{stamp}_{safe_name(name)}.ser", name,
                        edge=EdgeClock(*self.sensor))
        self._stop.clear()
        self.current = rec
        log.info("recording", extra={"data": {"name": name, "seconds": seconds,
                                              "sensor_temp_c": self.camera.temperature_c()}})
        threading.Thread(target=self._run, args=(rec, roi, seconds), daemon=True).start()
        return rec

    def _find_planet(self) -> Roi:
        """Planet settings, then an ROI around the brightest thing in the full frame."""
        try:
            self.camera.set_roi(None)
            # Short exposures freeze the seeing and keep the planet from burning out; the video's
            # 0.25 s gave white blobs (2026-10-10).
            self.camera.set_exposure(PLANET_EXPOSURE_S)
            self.camera.set_gain(PLANET_GAIN)
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
        return roi

    def _restore_view(self) -> None:
        """Back to the live view's full frame and settings (the main view stays useful)."""
        self.camera.set_roi(None)
        self.camera.set_exposure(self._restore[0])
        self.camera.set_gain(self._restore[1])

    def stop(self) -> None:
        self._stop.set()

    def pause(self) -> None:
        """Recenter: stop writing, show the whole field until resume()."""
        if self.current is not None:
            self.current.paused.set()

    def resume(self) -> None:
        if self.current is not None:
            self.current.paused.clear()

    def _run(self, rec: Recording, roi: Roi | None, seconds: float) -> None:
        end = time.monotonic() + seconds
        size = (roi.height, roi.width)  # every frame in the file has the ROI's size
        try:
            with SerWriter(rec.path, roi.width, roi.height, bayer=self.camera.bayer) as ser:
                while time.monotonic() < end and not self._stop.is_set():
                    if rec.paused.is_set():  # recentering: the whole field on the main view
                        if roi is not None:
                            self.camera.set_roi(None)
                            roi = None
                            rec.edge.reset()
                        self._stop.wait(PAUSED_WAIT_S)
                        continue
                    if roi is None:  # resumed: find the planet again and follow it
                        blob = brightest_blob(self.camera.capture())
                        if blob is None:
                            self._stop.wait(PAUSED_WAIT_S)
                            continue
                        roi = roi_around(blob, size[1], self.sensor)
                        self.camera.set_roi(roi)
                    frame = self.camera.capture()
                    if frame.shape != size:  # taken before the ROI changed
                        continue
                    ser.write(frame)
                    rec.frames += 1
                    if rec.frames % EDGE_EVERY == 0 and (blob := brightest_blob(frame)) is not None:
                        rec.edge.add(time.monotonic(), roi.x + blob[0], roi.y + blob[1])
                    if rec.frames % RECENTER_EVERY == 0:
                        roi = self._recenter(frame, roi, rec)
                        if rec.lost:
                            rec.error = "The planet drifted out of view, so I stopped early."
                            break
        except (RuntimeError, OSError, ValueError) as e:  # SDK error, disk full, bad frame
            rec.error = f"Recording failed: {e}"
        finally:
            try:
                self._restore_view()
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
