"""Background planetary recording: SER video with an ROI that follows the planet."""

import logging
import re
import shutil
import threading
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from astro.capture.edge import EdgeClock
from astro.capture.roi import brightest_blob, companions, roi_around
from astro.capture.ser import SerWriter
from astro.devices.base import Camera, Roi

log = logging.getLogger(__name__)


def safe_name(name: str) -> str:
    """A target name as a file name part: "Barnard's Star" -> "Barnards_Star"."""
    return re.sub(r"[^A-Za-z0-9_-]", "", name.replace(" ", "_")) or "target"


ROI_PX = 512  # the smallest planet ROI, sensor pixels (room for the seeing and the drift)
MOON_REACH_PX = 1400  # moons this far from the planet join the ROI (Callisto: ~10' at 0.5"/px)
MAX_ROI_PIXELS = 2048 * 640  # the farthest moons drop out past this (disk space, frame rate)
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
PLANET_PEAK = (150, 230)  # 8-bit: the disk's brightest pixels land here (Jupiter clipped at 20 ms)
MIN_PLANET_EXPOSURE_S = 0.0002


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
        self.roi: Roi | None = None  # the ROI being recorded (the main view pastes it in place)
        self.background: np.ndarray | None = None  # the full live view from just before recording
        self._planet_in_roi = (ROI_PX / 2, ROI_PX / 2)  # where the planet sits in the ROI

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
        """Planet settings, then an ROI around the brightest thing in the full frame and any
        moons near it."""
        try:
            self.camera.set_roi(None)
            last = getattr(self.camera, "last", None)  # the live view, before planet settings
            self.background = None if last is None else last.copy()
            # Short exposures freeze the seeing and keep the planet from burning out; the video's
            # 0.25 s gave white blobs (2026-10-10).
            self.camera.set_exposure(PLANET_EXPOSURE_S)
            self.camera.set_gain(PLANET_GAIN)
            full = self.camera.capture()
            center = brightest_blob(full)
        except (RuntimeError, OSError) as e:  # SDK gave up after its retry
            raise CaptureRefused(f"The telescope camera isn't responding: {e}") from e
        if center is None:
            raise CaptureRefused("I don't see anything bright in the telescope view. "
                                 "Let's center it first.")
        roi = self._roi_with_moons(center, companions(full, center, MOON_REACH_PX))
        try:
            self._set_roi(roi)
            self._meter()
        except (RuntimeError, OSError) as e:
            raise CaptureRefused(f"The telescope camera isn't responding: {e}") from e
        return roi

    def _roi_with_moons(self, planet, moons: list) -> Roi:
        """The ROI_PX box around the planet, grown to take in its moons (brightest first, while
        it stays under MAX_ROI_PIXELS), with ROI_PX / 2 to spare around the planet and each
        moon: the planet drifts up to the recenter threshold before the ROI follows it.
        Remembers where the planet sits in it."""
        def box(pts):
            xs, ys = zip(*pts, strict=True)
            return (max(max(xs) - min(xs) + ROI_PX, ROI_PX), max(max(ys) - min(ys) + ROI_PX, ROI_PX),
                    (max(xs) + min(xs)) / 2, (max(ys) + min(ys)) / 2)
        pts = [planet]
        for moon in moons:
            w, h, _, _ = box([*pts, moon])
            if w * h > MAX_ROI_PIXELS:
                break
            pts.append(moon)
        w, h, cx, cy = box(pts)
        roi = roi_around((cx, cy), int(w), self.sensor, int(h))
        self._planet_in_roi = (planet[0] - roi.x, planet[1] - roi.y)
        return roi

    def _roi_for(self, planet) -> Roi:
        """The recording's ROI moved so the planet sits where it did at the start."""
        size = (self.roi.width, self.roi.height) if self.roi else (ROI_PX, ROI_PX)
        center = (planet[0] - self._planet_in_roi[0] + size[0] / 2, planet[1] - self._planet_in_roi[1] + size[1] / 2)
        return roi_around(center, size[0], self.sensor, size[1])

    def _set_roi(self, roi: Roi | None) -> None:
        self.camera.set_roi(roi)
        if roi is not None:
            self.roi = roi

    def _meter(self) -> None:
        """Exposure for the disk: its brightest pixels (0.1% of the ROI, inside the disk) in
        PLANET_PEAK, so the video keeps detail instead of a white disk (2026-10-10)."""
        exposure = PLANET_EXPOSURE_S
        for _ in range(6):
            peak = float(np.percentile(self.camera.capture(), 99.9))
            if PLANET_PEAK[0] <= peak <= PLANET_PEAK[1]:
                return
            exposure *= 0.25 if peak >= 250 else sum(PLANET_PEAK) / 2 / max(peak, 1.0)
            exposure = float(np.clip(exposure, MIN_PLANET_EXPOSURE_S, 4 * PLANET_EXPOSURE_S))
            self.camera.set_exposure(exposure)

    def _restore_view(self) -> None:
        """Back to the live view's full frame and settings (the main view stays useful)."""
        self.roi, self.background = None, None
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
                        roi = self._roi_for(blob)
                        self._set_roi(roi)
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
        dx, dy = blob[0] - self._planet_in_roi[0], blob[1] - self._planet_in_roi[1]
        if max(abs(dx), abs(dy)) < RECENTER_FRACTION * ROI_PX:
            return roi
        new = self._roi_for((roi.x + blob[0], roi.y + blob[1]))
        self._set_roi(new)  # real camera reopens here; a dropped frame is fine
        return new
