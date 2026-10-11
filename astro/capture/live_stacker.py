"""Background live stacking for deep-sky targets: short subs -> registered running stack.

The scope doesn't track, so subs stay short (stars drift ~30 px/s at prime focus) and the
target slides out of the 32' field within about two minutes; the stack waits then for the user
to nudge the scope back, and keeps adding to the same picture.
"""

import logging
import os
import threading
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
from PIL import Image

from astro.capture.edge import EdgeClock
from astro.capture.recorder import CaptureRefused, safe_name
from astro.devices.base import Camera
from astro.process.finish import finish, save_fits
from astro.process.livestack import LiveStack, stretch

SUB_EXPOSURE_S = 0.2  # ~6 px of drift at prime focus: still round-ish stars
# The gain is metered (0.1 dB steps) to put the sky near SKY_LEVEL of the 8-bit frame: at the old
# fixed gain 300 the sky sat at ~2 of 255 (2026-10-10), coarse and clipped at black.
START_GAIN = 300
GAIN_RANGE = (100, 500)
SKY_LEVEL = 12.0
CLIPPED_RAW = 250  # a raw level this high is (nearly) saturated
MAX_CLIPPED = 0.0005  # fraction of pixels allowed there (stars, a galaxy's core)
METER_FRAMES = 3
PREVIEW_EVERY_S = 3.0  # refresh the tablet's live view this often
MAX_SKIPS_IN_A_ROW = 10  # target drifted away, or clouds: ask for a nudge back
ADRIFT_GIVE_UP_S = 90.0  # still lost after this long: finish the picture
PREVIEW_WIDTH = 960  # the page's live-stack picture: a small JPEG (the full PNG was 4.5 MB)
PAUSED_WAIT_S = 0.1
log = logging.getLogger(__name__)


@dataclass
class LiveSession:
    name: str
    preview: Path  # rewritten while stacking (data/live/), not in the gallery
    picture: Path  # where the finished stack goes (the gallery)
    frames: int = 0
    skipped: int = 0
    error: str = ""
    preview_version: int = 0  # bumps whenever the preview file changes
    done: threading.Event = field(default_factory=threading.Event)
    paused: threading.Event = field(default_factory=threading.Event)  # Recenter: frames not added
    gain: int = 0  # the metered gain
    missed: int = 0  # frames that came in while another was being stacked
    adrift: bool = False  # lost the field (drifted out, clouds): waiting for a nudge back
    wide: bool = False  # through the finder camera (a target bigger than the telescope's view)
    edge: EdgeClock | None = None  # where the stack's field is in the frame; when it hits the edge


class LiveStacker:
    def __init__(self, camera: Camera, out_dir: Path,
                 preview_dir: Path | None = None):
        """`out_dir` gets finished pictures (the gallery); previews go to `preview_dir`."""
        self.camera, self.main_camera, self.out_dir = camera, camera, out_dir
        self.preview_dir = preview_dir or out_dir.parent / "live"
        self._stop = threading.Event()
        self.current: LiveSession | None = None
        self._restore: tuple[float, int] | None = None  # camera mode to go back to after a stack

    @property
    def busy(self) -> bool:
        return self.current is not None and not self.current.done.is_set()

    def start(self, name: str, seconds: float, wide_camera: Camera | None = None) -> LiveSession:
        """Stack the telescope camera's short frames, or `wide_camera`'s (the finder, for a target
        bigger than the telescope's view) at its own settings: they go on feeding the plate solver,
        and at ~30"/px its stars drift under a pixel a second."""
        if self.busy:
            raise CaptureRefused("I'm already stacking.")
        self.camera = wide_camera or self.main_camera
        self._restore = None
        if wide_camera is None:
            # Remember the camera's mode so planetary work afterwards isn't stuck at the stack's.
            self._restore = (self.camera.exposure_s, self.camera.gain)
            try:
                self.camera.set_roi(None)
                self.camera.set_exposure(SUB_EXPOSURE_S)
                self.camera.set_gain(START_GAIN)
            except (RuntimeError, OSError) as e:
                raise CaptureRefused(f"The telescope camera isn't responding: {e}") from e
        stamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S_%f")
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.preview_dir.mkdir(parents=True, exist_ok=True)
        file = f"{stamp}_{safe_name(name)}"
        live = LiveSession(name, self.preview_dir / f"{file}.jpg", self.out_dir / f"{file}.png",
                           wide=wide_camera is not None)
        self._stop.clear()
        self.current = live
        log.info("stacking", extra={"data": {"name": name, "seconds": seconds}})
        threading.Thread(target=self._run, args=(live, seconds), daemon=True).start()
        return live

    def stop(self) -> None:
        self._stop.set()

    def pause(self) -> None:
        """Recenter: keep the stack, add nothing until resume() (it re-registers then)."""
        if self.current is not None:
            self.current.paused.set()

    def resume(self) -> None:
        if self.current is not None:
            self.current.paused.clear()
            self._skips_in_a_row = 0

    def _run(self, live: LiveSession, seconds: float) -> None:
        stack = LiveStack(self.camera.bayer, full_res=live.wide)
        end = time.monotonic() + seconds
        saved = -1e9  # save on the first frame
        self._skips_in_a_row = 0
        adrift_at = 0.0
        try:
            if live.wide:
                live.gain = self.camera.gain
            else:
                live.gain, sky = meter_gain(self.camera, START_GAIN)
                log.info("stacking gain", extra={"data": {"gain": live.gain, "sky": sky}})
            seq = None
            while time.monotonic() < end and not self._stop.is_set():
                if live.paused.is_set():
                    live.edge and live.edge.reset()
                    seq = None  # take up again from the newest frame
                    self._stop.wait(PAUSED_WAIT_S)
                    continue
                raw, seq = self._next_frame(seq, live)
                if raw is None:
                    break
                if stack.add(raw):
                    live.frames, self._skips_in_a_row, live.adrift = stack.status.frames_added, 0, False
                    if (where := stack.reference_center()) is not None:
                        if live.edge is None:
                            live.edge = EdgeClock(*stack.frame_size)
                        live.edge.add(time.monotonic(), *where)
                else:
                    live.skipped = stack.status.frames_skipped
                    self._skips_in_a_row += 1
                    if self._skips_in_a_row >= MAX_SKIPS_IN_A_ROW and not live.adrift:
                        live.adrift, adrift_at = True, time.monotonic()  # the session asks for a nudge
                    if live.adrift and time.monotonic() - adrift_at > ADRIFT_GIVE_UP_S:
                        live.error = ("I lost the stars, maybe clouds, or the target drifted out "
                                      "of view, so I stopped stacking.")
                        break
                if stack.has_frames and time.monotonic() - saved >= PREVIEW_EVERY_S:
                    saved = time.monotonic()
                    self._save(stack, live)
        except (RuntimeError, OSError, ValueError) as e:
            log.exception("stacking failed")
            live.error = f"Stacking failed: {e}"
        finally:
            try:
                if stack.has_frames:
                    self._finish(stack, live)
            except (RuntimeError, OSError, ValueError) as e:
                log.exception("stacking: saving the picture failed")
                live.frames = 0  # no picture to announce
                live.error = f"I couldn't save the stacked picture: {e}"
            try:
                self._restore_mode()
            except (RuntimeError, OSError) as e:
                log.exception("stacking: camera did not reset")
                live.error = live.error or f"Stacking stopped, and the camera did not reset: {e}"
            finally:
                log.info("stacking done", extra={"data": {"name": live.name, "frames": live.frames,
                                                          "skipped": live.skipped, "missed": live.missed,
                                                          "gain": live.gain, "error": live.error}})
                live.done.set()

    def _next_frame(self, after: int | None, live: LiveSession) -> tuple[np.ndarray | None, int | None]:
        """The newest frame after sequence number `after` (None: the next one), and its number.
        A frame that came in while the last was being stacked is taken at once: capture() waits
        for the next exposure. Cameras without sequence numbers (tests) just capture."""
        if after is None or not hasattr(self.camera, "latest"):
            raw = self.camera.capture()
            return raw, getattr(self.camera, "seq", None) if hasattr(self.camera, "latest") else None
        while not self._stop.is_set():
            raw, seq, _ = self.camera.latest()
            if raw is not None and seq > after:
                live.missed += seq - after - 1
                return raw, seq
            self._stop.wait(0.01)
        return None, None

    def _restore_mode(self) -> None:
        if self._restore is None:  # a wide stack kept the finder's settings
            return
        exposure, gain = self._restore
        self.camera.set_exposure(exposure)
        self.camera.set_gain(gain)

    def _finish(self, stack: LiveStack, live: LiveSession) -> None:
        """The gallery picture: gradient removed, color balanced, stretched (C4). The linear
        stack is kept as FITS beside it, for Siril or GraXpert later."""
        linear = stack.picture()
        save_fits(linear, live.picture.with_suffix(".fits"))
        tmp = live.picture.with_name(live.picture.name + ".tmp")
        Image.fromarray(finish(linear)).save(tmp, format="PNG")
        os.replace(tmp, live.picture)
        live.preview.unlink(missing_ok=True)

    def _save(self, stack: LiveStack, live: LiveSession) -> None:
        """Write next to the preview, then swap it in: the tablet never reads a half file."""
        tmp = live.preview.with_name(live.preview.name + ".tmp")
        img = Image.fromarray(stretch(stack.picture()))
        if img.width > PREVIEW_WIDTH:
            img = img.resize((PREVIEW_WIDTH, round(img.height * PREVIEW_WIDTH / img.width)))
        img.save(tmp, format="JPEG", quality=85)
        os.replace(tmp, live.preview)
        live.preview_version += 1


def meter_gain(camera: Camera, gain: int) -> tuple[int, float]:
    """(gain, sky level) that puts the sky near SKY_LEVEL of 255 with at most MAX_CLIPPED
    pixels saturated, in up to METER_FRAMES frames. The camera's gain is in 0.1 dB: 200 steps
    is 10x (the 2026-10-10 stacks at 300 and the night view at 480 agree)."""
    sky = 0.0
    for _ in range(METER_FRAMES):
        camera.set_gain(gain)
        raw = camera.capture()
        sky = float(np.median(raw[::4, ::4]))  # one Bayer color (green), stars don't move a median
        want = gain + 200 * np.log10(SKY_LEVEL / max(sky, 0.5))
        if np.mean(raw[::2, ::2] >= CLIPPED_RAW) > MAX_CLIPPED:
            want = min(want, gain - 60)  # stars or a core burning out: back off
        want = int(np.clip(round(want), *GAIN_RANGE))
        if abs(want - gain) < 20:
            break
        gain = want
    camera.set_gain(gain)
    return gain, sky
