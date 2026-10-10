"""Background live stacking for deep-sky targets: short subs -> registered running stack.

The scope doesn't track, so subs stay short (stars drift ~30 px/s at prime focus) and the
target slides out of the 32' field within about two minutes; stacking stops then.
"""

import os
import threading
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from PIL import Image

from astro.capture.edge import EdgeClock
from astro.capture.recorder import CaptureRefused, safe_name
from astro.devices.base import Camera
from astro.process.finish import finish, save_fits
from astro.process.livestack import LiveStack, stretch

SUB_EXPOSURE_S = 0.2  # ~6 px of drift at prime focus: still round-ish stars
SUB_GAIN = 300
PREVIEW_EVERY_S = 3.0  # refresh the tablet's live view this often
MAX_SKIPS_IN_A_ROW = 10  # target drifted away, or clouds
PREVIEW_WIDTH = 960  # the page's live-stack picture: a small JPEG (the full PNG was 4.5 MB)
PAUSED_WAIT_S = 0.1


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
    edge: EdgeClock | None = None  # where the stack's field is in the frame; when it hits the edge


class LiveStacker:
    def __init__(self, camera: Camera, out_dir: Path,
                 preview_dir: Path | None = None):
        """`out_dir` gets finished pictures (the gallery); previews go to `preview_dir`."""
        self.camera, self.out_dir = camera, out_dir
        self.preview_dir = preview_dir or out_dir.parent / "live"
        self._stop = threading.Event()
        self.current: LiveSession | None = None
        self._restore: tuple[float, int] = (0.0, 0)  # camera mode to go back to after a stack

    @property
    def busy(self) -> bool:
        return self.current is not None and not self.current.done.is_set()

    def start(self, name: str, seconds: float) -> LiveSession:
        if self.busy:
            raise CaptureRefused("I'm already stacking.")
        # Remember the camera's mode so planetary work afterwards isn't stuck at 0.2 s / gain 300.
        self._restore = (self.camera.exposure_s, self.camera.gain)
        try:
            self.camera.set_roi(None)
            self.camera.set_exposure(SUB_EXPOSURE_S)
            self.camera.set_gain(SUB_GAIN)
        except (RuntimeError, OSError) as e:
            raise CaptureRefused(f"The main camera isn't responding: {e}") from e
        stamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S_%f")
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.preview_dir.mkdir(parents=True, exist_ok=True)
        file = f"{stamp}_{safe_name(name)}"
        live = LiveSession(name, self.preview_dir / f"{file}.jpg", self.out_dir / f"{file}.png")
        self._stop.clear()
        self.current = live
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
        stack = LiveStack(self.camera.bayer)
        end = time.monotonic() + seconds
        saved = -1e9  # save on the first frame
        self._skips_in_a_row = 0
        try:
            while time.monotonic() < end and not self._stop.is_set():
                if live.paused.is_set():
                    live.edge and live.edge.reset()
                    self._stop.wait(PAUSED_WAIT_S)
                    continue
                if stack.add(self.camera.capture()):
                    live.frames, self._skips_in_a_row = stack.status.frames_added, 0
                    if (where := stack.reference_center()) is not None:
                        if live.edge is None:
                            live.edge = EdgeClock(*stack.frame_size)
                        live.edge.add(time.monotonic(), *where)
                else:
                    live.skipped = stack.status.frames_skipped
                    self._skips_in_a_row += 1
                    if self._skips_in_a_row >= MAX_SKIPS_IN_A_ROW:
                        live.error = ("I lost the stars, maybe clouds, or the target drifted out "
                                      "of view, so I stopped stacking.")
                        break
                if stack.has_frames and time.monotonic() - saved >= PREVIEW_EVERY_S:
                    saved = time.monotonic()
                    self._save(stack, live)
        except (RuntimeError, OSError, ValueError) as e:
            live.error = f"Stacking failed: {e}"
        finally:
            try:
                if stack.has_frames:
                    self._finish(stack, live)
            except (RuntimeError, OSError, ValueError) as e:
                live.frames = 0  # no picture to announce
                live.error = f"I couldn't save the stacked picture: {e}"
            try:
                self._restore_mode()
            except (RuntimeError, OSError) as e:
                live.error = live.error or f"Stacking stopped, and the camera did not reset: {e}"
            finally:
                live.done.set()

    def _restore_mode(self) -> None:
        exposure, gain = self._restore
        self.camera.set_exposure(exposure)
        self.camera.set_gain(gain)

    def _finish(self, stack: LiveStack, live: LiveSession) -> None:
        """The gallery picture: gradient removed, color balanced, stretched (C4). The linear
        stack is kept as FITS beside it, for Siril or GraXpert later."""
        linear = stack.image()
        save_fits(linear, live.picture.with_suffix(".fits"))
        tmp = live.picture.with_name(live.picture.name + ".tmp")
        Image.fromarray(finish(linear)).save(tmp, format="PNG")
        os.replace(tmp, live.picture)
        live.preview.unlink(missing_ok=True)

    def _save(self, stack: LiveStack, live: LiveSession) -> None:
        """Write next to the preview, then swap it in: the tablet never reads a half file."""
        tmp = live.preview.with_name(live.preview.name + ".tmp")
        img = Image.fromarray(stretch(stack.image()))
        if img.width > PREVIEW_WIDTH:
            img = img.resize((PREVIEW_WIDTH, round(img.height * PREVIEW_WIDTH / img.width)))
        img.save(tmp, format="JPEG", quality=85)
        os.replace(tmp, live.preview)
        live.preview_version += 1
