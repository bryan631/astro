"""Background live stacking for deep-sky targets: short subs -> registered running stack.

The scope doesn't track, so subs stay short (stars drift ~30 px/s at prime focus) and the
target slides out of the 32' field within about two minutes; stacking stops then.
"""

import threading
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from PIL import Image

from astro.capture.recorder import CaptureRefused, SafetyCheck
from astro.devices.base import Camera
from astro.process.livestack import LiveStack, stretch

SUB_EXPOSURE_S = 0.2  # ~6 px of drift at prime focus: still round-ish stars
SUB_GAIN = 300
PREVIEW_EVERY_S = 3.0  # refresh the tablet's live view this often
MAX_SKIPS_IN_A_ROW = 10  # target drifted away, or clouds
SAFETY_CHECK_S = 1.0


@dataclass
class LiveSession:
    name: str
    preview: Path
    frames: int = 0
    skipped: int = 0
    error: str = ""
    preview_version: int = 0  # bumps whenever the preview file changes
    done: threading.Event = field(default_factory=threading.Event)


class LiveStacker:
    def __init__(self, camera: Camera, out_dir: Path, safety: SafetyCheck):
        self.camera, self.out_dir, self.safety = camera, out_dir, safety
        self._stop = threading.Event()
        self.current: LiveSession | None = None

    @property
    def busy(self) -> bool:
        return self.current is not None and not self.current.done.is_set()

    def start(self, name: str, seconds: float) -> LiveSession:
        if self.busy:
            raise CaptureRefused("I'm already stacking.")
        if reason := self.safety():
            raise CaptureRefused(f"I can't take pictures now: {reason}.")
        try:
            self.camera.set_roi(None)
            self.camera.set_exposure(SUB_EXPOSURE_S)
            self.camera.set_gain(SUB_GAIN)
        except (RuntimeError, OSError) as e:
            raise CaptureRefused(f"The main camera isn't responding: {e}") from e
        stamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S_%f")
        self.out_dir.mkdir(parents=True, exist_ok=True)
        live = LiveSession(name, self.out_dir / f"{stamp}_{name.replace(' ', '_')}.png")
        self._stop.clear()
        self.current = live
        threading.Thread(target=self._run, args=(live, seconds), daemon=True).start()
        return live

    def stop(self) -> None:
        self._stop.set()

    def _run(self, live: LiveSession, seconds: float) -> None:
        stack = LiveStack(self.camera.bayer)
        end = time.monotonic() + seconds
        checked = saved = -1e9
        skips_in_a_row = 0
        try:
            while time.monotonic() < end and not self._stop.is_set():
                now = time.monotonic()
                if now - checked >= SAFETY_CHECK_S:
                    checked = now
                    if reason := self.safety():
                        live.error = f"I stopped stacking: {reason}."
                        break
                if stack.add(self.camera.capture()):
                    live.frames, skips_in_a_row = stack.status.frames_added, 0
                else:
                    live.skipped, skips_in_a_row = live.skipped + 1, skips_in_a_row + 1
                    if skips_in_a_row >= MAX_SKIPS_IN_A_ROW:
                        live.error = ("I lost the stars, maybe clouds, or the target drifted out "
                                      "of view, so I stopped stacking.")
                        break
                if time.monotonic() - saved >= PREVIEW_EVERY_S:
                    saved = time.monotonic()
                    self._save(stack, live)
        except (RuntimeError, OSError, ValueError) as e:
            live.error = f"Stacking failed: {e}"
        finally:
            try:
                if live.frames:
                    self._save(stack, live)
            finally:
                live.done.set()

    def _save(self, stack: LiveStack, live: LiveSession) -> None:
        Image.fromarray(stretch(stack.image())).save(live.preview)
        live.preview_version += 1
