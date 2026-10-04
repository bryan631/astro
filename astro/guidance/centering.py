"""'Center it': finish guidance using the main camera itself.

The finder model gets the target into the main camera's small field, but usually off-center
(finder-to-main offset). This step learns the main camera's orientation from two small pushes
(once per session), then guides the target to the image center and remembers the offset so
later gotos land centered.
"""

from dataclasses import dataclass, field

import numpy as np

from astro.pointing.main_offset import MIN_MOTION_DEG, CameraAxes, MainOffset

CENTERED_FRACTION = 0.05  # within 5% of the image width of center counts as centered
REASK_AFTER = 4  # updates without enough motion before asking again


@dataclass
class Step:
    say: str | None
    done: bool = False


@dataclass
class Centerer:
    image_size: tuple[int, int]  # (w, h) of the frames passed to update()
    axes: CameraAxes = field(default_factory=CameraAxes)
    offset: MainOffset = field(default_factory=MainOffset)
    _last: tuple[np.ndarray, np.ndarray] | None = None  # (sky position, target pixel)
    _asked: str = ""
    _start_px: np.ndarray | None = None  # target pixel when the finder model said "on target"
    _observed: bool = False
    _waiting: int = 0  # updates since the last request without enough motion

    @property
    def calibrated(self) -> bool:
        return self.axes.matrix is not None

    def update(self, position: tuple[float, float], target_px: tuple[float, float] | None) -> Step:
        """Feed the scope's (alt, az) and the target's pixel position (None if not visible)."""
        if target_px is None:
            return Step("I can't see it in the camera. Let's go back to the last spot.")
        alt, az = position
        sky = np.array([az * np.cos(np.radians(alt)), alt])
        px = np.asarray(target_px, float)
        center = np.asarray(self.image_size, float) / 2
        if self._start_px is None:
            self._start_px = px
        if self.calibrated and not self._observed:  # the start view shows the camera's offset
            self.offset.observe(self.axes, *(self._start_px - center))
            self._observed = True
        if np.hypot(*(px - center)) < CENTERED_FRACTION * self.image_size[0]:
            return Step("stop, it's centered", done=True)
        if not self.calibrated:
            return self._calibrate(sky, px)
        d_az, d_alt = self.axes.sky_offset(*(px - center))  # target's offset from the aim
        if abs(d_az) >= abs(d_alt):
            return Step("push right a little" if d_az > 0 else "push left a little")
        return Step("push up a little" if d_alt > 0 else "push down a little")

    def restart(self) -> None:
        """New target: keep what was learned (axes, offset), forget this attempt."""
        self._last, self._asked, self._start_px, self._observed = None, "", None, False

    def _calibrate(self, sky: np.ndarray, px: np.ndarray) -> Step:
        """Ask for one small push per axis and pair encoder motion with image motion."""
        moved = 0.0
        if self._last is not None:
            d_sky, d_px = sky - self._last[0], px - self._last[1]
            moved = float(np.hypot(*d_sky))
            if moved >= MIN_MOTION_DEG:  # big enough for the encoders to register (2+ counts)
                self.axes.add_move(*d_sky, *d_px)
                self._last, self._waiting = (sky, px), 0
                if self.calibrated:
                    return Step("thanks, now I can center it")
        else:
            self._last = (sky, px)
        direction = "left" if not self.axes._sky else "up"
        ask = f"push {direction} a tiny bit, then stop"
        if ask != self._asked:
            self._asked, self._waiting = ask, 0
            return Step(ask if direction == "left" else "now " + ask)
        self._waiting += 1
        if self._waiting >= REASK_AFTER:  # the push was too small, or didn't happen
            self._waiting = 0
            return Step(f"push a little more {direction}, then stop" if moved > 0 else ask)
        return Step(None)
