"""'Center it': finish guidance using the main camera itself.

The finder model gets the target into the main camera's small field, but usually off-center
(finder-to-main offset). This step learns the main camera's orientation from two small pushes
(once per session), then guides the target to the image center and remembers the offset so
later gotos land centered.
"""

from dataclasses import dataclass, field

import numpy as np

from astro.pointing.main_offset import MIN_MOTION_DEG, CameraAxes, MainOffset, local_delta

CENTERED_FRACTION = 0.05  # within 5% of the image width of center counts as centered
REASK_AFTER = 4  # updates without enough motion before asking again
LOST = "I can't see it in the telescope view. Let's find it again with the finder."
CENTERED = "stop, it's centered"
CALIBRATED = "thanks, now I can center it"


def centering_phrases() -> list[str]:
    """Everything centering can say beyond the guide's cues, for pre-rendering speech."""
    asks = [f"{p}push {d} a tiny bit, then stop" for d in ("left", "up") for p in ("", "now ")]
    more = [f"push a little more {d}, then stop" for d in ("left", "up")]
    return [LOST, CENTERED, CALIBRATED, *asks, *more]


@dataclass
class Step:
    say: str | None
    done: bool = False
    lost: bool = False  # the target isn't in the main camera's view


@dataclass
class Centerer:
    image_size: tuple[int, int]  # (w, h) of the frames passed to update()
    axes: CameraAxes = field(default_factory=CameraAxes)
    offset: MainOffset = field(default_factory=MainOffset)
    _target: tuple[float, float] | None = None  # true (alt, az) of the target being centered
    _start: tuple[tuple[float, float], np.ndarray] | None = None  # (position, target px) at handover
    _last: tuple[tuple[float, float], np.ndarray] | None = None  # last calibration point
    _asked: str = ""
    _waiting: int = 0  # updates since the last request without enough motion
    _observed: bool = False

    @property
    def calibrated(self) -> bool:
        return self.axes.matrix is not None

    def restart(self, target_altaz: tuple[float, float]) -> None:
        """New attempt at `target_altaz` (true, uncorrected); keeps what was learned."""
        self._target, self._start, self._last = target_altaz, None, None
        self._asked, self._waiting, self._observed = "", 0, False

    def update(self, position: tuple[float, float], target_px: tuple[float, float] | None) -> Step:
        """Feed the scope's (alt, az) and the target's pixel position (None if not visible)."""
        if target_px is None:
            return Step(LOST, lost=True)
        px = np.asarray(target_px, float)
        center = np.asarray(self.image_size, float) / 2
        if self._start is None:
            self._start = (position, px)
        if self.calibrated and not self._observed and self._target is not None:
            # Exact camera offset from the handover view: the finder's leftover error and any
            # correction already applied are in (target - position), not in the camera offset.
            p0, px0 = self._start
            self.offset.observe(self.axes, *(px0 - center),
                                target_from_model=tuple(local_delta(*p0, *self._target)))
            self._observed = True
        if np.hypot(*(px - center)) < CENTERED_FRACTION * self.image_size[0]:
            return Step(CENTERED, done=True)
        if not self.calibrated:
            return self._calibrate(position, px)
        d_az, d_alt = self.axes.sky_offset(*(px - center))  # target's offset from the aim
        if abs(d_az) >= abs(d_alt):  # same words as the guide ("right a little")
            return Step("right a little" if d_az > 0 else "left a little")
        return Step("up a little" if d_alt > 0 else "down a little")

    def _calibrate(self, position: tuple[float, float], px: np.ndarray) -> Step:
        """Ask for one small push per axis and pair encoder motion with image motion."""
        moved = 0.0
        if self._last is not None:
            d_sky = local_delta(*self._last[0], *position)
            moved = float(np.hypot(*d_sky))
            if moved >= MIN_MOTION_DEG:  # big enough for the encoders to register (2+ counts)
                self.axes.add_move(*d_sky, *(px - self._last[1]))
                self._last, self._waiting = (position, px), 0
                if self.calibrated:
                    return Step(CALIBRATED)
        else:
            self._last = (position, px)
        direction = "left" if not self.axes.moves else "up"
        ask = f"push {direction} a tiny bit, then stop"
        if ask != self._asked:
            self._asked, self._waiting = ask, 0
            return Step(ask if direction == "left" else "now " + ask)
        self._waiting += 1
        if self._waiting >= REASK_AFTER:  # the push was too small, or didn't happen
            self._waiting = 0
            return Step(f"push a little more {direction}, then stop" if moved > 0 else ask)
        return Step(None)
