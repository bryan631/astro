"""Push-to guidance: current vs target alt/az -> spoken cues plus UI arrow state.

Deterministic local code; the LLM never drives pointing. Call `update` at ~10 Hz.
One axis is coached at a time (the larger sky error first), with coarse/fine wording,
"slower" when closing fast, "back" after an overshoot, and "stop" with hysteresis.
"""

import math
from dataclasses import dataclass

COARSE_DEG = 3.0
FINE_DEG = 0.5
MIN_SPEAK_INTERVAL_S = 1.5
REPEAT_INTERVAL_S = 4.0
SLOW_DOWN_TIME_S = 1.0  # say "slower" if we'd reach the target within this time


def _directional(text: str) -> bool:
    """Movement instructions, which may repeat as "keep going" (never "stop", "slower", ...)."""
    return text.startswith("push ") or text.endswith((", getting closer", " a little")) \
        and not text.startswith("passed it")


def cue_phrases() -> list[str]:
    """Every phrase the guide can say, for pre-rendering speech."""
    dirs = ("left", "right", "up", "down")
    return ["stop", "slower", "keep going", *(f"push {d}" for d in dirs),
            *(f"{d}, getting closer" for d in dirs), *(f"{d} a little" for d in dirs),
            *(f"passed it, back {d} a little" for d in dirs)]


def wrap180(deg: float) -> float:
    return (deg + 180.0) % 360.0 - 180.0


@dataclass(frozen=True)
class GuideState:
    """What the UI shows every update."""

    d_alt_deg: float  # + means push up
    d_az_sky_deg: float  # + means push toward increasing azimuth, scaled by cos(alt)
    distance_deg: float
    on_target: bool
    beep_hz: float  # 0 = silent; faster beeps when closer


@dataclass(frozen=True)
class Cue:
    text: str
    state: GuideState


class CueLimiter:
    """Keeps cues from talking over each other: a minimum gap between cues, no quick repeats,
    and a repeated movement instruction becomes "keep going". Urgent cues always go through."""

    def __init__(self) -> None:
        self.last_text = ""
        self._last_t = -math.inf

    def speak(self, text: str, t: float, urgent: bool = False) -> str | None:
        """What to say now for `text`, or None to stay quiet."""
        if not urgent:
            if t - self._last_t < MIN_SPEAK_INTERVAL_S:
                return None
            if text == self.last_text and t - self._last_t < REPEAT_INTERVAL_S:
                return None
        spoken = "keep going" if text == self.last_text and _directional(text) else text
        self.last_text, self._last_t = text, t
        return spoken


class Guide:
    def __init__(self, target_alt: float, target_az: float, tolerance_arcmin: float = 4.0,
                 right_is_plus_az: bool = True):
        self.target = (target_alt, target_az)
        self.tol_deg = tolerance_arcmin / 60.0
        self.right_is_plus_az = right_is_plus_az
        self.on_target = False
        self._limiter = CueLimiter()
        self._prev: tuple[float, float, float] | None = None  # (t, d_alt, d_az)
        self._axis: str | None = None

    def update(self, alt: float, az: float, t: float) -> tuple[GuideState, Cue | None]:
        d_alt = self.target[0] - alt
        d_az = wrap180(self.target[1] - az) * math.cos(math.radians(self.target[0]))
        dist = math.hypot(d_alt, d_az)

        # Hysteresis: lock on inside tolerance, release only beyond 1.5x.
        was_on = self.on_target
        self.on_target = dist <= self.tol_deg or (was_on and dist <= 1.5 * self.tol_deg)
        state = GuideState(d_alt, d_az, dist, self.on_target, self._beep(dist))

        text, urgent = self._choose_text(t, d_alt, d_az, was_on)
        self._prev = (t, d_alt, d_az)
        spoken = self._limiter.speak(text, t, urgent) if text else None
        return state, Cue(spoken, state) if spoken else None

    def _choose_text(self, t: float, d_alt: float, d_az: float, was_on: bool
                     ) -> tuple[str | None, bool]:
        """Return (text, urgent). Urgent cues bypass the rate limit."""
        if self.on_target:
            self._axis = None
            return ("stop", True) if not was_on else (None, False)

        axis_tol = self.tol_deg * 0.7  # both axes within this -> total within tolerance
        if self._axis is not None:
            err = d_alt if self._axis == "alt" else d_az
            if abs(err) <= axis_tol:
                # This axis is lined up: stop the user before switching to the other one.
                # Always urgent: it's only reached on the transition, and it must never be
                # rate-limited away while the user is still pushing.
                self._axis = None
                return ("stop", True)
        if self._axis is None:
            self._axis = "alt" if abs(d_alt) >= abs(d_az) else "az"
        err = d_alt if self._axis == "alt" else d_az
        direction = self._direction(self._axis, err)

        if self._prev is not None:
            pt, pd_alt, pd_az = self._prev
            prev_err = pd_alt if self._axis == "alt" else pd_az
            dt = t - pt
            if prev_err * err < 0:
                return f"passed it, back {direction} a little", True
            closing = (abs(prev_err) - abs(err)) / dt if dt > 0 else 0.0
            if closing > 0 and abs(err) / closing < SLOW_DOWN_TIME_S:
                return "slower", self._limiter.last_text != "slower"

        if abs(err) > COARSE_DEG:
            return f"push {direction}", False
        if abs(err) > FINE_DEG:
            return f"{direction}, getting closer", False
        return f"{direction} a little", False

    def _direction(self, axis: str, err: float) -> str:
        if axis == "alt":
            return "up" if err > 0 else "down"
        return "right" if (err > 0) == self.right_is_plus_az else "left"

    def _beep(self, dist: float) -> float:
        if dist > COARSE_DEG:
            return 0.0
        return 1.0 + 7.0 * (1.0 - dist / COARSE_DEG)  # 1-8 beeps/s


class DirectionLearner:
    """Learn whether 'right' means increasing azimuth from the first az push.

    Call `observe(cue_direction, d_az_moved_deg)` after a left/right cue; returns the
    learned `right_is_plus_az` once motion is unambiguous, else None.
    """

    def __init__(self, min_motion_deg: float = 0.5):
        self.min_motion = min_motion_deg

    def observe(self, said: str, moved_az_deg: float) -> bool | None:
        if said not in ("left", "right") or abs(moved_az_deg) < self.min_motion:
            return None
        return (said == "right") == (moved_az_deg > 0)
