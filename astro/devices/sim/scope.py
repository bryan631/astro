"""Simulated Dobsonian plus a simulated user who follows spoken cues."""

from dataclasses import dataclass
from typing import ClassVar


@dataclass
class SimScope:
    alt: float = 30.0
    az: float = 0.0

    def step(self, v_alt: float, v_az: float, dt: float) -> None:
        self.alt = max(0.0, min(90.0, self.alt + v_alt * dt))
        self.az = (self.az + v_az * dt) % 360.0


class SimUser:
    """Pushes at a speed that depends on the last cue; reacts after a delay.

    Models an imperfect user: overshoots because of reaction time.
    """

    SPEEDS: ClassVar[dict[str, float]] = {"push": 6.0, "getting closer": 1.5, "a little": 0.3}

    def __init__(self, right_is_plus_az: bool = True, reaction_s: float = 0.4):
        self.right_sign = 1 if right_is_plus_az else -1
        self.reaction_s = reaction_s
        self.v = (0.0, 0.0)
        self._pending: list[tuple[float, str]] = []

    def hear(self, text: str, t: float) -> None:
        self._pending.append((t + self.reaction_s, text))

    def act(self, t: float) -> tuple[float, float]:
        while self._pending and self._pending[0][0] <= t:
            self.v = self._interpret(self._pending.pop(0)[1])
        return self.v

    def _interpret(self, text: str) -> tuple[float, float]:
        if text == "stop":
            return (0.0, 0.0)
        if text == "keep going":
            return self.v
        if text == "slower":
            return (self.v[0] / 3, self.v[1] / 3)
        speed = next((s for k, s in self.SPEEDS.items() if k in text), 1.0)
        if "back" in text:
            speed = self.SPEEDS["a little"]
        if " up" in f" {text}":
            return (speed, 0.0)
        if " down" in f" {text}":
            return (-speed, 0.0)
        sign = self.right_sign if "right" in text else -self.right_sign
        return (0.0, sign * speed)


@dataclass
class SimEncoders:
    """Encoder angles for a SimScope, offset like an uncalibrated real mount."""

    scope: SimScope
    alt_offset: float = -1.2
    az_offset: float = 123.4

    def __call__(self) -> tuple[float, float]:
        return self.scope.alt - self.alt_offset, (self.scope.az - self.az_offset) % 360
