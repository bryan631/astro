"""When a drifting target reaches the edge of the frame.

The scope doesn't track, so a capture ends when the target drifts out (about two minutes across
the main camera). From the target's recent positions in the frame, a straight-line fit gives its
velocity and the seconds left before it reaches the edge margin, so the page can say when to
stop or recenter.
"""

from collections import deque

import numpy as np

WINDOW_S = 20.0  # fit over the last 20 s of positions (the drift is steady)
MIN_SPAN_S = 2.0  # need this much time between the first and last position for a speed
MIN_SPEED_PX_S = 0.05  # slower than this isn't drifting toward any edge
MARGIN = 0.05  # the "edge" is this fraction inside the frame on each side


class EdgeClock:
    def __init__(self, width: float, height: float):
        self.width, self.height = width, height
        self._points: deque = deque()

    def reset(self) -> None:
        """After a recenter: old positions say nothing about the new ones."""
        self._points.clear()

    def add(self, t: float, x: float, y: float) -> None:
        self._points.append((t, x, y))
        while self._points and t - self._points[0][0] > WINDOW_S:
            self._points.popleft()

    def position(self) -> tuple[float, float] | None:
        """The newest position, as fractions of the frame (for a marker on the page)."""
        if not self._points:
            return None
        _, x, y = self._points[-1]
        return x / self.width, y / self.height

    def seconds_left(self) -> float | None:
        """Seconds until the target crosses the edge margin, or None (no drift measured yet)."""
        if len(self._points) < 3 or self._points[-1][0] - self._points[0][0] < MIN_SPAN_S:
            return None
        t, x, y = np.array(self._points).T
        (vx, x0), (vy, y0) = np.polyfit(t - t[-1], x, 1), np.polyfit(t - t[-1], y, 1)
        if np.hypot(vx, vy) < MIN_SPEED_PX_S:
            return None
        left = [_to_edge(x0, vx, self.width), _to_edge(y0, vy, self.height)]
        return max(0.0, min(s for s in left if s is not None))


def _to_edge(p: float, v: float, size: float) -> float | None:
    """Seconds until position p moving at v leaves [margin, size - margin], or None if still."""
    lo, hi = MARGIN * size, (1 - MARGIN) * size
    if v > 0:
        return (hi - p) / v
    if v < 0:
        return (lo - p) / v
    return None
