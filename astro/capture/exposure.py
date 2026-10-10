"""Exposure for the page's camera views: day and night settings, and auto-exposure for the
finder by day and for either camera once dawn clips its night settings.

One step at a time from the latest frame: brighter or darker by STEP (faster when far off), and
in daylight the gain comes down once the exposure is at its shortest, and back up first when it
gets dim again. A capped lens stays dark at the longest exposure instead of hunting.
"""

import logging

import numpy as np

LOW_PEAK, HIGH_PEAK = 60, 220  # 8-bit brightest 0.5% of pixels: outside this, change the exposure
STEP = 2.0  # exposure change per step
FINDER_DAY_RANGE = (0.0001, 0.5)  # seconds; the finder reaches 0.1 ms in sunshine
MAIN_TWILIGHT_RANGE = (0.0005, 0.25)  # seconds: the main view auto-exposes once dawn clips it
MAIN_DAY = (0.004, 0)  # measured on a sunny tree
SATURATED_RAW = 250  # 8-bit raw: clipped
EVERY_S = 1.0  # ViewSettings checks this often
log = logging.getLogger(__name__)


def next_settings(frame: np.ndarray, exposure_s: float, gain: int,
                  limits: tuple[float, float] = FINDER_DAY_RANGE,
                  max_gain: int = 400) -> tuple[float, int] | None:
    """New (exposure, gain) for a frame that's too dark or too bright, or None if it's fine."""
    low, high = limits
    peak = float(np.percentile(frame[::4, ::4], 99.5))
    step = STEP * STEP if peak >= 250 or peak < LOW_PEAK / 4 else STEP  # far off: hurry
    if peak > HIGH_PEAK and exposure_s <= low and gain > 0:  # daylight: lower the gain too
        return exposure_s, int(gain / step)
    if peak < LOW_PEAK and gain < max_gain:  # dim again: gain back first
        return exposure_s, min(round(gain * step) + 1, max_gain)
    if peak < LOW_PEAK:
        new = min(exposure_s * step, high)
    elif peak > HIGH_PEAK:
        new = max(exposure_s / step, low)
    else:
        return None
    return (new, gain) if new != exposure_s else None


class ViewSettings:
    """Day or night settings for the streamed cameras (the page's views), once per switch.

    By day the finder's view auto-exposes; the main camera gets fixed daylight settings (every
    exposure change reopens it in the SDK, which froze its view in the field). At night both go
    back to the settings their streams started with (the finder's are the plate solver's), and
    either auto-exposes once dawn clips them (2026-10-10: both views all white at 6:50 AM)."""

    def __init__(self, cams: dict):
        self.cams = {name: cam for name, cam in cams.items() if hasattr(cam, "latest")}
        self._night = {name: (cam.exposure_s, cam.gain) for name, cam in self.cams.items()}
        self._day: dict[str, bool] = {}  # per camera: daytime when its settings last switched
        self._seen: dict[str, int] = {}  # per camera: the frame auto-exposure last looked at
        self._at = -1e9

    def update(self, t: float, day: bool, main_busy: bool, hold: bool = False) -> None:
        """At most every EVERY_S. A capture (`main_busy`) owns the main camera's settings;
        `hold` (Align) keeps exposures steady."""
        if t - self._at < EVERY_S:
            return
        self._at = t
        for name, cam in self.cams.items():
            if self._day.get(name) == day or (name == "main" and main_busy):
                continue  # a capture set its own: caught up once it ends
            self._day[name] = day
            exposure, gain = MAIN_DAY if day and name == "main" else self._night[name]
            log.info("camera settings", extra={"data": {"camera": name, "day": day,
                                                        "exposure_s": exposure, "gain": gain}})
            cam.set_exposure(exposure)
            cam.set_gain(gain)
        if hold:
            return
        for name, cam in self.cams.items():
            if name == "main" and (day or main_busy):
                continue  # main by day: MAIN_DAY; a capture sets its own
            frame, seq, _ = cam.latest()
            if frame is None or seq == self._seen.get(name):
                continue
            self._seen[name] = seq
            adjusting = (cam.exposure_s, cam.gain) != self._night[name]
            if not ((day and name == "finder") or adjusting or np.median(frame[::8, ::8]) >= SATURATED_RAW):
                continue
            limits = FINDER_DAY_RANGE if name == "finder" else MAIN_TWILIGHT_RANGE
            if (new := next_settings(frame, cam.exposure_s, cam.gain, limits, self._night[name][1])) is not None:
                cam.set_exposure(new[0])
                cam.set_gain(new[1])
