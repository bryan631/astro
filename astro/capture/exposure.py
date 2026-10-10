"""Daylight auto-exposure for the finder's view (night settings are fixed: the solver's).

One step at a time from the latest frame: brighter or darker by STEP (faster when far off), and
in daylight the gain comes down once the exposure is at its shortest, and back up first when it
gets dim again. A capped lens stays dark at the longest exposure instead of hunting.
"""

import numpy as np

LOW_PEAK, HIGH_PEAK = 60, 220  # 8-bit brightest 0.5% of pixels: outside this, change the exposure
STEP = 2.0  # exposure change per step
FINDER_DAY_RANGE = (0.0001, 0.5)  # seconds; the finder reaches 0.1 ms in sunshine


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
