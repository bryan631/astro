"""Bahtinov mask focus: found automatically, read as one number.

The mask turns a bright star into three spikes: two crossing at +-theta and a middle one that
passes through their crossing only at focus. Out of focus it slides sideways, one way inside
focus and the other way outside. This finds the three spikes around the brightest star (their
angles from rays out of the star, their sideways positions from strips along each) and returns
how far the middle spike misses the crossing: zero at focus, signed, in pixels.

No mask (the spider's four spikes, a plain star field) finds nothing, and the focus number
falls back to the stars' widths (astro/capture/focus.py).
"""

from dataclasses import dataclass

import numpy as np
from scipy import ndimage

WINDOW_PX = 60  # the spikes are looked for this far out from the star (view pixels)
CORE_PX = 8  # the star's burnt-out core and its first rings are left out
CENTER_BOX_PX = 25  # the star's center: the light's centroid in a box this big (each way)
MIN_PEAK = 120.0  # 8-bit: the star must be bright (saturated, in practice) to show spikes
SATURATED = 245.0
MAX_CLIPPED_PX = 1200  # clipped pixels around the star: a magnitude 0 star clips ~900 (model)
STEP_DEG = 0.5
OUTER_DEG = (16.0, 70.0)  # angle between the two outer spikes (2 theta; masks use 30-50)
BISECT_TOL_DEG = 3.0  # the middle spike bisects the outer two within this
SPIKE_SIGMA = 3.0  # a spike's ray sum stands this many noise sigmas above the typical ray
STRIP_HALF_PX = 7  # across a spike, its profile is sampled this far each side
SCORE_PER_PX = 40.0  # the focus number drops this much per pixel of miss: 100 at focus, a sharp V
MAX_TOP_PX = 5  # a spike's profile within 80% of its top this wide: clipped wider than we can read


@dataclass(frozen=True)
class Bahtinov:
    offset_px: float  # the middle spike's miss of the outer two's crossing (signed)
    angles_deg: tuple[float, float, float]  # outer, middle, outer
    center: tuple[float, float]  # the star (x, y) in the frame


def measure_bahtinov(lum: np.ndarray) -> Bahtinov | None:
    """The mask's pattern around the brightest star of `lum` (8-bit scale gray), or None."""
    smooth = ndimage.gaussian_filter(lum.astype(np.float32), 2.0)
    peak = np.unravel_index(np.argmax(smooth), smooth.shape)
    if smooth[peak] < MIN_PEAK:
        return None
    # The star: the pattern is point-symmetric about it, so the light's centroid in a box finds
    # it (the brightest pixel can sit on a bright diffraction order, 13 px down a spike).
    cy, cx = peak
    for _ in range(4):
        box = smooth[max(cy - CENTER_BOX_PX, 0):cy + CENTER_BOX_PX + 1, max(cx - CENTER_BOX_PX, 0):cx + CENTER_BOX_PX + 1]
        light = np.maximum(box - np.median(smooth), 0)
        dy, dx = ndimage.center_of_mass(light)
        cy, cx = round(max(cy - CENTER_BOX_PX, 0) + dy), round(max(cx - CENTER_BOX_PX, 0) + dx)
    w = WINDOW_PX
    if not (w <= cy < lum.shape[0] - w and w <= cx < lum.shape[1] - w):
        return None
    win = lum[cy - w:cy + w + 1, cx - w:cx + w + 1].astype(np.float32)
    if (win >= SATURATED).sum() > MAX_CLIPPED_PX:
        return None  # too bright (magnitude ~-1 at the night view's settings): spikes clip wide
    # Left out: the core, grown to cover a bright star's burnt-out middle (out to where most of
    # a ring is unclipped: the spikes clip much farther out, but only along a few angles)
    ring = np.hypot(*np.mgrid[-w:w + 1, -w:w + 1]).astype(int)
    clipped = np.bincount(ring.ravel(), weights=(win >= SATURATED).ravel()) / np.bincount(ring.ravel())
    core = max(CORE_PX, int(np.argmax(clipped[:w] < 0.5)) + 3)
    if core > w / 2:
        return None  # the star fills the window: far too bright, or a planet
    win = win - np.median(win)
    angles = _spike_angles(win, core)
    if angles is None:
        return None
    lines = [_line(win, a, core) for a in angles]  # (angle, offset): x cos a + y sin a = offset
    if any(line is None for line in lines):
        return None
    (a1, s1), (am, sm), (a3, s3) = lines
    n1, n3 = _normal(a1), _normal(a3)
    crossing = np.linalg.solve(np.array([n1, n3]), [s1, s3])
    miss = float(crossing @ _normal(am) - sm)
    return Bahtinov(miss, (float(a1), float(am), float(a3)), (cx + float(crossing[0]), cy + float(crossing[1])))


def focus_number(found: Bahtinov) -> float:
    """The mask's focus number, like the stars' (higher is sharper): 100 at focus, falling
    straight off either side, so the peak is easy to find by turning the knob. On the XT8 at
    prime focus a pixel of miss is ~60 um of focuser travel; focus is good within ~0.75 px."""
    return max(0.0, 100.0 - SCORE_PER_PX * abs(found.offset_px))


def _normal(angle_deg: float) -> np.ndarray:
    """Unit normal (x, y) of a line running at `angle_deg` (0: along +x, 90: along +y)."""
    a = np.radians(angle_deg)
    return np.array([-np.sin(a), np.cos(a)])


def _spike_angles(win: np.ndarray, core: int) -> tuple[float, float, float] | None:
    """The three spikes' angles (0-180, outer-middle-outer), from rays out of the window's
    center: a spike lights its ray on both sides of the star."""
    w = win.shape[0] // 2
    r = np.arange(core, w)
    phi = np.radians(np.arange(0, 360, STEP_DEG))
    rays = ndimage.map_coordinates(win, [w + np.outer(np.sin(phi), r), w + np.outer(np.cos(phi), r)], order=1)
    one_way = rays.mean(axis=1)
    half = len(phi) // 2
    both = np.minimum(one_way[:half], one_way[half:])  # lit on both sides: a line through the star
    noise = 1.4826 * np.median(np.abs(both - np.median(both))) + 1e-6
    level = (both - np.median(both)) / noise
    peaks = [i for i in range(half) if level[i] > SPIKE_SIGMA
             and level[i] >= level[i - 1] and level[i] >= level[(i + 1) % half]]
    best = None
    for i in peaks:  # the middle one
        for j in peaks:
            for k in peaks:
                lo, mid, hi = (np.degrees(phi[x]) for x in (j, i, k))
                spread = (hi - lo) % 180
                if not OUTER_DEG[0] <= spread <= OUTER_DEG[1]:
                    continue
                if abs(((mid - lo) % 180) - spread / 2) > BISECT_TOL_DEG:
                    continue
                strength = min(level[i], level[j], level[k])
                if best is None or strength > best[0]:
                    best = (strength, lo, mid, hi)
    return None if best is None else best[1:]


def _line(win: np.ndarray, angle_deg: float, core: int) -> tuple[float, float] | None:
    """A spike refined: its angle (searched +-1 degree) and its offset from the window center,
    from the profile across a strip along it (both sides of the star, core left out). None if
    the spike's profile is a wide flat top: clipped wider than the strip, its middle can't be
    found (stars brighter than magnitude ~-1 at the night view's settings)."""
    w = win.shape[0] // 2
    t = np.concatenate([-np.arange(core, w - STRIP_HALF_PX), np.arange(core, w - STRIP_HALF_PX)])
    across = np.arange(-STRIP_HALF_PX, STRIP_HALF_PX + 1)
    best = None
    for a in angle_deg + np.arange(-1.0, 1.01, 0.1):
        d, n = np.array([np.cos(np.radians(a)), np.sin(np.radians(a))]), _normal(a)
        pts = t[:, None, None] * d + across[None, :, None] * n  # (along, across, xy)
        prof = ndimage.map_coordinates(win, [w + pts[..., 1], w + pts[..., 0]], order=1).mean(axis=0)
        sharp = float((np.maximum(prof - np.median(prof), 0) ** 2).sum())  # light most concentrated:
        if best is None or sharp > best[0]:  # a clipped spike's flat top is as high a degree off
            best = (sharp, a, prof)
    _, a, prof = best

    prof = prof - np.median(prof)
    i = int(np.argmax(prof))
    if i in (0, len(prof) - 1) or (prof > 0.8 * prof[i]).sum() > MAX_TOP_PX:
        return None  # no peak in the strip, or a flat top as wide as it (clipped wider than it)
    # The centroid of the spike's upper half: a bright star's spikes clip flat on top, where
    # a parabola through the top three samples lands anywhere along it.
    top = np.where(prof > 0.5 * prof[i], prof - 0.5 * prof[i], 0.0)
    return float(a), float((top * across).sum() / top.sum())
