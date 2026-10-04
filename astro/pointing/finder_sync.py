"""Find where the scope points: finder frame -> focus check -> plate solve -> mount model sync.

Every failure comes back with a plain-language reason the tablet can speak, instead of a
silent solve failure (pre-flight focus gate, docs/plan.md Phase 1 step 3).
"""

from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime

import numpy as np
from scipy import ndimage
from scipy.stats import norm

from astro.capture.focus import half_flux_radius
from astro.devices.base import Camera
from astro.pointing.coords import Site, radec_to_altaz
from astro.pointing.mount_model import MountModel, Sync
from astro.pointing.platesolve import FinderSolver, finder_gray

# Robust noise: sigma = MAD / Phi^-1(3/4) for Gaussian noise (the familiar 1.4826).
MAD_TO_SIGMA = 1 / norm.ppf(0.75)
DETECT_SIGMA = 5.0  # a star is a blob brighter than background + 5 sigma
CLIP_SIGMA = 3.0  # inside a star cutout, pixels below 3 sigma count as background
CUTOUT_RADIUS_PX = 7  # star cutout is 15x15 binned pixels, wide enough for a soft star
# Dark 8-bit frames are mostly one value, so the MAD can be exactly 0 (a capped frame at
# gain 1000 gave 33,000 "stars"). Fall back to the spread without the brightest 1%, and
# never below 1 ADU.
CLIP_PERCENTILE = 99
MIN_NOISE_ADU = 1.0

# Warm pixels below the hot-pixel cut still pass 5 sigma, but only as single binned pixels
# (186 of them on a capped frame); real stars always cover at least 2.
MIN_STAR_AREA_PX = 2

# Saturated stars look wider than they are; skip them for HFR when others exist.
# Binned pixels sum 4 raw 8-bit pixels, so ~1000 means the core is clipped.
SATURATED_BINNED = 4 * 250

MIN_STARS = 4  # the solver matches 4-star patterns; don't gate stricter than it
FEW_STARS = 3
# Binned pixels (~62"/px). Defocus first hides faint stars, then grows the HFR, so a short
# star count with a slightly soft HFR also means "focus". Tuned on simulation; retune on real sky.
HFR_MAX_PX = 2.0
HFR_SOFT_PX = 1.3
MAX_STARS_MEASURED = 10
MAX_PLAUSIBLE_STARS = 1000  # far more "stars" than an 11-degree suburban field shows
LOOSE_SYNC_SYNCS = 3  # once the model has this many syncs, warn if they disagree by more than
LOOSE_SYNC_ARCMIN = 10.0  # ... this RMS (arcmin): the mount model is still rough


@dataclass(frozen=True)
class FocusReport:
    stars: int
    hfr_px: float  # median half-flux radius of the brightest stars; inf if none
    ok: bool
    reason: str = ""  # spoken explanation when not ok
    outcome: str = "ok"  # machine-readable: ok, no_stars, not_sky, few_stars, out_of_focus


def _clipped_std(gray: np.ndarray) -> float:
    return float(gray[gray <= np.percentile(gray, CLIP_PERCENTILE)].std())


def check_focus(gray: np.ndarray) -> FocusReport:
    """Count stars and measure their sharpness on a (binned) gray finder frame."""
    bg = np.median(gray)
    noise = MAD_TO_SIGMA * np.median(np.abs(gray - bg)) or _clipped_std(gray)
    noise = max(noise, MIN_NOISE_ADU)
    labels, n = ndimage.label(gray > bg + DETECT_SIGMA * noise)
    areas = np.bincount(labels.ravel())[1:]
    keep = np.flatnonzero(areas >= MIN_STAR_AREA_PX) + 1  # label ids of star-sized blobs
    n = len(keep)
    if n == 0:
        return FocusReport(0, float("inf"), False,
                           "I can't see any stars. Is the finder lens cap off, is it cloudy, "
                           "or is the finder far out of focus?", "no_stars")
    if n > MAX_PLAUSIBLE_STARS:
        return FocusReport(n, float("inf"), False,
                           "That doesn't look like a starry sky. It may still be too bright out, "
                           "or the finder is seeing something nearby.", "not_sky")
    peaks = ndimage.maximum_position(gray, labels, keep)
    unsaturated = [p for p in peaks if gray[p] < SATURATED_BINNED]
    brightest = sorted(unsaturated or peaks, key=lambda p: -gray[p])[:MAX_STARS_MEASURED]
    hfrs = []
    r = CUTOUT_RADIUS_PX
    for y, x in brightest:
        cut = gray[max(y - r, 0):y + r + 1, max(x - r, 0):x + r + 1] - bg
        cut[cut < CLIP_SIGMA * noise] = 0  # keep background noise from inflating the radius
        hfrs.append(half_flux_radius(cut))
    hfr = float(np.median(hfrs))
    if n < FEW_STARS:  # too few stars to judge focus; say what we see
        return FocusReport(n, hfr, False,
                           "I only see a star or two. Clouds or trees may be in the way.",
                           "few_stars")
    if hfr > HFR_MAX_PX or (n < MIN_STARS and hfr > HFR_SOFT_PX):
        return FocusReport(n, hfr, False,
                           "The finder looks out of focus. Say 'focus the finder' and I'll help.",
                           "out_of_focus")
    if n < MIN_STARS:
        return FocusReport(n, hfr, False,
                           "I only see a few stars. Try pointing higher, away from trees and lights.",
                           "few_stars")
    return FocusReport(n, hfr, True)


class FinderSync:
    def __init__(self, camera: Camera, solver: FinderSolver, model: MountModel,
                 encoders: Callable[[], tuple[float, float]], site: Site,
                 clock: Callable[[], datetime]):
        """`encoders()` returns encoder (alt, az) angles in degrees, before the mount model."""
        self.camera, self.solver, self.model = camera, solver, model
        self.encoders, self.site, self.clock = encoders, site, clock
        self.synced = False
        self.last_rms: float | None = None  # arcmin, mount model fit after the latest sync

    def alignment(self) -> tuple[int, float | None]:
        """(number of syncs, model RMS in arcmin or None) for the setup wizard."""
        return len(self.model.syncs), self.last_rms

    def reset(self, site: Site) -> None:
        """New site: the old mount model's alt/az frame no longer applies; re-sync from scratch."""
        self.site, self.model, self.synced = site, MountModel(), False

    def position(self) -> tuple[float, float]:
        """Current true (alt, az) through the mount model."""
        return self.model.to_sky(*self.encoders())

    def focus_report(self) -> FocusReport:
        return check_focus(finder_gray(self.camera.capture()))

    def sync(self) -> tuple[bool, str]:
        """Solve the current finder view and refine the mount model. Returns (ok, message)."""
        enc = self.encoders()  # read encoders at exposure time, not after the solve
        gray = finder_gray(self.camera.capture())
        focus = check_focus(gray)
        if not focus.ok:
            return False, focus.reason
        sol = self.solver.solve(gray, bayer=False)
        if sol is None:
            return False, ("I can see stars but couldn't recognize the pattern. "
                           "Something may be blocking part of the view.")
        alt, az = radec_to_altaz(sol.ra_deg, sol.dec_deg, self.site, self.clock())
        rms = self.model.add_sync(Sync(enc[0], enc[1], alt, az))
        self.last_rms = rms
        self.synced = True
        msg = "Got it, I know where we're pointing."
        if len(self.model.syncs) >= LOOSE_SYNC_SYNCS and rms > LOOSE_SYNC_ARCMIN:
            msg += " The alignment is still rough; another sync in a different part of the sky helps."
        return True, msg
