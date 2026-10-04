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
from astro.pointing.platesolve import FinderSolver, bin2x2

# Robust noise: sigma = MAD / Phi^-1(3/4) for Gaussian noise (the familiar 1.4826).
MAD_TO_SIGMA = 1 / norm.ppf(0.75)
DETECT_SIGMA = 5.0  # a star is a blob brighter than background + 5 sigma
CLIP_SIGMA = 3.0  # inside a star cutout, pixels below 3 sigma count as background
CUTOUT_RADIUS_PX = 7  # star cutout is 15x15 binned pixels, wide enough for a soft star
TINY = 1e-6  # avoids a zero noise estimate on synthetic, noise-free frames

MIN_STARS = 6
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
    reason: str = ""


def check_focus(gray: np.ndarray) -> FocusReport:
    """Count stars and measure their sharpness on a (binned) gray finder frame."""
    bg = np.median(gray)
    noise = MAD_TO_SIGMA * np.median(np.abs(gray - bg)) + TINY
    labels, n = ndimage.label(gray > bg + DETECT_SIGMA * noise)
    if n == 0:
        return FocusReport(0, float("inf"), False,
                           "I can't see any stars. Is the finder lens cap off, or is it cloudy?")
    if n > MAX_PLAUSIBLE_STARS:
        return FocusReport(n, float("inf"), False,
                           "That doesn't look like a starry sky. It may still be too bright out, "
                           "or the finder is seeing something nearby.")
    peaks = ndimage.maximum_position(gray, labels, range(1, n + 1))
    brightest = sorted(peaks, key=lambda p: -gray[p])[:MAX_STARS_MEASURED]
    hfrs = []
    r = CUTOUT_RADIUS_PX
    for y, x in brightest:
        cut = gray[max(y - r, 0):y + r + 1, max(x - r, 0):x + r + 1] - bg
        cut[cut < CLIP_SIGMA * noise] = 0  # keep background noise from inflating the radius
        hfrs.append(half_flux_radius(cut))
    hfr = float(np.median(hfrs))
    if hfr > HFR_MAX_PX or (n < MIN_STARS and hfr > HFR_SOFT_PX):
        return FocusReport(n, hfr, False,
                           "The finder looks out of focus. Say 'focus the finder' and I'll help.")
    if n < FEW_STARS:
        return FocusReport(n, hfr, False,
                           "I only see a star or two. Clouds or trees may be in the way.")
    if n < MIN_STARS:
        return FocusReport(n, hfr, False,
                           "I only see a few stars. Try pointing higher, away from trees and lights.")
    return FocusReport(n, hfr, True)


class FinderSync:
    def __init__(self, camera: Camera, solver: FinderSolver, model: MountModel,
                 encoders: Callable[[], tuple[float, float]], site: Site,
                 clock: Callable[[], datetime]):
        """`encoders()` returns encoder (alt, az) angles in degrees, before the mount model."""
        self.camera, self.solver, self.model = camera, solver, model
        self.encoders, self.site, self.clock = encoders, site, clock
        self.synced = False

    def position(self) -> tuple[float, float]:
        """Current true (alt, az) through the mount model."""
        return self.model.to_sky(*self.encoders())

    def focus_report(self) -> FocusReport:
        return check_focus(bin2x2(self.camera.capture()))

    def sync(self) -> tuple[bool, str]:
        """Solve the current finder view and refine the mount model. Returns (ok, message)."""
        enc = self.encoders()  # read encoders at exposure time, not after the solve
        gray = bin2x2(self.camera.capture())
        focus = check_focus(gray)
        if not focus.ok:
            return False, focus.reason
        sol = self.solver.solve(gray, bayer=False)
        if sol is None:
            return False, ("I can see stars but couldn't recognize the pattern. "
                           "Something may be blocking part of the view.")
        alt, az = radec_to_altaz(sol.ra_deg, sol.dec_deg, self.site, self.clock())
        rms = self.model.add_sync(Sync(enc[0], enc[1], alt, az))
        self.synced = True
        msg = "Got it, I know where we're pointing."
        if len(self.model.syncs) >= LOOSE_SYNC_SYNCS and rms > LOOSE_SYNC_ARCMIN:
            msg += " The alignment is still rough; another sync in a different part of the sky helps."
        return True, msg
