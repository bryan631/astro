"""Solar safety: never guide or capture near the Sun, and lock out daytime use.

The guidance and capture layers must call `check_target` before every move or exposure.
"""

from dataclasses import dataclass
from datetime import datetime

from astro.pointing.coords import Site, body_altaz
from astro.pointing.geometry import separation_deg

SUN_EXCLUSION_DEG = 20.0
DAYTIME_SUN_ALT_DEG = 0.0  # sun above the horizon counts as daytime


@dataclass(frozen=True)
class SafetyResult:
    ok: bool
    reason: str = ""


def check_target(
    alt_deg: float,
    az_deg: float,
    site: Site,
    when: datetime,
    developer_override: bool = False,
    exclusion_deg: float = SUN_EXCLUSION_DEG,
) -> SafetyResult:
    """Refuse targets near the Sun, below the horizon, or during daytime.

    `developer_override` relaxes only the daytime lockout; the Sun exclusion zone always applies.
    """
    sun_alt, sun_az = body_altaz("sun", site, when)
    if separation_deg(alt_deg, az_deg, sun_alt, sun_az) < max(exclusion_deg, SUN_EXCLUSION_DEG):
        return SafetyResult(False, "too close to the Sun")
    if alt_deg < 0:
        return SafetyResult(False, "below the horizon")
    if sun_alt > DAYTIME_SUN_ALT_DEG and not developer_override:
        return SafetyResult(False, "daytime lockout")
    return SafetyResult(True)
