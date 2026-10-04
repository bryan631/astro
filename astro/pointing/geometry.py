"""Small spherical-geometry helpers on (alt, az) in degrees."""

import math


def separation_deg(alt1: float, az1: float, alt2: float, az2: float) -> float:
    """Great-circle angle between two alt/az points (haversine, stable for small angles)."""
    a1, a2 = math.radians(alt1), math.radians(alt2)
    dalt, daz = a2 - a1, math.radians(az2 - az1)
    h = math.sin(dalt / 2) ** 2 + math.cos(a1) * math.cos(a2) * math.sin(daz / 2) ** 2
    return math.degrees(2 * math.asin(min(1.0, math.sqrt(h))))
