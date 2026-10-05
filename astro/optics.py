"""Optics math for the main and finder cameras.

Reference math: runtime code uses the measured or derived numbers (noted next to each
constant, e.g. FINDER_FOV_DEG, MIN_MOTION_DEG), and the tests check them against this."""

import math

SIDEREAL_ARCSEC_PER_S = 15.04
MAIN_SENSOR_PX = (3856, 2180)  # SV705C full frame (width, height)


def plate_scale(pixel_um: float, focal_mm: float) -> float:
    """Arcseconds per pixel."""
    return 206.265 * pixel_um / focal_mm


def fov_arcmin(width_px: int, height_px: int, scale_arcsec: float) -> tuple[float, float]:
    """Field of view (width, height) in arcminutes."""
    return width_px * scale_arcsec / 60, height_px * scale_arcsec / 60


def drift_px_per_s(dec_deg: float, scale_arcsec: float) -> float:
    """Sidereal drift across the sensor in pixels per second."""
    return SIDEREAL_ARCSEC_PER_S * math.cos(math.radians(dec_deg)) / scale_arcsec
