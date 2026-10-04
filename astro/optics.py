"""Optics math for the main and finder cameras."""

import math

SIDEREAL_ARCSEC_PER_S = 15.04


def plate_scale(pixel_um: float, focal_mm: float) -> float:
    """Arcseconds per pixel."""
    return 206.265 * pixel_um / focal_mm


def fov_arcmin(width_px: int, height_px: int, scale_arcsec: float) -> tuple[float, float]:
    """Field of view (width, height) in arcminutes."""
    return width_px * scale_arcsec / 60, height_px * scale_arcsec / 60


def drift_px_per_s(dec_deg: float, scale_arcsec: float) -> float:
    """Sidereal drift across the sensor in pixels per second."""
    return SIDEREAL_ARCSEC_PER_S * math.cos(math.radians(dec_deg)) / scale_arcsec
