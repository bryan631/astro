"""RA/Dec <-> Alt/Az for a site and time, with atmospheric refraction.

Works offline: astropy's IERS auto-download is disabled. Accuracy without fresh IERS
data is a few arcseconds at worst, far below what push-to guidance needs.
"""

import functools
from dataclasses import dataclass
from datetime import datetime
from zoneinfo import ZoneInfo

import astropy.units as u
from astropy.coordinates import AltAz, EarthLocation, SkyCoord, get_body
from astropy.time import Time
from astropy.utils import iers
from timezonefinder import TimezoneFinder  # offline lookup; loaded once

iers.conf.auto_download = False
iers.conf.auto_max_age = None


@functools.cache
def _timezone_finder():
    return TimezoneFinder()


@functools.cache
def _zone_at(lat: float, lon: float) -> ZoneInfo:
    """Exact coordinates: rounding could cross a nearby time-zone border."""
    name = _timezone_finder().timezone_at(lat=lat, lng=lon)
    return ZoneInfo(name) if name else ZoneInfo("UTC")


@dataclass(frozen=True)
class Site:
    lat_deg: float
    lon_deg: float
    elevation_m: float = 0.0
    pressure_hpa: float = 1013.0  # 0 disables refraction
    temperature_c: float = 15.0
    humidity: float = 0.7  # 0-1

    @property
    def timezone(self) -> ZoneInfo:
        """Local time zone, from the coordinates (no network needed)."""
        return _zone_at(self.lat_deg, self.lon_deg)

    def frame(self, when: datetime) -> AltAz:
        return AltAz(
            obstime=Time(when),
            location=EarthLocation(
                lat=self.lat_deg * u.deg, lon=self.lon_deg * u.deg, height=self.elevation_m * u.m
            ),
            pressure=self.pressure_hpa * u.hPa,
            temperature=self.temperature_c * u.deg_C,
            relative_humidity=self.humidity,
            obswl=0.55 * u.micron,
        )


def radec_to_altaz(ra_deg: float, dec_deg: float, site: Site, when: datetime) -> tuple[float, float]:
    """ICRS RA/Dec (degrees) -> apparent (alt, az) in degrees, az measured N through E."""
    aa = SkyCoord(ra=ra_deg * u.deg, dec=dec_deg * u.deg).transform_to(site.frame(when))
    return aa.alt.deg, aa.az.deg


def altaz_to_radec(alt_deg: float, az_deg: float, site: Site, when: datetime) -> tuple[float, float]:
    """Apparent (alt, az) in degrees -> ICRS (ra, dec) in degrees."""
    c = SkyCoord(alt=alt_deg * u.deg, az=az_deg * u.deg, frame=site.frame(when)).icrs
    return c.ra.deg, c.dec.deg


def body_altaz(name: str, site: Site, when: datetime) -> tuple[float, float]:
    """Alt/az of a solar-system body ('sun', 'moon', 'jupiter', ...), using built-in ephemeris."""
    frame = site.frame(when)
    aa = get_body(name, frame.obstime, frame.location).transform_to(frame)
    return aa.alt.deg, aa.az.deg
