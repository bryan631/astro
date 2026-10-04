from datetime import UTC, datetime

import pytest

from astro.pointing.coords import Site, altaz_to_radec, body_altaz, radec_to_altaz
from astro.pointing.encoders import EncoderAxis
from astro.pointing.geometry import separation_deg

WPB = Site(lat_deg=26.7, lon_deg=-80.1)
NIGHT = datetime(2026, 6, 21, 4, 0, tzinfo=UTC)  # midnight EDT
POLARIS = (37.95, 89.26)


def test_polaris_altitude_near_latitude():
    alt, az = radec_to_altaz(*POLARIS, WPB, NIGHT)
    assert alt == pytest.approx(WPB.lat_deg, abs=1.0)
    assert min(az, 360 - az) < 2


def test_roundtrip_radec_altaz():
    ra, dec = 279.23, 38.78  # Vega, high at midnight in June
    alt, az = radec_to_altaz(ra, dec, WPB, NIGHT)
    ra2, dec2 = altaz_to_radec(alt, az, WPB, NIGHT)
    assert (ra2, dec2) == (pytest.approx(ra, abs=1e-4), pytest.approx(dec, abs=1e-4))


def test_refraction_lifts_low_objects():
    no_air = Site(WPB.lat_deg, WPB.lon_deg, pressure_hpa=0)
    ra, dec = altaz_to_radec(10.0, 90.0, no_air, NIGHT)
    alt_refracted, _ = radec_to_altaz(ra, dec, WPB, NIGHT)
    assert 0.07 < alt_refracted - 10.0 < 0.11  # ~5 arcmin at 10 deg


def test_sun_below_horizon_at_midnight():
    alt, _ = body_altaz("sun", WPB, NIGHT)
    assert alt < -30


@pytest.mark.parametrize("counts,expected", [(0, 0), (2304, 90), (9216, 0), (-2304, 270)])
def test_encoder_degrees(counts, expected):
    assert EncoderAxis().to_degrees(counts) == pytest.approx(expected)


def test_encoder_sign_flips_direction():
    assert EncoderAxis(sign=-1).to_degrees(2304) == pytest.approx(270)


def test_separation():
    assert separation_deg(0, 0, 0, 90) == pytest.approx(90)
    assert separation_deg(89, 0, 89, 180) == pytest.approx(2)
    assert separation_deg(45, 359.9, 45, 0.1) == pytest.approx(0.2 * 0.7071, abs=1e-3)
