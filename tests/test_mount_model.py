import pytest

from astro.pointing.geometry import separation_deg
from astro.pointing.mount_model import MountModel, Sync

TRUE = MountModel(az_offset_deg=123.4, alt_offset_deg=-1.5, tilt_n_deg=0.8, tilt_e_deg=-0.5)
POINTS = [(30, 10), (60, 100), (20, 200), (45, 290), (75, 45)]  # encoder (alt, az)


def make_sync(enc_alt, enc_az):
    return Sync(enc_alt, enc_az, *TRUE.to_sky(enc_alt, enc_az))


def max_error_arcmin(model):
    errs = [separation_deg(*model.to_sky(a, z), *TRUE.to_sky(a, z)) for a, z in POINTS]
    return max(errs) * 60


def test_identity_model():
    assert MountModel().to_sky(40, 120) == (pytest.approx(40), pytest.approx(120))


def test_one_sync_fixes_offsets_locally():
    m = MountModel()
    m.add_sync(make_sync(*POINTS[0]))
    assert separation_deg(*m.to_sky(*POINTS[0]), *TRUE.to_sky(*POINTS[0])) * 60 < 0.1


def test_one_sync_near_zenith_does_not_flip():
    m = MountModel()
    m.add_sync(Sync(61.2, 76.6, 60.0, 200.0))  # encoders read 61.2 deg alt
    assert m.alt_offset_deg == pytest.approx(-1.2)
    assert separation_deg(*m.to_sky(31.2, 76.6), 30.0, 200.0) < 1e-6


def test_three_syncs_recover_tilt_everywhere():
    m = MountModel()
    for p in POINTS[:3]:
        rms = m.add_sync(make_sync(*p))
    assert rms < 0.1
    assert max_error_arcmin(m) < 0.5


def test_noisy_syncs_stay_within_guidance_tolerance():
    import numpy as np

    rng = np.random.default_rng(1)
    m = MountModel()
    for a, z in POINTS:
        s = make_sync(a, z)
        noise = rng.normal(0, 2 / 60, 2)  # 2 arcmin solve/encoder noise
        m.add_sync(Sync(a, z, s.true_alt_deg + noise[0], s.true_az_deg + noise[1]))
    assert max_error_arcmin(m) < 4  # default guidance tolerance at prime focus


def test_false_solve_is_dropped():
    m = MountModel()
    for p in POINTS[:4]:
        m.add_sync(make_sync(*p))
    m.add_sync(Sync(50, 150, *TRUE.to_sky(50, 170)))  # solved the wrong field, 20 deg off
    assert len(m.syncs) == 4 and max_error_arcmin(m) < 1


def test_history_is_capped():
    from astro.pointing.mount_model import MAX_SYNCS

    m = MountModel()
    for i in range(MAX_SYNCS + 5):
        m.add_sync(make_sync(20 + i * 3, i * 25))
    assert len(m.syncs) == MAX_SYNCS
