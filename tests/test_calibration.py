"""CV7: calibration survives a restart."""

import os

import numpy as np
import pytest

if not os.environ.get("ASTRO_REQUIRE_SOLVER"):
    pytest.importorskip("tetra3", reason="run scripts/install-solver.sh")

from astro import calibration_store
from astro.pointing.coords import Site
from astro.pointing.geometry import separation_deg
from astro.pointing.platesolve import FinderSolver
from astro.session import Session
from tests.test_finder_sync import EVENING, WPB, make


@pytest.fixture(scope="module")
def solver():
    return FinderSolver()


def test_store_roundtrip_and_corrupt_file(tmp_path):
    calibration_store.save(tmp_path, {"right_is_plus_az": False})
    assert calibration_store.load(tmp_path)["right_is_plus_az"] is False
    (tmp_path / calibration_store.SAVED).write_text("{not json")
    assert calibration_store.load(tmp_path) == {}


def test_restart_at_same_site_needs_no_sync(solver):
    saved = []
    scope, finder = make(solver, 60, 200)
    s = Session(WPB, clock=lambda: EVENING, finder=finder, on_calibration_change=saved.append)
    s.right_is_plus_az, s._direction_known = False, True
    s.centerer.axes.matrix = np.eye(2) * 3600
    assert s.handle("sync")[0]["text"].startswith("Got it")
    assert saved and saved[-1]["mount"] is not None  # the sync was saved

    # "Restart": a fresh finder with the same encoders (the encoder board kept its counts).
    _, finder2 = make(solver, 60, 200)
    finder2.encoders = finder.encoders
    s2 = Session(WPB, clock=lambda: EVENING, finder=finder2, calibration=saved[-1])
    assert finder2.synced and s2.right_is_plus_az is False
    assert s2.centerer.axes.matrix.tolist() == [[3600, 0], [0, 3600]]
    scope.alt, scope.az = 40, 300
    assert separation_deg(*finder2.position(), 40, 300) < 0.2  # points right without a sync


def test_moved_site_keeps_camera_calibration_but_not_the_mount_model(solver):
    data = {"right_is_plus_az": False, "camera_axes": [[1, 0], [0, 1]],
            "main_offset": {"d_az_sky_deg": 0.1, "d_alt_deg": -0.05, "observations": 2},
            "mount": {"site": [26.7, -80.1], "az_offset_deg": 5, "alt_offset_deg": 1,
                      "tilt_n_deg": 0, "tilt_e_deg": 0, "syncs": []}}
    _, finder = make(solver)
    s = Session(Site(40.0, -105.0), clock=lambda: EVENING, finder=finder, calibration=data)
    assert not finder.synced and s.right_is_plus_az is False
    assert s.centerer.offset.d_az_sky_deg == 0.1 and s.centerer.offset.observations == 2
