"""CV7: calibration survives a restart."""

import os

import numpy as np
import pytest

if not os.environ.get("ASTRO_REQUIRE_SOLVER"):
    pytest.importorskip("tetra3", reason="run scripts/install-solver.sh")

import threading

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


@pytest.mark.parametrize("content", ['[]', '{"version": 1, "mount": {"site": [1, 2]}}',
                                     '{"version": 1, "camera_axes": [1, 2, 3]}',
                                     '{"version": 1, "right_is_plus_az": "yes"}'])
def test_wrong_shapes_start_fresh(tmp_path, content):
    (tmp_path / calibration_store.SAVED).parent.mkdir(parents=True, exist_ok=True)
    (tmp_path / calibration_store.SAVED).write_text(content)
    assert calibration_store.load(tmp_path) == {}


def test_concurrent_saves_dont_clobber(tmp_path):
    threads = [threading.Thread(target=calibration_store.save, args=(tmp_path, {"n": i}))
               for i in range(20)]
    for th in threads:
        th.start()
    for th in threads:
        th.join()
    assert calibration_store.load(tmp_path) == {} or "n" in __import__("json").loads(
        (tmp_path / calibration_store.SAVED).read_text())
    assert not list((tmp_path / "data").glob("*.tmp"))


def test_boot_before_restore_skips_the_saved_model(solver):
    saved_model = {"site": [26.7, -80.1], "az_offset_deg": 5.0, "alt_offset_deg": 1.0,
                   "tilt_n_deg": 0.0, "tilt_e_deg": 0.0, "syncs": []}
    _, finder = make(solver)
    finder.encoder_boots = lambda: 1  # the board booted before the session restored
    Session(WPB, clock=lambda: EVENING, finder=finder,
            calibration={"version": 1, "mount": saved_model})
    assert not finder.synced


def test_restored_model_keeps_its_origin_for_move_checks(solver):
    origin = {"site": [26.7, -80.1], "az_offset_deg": 0.0, "alt_offset_deg": 0.0,
              "tilt_n_deg": 0.0, "tilt_e_deg": 0.0, "syncs": []}
    _, finder = make(solver)
    s = Session(Site(26.7081, -80.1), clock=lambda: EVENING, finder=finder,
                calibration={"version": 1, "mount": origin})  # 0.9 km north of the origin
    assert finder.synced and s._model_site.lat_deg == 26.7
    s.set_location(26.7162, -80.1, None, None)  # another 0.9 km: 1.8 km from the origin
    assert not finder.synced
