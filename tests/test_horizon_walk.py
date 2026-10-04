from datetime import UTC, datetime

from astro.intents import Intent, parse
from astro.planner import horizon_store
from astro.planner.horizon import HorizonMask
from astro.pointing.coords import Site
from astro.session import Session

NIGHT = datetime(2026, 10, 4, 2, 0, tzinfo=UTC)


def texts(msgs):
    return [m["text"] for m in msgs if m["type"] == "say"]


def walk_session(saved):
    pos = {"alt": 25.0, "az": 0.0}
    s = Session(Site(26.7, -80.1), lambda: (pos["alt"], pos["az"]), clock=lambda: NIGHT,
                on_horizon_change=saved.append)
    return s, pos


def test_intents():
    assert parse("start the horizon walk") == Intent("horizon_start")
    assert parse("mark") == Intent("horizon_mark")
    assert parse("done") == Intent("stop")


def test_walk_records_marks_and_saves_mask():
    saved = []
    s, pos = walk_session(saved)
    assert "treeline" in texts(s.handle("start the horizon walk"))[0]
    for az, alt in [(0, 25), (90, 35), (180, 15), (270, 30)]:
        pos.update(alt=alt, az=az)
        assert texts(s.handle("mark"))[0].startswith("Marked")
    assert "from 4 marks" in texts(s.handle("done"))[0]
    assert saved[0].points == ((0, 25), (90, 35), (180, 15), (270, 30))
    assert s.horizon.min_alt(45) == 30  # interpolated between the marks


def test_too_few_marks_keeps_old_horizon():
    saved = []
    s, _ = walk_session(saved)
    s.handle("horizon walk")
    s.handle("mark")
    assert "kept the old horizon" in texts(s.handle("done"))[0]
    assert saved == [] and s.horizon == HorizonMask()


def test_planner_respects_walked_horizon():
    s, _ = walk_session([])
    s.horizon = HorizonMask(((0, 89.0),))  # a wall of trees everywhere
    assert texts(s.handle("what's good tonight"))[0] == "Nothing good is up right now."


def test_store_roundtrip(tmp_path):
    mask = HorizonMask(((0.0, 22.5), (180.0, 31.0)))
    horizon_store.save(tmp_path, mask)
    assert horizon_store.load(tmp_path) == mask
    assert (tmp_path / horizon_store.STELLARIUM).read_text() == "0.0 22.5\n180.0 31.0\n"
    assert horizon_store.load(tmp_path / "nowhere") == HorizonMask()
