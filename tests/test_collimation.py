import numpy as np
import pytest

from astro.capture.collimation import CollimationCoach, analyze


def donut(shadow_dx=0.0, shadow_dy=0.0, radius=40, size=160):
    """A defocused star: a lit disk with the secondary's shadow cut out of it."""
    y, x = np.mgrid[:size, :size] - size / 2
    ring = (np.hypot(x, y) < radius) & (np.hypot(x - shadow_dx, y - shadow_dy) > radius * 0.35)
    rng = np.random.default_rng(0)
    return (20 + 180 * ring + rng.normal(0, 3, ring.shape)).clip(0, 255).astype(np.uint8)


def test_centered_shadow_is_collimated():
    d = analyze(donut())
    assert d.off < 0.03 and d.radius_px == pytest.approx(40, rel=0.05)


def test_offset_shadow_is_found_with_its_clock_position():
    d = analyze(donut(shadow_dx=8))  # 8 px right of a 40 px ring: 20 percent, 3 o'clock
    assert d.off == pytest.approx(0.2, abs=0.04) and d.clock == 3
    assert analyze(donut(shadow_dy=-8)).clock == 12  # image y up is 12 o'clock


def test_in_focus_star_asks_for_a_donut():
    star = np.full((160, 160), 20, np.uint8)
    star[78:82, 78:82] = 250
    assert "donut" in analyze(star)


def test_coach_says_better_worse_and_done():
    coach = CollimationCoach()
    assert "toward 3 o'clock" in coach.update(analyze(donut(shadow_dx=8)))
    assert coach.update(analyze(donut(shadow_dx=4))) == "Better, keep going."
    assert coach.update(analyze(donut(shadow_dx=8))).startswith("Worse")
    assert "centered" in coach.update(analyze(donut()))


def test_session_coaches_until_centered_then_asks_to_refocus():
    from datetime import datetime, timedelta, timezone

    from astro.pointing.coords import Site
    from astro.session import Session

    frames = [donut(shadow_dx=8), donut(shadow_dx=4), donut()]

    class Camera:
        def capture(self):
            return frames.pop(0)

    evening = datetime(2026, 10, 3, 21, 0, tzinfo=timezone(timedelta(hours=-4)))
    s = Session(Site(26.7, -80.1), lambda: (45, 180), clock=lambda: evening,
                main_camera=Camera())
    assert "donut" in s.handle("check the collimation")[0]["text"]
    said = [m["text"] for t in (10, 12, 14) for m in s.tick(t)]
    assert "3 o'clock" in said[0] and said[1] == "Better, keep going."
    assert "centered" in said[2] and "sharp point" in said[2]
    assert s._collimation is None and not s.main_focus_ok


def test_small_turns_add_up_to_better():
    coach = CollimationCoach()
    coach.update(analyze(donut(shadow_dx=8)))  # ~0.20
    said = [coach.update(analyze(donut(shadow_dx=dx))) for dx in (7.6, 7.2, 6.8)]
    assert "Better, keep going." in said  # each step < CHANGE, together > CHANGE


def test_collimation_blocks_capture_and_drops_the_focus_gate():
    from datetime import datetime, timedelta, timezone

    from astro.pointing.coords import Site
    from astro.session import Session

    evening = datetime(2026, 10, 3, 21, 0, tzinfo=timezone(timedelta(hours=-4)))
    s = Session(Site(26.7, -80.1), lambda: (45, 180), clock=lambda: evening,
                main_camera=object())
    s.main_focus_ok = True
    s.handle("collimate")
    assert not s.main_focus_ok
    for cmd in ("take a picture", "focus", "start the horizon walk", "set up the telescope"):
        assert "checking collimation" in s.handle(cmd)[0]["text"]
