import pytest

from astro.capture.edge import EdgeClock


def test_seconds_left_from_a_steady_drift():
    clock = EdgeClock(1000, 500)
    for t in range(6):
        clock.add(float(t), 500 + 30 * t, 250 - 2 * t)  # 30 px/s right (sidereal at 0.5"/px)
    # now at x=650; the edge margin is at 950: 300 px at 30 px/s
    assert clock.seconds_left() == pytest.approx(10.0, abs=0.1)
    assert clock.position() == pytest.approx((0.65, 0.48))


def test_no_answer_until_it_has_seen_drift():
    clock = EdgeClock(1000, 500)
    assert clock.seconds_left() is None
    clock.add(0.0, 500, 250)
    clock.add(0.5, 515, 250)
    assert clock.seconds_left() is None  # half a second isn't enough
    for t in range(1, 5):
        clock.add(float(t), 500, 250)  # parked
    clock.reset()
    assert clock.seconds_left() is None and clock.position() is None


def test_already_past_the_margin_is_zero():
    clock = EdgeClock(1000, 500)
    for t in range(4):
        clock.add(float(t), 20 - 5 * t, 250)  # drifting off the left side
    assert clock.seconds_left() == 0.0
