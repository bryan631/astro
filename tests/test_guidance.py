import pytest

from astro.devices.sim.scope import SimScope, SimUser
from astro.guidance.engine import DirectionLearner, Guide, wrap180


def run_session(start, target, seconds=120, dt=0.1, **guide_kw):
    scope, user = SimScope(*start), SimUser()
    guide = Guide(*target, **guide_kw)
    spoken = []
    t = 0.0
    while t < seconds:
        state, cue = guide.update(scope.alt, scope.az, t)
        if cue:
            spoken.append(cue.text)
            user.hear(cue.text, t)
        scope.step(*user.act(t), dt)
        t += dt
        settled = state.on_target and user.v == (0.0, 0.0) and t > 1
        # Stay put 2 s to confirm it is stable.
        if settled and all(guide.update(scope.alt, scope.az, t + k / 10)[0].on_target
                           for k in range(20)):
            return state, spoken
    return state, spoken


def test_wrap180():
    assert wrap180(350) == -10 and wrap180(-190) == 170


@pytest.mark.parametrize("start,target", [
    ((20, 10), (55, 80)),
    ((60, 350), (40, 20)),  # crosses north
    ((30, 200), (31, 199)),  # already close
])
def test_simulated_user_converges(start, target):
    state, spoken = run_session(start, target)
    assert state.on_target, spoken
    assert spoken[-1] == "stop"
    assert len(spoken) < 40


def test_says_stop_immediately_without_rate_limit():
    g = Guide(45, 100)
    g.update(44, 100, 0.0)
    _, cue = g.update(45, 100, 0.05)
    assert cue and cue.text == "stop"


def test_does_not_repeat_itself_rapidly():
    g = Guide(45, 100)
    texts = [c.text for t in range(30) if (c := g.update(10, 100, t * 0.1)[1])]
    assert texts == ["push up"]


def test_repeats_become_keep_going():
    g = Guide(45, 100)
    texts = [c.text for t in range(60) if (c := g.update(10, 100, t * 0.1)[1])]
    assert texts[:2] == ["push up", "keep going"]


def test_hysteresis():
    g = Guide(45, 100, tolerance_arcmin=4)
    g.update(45, 100, 0)
    state, _ = g.update(45 + 5 / 60, 100, 0.1)  # 5' off: still within 1.5x
    assert state.on_target
    state, _ = g.update(45 + 7 / 60, 100, 0.2)
    assert not state.on_target


def test_left_right_follows_learned_direction():
    _, cue = Guide(45, 110, right_is_plus_az=False).update(45, 100, 0)
    assert cue.text == "push left"


def test_direction_learner():
    learn = DirectionLearner()
    assert learn.observe("right", 0.1) is None
    assert learn.observe("right", 2.0) is True
    assert learn.observe("right", -2.0) is False


def test_beep_rate_increases_when_closer():
    g = Guide(45, 100)
    far, _ = g.update(43, 100, 0)
    near, _ = g.update(44.9, 100, 1)
    assert near.beep_hz > far.beep_hz > 0


def test_never_says_keep_going_when_it_means_stop():
    """Review H1: an axis 'stop', a rate-limited cue, then arriving on target."""
    g = Guide(45.0, 100.0, tolerance_arcmin=4)
    cues = [g.update(alt, az, t) for alt, az, t in
            [(44.0, 100.14, 0), (44.99, 100.14, 2.0), (44.99, 100.14, 2.1), (44.99, 100.07, 2.5)]]
    state, cue = cues[-1]
    assert state.on_target and cue.text == "stop"


def test_stop_nudge_back_on_target_says_stop_again():
    """On target, nudged off (cue rate-limited), back on target 0.6 s later: say "stop"."""
    g = Guide(45.0, 100.0, tolerance_arcmin=4)
    said = [g.update(alt, az, t)[1] for t, (alt, az) in
            [(0, (44.0, 100.0)), (1.0, (44.99, 100.0)), (1.2, (44.99, 100.3)),
             (1.6, (44.99, 100.02))]]
    assert said[1].text == "stop" and said[3].text == "stop"
