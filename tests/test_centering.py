import numpy as np
import pytest

from astro.guidance.centering import Centerer

SIZE = (1928, 1090)
SCALE = 3600.0  # px per degree (sim main camera: 1"/px)


def rotation(deg, mirrored=False):
    c, s = np.cos(np.radians(deg)), np.sin(np.radians(deg))
    m = SCALE * np.array([[c, -s], [s, c]]) @ np.diag([1, -1])
    return m @ np.diag([-1, 1]) if mirrored else m


class Rig:
    """Scope + main camera: the camera aims `aim_offset` (deg, sky) away from the finder model."""

    def __init__(self, rot, mirrored, aim_offset):
        self.A, self.offset = rotation(rot, mirrored), np.array(aim_offset)
        self.alt, self.az = 40.0, 100.0  # where the finder model says we point (on target)
        self.target = np.array([100.0 * np.cos(np.radians(40)), 40.0])  # target in sky coords

    def sky(self):
        return np.array([self.az * np.cos(np.radians(self.alt)), self.alt])

    def target_px(self):
        aim = self.sky() + self.offset
        return tuple(np.array(SIZE) / 2 + self.A @ (self.target - aim))

    def obey(self, cue, step=0.05):  # a "tiny bit" push is about 1-2 encoder counts
        if not cue or "push" not in cue:
            return
        d = {"left": (-step, 0), "right": (step, 0), "up": (0, step), "down": (0, -step)}
        dx, dy = next(v for k, v in d.items() if k in cue)
        self.alt += dy
        self.az += dx / np.cos(np.radians(self.alt))


@pytest.mark.parametrize("rot,mirrored", [(0, False), (73, False), (200, True)])
def test_calibrates_then_centers_and_learns_offset(rot, mirrored):
    rig = Rig(rot, mirrored, aim_offset=(0.06, -0.04))  # target starts ~4 arcmin off-center
    centerer = Centerer(SIZE)
    said = []
    for _ in range(200):
        step = centerer.update((rig.alt, rig.az), rig.target_px())
        said.append(step.say)
        if step.done:
            break
        rig.obey(step.say)
    assert step.done, said
    assert "push left a tiny bit, then stop" in said and centerer.calibrated
    learned = (centerer.offset.d_az_sky_deg, centerer.offset.d_alt_deg)
    assert learned == (pytest.approx(0.06, abs=0.003), pytest.approx(-0.04, abs=0.003))


def test_second_target_needs_no_calibration():
    rig = Rig(30, False, aim_offset=(-0.12, 0.08))
    centerer = Centerer(SIZE)
    for _ in range(200):
        step = centerer.update((rig.alt, rig.az), rig.target_px())
        if step.done:
            break
        rig.obey(step.say)
    assert step.done
    assert centerer.calibrated
    centerer.restart()
    rig.alt, rig.az = 40.0, 100.0
    first = centerer.update((rig.alt, rig.az), rig.target_px())
    assert first.say.startswith("push") and "tiny bit" not in first.say


def test_lost_target_is_spoken():
    assert "can't see it" in Centerer(SIZE).update((40, 100), None).say
