import numpy as np
import pytest

from astro.guidance.centering import Centerer, centering_phrases
from astro.guidance.engine import cue_phrases
from astro.pointing.main_offset import local_delta

SIZE = (1928, 1090)
SCALE = 3600.0  # px per degree (sim main camera: 1"/px)


def rotation(deg, mirrored=False):
    c, s = np.cos(np.radians(deg)), np.sin(np.radians(deg))
    m = SCALE * np.array([[c, -s], [s, c]]) @ np.diag([1, -1])
    return m @ np.diag([-1, 1]) if mirrored else m


class Rig:
    """Scope + main camera in local sky geometry. The finder model says the scope points at
    (alt, az); the main camera actually aims `aim_offset` (local d_az_sky, d_alt deg) from there."""

    def __init__(self, rot, mirrored, aim_offset, target=(40.0, 100.0)):
        self.A, self.offset = rotation(rot, mirrored), np.array(aim_offset)
        self.target = target  # true (alt, az)
        self.alt, self.az = target  # finder model says "on target"

    def aim(self):
        alt = self.alt + self.offset[1]
        return alt, self.az + self.offset[0] / np.cos(np.radians(self.alt))

    def target_px(self):
        return tuple(np.array(SIZE) / 2 + self.A @ local_delta(*self.aim(), *self.target))

    def obey(self, cue):
        if not cue or not any(d in cue for d in ("left", "right", "up", "down")):
            return
        # Calibration "tiny bit" pushes are ~1-2 encoder counts; centering nudges are gentler.
        step = 0.05 if "tiny bit" in cue or "more" in cue else 0.02
        d = {"left": (-step, 0), "right": (step, 0), "up": (0, step), "down": (0, -step)}
        dx, dy = next(v for k, v in d.items() if k in cue)
        self.az += dx / np.cos(np.radians(self.alt))
        self.alt += dy


@pytest.mark.parametrize("rot,mirrored", [(0, False), (73, False), (200, True)])
def test_calibrates_then_centers_and_learns_offset(rot, mirrored):
    rig = Rig(rot, mirrored, aim_offset=(0.06, -0.04))  # target starts ~4 arcmin off-center
    centerer = Centerer(SIZE)
    centerer.restart(rig.target)
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
    centerer.restart(rig.target)
    for _ in range(200):
        step = centerer.update((rig.alt, rig.az), rig.target_px())
        if step.done:
            break
        rig.obey(step.say)
    assert step.done
    assert centerer.calibrated
    centerer.restart(rig.target)
    rig.alt, rig.az = rig.target
    first = centerer.update((rig.alt, rig.az), rig.target_px())
    assert first.say.endswith("a little") and "tiny bit" not in first.say  # no recalibration


def test_lost_target_is_spoken():
    assert "can't see it" in Centerer(SIZE).update((40, 100), None).say


def test_pure_altitude_push_has_no_azimuth_component():
    assert local_delta(40.0, 100.0, 40.1, 100.0) == pytest.approx([0.0, 0.1])
    assert local_delta(40.0, 359.95, 40.0, 0.05)[0] == pytest.approx(0.1 * np.cos(np.radians(40)))


def test_offset_exact_despite_finder_residual_and_wraparound():
    """Finder stopped 3' short, near az 0: the learned offset is still the camera's own."""
    rig = Rig(55, False, aim_offset=(0.07, -0.05), target=(35.0, 359.98))
    rig.alt, rig.az = 35.0 - 0.05, 359.98  # finder's leftover error (within its tolerance)
    centerer = Centerer(SIZE)
    centerer.restart(rig.target)
    for _ in range(200):
        step = centerer.update((rig.alt, rig.az), rig.target_px())
        if step.done:
            break
        rig.obey(step.say)
    learned = (centerer.offset.d_az_sky_deg, centerer.offset.d_alt_deg)
    assert learned == (pytest.approx(0.07, abs=0.003), pytest.approx(-0.05, abs=0.003))


def test_centering_phrases_cover_what_centering_says():
    rig = Rig(73, False, aim_offset=(0.06, -0.04))
    centerer = Centerer(SIZE)
    centerer.restart(rig.target)
    said = set()
    for _ in range(200):
        step = centerer.update((rig.alt, rig.az), rig.target_px())
        said.add(step.say)
        if step.done:
            break
        rig.obey(step.say)
    said.add(centerer.update((40, 100), None).say)
    assert said - {None} <= set(cue_phrases()) | set(centering_phrases())
