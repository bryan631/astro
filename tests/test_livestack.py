from types import SimpleNamespace

import numpy as np
import pytest
from scipy import ndimage

from astro.capture import live_stacker
from astro.process import livestack
from astro.process.livestack import LiveStack, stretch
from astro.process.planet import _LAYOUT

SIZE = 256  # raw pixels; stacks are half that after superpixel debayer


def sky_scene(rng):
    """Gray-ish star field (40 stars) plus a faint reddish nebula, as float RGB (h, w, 3)."""
    img = np.zeros((SIZE, SIZE))
    ys, xs = rng.uniform(20, SIZE - 20, (2, 40))
    img[ys.astype(int), xs.astype(int)] = rng.uniform(300, 3000, 40)
    stars = ndimage.gaussian_filter(img, 1.5)
    y, x = np.mgrid[0:SIZE, 0:SIZE]
    nebula = 6 * np.exp(-((x - 128) ** 2 + (y - 128) ** 2) / (2 * 30**2))
    return np.dstack([stars + nebula, stars + 0.4 * nebula, stars + 0.3 * nebula])


def observe(scene, angle_deg, shift, rng, noise=8.0):
    """Rotate/shift the sky (field rotation + drift), mosaic to GRBG, add sky + read noise."""
    moved = np.dstack([ndimage.shift(ndimage.rotate(scene[..., c], angle_deg, reshape=False,
                                                    order=1), shift, order=1)
                       for c in range(3)])
    raw = np.zeros((SIZE, SIZE))
    for (y, x), c in zip(_LAYOUT["GRBG"], (0, 1, 1, 2), strict=True):
        raw[y::2, x::2] = moved[y::2, x::2, c]
    return (raw + 20 + rng.normal(0, noise, raw.shape)).clip(0, 255).astype(np.uint8)


def test_stack_registers_rotated_frames_and_cuts_noise():
    rng = np.random.default_rng(3)
    scene = sky_scene(rng)
    stack = LiveStack("GRBG")
    for i in range(16):
        assert stack.add(observe(scene, angle_deg=0.25 * i, shift=(i * 0.8, -i * 0.5), rng=rng))
    single = LiveStack("GRBG")
    single.add(observe(scene, 0, (0, 0), rng))
    sky = (slice(5, 25), slice(5, 25))  # star-free corner
    noise_single = single.image()[sky].mean(axis=-1).std()
    noise_stack = stack.image()[sky].mean(axis=-1).std()
    assert stack.status.frames_added == 16
    assert noise_stack < noise_single / 2.5  # ~sqrt(16) = 4 in theory


def test_cloudy_frame_is_skipped():
    rng = np.random.default_rng(4)
    stack = LiveStack("GRBG")
    stack.add(observe(sky_scene(rng), 0, (0, 0), rng))
    cloud = (20 + rng.normal(0, 8, (SIZE, SIZE))).clip(0, 255).astype(np.uint8)
    assert not stack.add(cloud)
    assert stack.status.frames_skipped == 1 and stack.status.frames_added == 1


def test_shift_fallback_stacks_frames_triangles_cannot_match(monkeypatch):
    """2026-10-10: triangle matching dropped most 0.2 s frames of the Dumbbell; a drifting
    field is nearly a pure shift, which phase correlation measures from the whole picture."""
    def no_triangles(*a, **k):
        raise ValueError("List of matching triangles exhausted")

    monkeypatch.setattr(livestack.astroalign, "find_transform", no_triangles)
    rng = np.random.default_rng(5)
    scene = sky_scene(rng)
    stack = LiveStack("GRBG")
    for i in range(6):
        assert stack.add(observe(scene, 0, (-2.0 * i, 3.0 * i), rng))  # rows, cols in raw px
    assert stack.reference_center() == pytest.approx((64 + 1.5 * 5, 64 - 1.0 * 5), abs=0.3)
    cloud = (20 + rng.normal(0, 8, (SIZE, SIZE))).clip(0, 255).astype(np.uint8)
    assert not stack.add(cloud)  # noise has no real peak: still skipped


def test_stretch_maps_to_8bit():
    out = stretch(np.random.default_rng(0).normal(100, 5, (32, 32, 3)))
    assert out.dtype == np.uint8 and out.max() == 255 and out.min() == 0


def test_first_frame_needs_stars():
    rng = np.random.default_rng(5)
    stack = LiveStack("GRBG")
    cloud = (20 + rng.normal(0, 8, (SIZE, SIZE))).clip(0, 255).astype(np.uint8)
    assert not stack.add(cloud) and not stack.has_frames
    assert stack.add(observe(sky_scene(rng), 0, (0, 0), rng))  # first clear frame anchors


class SkyCamera:
    """Main-camera stub for LiveStacker: a slowly rotating star field."""

    bayer = "GRBG"

    def __init__(self):
        self.rng, self.n = np.random.default_rng(6), 0
        self.scene = sky_scene(self.rng)
        self.exposure_s, self.gain = 0.01, 0

    def set_roi(self, roi):
        pass

    def set_exposure(self, s):
        self.exposure_s = s

    def set_gain(self, g):
        self.gain = g

    def capture(self):
        self.n += 1
        return observe(self.scene, 0.2 * self.n, (0.5 * self.n, 0), self.rng)


def test_four_stars_anchor_the_stack_three_do_not(monkeypatch):
    frame = np.zeros((64, 64), np.uint8)
    for stars, anchored in [(3, False), (4, True)]:
        monkeypatch.setattr(livestack, "check_focus", lambda lum, n=stars: SimpleNamespace(stars=n))
        assert LiveStack("GRBG").add(frame) is anchored  # a double star plus two field stars


class MeterCamera:
    """A camera whose sky brightens 10x per 200 gain steps (0.1 dB), with a star field that
    saturates in a patch above `burns_at`."""

    def __init__(self, sky_at_300=2.4, burns_at=None):
        self.gain, self.sky_at_300, self.burns_at = 300, sky_at_300, burns_at

    def set_gain(self, gain):
        self.gain = gain

    def capture(self):
        rng = np.random.default_rng(self.gain)
        sky = self.sky_at_300 * 10 ** ((self.gain - 300) / 200)
        raw = rng.poisson(sky, (400, 400)).clip(0, 255).astype(np.uint8)
        if self.burns_at is not None and self.gain > self.burns_at:
            raw[:20, :20] = 255  # a bright core
        return raw


def test_gain_is_metered_to_put_the_sky_above_black():
    """2026-10-10: at gain 300 the stacks' sky sat at ~2 of 255. The meter raises it to ~12."""
    cam = MeterCamera()
    gain, _ = live_stacker.meter_gain(cam, 300)
    assert cam.gain == gain and abs(gain - 440) <= 25  # 300 + 200 log10(12 / 2.4)
    assert 9 < cam.sky_at_300 * 10 ** ((gain - 300) / 200) < 16


def test_meter_backs_off_when_a_core_burns_out():
    cam = MeterCamera(burns_at=360)
    gain, _ = live_stacker.meter_gain(cam, 300)
    assert gain <= 360


def test_stack_waits_through_a_drift_out_and_keeps_adding_after_the_nudge(tmp_path, monkeypatch):
    """2026-10-10: stacks stopped at "lost the stars" after ~1.5 minutes of drift. Now the
    stack asks for a nudge and adds to the same picture once the field is back."""
    rng = np.random.default_rng(5)
    scene = sky_scene(rng)
    cloud = (20 + rng.normal(0, 8, (SIZE, SIZE))).clip(0, 255).astype(np.uint8)
    frames = [observe(scene, 0, (i, 0), rng) for i in range(3)] + [cloud] * 12 + \
             [observe(scene, 0, (-5 + i, 2), rng) for i in range(3)]
    adrift = []

    class Cam:
        bayer, exposure_s, gain = "GRBG", 0.2, 300

        def set_roi(self, roi): pass
        def set_exposure(self, s): pass
        def set_gain(self, g): pass

        def capture(self):
            adrift.append(stacker.current.adrift)
            if not frames:
                stacker.stop()
                return cloud
            return frames.pop(0)

    monkeypatch.setattr(live_stacker, "meter_gain", lambda cam, gain: (gain, 12.0))
    stacker = live_stacker.LiveStacker(Cam(), tmp_path / "gallery", tmp_path / "live")
    live = stacker.start("M99", 60)
    assert live.done.wait(30)
    assert live.frames == 6 and not live.error and live.picture.exists()
    assert True in adrift and not live.adrift  # it asked for a nudge, then found the field again


def test_picture_drops_the_edges_few_frames_reached():
    """2026-10-11: the drift left noisy color bands along two edges of Andromeda."""
    rng = np.random.default_rng(6)
    scene = sky_scene(rng)
    stack = LiveStack("GRBG")
    for i in range(8):
        assert stack.add(observe(scene, 0, (0, i * 6), rng))  # drifts 42 raw px: 21 stacked px
    full, cropped = stack.image(), stack.picture()
    assert full.shape[0] - cropped.shape[0] <= 2 and full.shape[1] - 21 <= cropped.shape[1] < full.shape[1] - 5
