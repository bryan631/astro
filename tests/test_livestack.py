import time
from types import SimpleNamespace

import numpy as np
from scipy import ndimage

from astro.capture.live_stacker import LiveStacker
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


def test_safety_stop_mid_stack_keeps_frames_and_restores_camera(tmp_path):
    unsafe = {"reason": None}
    cam = SkyCamera()
    stacker = LiveStacker(cam, tmp_path, lambda: unsafe["reason"])
    live = stacker.start("M27", 30)
    assert cam.exposure_s == 0.2 and cam.gain == 300
    while live.frames < 3:
        time.sleep(0.02)
    unsafe["reason"] = "daytime lockout"
    assert live.done.wait(5)
    assert "daytime lockout" in live.error and live.frames >= 3
    assert live.picture.exists() and not live.preview.exists()  # moved into the gallery
    assert live.picture.with_suffix(".fits").exists()  # linear stack kept for Siril/GraXpert
    assert not list(live.preview.parent.glob("*.tmp"))  # no half-written previews left behind
    assert (cam.exposure_s, cam.gain) == (0.01, 0)  # planetary mode restored


def test_four_stars_anchor_the_stack_three_do_not(monkeypatch):
    frame = np.zeros((64, 64), np.uint8)
    for stars, anchored in [(3, False), (4, True)]:
        monkeypatch.setattr(livestack, "check_focus", lambda lum, n=stars: SimpleNamespace(stars=n))
        assert LiveStack("GRBG").add(frame) is anchored  # a double star plus two field stars
