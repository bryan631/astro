"""A bright star through the XT8 with a Bahtinov mask, by Fourier optics, for testing.

The pupil (8" mirror, 47 mm secondary shadow, four spider vanes) times the mask's three slit
gratings and a defocus phase; its Fourier transform is the star's image, summed over visible
wavelengths, blurred by seeing, then binned to camera pixels with sky, noise and 8-bit clipping.
Without the mask it renders the plain star (spider spikes only), for false-positive tests.
"""

from functools import lru_cache

import numpy as np
from scipy import ndimage

D_M, OBSTRUCTION_M, VANE_M, F_RATIO = 0.203, 0.047, 0.001, 5.9
PITCH_M = 0.0075  # slit period: ~focal length / 160 (Bahtinov's rule of thumb), half open
WAVELENGTHS_M = np.linspace(450e-9, 650e-9, 9)
N = 512  # FFT grid
SAMPLE_ARCSEC = 0.25  # focal-plane sampling before binning to camera pixels (Airy core: 0.57")


def pupil(x, y, mask, theta_deg, mask_angle_deg, defocus_waves_m, wavelength):
    """Complex pupil on (x, y) meters: aperture, obstruction, spider, mask, defocus."""
    r = np.hypot(x, y)
    open_ = (r <= D_M / 2) & (r >= OBSTRUCTION_M / 2)
    open_ &= (np.abs(x) > VANE_M / 2) & (np.abs(y) > VANE_M / 2)  # four vanes on the axes
    if mask:
        a = np.radians(mask_angle_deg)
        u, v = x * np.cos(a) + y * np.sin(a), -x * np.sin(a) + y * np.cos(a)  # the mask's own axes
        t = np.radians(theta_deg)
        # The mask is split along v = 0. One half: slits across that line (along v), so its
        # spike runs along the line and defocus, which moves the half's light across the line,
        # slides it sideways. The other half's two quarters: slits turned +-theta. Each
        # grating is half open; bars along v = 0 and (in the second half) u = 0 hold them.
        g_half = np.cos(2 * np.pi * u / PITCH_M) > 0
        g_plus = np.cos(2 * np.pi * (u * np.cos(t) + v * np.sin(t)) / PITCH_M) > 0
        g_minus = np.cos(2 * np.pi * (u * np.cos(t) - v * np.sin(t)) / PITCH_M) > 0
        grating = np.where(v > 0, g_half, np.where(u > 0, g_plus, g_minus))
        open_ &= grating & (np.abs(v) > 0.003) & ~((v < 0) & (np.abs(u) < 0.0015))
    rho2 = (r / (D_M / 2)) ** 2
    phase = 2 * np.pi * defocus_waves_m / wavelength * rho2
    return open_ * np.exp(1j * phase)


@lru_cache(maxsize=64)
def psf(defocus_um=0.0, mask=True, theta_deg=20.0, mask_angle_deg=0.0) -> np.ndarray:
    """Polychromatic intensity on a SAMPLE_ARCSEC grid (N x N), total 1."""
    w020 = defocus_um * 1e-6 / (8 * F_RATIO**2)  # focal shift -> wavefront defocus at the edge
    out = np.zeros((N, N))
    rad = np.radians(SAMPLE_ARCSEC / 3600)
    for lam in WAVELENGTHS_M:
        dx = lam / (N * rad)  # pupil sampling that puts SAMPLE_ARCSEC on each focal-plane pixel
        c = (np.arange(N) - N / 2) * dx
        x, y = np.meshgrid(c, c)
        field = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(pupil(x, y, mask, theta_deg, mask_angle_deg, w020, lam))))
        out += np.abs(field) ** 2
    return out / out.sum()


def frame(defocus_um=0.0, seeing_arcsec=2.0, mask=True, theta_deg=20.0, mask_angle_deg=0.0,
          flux=2e6, scale_arcsec_px=1.0, size=125, sky=20.0, noise=5.0, offset_px=(0.0, 0.0),
          trail_px=0.0, trail_deg=0.0, seed=0) -> np.ndarray:
    """An 8-bit-scale camera frame (size x size, at most 127 at 1"/px) of the star, binned to
    `scale_arcsec_px`, drifted `trail_px` camera pixels at `trail_deg` during the exposure."""
    img = psf(defocus_um, mask, theta_deg, mask_angle_deg)
    img = ndimage.gaussian_filter(img, seeing_arcsec / 2.355 / SAMPLE_ARCSEC)
    if trail_px:
        step = np.array([np.sin(np.radians(trail_deg)), np.cos(np.radians(trail_deg))]) * trail_px * scale_arcsec_px / SAMPLE_ARCSEC
        img = np.mean([ndimage.shift(img, step * f, order=1) for f in np.linspace(-0.5, 0.5, 7)], axis=0)
    img = ndimage.shift(img, np.array(offset_px) * scale_arcsec_px / SAMPLE_ARCSEC, order=1)
    k = round(scale_arcsec_px / SAMPLE_ARCSEC)
    n = N // k * k
    binned = img[:n, :n].reshape(n // k, k, n // k, k).sum(axis=(1, 3)) * flux
    c = binned.shape[0] // 2
    cut = binned[c - size // 2:c + size // 2 + 1, c - size // 2:c + size // 2 + 1]
    rng = np.random.default_rng(seed)
    return np.clip(rng.poisson(np.maximum(cut, 0)) + sky + rng.normal(0, noise, cut.shape), 0, 255)
