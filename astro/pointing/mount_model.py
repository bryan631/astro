"""Alt-az mount model: encoder angles -> true sky alt/az.

Parameters: azimuth zero offset, altitude zero offset, and base tilt (small rotations
about the north and east horizontal axes). Fitted by least squares from syncs, where a
sync pairs the encoder angles with the true alt/az from a plate solve.
One sync fits only the offsets; two or more also fit the tilt.
"""

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import least_squares

MAX_SYNCS = 12  # newest kept: an old sync can't outvote a bumped base forever
OUTLIER_X = 3.0  # a sync this many times worse than the median (and over 30') is dropped
OUTLIER_MIN_ARCMIN = 30.0
SHIFT_SYNCS = 2  # this many new syncs rejected in a row: the base moved, start over from them


def _vec(alt_deg: float, az_deg: float) -> np.ndarray:
    alt, az = np.radians(alt_deg), np.radians(az_deg)
    return np.array([np.cos(alt) * np.cos(az), np.cos(alt) * np.sin(az), np.sin(alt)])  # N, E, up


def _altaz(v: np.ndarray) -> tuple[float, float]:
    alt = np.degrees(np.arcsin(np.clip(v[2], -1, 1)))
    az = np.degrees(np.arctan2(v[1], v[0])) % 360
    return float(alt), float(az)


def _tilt_matrix(tilt_n_deg: float, tilt_e_deg: float) -> np.ndarray:
    """Rotation about the north axis, then the east axis."""
    a, b = np.radians(tilt_n_deg), np.radians(tilt_e_deg)
    rot_n = np.array([[1, 0, 0], [0, np.cos(a), -np.sin(a)], [0, np.sin(a), np.cos(a)]])
    rot_e = np.array([[np.cos(b), 0, np.sin(b)], [0, 1, 0], [-np.sin(b), 0, np.cos(b)]])
    return rot_e @ rot_n


@dataclass(frozen=True)
class Sync:
    enc_alt_deg: float
    enc_az_deg: float
    true_alt_deg: float
    true_az_deg: float


@dataclass
class MountModel:
    az_offset_deg: float = 0.0
    alt_offset_deg: float = 0.0
    tilt_n_deg: float = 0.0
    tilt_e_deg: float = 0.0
    syncs: list[Sync] = field(default_factory=list)
    rejected: list[Sync] = field(default_factory=list, repr=False)  # newest syncs dropped

    def to_sky(self, enc_alt_deg: float, enc_az_deg: float) -> tuple[float, float]:
        """Encoder angles -> true (alt, az) in degrees."""
        v = _vec(enc_alt_deg + self.alt_offset_deg, enc_az_deg + self.az_offset_deg)
        return _altaz(_tilt_matrix(self.tilt_n_deg, self.tilt_e_deg) @ v)

    def add_sync(self, sync: Sync) -> float:
        """Add a sync, refit, and return the RMS residual in arcminutes."""
        self.syncs = [*self.syncs, sync][-MAX_SYNCS:]
        rms = self.fit()
        if len(self.syncs) >= 4:  # enough to tell which one is wrong
            errs = self.residuals_arcmin()
            worst = int(np.argmax(errs))
            if errs[worst] > max(OUTLIER_MIN_ARCMIN, OUTLIER_X * float(np.median(errs))):
                dropped = self.syncs.pop(worst)  # a false solve, or the base moved
                self.rejected = [*self.rejected, dropped] if dropped is sync else []
                if len(self.rejected) >= SHIFT_SYNCS:  # newest keep disagreeing: base moved
                    self.syncs, self.rejected = self.rejected, []
                return self.fit()
        self.rejected = []
        return rms

    def residuals_arcmin(self) -> np.ndarray:
        """Per-sync pointing error of the current fit, arcminutes."""
        return np.array([np.degrees(np.linalg.norm(
            _vec(*self.to_sky(s.enc_alt_deg, s.enc_az_deg)) - _vec(s.true_alt_deg, s.true_az_deg)))
            * 60 for s in self.syncs])

    def fit(self) -> float:
        if len(self.syncs) == 1:
            # Exact closed form. A least-squares start from zero can land on the equivalent
            # "flipped over the zenith" solution, which only matches this one point.
            s = self.syncs[0]
            self.alt_offset_deg = s.true_alt_deg - s.enc_alt_deg
            self.az_offset_deg = (s.true_az_deg - s.enc_az_deg) % 360
            self.tilt_n_deg = self.tilt_e_deg = 0.0
            return 0.0
        x0 = [self.az_offset_deg, self.alt_offset_deg, self.tilt_n_deg, self.tilt_e_deg]
        targets = [_vec(s.true_alt_deg, s.true_az_deg) for s in self.syncs]

        def residuals(x: np.ndarray) -> np.ndarray:
            self._set(x)
            return np.concatenate(
                [_vec(*self.to_sky(s.enc_alt_deg, s.enc_az_deg)) - t for s, t in zip(self.syncs, targets, strict=True)]
            )

        # Robust loss: one bad sync can't drag the fit, so it stands out to be dropped.
        result = least_squares(residuals, x0, loss="soft_l1", f_scale=np.radians(0.5))
        self._set(result.x)
        # Chord length ~ angle (radians) for small errors: RMS of the per-sync angular error.
        rms_rad = np.sqrt(np.sum(result.fun**2) / len(self.syncs))
        return float(np.degrees(rms_rad) * 60)

    def _set(self, x: np.ndarray) -> None:
        self.az_offset_deg, self.alt_offset_deg, self.tilt_n_deg, self.tilt_e_deg = map(float, x)
