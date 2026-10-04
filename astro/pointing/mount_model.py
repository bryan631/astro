"""Alt-az mount model: encoder angles -> true sky alt/az.

Parameters: azimuth zero offset, altitude zero offset, and base tilt (small rotations
about the north and east horizontal axes). Fitted by least squares from syncs, where a
sync pairs the encoder angles with the true alt/az from a plate solve.
One sync fits only the offsets; two or more also fit the tilt.
"""

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import least_squares


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

    def to_sky(self, enc_alt_deg: float, enc_az_deg: float) -> tuple[float, float]:
        """Encoder angles -> true (alt, az) in degrees."""
        v = _vec(enc_alt_deg + self.alt_offset_deg, enc_az_deg + self.az_offset_deg)
        return _altaz(_tilt_matrix(self.tilt_n_deg, self.tilt_e_deg) @ v)

    def add_sync(self, sync: Sync) -> float:
        """Add a sync, refit, and return the RMS residual in arcminutes."""
        self.syncs.append(sync)
        return self.fit()

    def fit(self) -> float:
        fit_tilt = len(self.syncs) >= 2
        x0 = [self.az_offset_deg, self.alt_offset_deg]
        if fit_tilt:
            x0 += [self.tilt_n_deg, self.tilt_e_deg]
        targets = [_vec(s.true_alt_deg, s.true_az_deg) for s in self.syncs]

        def residuals(x: np.ndarray) -> np.ndarray:
            self._set(x)
            return np.concatenate(
                [_vec(*self.to_sky(s.enc_alt_deg, s.enc_az_deg)) - t for s, t in zip(self.syncs, targets)]
            )

        result = least_squares(residuals, x0)
        self._set(result.x)
        # Chord length ~ angle (radians) for small errors; 2 independent components per sync.
        rms_rad = np.sqrt(np.sum(result.fun**2) / len(self.syncs))
        return float(np.degrees(rms_rad) * 60)

    def _set(self, x: np.ndarray) -> None:
        self.az_offset_deg, self.alt_offset_deg = x[0], x[1]
        if len(x) == 4:
            self.tilt_n_deg, self.tilt_e_deg = x[2], x[3]
