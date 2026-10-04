"""IntelliScope encoder counts -> mount angles."""

from dataclasses import dataclass

COUNTS_PER_REV = 9216


@dataclass(frozen=True)
class EncoderAxis:
    counts_per_rev: int = COUNTS_PER_REV
    sign: int = 1  # +1 or -1; learned on first push

    def to_degrees(self, counts: int) -> float:
        """Raw counts -> degrees in [0, 360). Zero offsets belong to the mount model."""
        return (self.sign * counts * 360.0 / self.counts_per_rev) % 360.0
