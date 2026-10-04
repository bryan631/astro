"""IntelliScope handset fallback: encoder positions over RS-232 (Tangent/BBox "Q" query).

Used if the encoder signals aren't clean quadrature for the Nano Every (plan Phase 3 step 1).
The handset answers "Q" with both axes' counts, e.g. "+00123\t-04567\r" (Tangent/BBox convention;
UNVERIFIED on the IntelliScope until Phase 3 — adjust `_REPLY` if it differs). Connect through a
USB-RS232 adapter; 9600 baud 8N1. Implements `MountEncoders`.
"""

import re

import serial

BAUD = 9600
QUERY = b"Q"
MAX_REPLY = 64  # bytes; a reply is 14, leaving room for leading noise (timeout bounds the wait)
_REPLY = re.compile(rb"([+-]\d{5})\t([+-]\d{5})\r")


def parse_reply(data: bytes) -> tuple[int, int] | None:
    """(az, alt) counts from a Q reply, or None if garbled/incomplete."""
    m = _REPLY.search(data)
    return (int(m.group(1)), int(m.group(2))) if m else None


class Handset:
    def __init__(self, port: str, serial_factory=serial.Serial):
        self._ser = serial_factory(port, BAUD, timeout=0.3)
        self._last = (0, 0)
        self.failures = 0  # consecutive unanswered queries (cable out, handset off)

    def counts(self) -> tuple[int, int]:
        """MountEncoders: query the handset; keep the last good reading if it doesn't answer."""
        try:
            self._ser.reset_input_buffer()
            self._ser.write(QUERY)
            reply = parse_reply(self._ser.read_until(b"\r", MAX_REPLY))
        except (serial.SerialException, OSError):  # adapter unplugged: same as no answer
            reply = None
        if reply is None:
            self.failures += 1
            return self._last
        self.failures, self._last = 0, reply
        return reply

    def close(self) -> None:
        self._ser.close()
