"""Line protocol between the host and the Nano Every (firmware/astro_mcu).

Every line is ASCII `<BODY>*<CS>\n`, where CS is the XOR of BODY's bytes as two hex digits
(NMEA style), so a garbled line from a loose USB cable is dropped rather than misread.

MCU -> host:  POS <az_counts> <alt_counts>          (20 Hz)
              ENV <temp_c> <rh_pct> <optic_c|nan>   (1 Hz; optic from the DS18B20, if fitted)
              VER <name> <version>                  (reply to VER?)
              BOOT <name> <version>                 (sent once at every start-up: counts are 0)
              ERR <text>
Host -> MCU:  HEAT <channel 0|1> <percent 0-100>
              ZERO                                  (encoder counts to 0)
              VER?
              PING                                  (heartbeat; heaters off after 10 s without one)
"""

import math
from dataclasses import dataclass


def checksum(body: str) -> str:
    cs = 0
    for b in body.encode("ascii", "replace"):  # garbled input just fails the comparison
        cs ^= b
    return f"{cs:02X}"


def frame(body: str) -> bytes:
    return f"{body}*{checksum(body)}\n".encode("ascii")


@dataclass(frozen=True)
class Position:
    az_counts: int
    alt_counts: int


@dataclass(frozen=True)
class Environment:
    temp_c: float
    rh_pct: float
    optic_c: float  # nan if no optic sensor


@dataclass(frozen=True)
class Version:
    name: str
    version: str


@dataclass(frozen=True)
class Boot:
    name: str
    version: str


@dataclass(frozen=True)
class McuError:
    text: str


Message = Position | Environment | Version | Boot | McuError


def parse(line: bytes | str) -> Message | None:
    """Parse one line; None if it is garbled, has a bad checksum, or is unknown."""
    text = line.decode("ascii", "replace") if isinstance(line, bytes) else line
    body, sep, cs = text.strip().rpartition("*")
    if not sep or cs.upper() != checksum(body):
        return None
    tokens = body.split()
    if not tokens:  # e.g. "*00": valid checksum, nothing in it
        return None
    kind, *args = tokens
    try:
        if kind == "POS" and len(args) == 2:
            return Position(int(args[0]), int(args[1]))
        if kind == "ENV" and len(args) == 3:
            return Environment(float(args[0]), float(args[1]), float(args[2]))
        if kind == "VER" and len(args) == 2:
            return Version(*args)
        if kind == "BOOT" and len(args) == 2:
            return Boot(*args)
        if kind == "ERR":
            return McuError(" ".join(args))
    except ValueError:
        return None
    return None


def heat(channel: int, percent: float) -> bytes:
    return frame(f"HEAT {channel} {round(min(max(percent, 0), 100))}")


def dew_point_c(temp_c: float, rh_pct: float) -> float:
    """Magnus formula (Alduchov & Eskridge constants), good to ~0.4 C over -40..50 C."""
    a, b = 17.625, 243.04
    gamma = math.log(max(rh_pct, 0.1) / 100) + a * temp_c / (b + temp_c)
    return b * gamma / (a - gamma)
