"""Host driver for the Nano Every: encoder counts, environment, dew heaters, heartbeat.

Implements `MountEncoders`. A reader thread parses lines; every ENV reading drives the dew
heaters; a PING every couple of seconds keeps the firmware's failsafe from cutting them.
"""

import glob
import os
import threading
import time

import serial

from astro.devices import mcu_protocol as proto
from astro.devices.dew import heater_percent

BAUD = 115200
PING_EVERY_S = 2.0  # firmware turns heaters off after 10 s without a PING
HEATER_CHANNELS = (0, 1)  # secondary holder, finder lens


def default_port() -> str:
    ports = sorted(glob.glob("/dev/ttyACM*"))
    return os.environ.get("ASTRO_MCU_PORT") or (ports[0] if ports else "/dev/ttyACM0")


class Mcu:
    def __init__(self, port=None, serial_factory=serial.Serial):
        """`serial_factory(port, baud, timeout=...)` is replaceable for tests."""
        self._ser = serial_factory(port or default_port(), BAUD, timeout=0.5)
        self._lock = threading.Lock()
        self.position = proto.Position(0, 0)
        self.environment: proto.Environment | None = None
        self.version: proto.Version | None = None
        self.errors: list[str] = []
        self.heat = 0
        self._stop = threading.Event()
        self._reader = threading.Thread(target=self._read_loop, daemon=True)
        self._pinger = threading.Thread(target=self._ping_loop, daemon=True)

    def start(self) -> "Mcu":
        self._send(proto.frame("VER?"))
        self._reader.start()
        self._pinger.start()
        return self

    def close(self) -> None:
        self._stop.set()
        for worker in (self._reader, self._pinger):  # no late ENV may turn heat back on
            if worker.is_alive():
                worker.join(timeout=2)
        for ch in HEATER_CHANNELS:
            self._send(proto.heat(ch, 0))
        self._ser.close()

    def counts(self) -> tuple[int, int]:
        """MountEncoders: raw (azimuth, altitude) counts."""
        return self.position.az_counts, self.position.alt_counts

    def zero(self) -> None:
        self._send(proto.frame("ZERO"))

    def handle(self, msg: proto.Message) -> None:
        """Apply one parsed message (called by the reader thread; public for tests)."""
        if isinstance(msg, proto.Position):
            self.position = msg
        elif isinstance(msg, proto.Environment):
            self.environment = msg
            self.heat = heater_percent(msg)
            for ch in HEATER_CHANNELS:
                self._send(proto.heat(ch, self.heat))
        elif isinstance(msg, proto.Version):
            self.version = msg
        elif isinstance(msg, proto.McuError):
            self.errors = [*self.errors[-9:], msg.text]

    def _send(self, data: bytes) -> None:
        with self._lock:
            self._ser.write(data)

    def _read_loop(self) -> None:
        while not self._stop.is_set():
            try:
                line = self._ser.readline()
            except (serial.SerialException, OSError):
                if self._stop.is_set():
                    return
                time.sleep(1.0)  # cable bumped; keep trying
                continue
            if line and (msg := proto.parse(line)) is not None:
                self.handle(msg)

    def _ping_loop(self) -> None:
        while not self._stop.wait(PING_EVERY_S):
            try:
                self._send(proto.frame("PING"))
            except (serial.SerialException, OSError):
                pass
