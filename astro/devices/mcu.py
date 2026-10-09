"""Host driver for the Nano Every: encoder counts, environment, dew heaters, heartbeat.

Implements `MountEncoders`. A reader thread parses lines; every ENV reading drives the dew
heaters; a PING every couple of seconds keeps the firmware's failsafe from cutting them.
"""

import glob
import os
import threading
import time
from collections.abc import Callable

import serial

from astro.devices import mcu_protocol as proto
from astro.devices.dew import heater_percent

BAUD = 115200
PING_EVERY_S = 2.0  # firmware turns heaters off after 10 s without a PING (or HEAT)
ENV_STALE_S = 5.0  # no valid environment reading this long: heaters off
REOPEN_EVERY_S = 1.0  # retry opening the serial port this often after an unplug
HEATER_CHANNELS = (0, 1)  # secondary holder, finder lens


def default_port() -> str:
    # Nano Every enumerates as ttyACM*; a classic Nano (CH340) as ttyUSB*.
    ports = sorted(glob.glob("/dev/ttyACM*")) + sorted(glob.glob("/dev/ttyUSB*"))
    return os.environ.get("ASTRO_MCU_PORT") or (ports[0] if ports else "/dev/ttyACM0")


class _Unplugged:
    """Stands in for the serial port until the board is plugged in."""

    def readline(self) -> bytes:
        raise OSError("encoder board not connected")

    def write(self, data: bytes) -> int:
        raise OSError("encoder board not connected")

    def close(self) -> None:
        pass


class Mcu:
    def __init__(self, port=None, serial_factory=serial.Serial):
        """`serial_factory(port, baud, timeout=...)` is replaceable for tests."""
        self._port, self._factory = port, serial_factory
        try:
            self._ser = self._open()
        except (serial.SerialException, OSError):  # unplugged: the reader keeps reopening
            self._ser = _Unplugged()
        self._lock = threading.Lock()
        self.position = proto.Position(0, 0)
        self.environment: proto.Environment | None = None
        self.version: proto.Version | None = None
        self.errors: list[str] = []
        self.heat = 0
        self._last_pos = self._last_env = time.monotonic()  # start the staleness clocks now
        self.on_reboot: Callable[[], None] | None = None  # e.g. invalidate the mount model
        self.boots = 0  # BOOT lines seen, latched: a boot before anyone listened still counts
        self._stop = threading.Event()
        self._reader = threading.Thread(target=self._read_loop, daemon=True)
        self._pinger = threading.Thread(target=self._ping_loop, daemon=True)

    def start(self) -> "Mcu":
        try:
            self._send(proto.frame("VER?"))
        except (serial.SerialException, OSError):
            pass  # not plugged in yet: asked again on its BOOT
        self._reader.start()
        self._pinger.start()
        return self

    def close(self) -> None:
        self._stop.set()
        for worker in (self._reader, self._pinger):  # no late ENV may turn heat back on
            if worker.is_alive():
                worker.join(timeout=2)
        try:
            for ch in HEATER_CHANNELS:
                self._send(proto.heat(ch, 0))
        except (serial.SerialException, OSError):
            pass  # already unplugged: the firmware's heartbeat timeout turns them off
        try:
            self._ser.close()  # always attempted, even if the heater writes failed
        except (serial.SerialException, OSError):
            pass

    def position_age(self) -> float:
        """Seconds since the last POS line (guidance must not trust frozen counts)."""
        return time.monotonic() - self._last_pos

    def counts(self) -> tuple[int, int]:
        """MountEncoders: raw (azimuth, altitude) counts."""
        return self.position.az_counts, self.position.alt_counts

    def zero(self) -> None:
        self._send(proto.frame("ZERO"))

    def handle(self, msg: proto.Message) -> None:
        """Apply one parsed message (called by the reader thread; public for tests)."""
        if isinstance(msg, proto.Position):
            self.position, self._last_pos = msg, time.monotonic()
        elif isinstance(msg, proto.Environment):
            self.environment, self._last_env = msg, time.monotonic()
            self._set_heat(heater_percent(msg))
        elif isinstance(msg, proto.Version):
            self.version = msg
        elif isinstance(msg, proto.Boot):
            self.boots += 1
            # The board (re)started, so its counts are 0: any mount model built on earlier
            # counts is wrong, including one restored from disk at start-up (CV7).
            if self.on_reboot:
                self.on_reboot()
            # Opening the port resets a classic Nano (DTR), so the VER? from start() can be
            # lost in the bootloader; ask again now that the firmware is up. Last, so a failed
            # write can't skip the reboot handling above.
            self._send(proto.frame("VER?"))
        elif isinstance(msg, proto.McuError):
            self.errors = [*self.errors[-9:], msg.text]

    def _set_heat(self, percent: int) -> None:
        # Both channels get the same power for now. The finder lens and the secondary dew up
        # differently; give each its own percent if one fogs while the other doesn't.
        self.heat = percent
        for ch in HEATER_CHANNELS:
            self._send(proto.heat(ch, percent))

    def _open(self):
        return self._factory(self._port or default_port(), BAUD, timeout=0.5)

    def _send(self, data: bytes) -> None:
        with self._lock:
            self._ser.write(data)

    def _read_loop(self) -> None:
        while not self._stop.is_set():
            try:
                line = self._ser.readline()
                if line and (msg := proto.parse(line)) is not None:
                    self.handle(msg)  # may write (heaters): same recovery as a failed read
            except (serial.SerialException, OSError):
                if self._stop.is_set():
                    return
                self._reopen()  # unplugged: the old handle is dead, and the device may come
                continue  # back under another name (ttyACM1)

    def _reopen(self) -> None:
        with self._lock:
            try:
                self._ser.close()
            except (serial.SerialException, OSError):
                pass
            while not self._stop.is_set():
                try:
                    self._ser = self._open()
                    return
                except (serial.SerialException, OSError):
                    self._stop.wait(REOPEN_EVERY_S)

    def _ping_loop(self) -> None:
        while not self._stop.wait(PING_EVERY_S):
            try:
                if time.monotonic() - self._last_env > ENV_STALE_S and self.heat:
                    self._set_heat(0)  # no valid readings: don't heat blind
                self._send(proto.frame("PING"))
            except (serial.SerialException, OSError):
                pass
