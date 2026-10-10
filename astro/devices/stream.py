"""Always-on cameras: each camera captures continuously; everything else reads its frames.

Before this, one user at a time owned a camera (video, solving, focus, recording), and the
hand-offs froze views and lost settings in the field. Now a camera streams all the time and any
number of readers share it. A real camera runs in its own process: the SVBony SDK segfaulted
with two capture threads in one process, and is fine with one process per camera (2026-10-10,
scripts/dev/two_cams.py). Simulated cameras use a thread with the same interface.

`CameraStream` looks like a `Camera` to its readers: `capture()` waits for the next frame taken
with the current settings, so a reader that changes the exposure never gets an older frame.
"""

import logging
import multiprocessing as mp
import queue
import threading
import time
from multiprocessing import shared_memory

import numpy as np

from astro.devices.base import Roi
from astro.devices.svbony import SvbonyCamera  # no SDK load until a camera connects

log = logging.getLogger(__name__)
MAX_PIXELS = 4096 * 2304  # frame buffer: the largest sensor here (SV705C 3856 x 2180) fits
CAPTURE_WAIT_S = 30.0  # longest exposure plus a camera reopen
RETRY_S = 0.5  # after a failed capture (unplugged: the driver keeps trying to reconnect)
IDLE_S = 10.0  # simulators: stop streaming when nobody has read a frame for this long
# Shared frame header: sequence number, mid-exposure time (Unix), height, width, settings
# generation of that frame, connected flag, newest settings generation applied.
SEQ, T_MID, HEIGHT, WIDTH, GEN, CONNECTED, APPLIED = range(7)
HEADER = 7


class _Slot:
    """The latest frame and its header, shared by the capture loop and the readers."""

    def __init__(self, header, frame_buf: np.ndarray, lock):
        self.header, self.buf, self.lock = header, frame_buf, lock
        self.cond = threading.Condition()  # readers in this process wait here (thread backend)

    def publish(self, frame: np.ndarray, t_mid: float, gen: int) -> None:
        h, w = frame.shape
        with self.lock:
            self.buf[:h * w] = frame.ravel()
            self.header[HEIGHT], self.header[WIDTH] = h, w
            self.header[T_MID], self.header[GEN] = t_mid, gen
            self.header[SEQ] += 1
            self.header[CONNECTED] = 1
        with self.cond:
            self.cond.notify_all()

    def read(self) -> tuple[np.ndarray | None, int, float, int]:
        """(copy of the frame or None, sequence number, mid-exposure time, settings generation)."""
        with self.lock:
            seq, h, w = int(self.header[SEQ]), int(self.header[HEIGHT]), int(self.header[WIDTH])
            frame = self.buf[:h * w].reshape(h, w).copy() if seq else None
            return frame, seq, float(self.header[T_MID]), int(self.header[GEN])


def _capture_loop(camera, slot: _Slot, commands, stop, info) -> None:
    """Capture until stopped, applying setting changes between frames. Each frame carries the
    generation of the newest settings applied before it."""
    gen = int(slot.header[APPLIED])  # a restarted simulator stream continues the count
    while not stop.is_set():
        try:
            while True:  # settings first: a frame published after this used them
                name, value, gen = commands.get_nowait()
                if name == "roi":
                    camera.set_roi(Roi(*value) if value else None)
                else:
                    getattr(camera, f"set_{name}")(value)
                slot.header[APPLIED] = gen
        except queue.Empty:
            pass
        try:
            frame = camera.capture()
        except (RuntimeError, OSError) as e:  # unplugged or a timeout: keep trying
            log.warning("capture failed: %r", e)
            slot.header[CONNECTED] = 0
            stop.wait(RETRY_S)
            continue
        if info is not None and info.get("bayer") is None:
            info.update(bayer=camera.bayer, sensor=tuple(camera.sensor_size))
        slot.publish(frame, time.time() - camera.exposure_s / 2, gen)


def _process_main(factory, args, exposure_s, gain, shm_name, header, lock, commands, stop,
                  info) -> None:
    """The child process: make the camera (`factory(*args)`) and stream it. For SVBony cameras
    the vendor SDK loads only in here."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s camera %(message)s")
    shm = shared_memory.SharedMemory(name=shm_name)
    slot = _Slot(header, np.ndarray((MAX_PIXELS,), np.uint8, shm.buf), lock)
    try:
        camera = factory(*args)
        try:
            camera.connect()
        except Exception:  # unplugged at start: the SVBony driver retries on every capture
            log.exception("camera %s not connected; will keep trying", args)
            camera._lost = True
        camera.set_exposure(exposure_s)
        camera.set_gain(gain)
        _capture_loop(camera, slot, commands, stop, info)
        camera.close()
    finally:
        slot.buf = None  # release the view of the shared memory before closing it
        shm.close()


class CameraStream:
    """A continuously capturing camera, shared by any number of readers (implements Camera)."""

    def __init__(self, exposure_s: float, gain: int, bayer: str | None = None,
                 sensor_size: tuple[int, int] | None = None):
        self.exposure_s, self.gain, self.roi = exposure_s, gain, None
        self._bayer, self._sensor = bayer, sensor_size
        self._gen = 0  # settings generation: bumped by every set_*
        self._mono_minus_unix = time.monotonic() - time.time()  # fixed, so last_at is stable

    # --- Camera interface -------------------------------------------------------------------
    @property
    def bayer(self) -> str:
        info = self._info()
        return info.get("bayer") or self._bayer or "RGGB"

    @property
    def sensor_size(self) -> tuple[int, int]:
        info = self._info()
        return tuple(info["sensor"]) if info.get("sensor") else (self._sensor or (0, 0))

    @property
    def connected(self) -> bool:
        return bool(self._slot.header[CONNECTED])

    def connect(self) -> None:  # streams connect themselves
        pass

    def set_exposure(self, seconds: float) -> None:
        self.exposure_s = seconds
        self._send("exposure", seconds)

    def set_gain(self, gain: int) -> None:
        self.gain = gain
        self._send("gain", gain)

    def set_roi(self, roi: Roi | None) -> None:
        self.roi = roi
        self._send("roi", (roi.x, roi.y, roi.width, roi.height) if roi else None)

    def temperature_c(self) -> float | None:
        return None  # the camera lives in another process; not needed by the readers

    def capture(self) -> np.ndarray:
        """The next frame taken with the current settings (blocks until it arrives)."""
        self._touch()
        want_gen, after = self._gen, self._slot.read()[1]
        deadline = time.monotonic() + CAPTURE_WAIT_S + self.exposure_s
        while time.monotonic() < deadline:
            frame, seq, _, gen = self._slot.read()
            if frame is not None and seq > after and gen >= want_gen:
                return frame
            self._wait(0.02)
        raise RuntimeError("the camera isn't sending frames")

    # --- for the views ---------------------------------------------------------------------
    @property
    def last(self) -> np.ndarray | None:
        self._touch()
        return self._slot.read()[0]

    @property
    def last_at(self) -> float:
        """time.monotonic() at the end of the latest frame's exposure (0 before the first), as
        TappedCamera kept it."""
        self._touch()
        _, seq, t_mid, _ = self._slot.read()
        return t_mid + self.exposure_s / 2 + self._mono_minus_unix if seq else 0.0

    @property
    def seq(self) -> int:
        """Frames published so far (cheap: no copy). Reading it keeps a simulator streaming."""
        self._touch()
        return int(self._slot.header[SEQ])

    def latest(self) -> tuple[np.ndarray | None, int, float]:
        """(frame, sequence number, mid-exposure Unix time) without waiting."""
        self._touch()
        frame, seq, t_mid, _ = self._slot.read()
        return frame, seq, t_mid

    # --- backend hooks ---------------------------------------------------------------------
    def _send(self, name: str, value) -> None:
        self._gen += 1
        self._commands.put((name, value, self._gen))

    def _info(self) -> dict:
        return {}

    def _touch(self) -> None:
        """A reader is here (simulators stream only while someone reads)."""

    def _wait(self, seconds: float) -> None:
        time.sleep(seconds)


class ThreadStream(CameraStream):
    """Streams an in-process camera (the simulators) from a thread. Simulators return frames
    instantly, so the thread paces itself to the exposure, and it sleeps while nobody reads."""

    def __init__(self, camera):
        super().__init__(camera.exposure_s, camera.gain, camera.bayer, tuple(camera.sensor_size))
        self.camera = camera
        self._commands = queue.Queue()
        self._slot = _Slot([0.0] * HEADER, np.zeros(MAX_PIXELS, np.uint8), threading.Lock())
        self._stop = threading.Event()
        self._read_at = time.monotonic()
        self._thread: threading.Thread | None = None
        self._start_lock = threading.Lock()

    @property
    def bayer(self) -> str:
        return self.camera.bayer

    @property
    def sensor_size(self) -> tuple[int, int]:
        return tuple(self.camera.sensor_size)

    def _touch(self) -> None:
        self._read_at = time.monotonic()
        with self._start_lock:
            if not self._stop.is_set() and (self._thread is None or not self._thread.is_alive()):
                self._thread = threading.Thread(target=self._run, daemon=True)
                self._thread.start()

    def _run(self) -> None:
        _capture_loop(_Paced(self.camera, self._stop), self._slot, self._commands,
                      _UntilIdle(self), None)

    def _wait(self, seconds: float) -> None:
        with self._slot.cond:
            self._slot.cond.wait(seconds)

    def close(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5)
        self.camera.close()

    def __getattr__(self, name):  # simulator extras
        return getattr(self.__dict__["camera"], name)


class _Paced:
    """A simulator that takes as long as its exposure, like a real camera."""

    def __init__(self, camera, stop: threading.Event):
        self.camera, self.stop = camera, stop

    def capture(self) -> np.ndarray:
        start = time.monotonic()
        frame = self.camera.capture()
        self.stop.wait(max(0.0, self.camera.exposure_s - (time.monotonic() - start)))
        return frame

    def __getattr__(self, name):  # exposure_s, set_exposure, set_roi, bayer, ...
        return getattr(self.camera, name)


class _UntilIdle:
    """The stop signal for a simulator stream: closed, or nobody has read for IDLE_S."""

    def __init__(self, stream: ThreadStream):
        self.stream = stream

    def is_set(self) -> bool:
        return self.stream._stop.is_set() or time.monotonic() - self.stream._read_at > IDLE_S

    def wait(self, seconds: float) -> None:
        self.stream._stop.wait(seconds)


class ProcessStream(CameraStream):
    """Streams a camera from its own process: `factory(*args)` runs in the child (SvbonyCamera
    and its model name on the Mele; a fake camera in the tests)."""

    def __init__(self, exposure_s: float, gain: int, factory=SvbonyCamera, args: tuple = ()):
        super().__init__(exposure_s, gain)
        ctx = mp.get_context("spawn")  # a fresh interpreter: no SDK or thread state inherited
        self._shm = shared_memory.SharedMemory(create=True, size=MAX_PIXELS)
        header, lock = ctx.Array("d", HEADER, lock=False), ctx.Lock()
        self._manager_info = ctx.Manager()  # bayer and sensor size, once the camera opens
        self._info_dict = self._manager_info.dict()
        self._slot = _Slot(header, np.ndarray((MAX_PIXELS,), np.uint8, self._shm.buf), lock)
        self._commands, self._stop = ctx.Queue(), ctx.Event()
        self._proc = ctx.Process(target=_process_main, daemon=True, name=f"camera-{args}",
                                 args=(factory, args, exposure_s, gain, self._shm.name, header,
                                       lock, self._commands, self._stop, self._info_dict))
        self._proc.start()
        self._cached_info: dict = {}

    def _info(self) -> dict:
        if not self._cached_info.get("bayer"):
            try:
                self._cached_info = dict(self._info_dict)
            except (OSError, EOFError):  # manager gone (shutting down)
                pass
        return self._cached_info

    @property
    def alive(self) -> bool:
        return self._proc.is_alive()

    def close(self) -> None:
        if self._slot.buf is None:  # closed already (two owners may both close it)
            return
        self._stop.set()
        self._proc.join(timeout=5)
        if self._proc.is_alive():
            self._proc.kill()
        self._slot.buf = None
        self._shm.close()
        self._shm.unlink()
        self._manager_info.shutdown()
