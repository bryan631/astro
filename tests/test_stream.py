import threading

import pytest

from astro.devices import stream
from astro.devices.base import Roi
from astro.devices.stream import ProcessStream, ThreadStream
from tests.fake_camera import FakeCamera


@pytest.fixture
def thread_stream():
    s = ThreadStream(FakeCamera())
    yield s
    s.close()


def test_capture_gets_new_frames(thread_stream):
    a, b = thread_stream.capture(), thread_stream.capture()
    assert a[0, 0] != b[0, 0]


def test_frames_after_a_setting_change_use_it(thread_stream):
    thread_stream.capture()
    thread_stream.set_exposure(0.042)
    assert thread_stream.capture()[1, 1] == 42  # never a frame from before the change
    thread_stream.set_roi(Roi(0, 0, 16, 8))
    assert thread_stream.capture().shape == (8, 16)


def test_many_readers_share_the_camera(thread_stream):
    got = []
    readers = [threading.Thread(target=lambda: got.append(thread_stream.capture())) for _ in range(4)]
    for r in readers:
        r.start()
    for r in readers:
        r.join(timeout=5)
    assert len(got) == 4


def test_last_at_is_stable_between_frames(thread_stream):
    thread_stream.set_exposure(0.3)  # slow frames: two reads see the same one
    thread_stream.capture()
    assert thread_stream.last_at == thread_stream.last_at


def test_simulator_stream_sleeps_without_readers_and_wakes(monkeypatch, thread_stream):
    monkeypatch.setattr(stream, "IDLE_S", 0.1)
    thread_stream.capture()
    thread_stream._thread.join(timeout=2)  # nobody reads: the thread ends
    assert not thread_stream._thread.is_alive()
    thread_stream.set_exposure(0.007)
    assert thread_stream.capture()[1, 1] == 7  # reading starts it again, settings kept


def test_process_stream():
    s = ProcessStream(0.01, 0, factory=FakeCamera, args=(0.01,))
    try:
        s.capture()
        s.set_exposure(0.05)
        assert s.capture()[1, 1] == 50
        assert s.connected and s.bayer == "RGGB" and s.sensor_size == (64, 48)
        frame, seq, t_mid = s.latest()
        assert frame is not None and seq > 0 and t_mid > 0
    finally:
        s.close()
    assert not s.alive
