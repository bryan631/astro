"""A camera for stream tests: frame pixels encode the exposure (ms) and a frame counter, so a
test can tell which settings a frame was taken with. Importable by a spawned child process."""

import time

import numpy as np


class FakeCamera:
    bayer = "RGGB"
    sensor_size = (64, 48)

    def __init__(self, delay_s: float = 0.01):
        self.exposure_s, self.gain, self.roi, self.count, self.delay_s = 0.01, 0, None, 0, delay_s

    def connect(self) -> None:
        pass

    def close(self) -> None:
        pass

    def set_exposure(self, seconds: float) -> None:
        self.exposure_s = seconds

    def set_gain(self, gain: int) -> None:
        self.gain = gain

    def set_roi(self, roi) -> None:
        self.roi = roi

    def capture(self) -> np.ndarray:
        time.sleep(self.delay_s)
        self.count += 1
        h, w = (self.roi.height, self.roi.width) if self.roi else self.sensor_size[::-1]
        frame = np.full((h, w), round(self.exposure_s * 1000) % 256, np.uint8)
        frame[0, 0] = self.count % 256
        return frame
