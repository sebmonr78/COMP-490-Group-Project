"""A pretend IMU chip. WORLD SIDE: it reads Habitat's true pose.

Ported unchanged from the glasses project -- an IMU does not care what it is
bolted to. It is the drone's fallback when the cameras cannot see: a blank
wall, a dark corner, or a frame with too few corners to track.

That is allowed for the same reason the renderer may read the scene: this
file plays the sensor, not the brain. The brain only ever sees its output,
which is the true motion turned into what a chip would feel, plus noise and
a slowly wandering bias.

What it outputs, per frame:
    gyro    rotation speed around the camera's x, y, z axes, rad/s
    accel   acceleration the chip feels, m/s^2, INCLUDING gravity -- a chip
            sitting still reads 9.81 pointing up

Axes are the left camera's OpenCV axes (x right, y down, z forward); the
chip is assumed to sit exactly on the left camera. A real IMU runs at
100-400 Hz; this one reads once per rendered frame, which is the honest
limit of driving it from a renderer.
"""

from collections import deque
from dataclasses import dataclass

import cv2
import numpy as np


GRAVITY_WORLD = np.array([0.0, -9.81, 0.0])     # Habitat: +y is up


@dataclass(frozen=True)
class ImuReading:
    t: float
    dt: float
    gyro: np.ndarray
    accel: np.ndarray = None


class FakeIMU:
    def __init__(self, config):
        c = config
        self.rng = np.random.default_rng(c["seed"])
        self.c = c
        self.gyro_bias = self.rng.normal(0.0, c["gyro_bias_start"], 3)
        self.accel_bias = self.rng.normal(0.0, c["accel_bias_start"], 3)
        self._poses = deque(maxlen=3)

    def read(self, t, rotation, position):
        """Feed the true pose (world <- camera, OpenCV camera axes). Returns a
        reading once it has two poses to compare, else None."""
        self._poses.append((t, np.asarray(rotation, float), np.asarray(position, float)))

        if len(self._poses) < 2:
            return None

        (t1, r1, p1), (t2, r2, p2) = self._poses[-2], self._poses[-1]
        dt = t2 - t1

        self._drift(dt)

        turn, _ = cv2.Rodrigues(r1.T @ r2)
        gyro = turn.ravel() / dt + self.gyro_bias + self.rng.normal(0.0, self.c["gyro_noise"], 3)

        accel = None

        if len(self._poses) == 3:
            t0, _, p0 = self._poses[0]
            v1 = (p1 - p0) / (t1 - t0)
            v2 = (p2 - p1) / dt
            felt = (v2 - v1) / ((t2 - t0) / 2.0) - GRAVITY_WORLD

            accel = r2.T @ felt + self.accel_bias + self.rng.normal(0.0, self.c["accel_noise"], 3)

        return ImuReading(t2, dt, gyro, accel)

    def _drift(self, dt):
        root = np.sqrt(max(dt, 0.0))
        self.gyro_bias += self.rng.normal(0.0, self.c["gyro_bias_walk"] * root, 3)
        self.accel_bias += self.rng.normal(0.0, self.c["accel_bias_walk"] * root, 3)
