"""Rectify and undistort -- OpenCV on the slide.

Turns the raw pair into an ideal one: no lens distortion, and rows lined up
so that a point on row 300 of the left image is on row 300 of the right one.
Every box after this assumes exactly that. SGBM searches along rows. RTAB-Map
is handed one focal length and a baseline. DA3 is told the right camera is
the left one moved sideways.

In simulation the pair already is ideal -- Habitat renders perfect pinholes,
mounted parallel -- so the maps come out as identity and this costs nothing.
It runs anyway, for the same reason `rig.undistort` exists: when checkerboard
numbers land in config/rig.json, nothing downstream has to change.
"""

import math
from dataclasses import dataclass

import cv2
import numpy as np

from delta import rig as rig_module


@dataclass
class RectifiedPair:
    left: np.ndarray        # H x W x 3 uint8, RGB
    right: np.ndarray
    K: np.ndarray           # 3x3, shared by both images after rectification
    baseline: float         # metres, left lens to right lens


def _habitat_rotation(camera):
    """A camera's rotation in the visor frame, built the way Habitat does:
    Rz * Ry * Rx, from the orientation the rig hands to Habitat."""
    pitch, yaw, roll = camera.orientation

    def about(axis, angle):
        c, s = math.cos(angle), math.sin(angle)

        if axis == "x":
            return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])

        if axis == "y":
            return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])

        return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])

    return about("z", roll) @ about("y", yaw) @ about("x", pitch)


class Rectifier:
    """Built once for a rig and an image size; then called on every pair."""

    def __init__(self, rig, size):
        width, height = size
        left_name, right_name = rig.stereo_pair
        left, right = rig.camera(left_name), rig.camera(right_name)

        K_left = rig_module.intrinsics_for(width, height, left.hfov)
        K_right = rig_module.intrinsics_for(width, height, right.hfov)

        D_left = np.asarray(rig.distortion.get(left_name, [0.0] * 5), dtype=float)
        D_right = np.asarray(rig.distortion.get(right_name, [0.0] * 5), dtype=float)

        # Where the right camera sits relative to the left, in OpenCV's
        # convention: p_right = R p_left + T, with optical axes. Habitat's
        # camera axes differ from optical ones by flipping y and z.
        flip = np.diag([1.0, -1.0, -1.0])

        R_left = _habitat_rotation(left)
        R_right = _habitat_rotation(right)

        R = flip @ R_right.T @ R_left @ flip
        T = flip @ R_right.T @ (np.array(left.position) - np.array(right.position))

        self.size = size
        self.identity = (
            not np.any(D_left)
            and not np.any(D_right)
            and np.allclose(R, np.eye(3))
            and np.allclose(K_left, K_right)
            and abs(T[1]) < 1e-9
            and abs(T[2]) < 1e-9
        )

        if self.identity:
            # Already ideal: nothing to remap, and the baseline is simply how
            # far apart the lenses are.
            self.K = K_left
            self.baseline = float(abs(T[0]))
            self.maps = None
            return

        R1, R2, P1, P2, _, _, _ = cv2.stereoRectify(
            K_left, D_left, K_right, D_right, (width, height), R, T,
            flags=cv2.CALIB_ZERO_DISPARITY, alpha=0,
        )

        self.maps = (
            cv2.initUndistortRectifyMap(K_left, D_left, R1, P1, (width, height), cv2.CV_16SC2),
            cv2.initUndistortRectifyMap(K_right, D_right, R2, P2, (width, height), cv2.CV_16SC2),
        )

        self.K = P1[:3, :3]
        self.baseline = float(-P2[0, 3] / P2[0, 0])

    def __call__(self, frame):
        if self.identity:
            return RectifiedPair(frame.left, frame.right, self.K, self.baseline)

        (left_x, left_y), (right_x, right_y) = self.maps

        return RectifiedPair(
            cv2.remap(frame.left, left_x, left_y, cv2.INTER_LINEAR),
            cv2.remap(frame.right, right_x, right_y, cv2.INTER_LINEAR),
            self.K,
            self.baseline,
        )
