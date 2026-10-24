"""Two RGB cameras: what arrives at the brain, from whichever source.

In simulation the frames come over the link from Habitat. On real hardware
they will come from the OAK-FFC-4P, and only this file will know. Either way
the rest of the brain sees the same StereoFrame.

The oracle fields are only ever filled in simulation, and only when asked
for. They let one box at a time be swapped for the truth, so its share of
the error can be measured -- never so that perception can lean on them.
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class StereoFrame:
    id: int
    stamp: float                         # seconds, the source's clock
    left: np.ndarray                     # H x W x 3 uint8, RGB
    right: np.ndarray

    # The IMU's heading, radians anticlockwise, from an arbitrary zero. The
    # OAK-FFC-4P carries one; the sim models it with drift and noise.
    imu_yaw: Optional[float] = None

    # The IMU's tilt from gravity, degrees: + pitch looks down, + roll tilts
    # the head right. None when the sim does not send it.
    imu_pitch: Optional[float] = None
    imu_roll: Optional[float] = None

    # The cage's axle encoder: metres rolled since power-on, a running total
    # (forward up, backward down). None while flying, when the cage does not
    # roll and the reading means nothing.
    odometer: Optional[float] = None

    # ORACLE, simulation only: Habitat's pose of the left lens as
    # (position xyz, quaternion wxyz), and its planar depth in metres.
    true_pose: Optional[tuple] = None
    true_depth: Optional[np.ndarray] = None

    @property
    def size(self):
        """(width, height)."""
        return self.left.shape[1], self.left.shape[0]


def from_message(message):
    """A StereoFrame from a `frame` message sent by the sim."""
    fields = message.fields
    arrays = message.arrays

    true_pose = None

    if fields.get("true_position") is not None:
        true_pose = (fields["true_position"], fields["true_rotation"])

    return StereoFrame(
        id=int(fields["id"]),
        stamp=float(fields["stamp"]),
        left=arrays["left"],
        right=arrays["right"],
        imu_yaw=fields.get("imu_yaw"),
        imu_pitch=fields.get("imu_pitch"),
        imu_roll=fields.get("imu_roll"),
        odometer=fields.get("odometer"),
        true_pose=true_pose,
        true_depth=arrays.get("true_depth"),
    )
