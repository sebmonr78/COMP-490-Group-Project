"""Coordinate frames, and the only place that converts between them.

Four frames meet in this project, and every bug of the "turns the wrong way"
kind is two of them being confused. So each has a one-letter name, used in
every variable that holds a pose: `world_from_base` means "takes a point in
the base frame and gives it in the world frame".

    H  Habitat world     x east, y up, z south. -Z is north on the layout map.
    W  brain world       x east, y north, z up. The map is the x-y plane.
                         RTAB-Map's map frame has the same axes (z up); only
                         its origin and heading differ, set where it started.
    C  camera optical    x right, y down, z forward. OpenCV, DA3 and YOLOE.
    B  base              x forward, y left, z up. ROS and RTAB-Map's body
                         frame, sitting on the left lens.

Numpy only, so the sim (Python 3.9) and the brain (Python 3.10) share it.
"""

import math

import numpy as np


# Habitat world to brain world: same east, Habitat's -z is north, y is up.
W_FROM_H = np.array(
    [
        [1.0, 0.0, 0.0],
        [0.0, 0.0, -1.0],
        [0.0, 1.0, 0.0],
    ]
)

# Optical to base: the camera looks along base x; image right is base -y;
# image down is base -z.
B_FROM_C = np.array(
    [
        [0.0, 0.0, 1.0],
        [-1.0, 0.0, 0.0],
        [0.0, -1.0, 0.0],
    ]
)

# Habitat's camera frame (x right, y up, z back) to optical: flip y and z.
C_FROM_HABITAT_CAMERA = np.diag([1.0, -1.0, -1.0])


def transform(rotation, translation):
    """A 4x4 rigid transform from a 3x3 rotation and a 3-vector."""
    matrix = np.eye(4)
    matrix[:3, :3] = rotation
    matrix[:3, 3] = translation

    return matrix


def invert(matrix):
    """Inverse of a rigid 4x4 transform, without a general matrix inverse."""
    rotation = matrix[:3, :3]

    return transform(rotation.T, -rotation.T @ matrix[:3, 3])


def quaternion_matrix(w, x, y, z):
    """Rotation matrix of a unit quaternion given as w, x, y, z."""
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ]
    )


def world_from_habitat_camera(position, quaternion_wxyz):
    """Habitat's pose of a camera sensor, as world_from_base.

    `position` and `quaternion_wxyz` are what Habitat reports for a sensor:
    where it is in the Habitat world, and how it is turned. The answer is the
    same camera as a pose in the brain's world, with base axes.
    """
    habitat_from_camera = quaternion_matrix(*quaternion_wxyz)

    world_from_optical = W_FROM_H @ habitat_from_camera @ C_FROM_HABITAT_CAMERA

    return transform(
        world_from_optical @ B_FROM_C.T,
        W_FROM_H @ np.asarray(position, dtype=float),
    )


def world_from_optical(world_from_base):
    """The same camera pose with optical axes, for projecting pixels."""
    return world_from_base @ transform(B_FROM_C, np.zeros(3))


def habitat_point(world_point):
    """A brain-world point back in Habitat coordinates."""
    return W_FROM_H.T @ np.asarray(world_point, dtype=float)


def world_point(habitat_point_):
    """A Habitat point in brain-world coordinates."""
    return W_FROM_H @ np.asarray(habitat_point_, dtype=float)


# ---------------------------------------------------------------------
# On the floor plane
# ---------------------------------------------------------------------


def planar(world_from_base):
    """(x, y, yaw) of a pose, on the floor. Yaw is anticlockwise from +x.

    The wearer is upright, so position on the floor and the direction the
    face points are all that guidance needs.
    """
    forward = world_from_base[:3, 0]

    return (
        float(world_from_base[0, 3]),
        float(world_from_base[1, 3]),
        math.atan2(forward[1], forward[0]),
    )


def from_planar(x, y, yaw, z=0.0):
    """A level pose from (x, y, yaw): the inverse of `planar`."""
    cos, sin = math.cos(yaw), math.sin(yaw)

    return transform(
        np.array(
            [
                [cos, -sin, 0.0],
                [sin, cos, 0.0],
                [0.0, 0.0, 1.0],
            ]
        ),
        [x, y, z],
    )


def wrap(angle):
    """An angle in radians, folded into (-pi, pi]."""
    return math.atan2(math.sin(angle), math.cos(angle))


def bearing(from_xy, to_xy):
    """Direction from one floor point to another, anticlockwise from +x."""
    return math.atan2(to_xy[1] - from_xy[1], to_xy[0] - from_xy[0])


def clock(relative):
    """A relative angle as a clock position: 12 ahead, 3 right, 9 left.

    Clock positions are how orientation and mobility instructors describe
    direction to blind travellers, and they need no sense of north.
    `relative` is radians anticlockwise from straight ahead, so positive is
    to the LEFT.
    """
    hour = int(round(-math.degrees(relative) / 30.0)) % 12

    return 12 if hour == 0 else hour


def heading_name(yaw):
    """Compass word for a yaw: 0 is east, pi/2 north."""
    names = [
        "east", "north-east", "north", "north-west",
        "west", "south-west", "south", "south-east",
    ]

    return names[int(round(math.degrees(yaw) / 45.0)) % 8]


# ---------------------------------------------------------------------
# Head tilt, from the IMU's gravity reading
# ---------------------------------------------------------------------


def level_from_tilted(pitch_degrees=0.0, roll_degrees=0.0):
    """3x3 rotation taking base-frame vectors of a TILTED head into the
    level body frame (x forward, y left, z up), heading unchanged.

    pitch: + looks down (rotation about base +y, left).
    roll:  + tilts the head to the right (rotation about base +x, forward).

    The IMU on the frame measures both from gravity; with them, a depth
    point seen by a camera looking down at the floor is put back where it
    is, instead of where a level camera would have seen it.
    """
    p, r = math.radians(pitch_degrees or 0.0), math.radians(roll_degrees or 0.0)
    cp, sp, cr, sr = math.cos(p), math.sin(p), math.cos(r), math.sin(r)

    about_left = np.array([[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]])
    about_forward = np.array([[1.0, 0.0, 0.0], [0.0, cr, -sr], [0.0, sr, cr]])

    return about_left @ about_forward


def level(world_from_base):
    """The same pose with pitch and roll taken out: heading and position only.
    For poses that carry the head's tilt (the oracle's), so that the IMU's
    tilt is applied exactly once."""
    x, y, yaw = planar(world_from_base)
    c, s = math.cos(yaw), math.sin(yaw)
    out = np.eye(4)
    out[:3, :3] = [[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]]
    out[:3, 3] = world_from_base[:3, 3]

    return out
