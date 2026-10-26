"""The window's colours."""

import numpy as np

KIND_COLOURS = {
    "reply": (110, 210, 255),
    "system": (135, 140, 150),
}

FREE_COLOUR = np.array([78, 92, 110], np.uint8)
OCCUPIED_COLOUR = np.array([235, 232, 220], np.uint8)

# Something there, above rolling height -- a table top, a bed -- with floor
# the ball can roll on underneath: occupied, but not to it.
ROLL_UNDER_COLOUR = np.array([150, 140, 175], np.uint8)
TRUTH_FLOOR = np.array([34, 40, 50], np.uint8)
TRUTH_EDGE = np.array([70, 78, 94], np.uint8)

DRONE_TRUE = (80, 220, 110)
DRONE_BELIEVED = (255, 150, 50)
ROUTE_COLOUR = (80, 150, 255)
TARGET_COLOUR = (255, 220, 80)
THING_COLOUR = (215, 120, 255)
PLACE_COLOUR = (255, 220, 80)

# The brain's rooms: one tint each, cycled, and the doors between them.
ROOM_COLOURS = [
    (230, 120, 110), (110, 190, 120), (110, 150, 235), (230, 190, 90), (190, 120, 220),
    (90, 200, 200), (235, 150, 60), (160, 200, 90), (220, 110, 170), (140, 140, 240),
]
DOOR_COLOUR = (255, 235, 120)
