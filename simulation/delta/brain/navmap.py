"""Create safe 2D navigation map -- OpenCV and NumPy on the slide.

The occupancy grid says what is solid. This says where a person can go, and
how comfortably:

    blocked     solid things grown by the wearer's half-width plus a margin,
                so a route is a line the middle of the body can follow.
    cost        rises close to anything solid, so routes keep to the middle
                of a corridor rather than skimming the wall.
    unknown     floor nobody has seen yet. Allowed only near seen floor, and
                expensive: guiding a blind person through unseen space is the
                last resort, not the shortcut.

Built on demand from the grid, for the part of it that holds anything.
"""

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class NavMap:
    blocked: np.ndarray       # bool, the window
    cost: np.ndarray          # float32, >= 1, per metre
    clearance: np.ndarray     # metres to the nearest solid cell
    offset: tuple             # (row0, col0) of the window in the grid
    origin: np.ndarray        # world x, y of grid cell (0, 0)
    resolution: float

    def to_cell(self, xy):
        """World x, y to (row, col) in this window, or None if outside."""
        col, row = np.round(
            (np.asarray(xy, dtype=np.float64) - self.origin) / self.resolution
        ).astype(int) - np.array([self.offset[1], self.offset[0]])

        if 0 <= row < self.blocked.shape[0] and 0 <= col < self.blocked.shape[1]:
            return int(row), int(col)

        return None

    def to_world(self, row, col):
        return (
            np.array([col + self.offset[1], row + self.offset[0]], dtype=np.float64)
            * self.resolution
            + self.origin
        )

    def is_blocked(self, xy):
        cell = self.to_cell(xy)
        return cell is None or bool(self.blocked[cell])


def build(grid, settings, wearer_xy=None, wearer_radius=0.3):
    """The navigation map for everything the grid knows, or None if empty."""
    fringe = float(settings.unknown_fringe)
    bounds = grid.known_bounds(margin=fringe + 0.5)

    if bounds is None:
        return None

    r0, r1, c0, c1 = bounds
    logodds = grid.logodds[r0:r1, c0:c1]
    resolution = grid.resolution

    occupied = logodds > grid.occupied_above
    free = logodds < grid.free_below
    unknown = ~occupied & ~free

    # Distance to the nearest solid cell, and to the nearest seen floor.
    # distanceTransform measures to the nearest ZERO pixel.
    clearance = cv2.distanceTransform((~occupied).astype(np.uint8), cv2.DIST_L2, 5) * resolution
    to_free = cv2.distanceTransform((~free).astype(np.uint8), cv2.DIST_L2, 5) * resolution

    margin = float(settings.clearance)
    comfort = float(settings.comfort)

    blocked = clearance < margin
    blocked |= unknown & (to_free > fringe)

    closeness = np.clip((comfort - clearance) / max(comfort - margin, 1e-6), 0.0, 1.0)
    cost = (1.0 + 3.0 * closeness).astype(np.float32)
    cost[unknown] *= float(settings.unknown_cost)

    navmap = NavMap(blocked, cost, clearance, (r0, c0), grid.origin, resolution)

    # Wherever the wearer is standing is, evidently, somewhere a person fits.
    if wearer_xy is not None:
        cell = navmap.to_cell(wearer_xy)

        if cell is not None:
            stand = np.zeros(blocked.shape, np.uint8)
            cv2.circle(stand, (cell[1], cell[0]), int(round(wearer_radius / resolution)), 1, -1)
            blocked &= ~stand.astype(bool)

    return navmap
