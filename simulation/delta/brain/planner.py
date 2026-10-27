"""Plan a path -- NumPy on the slide.

A* over the safe navigation map, from where the wearer stands to anywhere in
a ring around the target: you walk up to a fridge, not into it. Then the cell
path is pulled taut into a few straight legs, because guidance announces
corners, and a staircase of 5 cm steps has a corner in every cell.
"""

import heapq
import math
from dataclasses import dataclass, field

import numpy as np

NEIGHBOURS = [
    (-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0),
    (-1, -1, math.sqrt(2)), (-1, 1, math.sqrt(2)),
    (1, -1, math.sqrt(2)), (1, 1, math.sqrt(2)),
]

# Give up rather than freeze the brain on a hopeless search.
MAX_EXPANSIONS = 400_000


@dataclass
class Route:
    points: list                    # world (x, y) corners, start first
    target: tuple                   # what we are walking to
    label: str                      # what to call it out loud
    length: float = 0.0
    unknown: bool = False           # passes through floor nobody has seen
    planned_at: float = 0.0
    extra: dict = field(default_factory=dict)


def plan(navmap, start_xy, target_xy, near=(0.0, 0.3)):
    """Corners of the cheapest walk ending within `near` metres of the target.

    `near` is (closest, furthest): for an object, stop at arm's length; for a
    remembered spot, stand on it. Returns a list of world (x, y), or None.
    """
    blocked = navmap.blocked
    height, width = blocked.shape
    resolution = navmap.resolution

    start = navmap.to_cell(start_xy)

    if start is None:
        return None

    start = _nearest_open(blocked, start, int(round(0.5 / resolution)))

    if start is None:
        return None

    # The ring of acceptable end cells.
    rows, cols = np.mgrid[0:height, 0:width]
    target_col, target_row = (
        (np.asarray(target_xy, dtype=np.float64) - navmap.origin) / resolution
        - np.array([navmap.offset[1], navmap.offset[0]])
    )
    distance = np.hypot(rows - target_row, cols - target_col) * resolution
    goal = (distance >= near[0]) & (distance <= near[1]) & ~blocked

    if not goal.any():
        return None

    goal = goal.ravel()
    cost = navmap.cost.ravel()
    blocked_flat = blocked.ravel()

    def heuristic(row, col):
        # Straight-line metres to the ring, at the cheapest possible cost.
        return max(0.0, math.hypot(row - target_row, col - target_col) * resolution - near[1])

    start_index = start[0] * width + start[1]
    best = np.full(height * width, np.inf, np.float64)
    parent = np.full(height * width, -1, np.int64)
    best[start_index] = 0.0

    queue = [(heuristic(*start), 0.0, start_index)]
    expansions = 0
    reached = -1

    while queue and expansions < MAX_EXPANSIONS:
        _, so_far, index = heapq.heappop(queue)

        if so_far > best[index]:
            continue

        if goal[index]:
            reached = index
            break

        expansions += 1
        row, col = divmod(index, width)

        for dr, dc, step in NEIGHBOURS:
            r, c = row + dr, col + dc

            if not (0 <= r < height and 0 <= c < width):
                continue

            neighbour = r * width + c

            if blocked_flat[neighbour]:
                continue

            # No cutting corners between two blocked cells.
            if dr and dc and (blocked[row, c] or blocked[r, col]):
                continue

            through = so_far + step * resolution * 0.5 * (cost[index] + cost[neighbour])

            if through < best[neighbour]:
                best[neighbour] = through
                parent[neighbour] = index
                heapq.heappush(queue, (through + heuristic(r, c), through, neighbour))

    if reached < 0:
        return None

    cells = []
    index = reached

    while index >= 0:
        cells.append(divmod(int(index), width))
        index = parent[index]

    cells.reverse()

    corners = _pull_taut(blocked, cells)

    return [tuple(navmap.to_world(r, c)) for r, c in corners]


def _nearest_open(blocked, cell, radius):
    """The cell itself if open, else the nearest open one within `radius`."""
    if not blocked[cell]:
        return cell

    rows, cols = np.nonzero(~blocked[
        max(0, cell[0] - radius): cell[0] + radius + 1,
        max(0, cell[1] - radius): cell[1] + radius + 1,
    ])

    if len(rows) == 0:
        return None

    rows = rows + max(0, cell[0] - radius)
    cols = cols + max(0, cell[1] - radius)
    nearest = np.argmin((rows - cell[0]) ** 2 + (cols - cell[1]) ** 2)

    return int(rows[nearest]), int(cols[nearest])


def _clear_line(blocked, a, b):
    """Is the straight segment from cell a to cell b free of blocked cells?"""
    steps = int(max(abs(b[0] - a[0]), abs(b[1] - a[1])) * 2) + 1
    rows = np.round(np.linspace(a[0], b[0], steps)).astype(int)
    cols = np.round(np.linspace(a[1], b[1], steps)).astype(int)

    return not blocked[rows, cols].any()


def _pull_taut(blocked, cells):
    """Keep only the corners: from each, go as far as a straight line can."""
    if len(cells) <= 2:
        return cells

    corners = [cells[0]]
    anchor = 0
    index = 1

    while index < len(cells):
        if not _clear_line(blocked, cells[anchor], cells[index]):
            anchor = index - 1
            corners.append(cells[anchor])

        index += 1

    corners.append(cells[-1])

    return corners


def length(points):
    return sum(math.dist(a, b) for a, b in zip(points, points[1:]))
