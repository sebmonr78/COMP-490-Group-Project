"""Rooms: the robot-vacuum map -- the floor split into rooms at the doors.

The method is the distance-transform segmentation from the survey that
compared the standard ones (Bormann et al., "Room segmentation: Survey,
implementation, and analysis", ICRA 2016):

    1. the free floor's distance to the nearest wall, for every cell
    2. shrink the floor by more and more (keep only cells further than t
       from a wall): rooms come apart at their doors, because a doorway is
       narrow and a room is not. The shrink that gives the MOST separate
       pieces is the one that split every door without erasing rooms
    3. those pieces are the rooms' seeds; grow each back over the floor,
       one ring at a time, until every cell belongs to the nearest seed

Two things the survey's clean floor plans did not need, and a robot's map
does:

    The floor plan comes from WALL-HIGH things only (occupancy.walls, about
    0.9 to 1.8 m up). Seen from 15 cm, a sofa, a table or a bed is as solid
    as a wall, and a living room comes out as four rooms around its
    furniture. Measured as a floor plan, furniture is just floor.

    Pieces are merged afterwards. Two pieces sharing an opening wider than
    a door are one open-plan room -- a kitchen island makes two pieces of
    one kitchen otherwise -- and a piece smaller than a room joins its
    neighbour. What is left between rooms are the doors.

Indoors or out is decided per room, from two pieces of evidence:

    roofed    the share of its floor seen under a ceiling (occupancy.roof)
    enclosed  the share of its outer edge that is wall rather than unexplored

The ceiling decides; the walls are only the fallback when no ceiling has
been seen. Measured, not assumed: the default house's open-plan living room
was 100% roofed but only 46% enclosed, because its long glass wall is
invisible to depth -- the cameras see straight through it -- and the rest
of its edge had not been explored yet. Walls alone would have put it
outdoors. So:

    mostly roofed                    indoor     (a room; a walled garden is not)
    mostly open sky                  outdoor    (a garden, walled or not)
    some of each                     mixed      (a patio and the lawn it opens onto)
    no ceiling seen, walls all round indoor
    no ceiling seen, no walls seen   unsure

Nothing here reads the simulator: it is the brain's own map, cut up.
"""

from dataclasses import dataclass, field

import cv2
import numpy as np

from delta.brain.occupancy import ROOFED, UNKNOWN, UNROOFED

# How finely the shrink is swept, and from where, in metres.
SWEEP_STEP = 0.05
SWEEP_FROM = 0.25

# A seed smaller than this, m2, is a gap between pieces of furniture, not
# the core of a room.
MIN_SEED = 0.25


@dataclass
class Room:
    id: int
    cells: int
    area: float                      # m2
    point: tuple                     # (col, row): the cell furthest from its walls
    roofed: float = None             # share of the floor seen under a ceiling
    enclosed: float = None           # share of the edge that is wall
    kind: str = "unsure"             # indoor, outdoor, mixed, unsure


@dataclass
class Door:
    rooms: tuple                     # (room id, room id)
    point: tuple                     # (col, row): middle of the opening
    width: float                     # metres


@dataclass
class Rooms:
    labels: np.ndarray               # room id per cell, 0 for none
    rooms: list = field(default_factory=list)
    doors: list = field(default_factory=list)
    shrink: float = 0.0              # metres: the shrink that split them


def floor_plan(classes, walls):
    """Where a room can be: anything seen that is not wall-high.

    Floor seen free and furniture seen solid at rolling height both count;
    only walls (and things as tall) separate. Holes a little bigger than a
    chair -- the inside of a sofa base nobody can see into -- are filled, as
    long as the filling does not cross a wall.
    """
    space = (classes != UNKNOWN) & ~walls
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    closed = cv2.morphologyEx(space.astype(np.uint8), cv2.MORPH_CLOSE, kernel).astype(bool)

    return closed & ~walls


def segment(classes, walls, roof, resolution, settings):
    """Split the brain's map into rooms. All arrays are the same window.

    `classes` is occupancy.classes(), `walls` occupancy.wall_cells(), `roof`
    occupancy.roof_classes(). Returns a Rooms.
    """
    space = floor_plan(classes, walls)
    labels = np.zeros(space.shape, np.uint16)

    if space.sum() * resolution ** 2 < float(settings.min_room):
        return Rooms(labels)

    distance = cv2.distanceTransform(space.astype(np.uint8), cv2.DIST_L2, 5) * resolution
    shrink, seeds = _best_seeds(distance, resolution, settings)

    if seeds.max() == 0:
        return Rooms(labels, shrink=shrink)

    labels = _grow(seeds, space, distance)
    labels = _merge(labels, resolution, settings)

    rooms = _describe(labels, distance, walls, classes, roof, resolution, settings)
    doors = _doors(labels, resolution)

    return Rooms(labels, rooms, doors, shrink)


def _pieces(distance, t, min_cells):
    """The room-sized pieces of the floor shrunk by t: (labels, ids)."""
    count, components, stats, _ = cv2.connectedComponentsWithStats(
        (distance > t).astype(np.uint8), connectivity=4
    )

    return components, [i for i in range(1, count) if stats[i, cv2.CC_STAT_AREA] >= min_cells]


def _best_seeds(distance, resolution, settings):
    """Seeds: the pieces of the shrink that leaves the most of them, plus any
    narrower room that a smaller shrink found and that one erased.

    The best shrink for wide rooms is too much for narrow ones: in the
    default house it was 0.95 m, which split the open-plan space and wiped
    out a 1.8 m wide room completely -- that room was a piece of its own at
    0.5 m. So after the best shrink, smaller ones are looked at in turn, and
    a piece holding no seed yet becomes one.
    """
    min_cells = MIN_SEED / resolution ** 2
    highest = float(settings.max_room_width) / 2.0
    sweep = np.arange(SWEEP_FROM, highest + 1e-9, SWEEP_STEP)

    # Strictly more: on a tie the smaller shrink wins, which keeps more of
    # every room in its seed.
    counts = [len(_pieces(distance, t, min_cells)[1]) for t in sweep]
    best = int(np.argmax(counts))

    seeds = np.zeros(distance.shape, np.uint16)
    next_id = 1

    for t in sweep[best::-1]:
        components, ids = _pieces(distance, t, min_cells)

        for i in ids:
            piece = components == i

            if not seeds[piece].any():
                seeds[piece] = next_id
                next_id += 1

    return float(sweep[best]), seeds


def _grow(seeds, space, distance):
    """Grow every seed over the floor so that rooms meet at their doors.

    A watershed: the floor is flooded from the seeds in order of distance to
    the walls, deepest first, so two rooms' floods meet at the shallowest
    point between them -- the doorway. Grown ring by ring instead, they met
    wherever they happened to be equally far from both seeds, often in the
    middle of the bigger room, and the long border that made looked like an
    open-plan opening and merged them.
    """
    labels = seeds.copy()

    if seeds.max() == 0:
        return labels

    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))

    # Level by level, from the middle of the rooms down to their walls: at
    # each level every room claims the floor at least that deep next to what
    # it has, until nothing more joins. (OpenCV's cv2.watershed was tried
    # first: it floods by the image's EDGES, not its values, and on a smooth
    # distance map that let one room pour through a door into the next.)
    for level in np.arange(float(distance.max()), -SWEEP_STEP, -SWEEP_STEP):
        claimable = space & (distance >= level) & (labels == 0)

        while claimable.any():
            grown = cv2.dilate(labels, kernel)
            claim = claimable & (grown > 0)

            if not claim.any():
                break

            labels[claim] = grown[claim]
            claimable &= ~claim

    return labels


def _borders(labels):
    """{(a, b): cells of a that touch b, with a < b} -- the shared edges."""
    shared = {}

    for shifted, original in (
        (labels[1:, :], labels[:-1, :]),
        (labels[:, 1:], labels[:, :-1]),
    ):
        touching = (shifted != original) & (shifted > 0) & (original > 0)

        for a, b in zip(original[touching], shifted[touching]):
            key = (int(min(a, b)), int(max(a, b)))
            shared[key] = shared.get(key, 0) + 1

    return shared


def _merge(labels, resolution, settings):
    """Join open-plan pieces and too-small pieces, then number 1..n by size."""
    door_cells = float(settings.door_width) / resolution
    min_cells = float(settings.min_room) / resolution ** 2

    while True:
        ids, sizes = np.unique(labels[labels > 0], return_counts=True)
        size = dict(zip(ids.tolist(), sizes.tolist()))
        shared = _borders(labels)

        # An opening wider than any door: one room, not two.
        wide = [(n, pair) for pair, n in shared.items() if n > door_cells]

        if wide:
            _, (a, b) = max(wide)
            labels[labels == b] = a
            continue

        # The smallest piece too small to be a room joins the neighbour it
        # shares the most edge with.
        small = sorted((size[i], i) for i in size if size[i] < min_cells)
        merged = False

        for _, piece in small:
            neighbours = [(n, a if b == piece else b) for (a, b), n in shared.items() if piece in (a, b)]

            if neighbours:
                _, into = max(neighbours)
                labels[labels == piece] = into
                merged = True
                break

        if not merged:
            break

    # Biggest room first: room 1 is the main one.
    ids, sizes = np.unique(labels[labels > 0], return_counts=True)
    renumbered = np.zeros_like(labels)

    for n, i in enumerate(ids[np.argsort(-sizes)], start=1):
        renumbered[labels == i] = n

    return renumbered


def _describe(labels, distance, walls, classes, roof, resolution, settings):
    rooms = []
    outside = ~(labels > 0)
    wall_or_unknown = walls | (classes == UNKNOWN)

    for i in range(1, int(labels.max()) + 1):
        mine = labels == i
        cells = int(mine.sum())

        if cells == 0:
            continue

        # A point well inside it, for its label: the furthest from its walls.
        inside = np.where(mine, distance, -1.0)
        row, col = np.unravel_index(int(np.argmax(inside)), inside.shape)

        # Under a ceiling?
        known = mine & (roof != UNKNOWN)
        roofed = float((mine & (roof == ROOFED)).sum() / known.sum()) if known.sum() > 0.2 * cells else None

        # Its edge: cells just outside it. Wall, or unexplored? (Other
        # rooms, through doors, are neither and do not count.) Only the
        # OUTER edge: a pocket nobody has seen into inside the room -- under
        # a sofa, behind the island -- is not a way out of it, and counting
        # those called a whole kitchen and living room "covered".
        filled = _fill_holes(mine)
        ring = (cv2.dilate(filled.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
                & ~filled & outside & wall_or_unknown)
        wall_edge = int((ring & walls).sum())
        open_edge = int((ring & ~walls).sum())
        enclosed = wall_edge / max(1, wall_edge + open_edge)

        rooms.append(Room(
            id=i, cells=cells, area=cells * resolution ** 2, point=(int(col), int(row)),
            roofed=roofed, enclosed=enclosed,
            kind=_kind(roofed, enclosed, settings),
        ))

    return rooms


def _fill_holes(mask):
    """The mask with every hole inside it filled: its outer outline."""
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    filled = np.zeros(mask.shape, np.uint8)
    cv2.drawContours(filled, contours, -1, 1, thickness=cv2.FILLED)

    return filled.astype(bool)


def _kind(roofed, enclosed, settings):
    """indoor, outdoor, mixed or unsure -- see the module's docstring."""
    indoor, walled = float(settings.indoor), float(settings.enclosed)

    if roofed is None:
        return "indoor" if enclosed >= walled else "unsure"

    if roofed >= indoor:
        return "indoor"

    if roofed <= 1.0 - indoor:
        return "outdoor"

    return "mixed"


def _doors(labels, resolution):
    """One door per pair of rooms that touch: the middle of the opening."""
    doors = []

    for (a, b), _ in _borders(labels).items():
        here = ((labels[1:, :] == a) & (labels[:-1, :] == b)) | ((labels[1:, :] == b) & (labels[:-1, :] == a))
        rows, cols = np.nonzero(here)
        across = ((labels[:, 1:] == a) & (labels[:, :-1] == b)) | ((labels[:, 1:] == b) & (labels[:, :-1] == a))
        rows2, cols2 = np.nonzero(across)

        rows = np.concatenate([rows, rows2])
        cols = np.concatenate([cols, cols2])

        if len(rows) == 0:
            continue

        # The width is the opening's long side, not its cell count: a
        # diagonal doorway touches more cells than it is wide.
        width = (max(np.ptp(rows), np.ptp(cols)) + 1) * resolution

        doors.append(Door(rooms=(a, b), point=(float(cols.mean()), float(rows.mean())), width=float(width)))

    return doors
