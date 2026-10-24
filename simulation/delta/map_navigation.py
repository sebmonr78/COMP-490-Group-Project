"""Planning a route through the map the drone built itself.

`navigation.py` plans on Habitat's navmesh. That navmesh is the answer key:
the simulator hands over the exact shape of every floor and doorway in the
house, including the rooms the drone has never been in. It is the right tool
for checking our work and the wrong tool for doing it, because a real drone
in a real warehouse has no navmesh. This file is the same job done honestly
-- a route through nothing but what the drone has seen for itself.

Three differences follow from that, and all three are the point:

    it can be wrong          the map has holes, so a plan can fail or lead
                             somewhere that turns out to be blocked
    it knows about height    the navmesh is a floor; this is a stack of
                             layers, so a route can go OVER and UNDER things
    it chooses how to move   rolling is cheap, flying is expensive, and the
                             search prefers rolling because of the numbers in
                             config/map.json, not because of a rule

That last one is feature 4 of the proposal -- locomotion-mode selection --
and it falls out of A* rather than being decided anywhere. Give flying a
higher cost per metre than rolling and the cheapest route is a rolling route
wherever a rolling route exists. Flight appears exactly where the proposal
says it should: stairs, gaps, blocked paths, and things worth going over.

The search is A* over (row, column, level): a floor square and a height.
Horizontal moves stay in a layer, vertical moves change layer by one. A node
is only usable if the drone's whole body fits in free space there, which is
`GapMap.passable` -- so the search can never route through a layer that was
merely never seen.
"""

import heapq
import math

import numpy as np


# The eight horizontal moves, with what each costs in grid squares.
STEPS = [
    (-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0),
    (-1, -1, math.sqrt(2.0)), (-1, 1, math.sqrt(2.0)),
    (1, -1, math.sqrt(2.0)), (1, 1, math.sqrt(2.0)),
]

DEFAULT_COSTS = {
    "roll": 1.0,
    "fly": 4.0,
    "climb": 6.0,
    "_comment": "Per metre. Rolling is the unit; flying costs four times as "
                "much ground covered, and changing height costs six, which is "
                "what makes the drone commit to a climb rather than porpoise.",
}

# Refuse rather than hang. An unreachable goal in a large map would otherwise
# expand every node the drone has ever seen before admitting defeat.
MAX_EXPANSIONS = 400_000


class Plan:
    """A route through the map, and how to travel each part of it.

    Attributes:
        nodes    [(row, column, level), ...] in grid indices
        points   [(across, along, height), ...] in map metres
        modes    "roll" or "fly" for each point
        cost     what the search paid, in rolling-metre equivalents
        length   how far it actually is, in metres
    """

    def __init__(self, nodes, points, modes, cost, length):
        self.nodes = nodes
        self.points = points
        self.modes = modes
        self.cost = cost
        self.length = length

    def __len__(self):
        return len(self.points)

    @property
    def flies(self):
        return any(mode == "fly" for mode in self.modes)

    def legs(self):
        """The route as stretches of one mode: [(mode, [points...]), ...].

        What a route is actually made of, from the drone's point of view: a
        roll to the sofa, a flight over it, a roll to the far wall. Three
        instructions rather than four hundred.
        """
        if not self.points:
            return []

        out = [(self.modes[0], [self.points[0]])]

        for mode, point in zip(self.modes[1:], self.points[1:]):
            if mode == out[-1][0]:
                out[-1][1].append(point)
            else:
                out.append((mode, [point]))

        return out

    def describe(self):
        if not self.points:
            return "no route"

        parts = [
            f"{mode} {_leg_length(points):.1f} m"
            for mode, points in self.legs()
        ]

        return f"{self.length:.1f} m: " + " then ".join(parts)


def _leg_length(points):
    total = 0.0

    for a, b in zip(points, points[1:]):
        total += math.dist(a, b)

    return total


class MapPlanner:
    """Plans on one snapshot of the map.

    Built from a GapMap and then reused: the expensive part is working out
    which layers the drone fits in, and that does not change until the map
    does. Rebuild it after more mapping -- `is_stale` says when.
    """

    def __init__(self, gap_map, diameter, costs=None):
        self.map = gap_map
        self.diameter = float(diameter)
        self.costs = dict(DEFAULT_COSTS)
        self.costs.update(costs or {})

        self.built_at_frame = gap_map.frames

        grid = gap_map.states()

        self.fits = gap_map.passable(self.diameter, grid)
        self.ground = gap_map.ground_level(grid)
        self.roll_level = np.where(self.ground >= 0, self.ground + 1, -1)

    @property
    def is_stale(self):
        return self.map.frames != self.built_at_frame

    # -----------------------------------------------------------------
    # Asking about one spot
    # -----------------------------------------------------------------

    def is_rolling(self, row, column, level):
        """Is the drone sitting on the ground at this node, or flying?"""
        return level == self.roll_level[row, column]

    def usable(self, row, column, level):
        n, levels = self.map.n, self.map.levels

        if not (0 <= row < n and 0 <= column < n and 0 <= level < levels):
            return False

        return bool(self.fits[row, column, level])

    def snap(self, across, along, prefer_rolling=True):
        """The nearest usable node to a point on the floor, or None.

        A click lands on a square, not on a layer, and the square may be the
        middle of a sofa. Search outwards for somewhere the drone could
        actually be, preferring the ground: a destination you can roll to is
        worth more than one you would have to hover at.
        """
        row, column = self.map.to_grid(across, along)

        best = None

        for radius in range(0, int(self.map.g.get("snap_radius", 12)) + 1):
            for dr in range(-radius, radius + 1):
                for dc in range(-radius, radius + 1):
                    # Only the edge of each ring, so nearer rings win.
                    if radius > 0 and max(abs(dr), abs(dc)) != radius:
                        continue

                    r, c = row + dr, column + dc

                    if not (0 <= r < self.map.n and 0 <= c < self.map.n):
                        continue

                    if prefer_rolling:
                        level = self.roll_level[r, c]

                        if level >= 0 and self.usable(r, c, level):
                            return (r, c, int(level))

                    if best is None:
                        column_fits = np.nonzero(self.fits[r, c])[0]

                        if len(column_fits):
                            best = (r, c, int(column_fits[0]))

            if best is not None and not prefer_rolling:
                return best

        return best

    # -----------------------------------------------------------------
    # The search
    # -----------------------------------------------------------------

    def _move_cost(self, a, b):
        """What one step costs, in rolling-metre equivalents."""
        row_a, col_a, level_a = a
        row_b, col_b, level_b = b

        if level_a != level_b:
            return self.costs["climb"] * self.map.layer

        squares = math.dist((row_a, col_a), (row_b, col_b))
        metres = squares * self.map.cell

        # A step counts as rolling only if BOTH ends are on the ground. A
        # step from the floor onto thin air is the start of a flight, and
        # pricing it as a roll would let the search sneak across gaps.
        rolling = self.is_rolling(*a) and self.is_rolling(*b)

        return metres * self.costs["roll" if rolling else "fly"]

    def _heuristic(self, node, goal):
        """Straight-line distance at the cheapest possible price per metre.

        Admissible because no move anywhere costs less per metre than
        `roll`, so this can never overestimate what is left to pay.
        """
        dr = (node[0] - goal[0]) * self.map.cell
        dc = (node[1] - goal[1]) * self.map.cell
        dl = (node[2] - goal[2]) * self.map.layer

        return math.sqrt(dr * dr + dc * dc + dl * dl) * self.costs["roll"]

    def plan(self, start, goal, max_expansions=MAX_EXPANSIONS):
        """A* from one node to another. Returns a Plan, or None.

        `start` and `goal` are (row, column, level). None means no route
        exists through what has been mapped -- which is real information, and
        different from "there is no route": the honest answer is "not through
        anything I have seen", and the cure is to go and look.
        """
        if not (self.usable(*start) and self.usable(*goal)):
            return None

        frontier = [(self._heuristic(start, goal), 0.0, start)]
        came_from = {start: None}
        paid = {start: 0.0}
        expansions = 0

        while frontier:
            _, spent, node = heapq.heappop(frontier)

            if node == goal:
                return self._build(came_from, goal, spent)

            if spent > paid.get(node, math.inf):
                continue

            expansions += 1

            if expansions > max_expansions:
                return None

            for neighbour in self._neighbours(node):
                cost = spent + self._move_cost(node, neighbour)

                if cost < paid.get(neighbour, math.inf):
                    paid[neighbour] = cost
                    came_from[neighbour] = node

                    heapq.heappush(
                        frontier,
                        (cost + self._heuristic(neighbour, goal), cost, neighbour),
                    )

        return None

    def _neighbours(self, node):
        row, column, level = node

        for dr, dc, _ in STEPS:
            r, c = row + dr, column + dc

            if not self.usable(r, c, level):
                continue

            # No cutting corners diagonally through a doorframe: a diagonal
            # step is only allowed if both squares beside it are clear too.
            if dr and dc:
                if not (
                    self.usable(row, c, level)
                    and self.usable(r, column, level)
                ):
                    continue

            yield (r, c, level)

        for dl in (-1, 1):
            if self.usable(row, column, level + dl):
                yield (row, column, level + dl)

    def _build(self, came_from, goal, cost):
        nodes = []
        node = goal

        while node is not None:
            nodes.append(node)
            node = came_from[node]

        nodes.reverse()

        points = []
        modes = []

        for row, column, level in nodes:
            across, along, height = self.map.to_world(row, column, level)

            points.append((float(across), float(along), float(height)))
            modes.append("roll" if self.is_rolling(row, column, level) else "fly")

        length = _leg_length(points)

        return Plan(nodes, points, modes, float(cost), float(length))

    # -----------------------------------------------------------------
    # The convenient version
    # -----------------------------------------------------------------

    def plan_between(self, start_xy, goal_xy):
        """Plan between two points on the floor, snapping both. Plan or None."""
        start = self.snap(*start_xy)
        goal = self.snap(*goal_xy)

        if start is None or goal is None:
            return None

        return self.plan(start, goal)

    def frontier_cells(self):
        """Mapped squares that touch unmapped space: where to go to learn more.

        Not used by the planner. This is the hook for the proposal's
        exploration feature -- "work out which areas have not been explored
        and move toward them" is this list, plus a plan to the nearest one.
        """
        grid = self.map.states()

        seen = (grid != 0).any(axis=2)
        roll = self.map.rollable(self.diameter, grid)

        padded = np.zeros((self.map.n + 2, self.map.n + 2), bool)
        padded[1:-1, 1:-1] = seen

        unseen_beside = (
            ~padded[:-2, 1:-1] | ~padded[2:, 1:-1]
            | ~padded[1:-1, :-2] | ~padded[1:-1, 2:]
        )

        return roll & unseen_beside
