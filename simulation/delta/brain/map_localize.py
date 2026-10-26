"""Map-aware position: check RTAB-Map's pose against the map already built.

RTAB-Map tracks how the cameras moved. When it drifts, nothing compares
the result with the map: the wearer can be believed to stand inside a
wall, and new scans are drawn a little off, so walls come out doubled.

This takes each depth scan the glasses make, tries small shifts of the
pose -- up to +-30 cm and +-3 degrees -- and keeps the one where the walls
being seen land on the walls already mapped (a likelihood field: each
point scores by its distance to the nearest mapped obstacle). Two rules
make it careful:

  - the body can never be inside something solid: a pose with the body
    in an occupied cell is refused, and one that has drifted into a wall
    is pushed out to the best pose that is not;
  - it moves only part of the way each scan, only when the better fit is
    clear, and never piles up more than a metre: a wrong match fades
    rather than throws the wearer across the room.

The correction is a small rigid transform applied on top of RTAB-Map's
pose for what is about the WEARER: guidance, the beacon, where they are
drawn. The map itself is still drawn at RTAB-Map's pose: drawn at the
corrected one, the map followed each correction and the next scan agreed
with it -- the first version wandered a metre that way. With no clear
evidence the correction fades back to nothing. A loop closure -- RTAB-Map
fixing its own drift -- resets it.
"""

import math

import cv2
import numpy as np

from delta import frames

SHIFT, SHIFT_STEP = 0.30, 0.05                   # metres
TURN, TURN_STEP = math.radians(3.0), math.radians(1.0)
SIGMA = 0.15                                     # metres: how close counts as on a wall
BODY_CLEAR = 0.12                                # metres from the nearest solid cell
MIN_POINTS, SAMPLE = 60, 200
BETTER_BY = 0.04                                 # score gain needed to move
GOOD_FIT = 0.5                                   # a scan scoring this well confirms the pose
STEP = 0.3                                       # fraction of the found shift taken per scan
FADE = 0.03                                      # of the correction undone per scan without evidence
NEAR = 3.5                                       # metres: only points this close (depth is good there)
PRIOR = 0.12                                     # cost of the largest shift / turn
LIMIT = 1.0                                      # metres of total correction before giving up
FIELD_EVERY = 8                                  # grid versions between field rebuilds


def se2(dx, dy, dtheta):
    c, s = math.cos(dtheta), math.sin(dtheta)
    return np.array([[c, -s, dx], [s, c, dy], [0.0, 0.0, 1.0]])


class MapCorrector:
    def __init__(self, body_of):
        """`body_of(lens_pose)` -> ((x, y), yaw): where the body is."""
        self.body_of = body_of
        self.C = np.eye(3)                       # corrected_from_rtabmap, on the floor
        self._field = None
        self._field_version = -10**9
        self.stats = {"corrections": 0, "pushed_out": 0, "resets": 0,
                      "shift_m": 0.0, "turn_deg": 0.0, "score": 0.0,
                      "checked": 0, "raw_in_solid": 0, "in_solid": 0}

    # -----------------------------------------------------------------

    def apply(self, pose):
        """RTAB-Map's world_from_base, corrected."""
        if pose is None:
            return None

        lift = np.eye(4)
        lift[:2, :2] = self.C[:2, :2]
        lift[:2, 3] = self.C[:2, 2]

        return lift @ pose

    def reset(self):
        self.C = np.eye(3)
        self.stats["resets"] += 1
        self._sync_stats()

    # -----------------------------------------------------------------

    def update(self, grid, hits, raw_pose):
        """Fit one scan (`hits`: N x 2 floor-plan points, forward/left of the
        lens) to the grid, starting from the current correction."""
        if grid is None or hits is None or len(hits) < MIN_POINTS:
            return

        field = self._distance_field(grid)

        if field is None:
            return

        pose = self.apply(raw_pose)
        x, y, yaw = frames.planar(pose)

        hits = hits[np.hypot(hits[:, 0], hits[:, 1]) < NEAR]

        if len(hits) < MIN_POINTS:
            return

        # The complaint this answers: is the body inside something solid on
        # the glasses' own map? Counted before and after the correction.
        (rbx, rby), _ = self.body_of(raw_pose)
        (cbx, cby), _ = self.body_of(pose)
        self.stats["checked"] += 1
        self.stats["raw_in_solid"] += int(self._lookup(field, rbx, rby) < BODY_CLEAR)

        if len(hits) > SAMPLE:
            hits = hits[np.random.default_rng(len(hits)).choice(len(hits), SAMPLE, replace=False)]

        turns = np.arange(-TURN, TURN + 1e-9, TURN_STEP)
        shifts = np.arange(-SHIFT, SHIFT + 1e-9, SHIFT_STEP)
        dx, dy = np.meshgrid(shifts, shifts, indexing="ij")
        dx, dy = dx.ravel(), dy.ravel()

        best = None
        base_score, base_ok = 0.0, True
        (bx, by), _ = self.body_of(pose)

        for turn in turns:
            heading = yaw + turn
            c, s = math.cos(heading), math.sin(heading)
            px = x + c * hits[:, 0] - s * hits[:, 1]
            py = y + s * hits[:, 0] + c * hits[:, 1]

            # Candidates x points: how far each seen point is from a mapped wall.
            d = self._lookup(field, px[None, :] + dx[:, None], py[None, :] + dy[:, None])
            score = np.exp(-(d ** 2) / (2.0 * SIGMA ** 2)).mean(axis=1)

            # The body turns about the lens too.
            ox, oy = bx - x, by - y
            rbx = x + math.cos(turn) * ox - math.sin(turn) * oy
            rby = y + math.sin(turn) * ox + math.cos(turn) * oy
            body_clear = self._lookup(field, rbx + dx, rby + dy) >= BODY_CLEAR

            prior = PRIOR * ((np.hypot(dx, dy) / SHIFT) ** 2 + (turn / TURN) ** 2)
            total = np.where(body_clear, score - prior, -np.inf)

            if abs(turn) < 1e-9:
                zero = int(np.argmin(np.hypot(dx, dy)))
                base_score, base_ok = float(score[zero]), bool(body_clear[zero])

            i = int(np.argmax(total))

            if np.isfinite(total[i]) and (best is None or total[i] > best[0]):
                best = (float(total[i]), float(dx[i]), float(dy[i]), float(turn), float(score[i]))

        self.stats["score"] = round(base_score, 3)

        if best is None:
            return

        _, sx, sy, sturn, sscore = best
        pushed = not base_ok
        better = sscore - base_score > BETTER_BY

        if not (pushed or better):
            # Nothing better found. If the walls seen do not line up with the
            # map here either, there is no evidence for the correction: let it
            # fade towards none. If they do, it is right: keep it.
            if base_score < GOOD_FIT:
                self.C = se2(0.0, 0.0, 0.0) * FADE + self.C * (1.0 - FADE)
            self._sync_stats()
            self.stats["in_solid"] += int(self._lookup(field, cbx, cby) < BODY_CLEAR)
            return

        fraction = 1.0 if pushed else STEP
        # Turn about the lens point (x, y), then shift.
        step = se2(x + fraction * sx, y + fraction * sy, fraction * sturn) @ se2(-x, -y, 0.0)
        C = step @ self.C

        if math.hypot(C[0, 2], C[1, 2]) > LIMIT:
            self.reset()
            return

        self.C = C
        self.stats["corrections"] += 1
        self.stats["pushed_out"] += int(pushed)
        self._sync_stats()
        (nbx, nby), _ = self.body_of(self.apply(raw_pose))
        self.stats["in_solid"] += int(self._lookup(field, nbx, nby) < BODY_CLEAR)

    # -----------------------------------------------------------------

    def _sync_stats(self):
        self.stats["shift_m"] = round(math.hypot(self.C[0, 2], self.C[1, 2]), 3)
        self.stats["turn_deg"] = round(math.degrees(math.atan2(self.C[1, 0], self.C[0, 0])), 2)

    def _distance_field(self, grid):
        """Metres from each cell to the nearest confidently solid cell."""
        if self._field is None or self._field[1] is not grid or grid.version - self._field_version >= FIELD_EVERY:
            solid = grid.logodds > grid.occupied_above

            if solid.sum() < 50:
                return None

            dist = cv2.distanceTransform((~solid).astype(np.uint8), cv2.DIST_L2, 3)
            self._field = (dist * grid.resolution, grid)
            self._field_version = grid.version

        return self._field

    @staticmethod
    def _lookup(field, xs, ys):
        dist, grid = field
        xs, ys = np.broadcast_arrays(xs, ys)
        cells = grid.to_cells(np.stack([xs.ravel(), ys.ravel()], axis=1))
        cols = np.round(cells[:, 0]).astype(np.int64)
        rows = np.round(cells[:, 1]).astype(np.int64)
        inside = (rows >= 0) & (rows < dist.shape[0]) & (cols >= 0) & (cols < dist.shape[1])
        out = np.full(len(cols), 1.0, np.float32)            # off the grid: far from anything
        out[inside] = dist[rows[inside], cols[inside]]

        return out.reshape(xs.shape)
