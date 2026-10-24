"""Build occupancy grid -- NumPy on the slide.

A top-down grid of the home, 5 cm cells, built only from what the glasses
have seen. Each cell holds the log-odds that it is blocked. Evidence adds up
over many frames, so one noisy frame cannot invent a wall, and walking
through a doorway clears it.

What one frame says, as a Scan:

    hits    floor-plan points of everything 12 cm to 1.9 m above the floor:
            anything a 1.70 m person could walk into.
    free    the floor that was actually SEEN, out to the first thing in the
            way, in each direction.

"Seen" is the careful part. A camera 1.74 m up with a 94 degree vertical
field cannot see the floor within about 1.6 m of the wearer's feet. A ray to
the far wall passes over a footstool in that band without touching it, so
free space must start where the floor was visible, never at the lens, or
every step forward would erase the stool from the map. The wearer's own
footprint is the one place known free without being seen.

Scans are kept, not just added in. RTAB-Map corrects its past when a loop
closes, and `rebuild` redraws the grid from every kept scan at its corrected
pose, instead of leaving two copies of each wall.
"""

import math
from dataclasses import dataclass

import cv2
import numpy as np

from delta import frames

UNKNOWN, FREE, OCCUPIED = 0, 1, 2

# What the roof layer answers: whether a cell is under cover.
UNROOFED, ROOFED = 1, 2

# How far obstacles are mapped from a frame whose scale was borrowed.
BORROWED_RANGE = 1.0


@dataclass
class Scan:
    """One view, in the base frame of the left lens (x forward, y left)."""

    anchor: int
    anchor_from_base: np.ndarray    # 4x4
    free: np.ndarray                # P x 2 polygon, metres
    hits: np.ndarray                # H x 2 points, metres
    footprint: np.ndarray           # (x, y, radius) of the robot's body

    # What this view saw overhead: 1 a ceiling, 0 open sky, -1 no opinion.
    # The floor this view could see is under whatever is above the camera,
    # so one answer covers the whole free polygon.
    roofed: int = -1

    def world_from_base(self, node_poses):
        """Where the view was taken from, as the map currently has it."""
        if self.anchor not in node_poses:
            return None

        return node_poses[self.anchor] @ self.anchor_from_base


def make_scan(depth, K, lens_height, settings, footprint, anchor=0,
              anchor_from_base=None, step=2, free_space=True, tilt=None,
              footprint_free=True, roofed=-1):
    """Turn one metric depth map into a Scan.

    `depth` is planar depth in metres (already scale-corrected) and K its
    intrinsics. `tilt` is the head's tilt from the IMU
    (frames.level_from_tilted); None means level, when optical y down is
    straight down and heights need no rotation. `free_space=False`
    keeps only the obstacles: for a frame whose scale was borrowed rather
    than measured, nothing should be declared clear.

    `footprint_free=False` also drops the one cell the robot is standing in.
    For the glasses it was always true -- wherever the wearer is standing is,
    evidently, floor a person fits on, and that is the only free space you get
    for nothing. A FLYING drone is the exception: it is not standing on
    anything, and there may be nothing under it at all. Marking the cell below
    it free would quietly carve a floor across open stairwells.
    """
    height, width = depth.shape

    v, u = np.mgrid[0:height:step, 0:width:step].astype(np.float32)
    d = depth[::step, ::step]

    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]

    forward = d
    left = -(u + 0.5 - cx) / fx * d
    up = -(v + 0.5 - cy) / fy * d

    if tilt is not None:
        # Looking down or tilted: put every point back into the level frame.
        forward, left, up = (tilt[i, 0] * forward + tilt[i, 1] * left + tilt[i, 2] * up for i in range(3))

    above_floor = lens_height + up

    horizontal = np.hypot(forward, left)
    seen = (d > 0.1) & (horizontal < float(settings.max_range))

    floor_band = float(settings.floor_band)
    low, high = settings.obstacle_band

    floor = seen & (np.abs(above_floor) < floor_band)
    obstacle = seen & (above_floor > low) & (above_floor < high)

    # Per direction: free from the nearest floor seen to whichever comes
    # first, the first obstacle or the furthest floor seen.
    half_fov = math.atan((width / 2.0) / fx)
    bins = 90
    edges = np.linspace(-half_fov, half_fov, bins + 1)
    angle = np.arctan2(left, forward)

    def per_bin(mask, reduce, initial):
        out = np.full(bins, initial, dtype=np.float32)
        index = np.clip(np.searchsorted(edges, angle[mask]) - 1, 0, bins - 1)
        reduce.at(out, index, horizontal[mask])
        return out

    nearest_floor = per_bin(floor, np.minimum, np.inf)
    furthest_floor = per_bin(floor, np.maximum, 0.0)
    nearest_obstacle = per_bin(obstacle, np.minimum, np.inf)

    resolution = float(settings.resolution)

    outer = np.minimum(furthest_floor, nearest_obstacle - resolution)
    inner = nearest_floor
    has_free = np.isfinite(inner) & (outer > inner)

    centres = 0.5 * (edges[:-1] + edges[1:])
    inner = np.where(has_free, inner, 0.0)
    outer = np.where(has_free, outer, 0.0)

    # An annulus-shaped polygon: out along the near edge, back along the far.
    # Directions with nothing seen collapse to zero width.
    near_edge = np.stack([inner * np.cos(centres), inner * np.sin(centres)], axis=1)
    far_edge = np.stack([outer * np.cos(centres), outer * np.sin(centres)], axis=1)
    polygon = np.concatenate([near_edge, far_edge[::-1]]).astype(np.float32)

    if not free_space:
        polygon = np.zeros((0, 2), np.float32)

    # A borrowed scale is trusted only up close: 15% of 30 cm is nothing,
    # 15% of 3 m puts a sofa where the wearer is standing.
    if not free_space:
        obstacle &= horizontal < BORROWED_RANGE

    hits = np.stack([forward[obstacle], left[obstacle]], axis=1)
    hits = np.unique(np.round(hits / resolution).astype(np.int32), axis=0)
    hits = (hits * resolution).astype(np.float32)

    body = np.asarray(footprint, dtype=np.float32).copy()

    if not footprint_free:
        body[2] = 0.0

    return Scan(
        anchor=anchor,
        anchor_from_base=np.eye(4) if anchor_from_base is None else anchor_from_base,
        free=polygon,
        hits=hits,
        footprint=body,
        roofed=int(roofed),
    )


class OccupancyGrid:
    """A square grid of log-odds, centred where the brain started."""

    def __init__(self, settings, centre_xy):
        self.settings = settings
        self.resolution = float(settings.resolution)
        cells = int(round(float(settings.size) / self.resolution))

        self.logodds = np.zeros((cells, cells), np.float32)

        # A second layer of evidence, in the same cells: is this spot under
        # cover? Log-odds for the same reason the occupancy is -- one frame
        # that glimpsed a doorframe must not turn a garden into a room, and
        # the robot driving out through a door must be able to change the
        # answer back.
        self.roof = np.zeros((cells, cells), np.float32)
        self.origin = np.array(centre_xy, dtype=np.float64) - cells * self.resolution / 2.0

        self.hit = float(settings.hit)
        self.miss = float(settings.miss)
        self.limits = tuple(settings.limits)
        self.occupied_above = float(settings.occupied_above)
        self.free_below = float(settings.free_below)

        roof = settings.get("roof", {}) if hasattr(settings, "get") else {}
        self.roof_hit = float(roof.get("hit", 0.4))
        self.roof_miss = float(roof.get("miss", -0.4))
        self.roof_limits = tuple(roof.get("limits", (-2.5, 2.5)))
        self.roofed_above = float(roof.get("roofed_above", 0.8))
        self.unroofed_below = float(roof.get("unroofed_below", -0.8))

        self.scans = []
        self.version = 0

        spacing = settings.keyframe_spacing
        self._spacing = (float(spacing[0]), math.radians(float(spacing[1])))
        self._last_kept = None

    # -----------------------------------------------------------------
    # World and cells
    # -----------------------------------------------------------------

    def to_cells(self, xy):
        """World x, y (metres) to fractional (column, row)."""
        return (np.asarray(xy, dtype=np.float64) - self.origin) / self.resolution

    def to_world(self, cols_rows):
        return np.asarray(cols_rows, dtype=np.float64) * self.resolution + self.origin

    # -----------------------------------------------------------------
    # Adding evidence
    # -----------------------------------------------------------------

    def add(self, scan, world_from_base):
        """Add one scan's evidence at a pose, and keep it if it is a new view."""
        self._integrate(scan, world_from_base)

        x, y, yaw = frames.planar(world_from_base)

        if (
            self._last_kept is None
            or math.hypot(x - self._last_kept[0], y - self._last_kept[1]) >= self._spacing[0]
            or abs(frames.wrap(yaw - self._last_kept[2])) >= self._spacing[1]
        ):
            self.scans.append(scan)
            self._last_kept = (x, y, yaw)

        self.version += 1

    def rebuild(self, node_poses):
        """Redraw everything from the kept scans, at their current poses."""
        self.logodds[:] = 0.0
        self.roof[:] = 0.0

        for scan in self.scans:
            pose = scan.world_from_base(node_poses)

            if pose is not None:
                self._integrate(scan, pose)

        self.version += 1

    def _integrate(self, scan, world_from_base):
        x, y, yaw = frames.planar(world_from_base)
        cos, sin = math.cos(yaw), math.sin(yaw)

        def to_world(points):
            points = np.asarray(points, dtype=np.float64).reshape(-1, 2)

            return np.stack(
                [x + cos * points[:, 0] - sin * points[:, 1],
                 y + sin * points[:, 0] + cos * points[:, 1]],
                axis=1,
            )

        # Free: the seen floor, plus the footprint the wearer stands on.
        # Drawn in a window around the scan only -- the whole grid is a
        # million cells, and a rebuild draws every kept scan again.
        polygon = np.round(self.to_cells(to_world(scan.free))).astype(np.int32)
        body = np.round(self.to_cells(to_world(scan.footprint[:2]))[0]).astype(np.int32)
        radius = int(round(scan.footprint[2] / self.resolution))

        corners = np.vstack([polygon, body - radius, body + radius])
        rows, cols = self.logodds.shape
        c0, r0 = np.clip(corners.min(axis=0) - 1, 0, [cols, rows])
        c1, r1 = np.clip(corners.max(axis=0) + 2, 0, [cols, rows])

        if c1 > c0 and r1 > r0:
            free = np.zeros((r1 - r0, c1 - c0), np.uint8)
            offset = np.array([c0, r0])

            if len(polygon) >= 3:
                cv2.fillPoly(free, [polygon - offset], 1)

            if radius > 0:
                cv2.circle(free, tuple(int(v) for v in body - offset), radius, 1, -1)

            seen = free.astype(bool)

            window = self.logodds[r0:r1, c0:c1]
            window[seen] += self.miss
            np.clip(window, *self.limits, out=window)

            # The floor this view could see lies under whatever the view saw
            # overhead, so the same area gets the roof vote. A view with no
            # opinion -- too few upward pixels to tell -- votes for nothing,
            # which is different from voting for open sky.
            if scan.roofed >= 0:
                roof = self.roof[r0:r1, c0:c1]
                roof[seen] += self.roof_hit if scan.roofed else self.roof_miss
                np.clip(roof, *self.roof_limits, out=roof)

        # Hits after free, so a wall on the edge of the free area stays a wall.
        if len(scan.hits):
            cells = np.round(self.to_cells(to_world(scan.hits))).astype(np.int64)
            rows, cols = cells[:, 1], cells[:, 0]
            inside = (rows >= 0) & (rows < self.logodds.shape[0]) & (cols >= 0) & (cols < self.logodds.shape[1])
            flat = np.unique(rows[inside] * self.logodds.shape[1] + cols[inside])
            self.logodds.flat[flat] = np.minimum(
                self.logodds.flat[flat] + self.hit, self.limits[1]
            )

    # -----------------------------------------------------------------
    # Reading it
    # -----------------------------------------------------------------

    def classes(self):
        """UNKNOWN, FREE or OCCUPIED for every cell."""
        out = np.full(self.logodds.shape, UNKNOWN, np.uint8)
        out[self.logodds < self.free_below] = FREE
        out[self.logodds > self.occupied_above] = OCCUPIED
        return out

    def roof_classes(self):
        """UNKNOWN, UNROOFED or ROOFED for every cell.

        This is the house-versus-garden answer, and the drone worked it out
        for itself: nothing here came from the simulator.
        """
        out = np.full(self.roof.shape, UNKNOWN, np.uint8)
        out[self.roof < self.unroofed_below] = UNROOFED
        out[self.roof > self.roofed_above] = ROOFED
        return out

    def known_bounds(self, margin=1.0):
        """(row0, row1, col0, col1) around every cell with any evidence."""
        rows, cols = np.nonzero(self.logodds != 0.0)

        if len(rows) == 0:
            return None

        pad = int(round(margin / self.resolution))
        shape = self.logodds.shape

        return (
            max(0, rows.min() - pad), min(shape[0], rows.max() + pad + 1),
            max(0, cols.min() - pad), min(shape[1], cols.max() + pad + 1),
        )

    def is_occupied(self, xy):
        col, row = np.round(self.to_cells(xy)).astype(int)

        if not (0 <= row < self.logodds.shape[0] and 0 <= col < self.logodds.shape[1]):
            return False

        return bool(self.logodds[row, col] > self.occupied_above)

    # -----------------------------------------------------------------
    # Saving
    # -----------------------------------------------------------------

    def save(self, path):
        """Scans and grid to one .npz: polygons and hits flattened, with
        offsets, so no pickle is needed to read it back."""
        def flatten(arrays):
            lengths = np.array([len(a) for a in arrays], dtype=np.int64)
            data = np.concatenate(arrays) if arrays else np.zeros((0, 2), np.float32)
            return data, lengths

        free, free_lengths = flatten([s.free for s in self.scans])
        hits, hit_lengths = flatten([s.hits for s in self.scans])

        np.savez_compressed(
            path,
            logodds=self.logodds,
            roof=self.roof,
            roofed=np.array([s.roofed for s in self.scans], dtype=np.int64),
            origin=self.origin,
            resolution=self.resolution,
            anchors=np.array([s.anchor for s in self.scans], dtype=np.int64),
            anchor_from_base=np.array([s.anchor_from_base for s in self.scans]).reshape(-1, 4, 4),
            footprints=np.array([s.footprint for s in self.scans]).reshape(-1, 3),
            free=free, free_lengths=free_lengths,
            hits=hits, hit_lengths=hit_lengths,
        )

    def load(self, path):
        data = np.load(path)

        if data["logodds"].shape != self.logodds.shape or float(data["resolution"]) != self.resolution:
            raise ValueError("Saved grid has a different size or resolution than config/brain.json")

        self.logodds[:] = data["logodds"]
        self.origin = data["origin"]

        # Maps saved before the roof layer existed simply have no opinion.
        if "roof" in data:
            self.roof[:] = data["roof"]
        else:
            self.roof[:] = 0.0

        free = np.split(data["free"], np.cumsum(data["free_lengths"])[:-1])
        hits = np.split(data["hits"], np.cumsum(data["hit_lengths"])[:-1])

        roofed = (
            data["roofed"] if "roofed" in data
            else np.full(len(data["anchors"]), -1, dtype=np.int64)
        )

        self.scans = [
            Scan(int(anchor), pose, f, h, footprint, int(roof))
            for anchor, pose, f, h, footprint, roof in zip(
                data["anchors"], data["anchor_from_base"], free, hits,
                data["footprints"], roofed,
            )
        ]

        self.version += 1
