"""The replica: the place in 3D and in colour, as the drone has seen it.

Every depth pixel is a ray from the camera to a surface. The end of it is a
point on that surface, and the camera saw its colour. Each kept view
(occupancy keeps one every 0.3 m or so) gives a few tens of thousands of
such points; they are binned into 5 cm cubes (voxels), and a cube's colour
is the average of everything seen in it. A cube must be seen more than once
before it is shown, so a single bad depth pixel does not leave a speck.

That is how a mapping drone builds its live 3D map (Flyability's Elios 3
does it from LiDAR; this does it from depth), and how RTAB-Map's own clouds
are made. Two things carried over from the 2D map:

    Views hang off RTAB-Map nodes. When a loop closure corrects the past,
    the cubes are rebuilt from every kept view at its corrected place --
    re-integration, as BundleFusion does -- instead of leaving two copies
    of each wall.

    Heights are above the floor, as everywhere else in the brain: the floor
    is z = 0, whatever the drone's altitude when it looked.

The cubes are kept in a dict from a packed (x, y, z) index to a slot in a
few growing arrays, so adding a view costs only the cubes it touches.

Nothing here reads the simulator: depth, colour and position are the
brain's own.
"""

import math
import threading
from dataclasses import dataclass

import numpy as np

from delta import frames

# Packing three signed cube indices into one int64: 21 bits each, so up to a
# million cubes either way -- 50 km at 5 cm.
BITS = 21
OFFSET = 1 << (BITS - 1)
MASK = (1 << BITS) - 1


@dataclass
class View:
    """One kept view: surface points in the level base frame of the left
    lens (forward, left, height above the floor), their colours, and the
    RTAB-Map node they hang off."""

    anchor: int
    anchor_from_base: np.ndarray     # 4x4
    points: np.ndarray               # N x 3 float16: forward, left, up-from-floor
    colours: np.ndarray              # N x 3 uint8, RGB

    def world_from_base(self, node_poses):
        if self.anchor not in node_poses:
            return None

        return node_poses[self.anchor] @ self.anchor_from_base


def make_view(depth, K, rgb, lens_height, settings, anchor=0, anchor_from_base=None, tilt=None):
    """Surface points and their colours from one metric depth map.

    `rgb` is the left camera's picture at any resolution: each depth pixel
    takes the colour at the same place in it. Points further than
    settings.max_range are left out -- depth is least sure far away -- and so
    is the sky, anything above settings.max_height.
    """
    height, width = depth.shape
    step = int(settings.step)

    v, u = np.mgrid[0:height:step, 0:width:step].astype(np.float32)
    d = depth[::step, ::step]

    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]

    forward = d
    left = -(u + 0.5 - cx) / fx * d
    up = -(v + 0.5 - cy) / fy * d

    if tilt is not None:
        forward, left, up = (tilt[i, 0] * forward + tilt[i, 1] * left + tilt[i, 2] * up for i in range(3))

    above_floor = lens_height + up
    keep = (d > 0.1) & (np.hypot(forward, left) < float(settings.max_range)) & (above_floor < float(settings.max_height))

    # The colour at the same place in the picture.
    rows = np.clip(((v + 0.5) * rgb.shape[0] / height).astype(np.int32), 0, rgb.shape[0] - 1)
    cols = np.clip(((u + 0.5) * rgb.shape[1] / width).astype(np.int32), 0, rgb.shape[1] - 1)

    points = np.stack([forward[keep], left[keep], above_floor[keep]], axis=1)
    colours = rgb[rows[keep], cols[keep]][:, :3].astype(np.float32)

    # Thinned to one point per `thin`-metre cell, the average of what fell in
    # it: the floor right in front of the camera is hit by thousands of
    # pixels a few millimetres apart, and keeping them all made two minutes
    # of driving 69 MB of views.
    thin = float(settings.get("thin", 0.025)) if hasattr(settings, "get") else 0.025
    _, inverse, counts = np.unique(np.floor(points / thin).astype(np.int64), axis=0,
                                   return_inverse=True, return_counts=True)
    inverse = inverse.reshape(-1)
    mean_points = np.zeros((len(counts), 3), np.float64)
    mean_colours = np.zeros((len(counts), 3), np.float64)
    np.add.at(mean_points, inverse, points)
    np.add.at(mean_colours, inverse, colours)

    return View(
        anchor=anchor,
        anchor_from_base=np.eye(4) if anchor_from_base is None else anchor_from_base,
        points=(mean_points / counts[:, None]).astype(np.float16),
        colours=np.clip(mean_colours / counts[:, None], 0, 255).astype(np.uint8),
    )


class Replica:
    """Coloured 5 cm cubes, from every kept view. Safe to read from one
    thread while another adds to it."""

    def __init__(self, settings):
        self.settings = settings
        self.voxel = float(settings.voxel)
        self.min_seen = int(settings.min_seen)
        self.views = []
        self.version = 0
        self.lock = threading.Lock()
        self._clear()

    def _clear(self):
        self._slot = {}
        self._keys = np.zeros(1024, np.int64)
        self._seen = np.zeros(1024, np.int32)
        self._colour = np.zeros((1024, 3), np.int64)
        self._used = 0

    # -----------------------------------------------------------------
    # Adding
    # -----------------------------------------------------------------

    def add(self, view, world_from_base):
        """Keep a view and add its points at a pose."""
        with self.lock:
            self.views.append(view)
            self._integrate(view, world_from_base)
            self.version += 1

    def keep(self, view):
        """Keep a view without adding it now: the next rebuild will."""
        with self.lock:
            self.views.append(view)

    def reset(self):
        """Forget everything: a new map, in a new frame."""
        with self.lock:
            self.views = []
            self._clear()
            self.version += 1

    def rebuild(self, node_poses):
        """Redraw every cube from the kept views at their current poses."""
        with self.lock:
            self._clear()

            for view in self.views:
                pose = view.world_from_base(node_poses)

                if pose is not None:
                    self._integrate(view, pose)

            self.version += 1

    def _integrate(self, view, world_from_base):
        if not len(view.points):
            return

        x, y, yaw = frames.planar(world_from_base)
        cos, sin = math.cos(yaw), math.sin(yaw)
        points = view.points.astype(np.float32)

        world = np.stack([
            x + cos * points[:, 0] - sin * points[:, 1],
            y + sin * points[:, 0] + cos * points[:, 1],
            points[:, 2],
        ], axis=1)

        keys = _pack(np.floor(world / self.voxel).astype(np.int64))
        unique, inverse, counts = np.unique(keys, return_inverse=True, return_counts=True)
        colour = np.zeros((len(unique), 3), np.int64)
        np.add.at(colour, inverse, view.colours.astype(np.int64))

        slots = np.empty(len(unique), np.int64)

        for i, key in enumerate(unique.tolist()):
            slot = self._slot.get(key)

            if slot is None:
                slot = self._used
                self._used += 1
                self._slot[key] = slot

            slots[i] = slot

        self._grow(self._used)
        self._keys[slots] = unique
        self._seen[slots] += counts.astype(np.int32)
        self._colour[slots] += colour

    def _grow(self, needed):
        if needed <= len(self._keys):
            return

        size = max(needed, 2 * len(self._keys))
        self._keys = np.resize(self._keys, size)
        self._seen = np.concatenate([self._seen, np.zeros(size - len(self._seen), np.int32)])
        self._colour = np.concatenate([self._colour, np.zeros((size - len(self._colour), 3), np.int64)])

    # -----------------------------------------------------------------
    # Reading
    # -----------------------------------------------------------------

    def cloud(self):
        """(centres N x 3 float32 in metres, colours N x 3 uint8) of every
        cube seen at least min_seen times."""
        with self.lock:
            used = self._used
            keep = self._seen[:used] >= self.min_seen
            keys = self._keys[:used][keep]
            colour = self._colour[:used][keep] / np.maximum(self._seen[:used][keep], 1)[:, None]

        centres = (_unpack(keys).astype(np.float32) + 0.5) * self.voxel
        return centres, np.clip(colour, 0, 255).astype(np.uint8)

    def __len__(self):
        with self.lock:
            return int((self._seen[:self._used] >= self.min_seen).sum())

    # -----------------------------------------------------------------
    # Saving
    # -----------------------------------------------------------------

    def save(self, path):
        """The kept views, so the replica can be rebuilt, to one .npz."""
        with self.lock:
            views = list(self.views)

        lengths = np.array([len(v.points) for v in views], dtype=np.int64)
        np.savez_compressed(
            path,
            anchors=np.array([v.anchor for v in views], dtype=np.int64),
            anchor_from_base=np.array([v.anchor_from_base for v in views]).reshape(-1, 4, 4),
            lengths=lengths,
            points=np.concatenate([v.points for v in views]) if views else np.zeros((0, 3), np.float16),
            colours=np.concatenate([v.colours for v in views]) if views else np.zeros((0, 3), np.uint8),
        )

    def load(self, path):
        """Last session's views. Drawn at the next rebuild, once RTAB-Map
        has put its nodes back where they belong."""
        data = np.load(path)
        splits = np.cumsum(data["lengths"])[:-1]

        with self.lock:
            self.views = [
                View(int(a), pose, p, c)
                for a, pose, p, c in zip(
                    data["anchors"], data["anchor_from_base"],
                    np.split(data["points"], splits), np.split(data["colours"], splits),
                )
            ]

    def save_ply(self, path):
        """The cubes as a coloured point cloud, for any 3D viewer."""
        centres, colours = self.cloud()

        with open(path, "wb") as out:
            out.write((
                "ply\nformat binary_little_endian 1.0\n"
                f"element vertex {len(centres)}\n"
                "property float x\nproperty float y\nproperty float z\n"
                "property uchar red\nproperty uchar green\nproperty uchar blue\n"
                "end_header\n"
            ).encode())

            record = np.zeros(len(centres), dtype=[("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
                                                   ("r", "u1"), ("g", "u1"), ("b", "u1")])
            record["x"], record["y"], record["z"] = centres[:, 0], centres[:, 1], centres[:, 2]
            record["r"], record["g"], record["b"] = colours[:, 0], colours[:, 1], colours[:, 2]
            out.write(record.tobytes())


def _pack(index):
    """N x 3 signed cube indices -> N packed int64 keys."""
    shifted = (index + OFFSET) & MASK
    return (shifted[:, 0] << (2 * BITS)) | (shifted[:, 1] << BITS) | shifted[:, 2]


def _unpack(keys):
    """The inverse of _pack."""
    return np.stack([
        ((keys >> (2 * BITS)) & MASK) - OFFSET,
        ((keys >> BITS) & MASK) - OFFSET,
        (keys & MASK) - OFFSET,
    ], axis=1)
