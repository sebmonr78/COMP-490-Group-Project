"""Draw the brain's replica -- coloured 5 cm cubes -- from an orbiting camera.

Plain NumPy, no OpenGL: every cube's centre is projected into the picture
and drawn as a small square, nearest last, so near things hide far ones.
That is enough for a few hundred thousand cubes at several frames a second,
and it needs nothing from the graphics driver, which under WSL is the part
most likely to go wrong.

The camera orbits a point (the replica's middle, or the drone): drag to turn
it, scroll to come closer. Coordinates are the brain's: x and y on the
floor, z up from it.
"""

import math

import cv2
import numpy as np

FIELD_OF_VIEW = math.radians(60.0)


class OrbitView:
    """Where the replica is looked at from, and the picture of it."""

    def __init__(self):
        self.yaw = math.radians(-60.0)        # around the target, anticlockwise from +x
        self.pitch = math.radians(35.0)       # above the floor
        self.distance = 12.0
        self.target = np.zeros(3, np.float32)
        self.follow = True                    # keep the drone in the middle
        self._cached = None

    # -----------------------------------------------------------------
    # Steering
    # -----------------------------------------------------------------

    def drag(self, dx, dy):
        """A mouse drag of (dx, dy) pixels turns the view."""
        self.yaw -= dx * 0.008
        self.pitch = min(math.radians(89.0), max(math.radians(-10.0), self.pitch + dy * 0.008))
        self._cached = None

    def zoom(self, steps):
        """Scroll: closer (positive) or further."""
        self.distance = min(80.0, max(1.5, self.distance * (0.88 ** steps)))
        self._cached = None

    def look_at(self, point):
        point = np.asarray(point, np.float32)

        if not np.allclose(point, self.target, atol=0.05):
            self.target = point
            self._cached = None

    # -----------------------------------------------------------------
    # Drawing
    # -----------------------------------------------------------------

    def render(self, centres, colours, size, voxel=0.05, markers=(), version=None):
        """The picture, (height, width, 3) uint8 RGB.

        `markers`: (xyz, rgb, radius px) to draw on top -- the drone, where
        it really is. Redrawn only when the cubes, the view or a marker
        moved; otherwise the last picture is handed back.
        """
        key = (version, size, tuple((tuple(np.round(m[0], 2)), m[1]) for m in markers))

        if self._cached is not None and self._cached[0] == key:
            return self._cached[1]

        width, height = size
        image = np.zeros((height, width, 3), np.uint8)
        image[:] = (18, 20, 26)

        eye, rotation = self._camera()
        focal = (width / 2.0) / math.tan(FIELD_OF_VIEW / 2.0)

        self._floor_grid(image, eye, rotation, focal)

        if len(centres):
            camera = (centres - eye) @ rotation.T            # x right, y down, z forward
            ahead = camera[:, 2] > 0.2
            camera, shade = camera[ahead], colours[ahead]

            u = (focal * camera[:, 0] / camera[:, 2] + width / 2.0).astype(np.int32)
            v = (focal * camera[:, 1] / camera[:, 2] + height / 2.0).astype(np.int32)
            depth = camera[:, 2]

            # A cube is about voxel * focal / depth pixels across.
            radius = np.clip(np.round(voxel * focal / depth / 2.0), 0, 3).astype(np.int32)
            inside = (u >= -3) & (u < width + 3) & (v >= -3) & (v < height + 3)
            u, v, depth, radius, shade = u[inside], v[inside], depth[inside], radius[inside], shade[inside]

            # Every pixel of every square, then all of them written in one
            # go, furthest first, so a nearer square is always on top -- of
            # the middle AND the edges of anything behind it.
            xs, ys, ds, cs = [], [], [], []

            for r in range(int(radius.max()) + 1 if len(radius) else 0):
                pick = radius == r

                if not pick.any():
                    continue

                for du in range(-r, r + 1):
                    for dv in range(-r, r + 1):
                        xs.append(u[pick] + du)
                        ys.append(v[pick] + dv)
                        ds.append(depth[pick])
                        cs.append(shade[pick])

            if xs:
                x, y, d, c = (np.concatenate(a) for a in (xs, ys, ds, cs))
                ok = (x >= 0) & (x < width) & (y >= 0) & (y < height)
                x, y, d, c = x[ok], y[ok], d[ok], c[ok]
                order = np.argsort(-d)
                image[y[order], x[order]] = c[order]

        for xyz, rgb, radius in markers:
            camera = (np.asarray(xyz, np.float32) - eye) @ rotation.T

            if camera[2] > 0.2:
                point = (int(focal * camera[0] / camera[2] + width / 2.0),
                         int(focal * camera[1] / camera[2] + height / 2.0))
                cv2.circle(image, point, radius, tuple(int(c) for c in rgb), 2, cv2.LINE_AA)

        self._cached = (key, image)
        return image

    def _camera(self):
        """Eye position, and the rotation from world to camera axes."""
        offset = np.array([
            math.cos(self.pitch) * math.cos(self.yaw),
            math.cos(self.pitch) * math.sin(self.yaw),
            math.sin(self.pitch),
        ], np.float32) * self.distance
        eye = self.target + offset

        forward = -offset / np.linalg.norm(offset)
        right = np.cross(forward, [0.0, 0.0, 1.0])
        right /= max(np.linalg.norm(right), 1e-6)
        down = np.cross(forward, right)

        return eye, np.stack([right, down, forward]).astype(np.float32)

    def _floor_grid(self, image, eye, rotation, focal):
        """Faint metre lines on the floor around the target, for scale."""
        height, width = image.shape[:2]
        cx, cy = round(float(self.target[0])), round(float(self.target[1]))

        for n in range(-15, 16):
            for a, b in (((cx + n, cy - 15), (cx + n, cy + 15)), ((cx - 15, cy + n), (cx + 15, cy + n))):
                ends = []

                for x, y in (a, b):
                    camera = (np.array([x, y, 0.0], np.float32) - eye) @ rotation.T
                    ends.append(camera)

                p, q = ends

                # Clip the line to in front of the camera.
                if p[2] < 0.2 and q[2] < 0.2:
                    continue

                if p[2] < 0.2 or q[2] < 0.2:
                    t = (0.2 - p[2]) / (q[2] - p[2])
                    clipped = p + t * (q - p)
                    p, q = (clipped, q) if p[2] < 0.2 else (p, clipped)

                points = [(int(focal * c[0] / c[2] + width / 2.0), int(focal * c[1] / c[2] + height / 2.0)) for c in (p, q)]
                cv2.line(image, points[0], points[1], (38, 42, 52), 1, cv2.LINE_AA)
