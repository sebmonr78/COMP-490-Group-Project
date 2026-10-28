"""Draw the brain's replica -- coloured 5 cm cubes -- from an orbiting camera.

Plain NumPy, no OpenGL: every cube's centre is projected into the picture
and drawn as a small square, with a depth buffer, so near things hide far
ones. It needs nothing from the graphics driver, which under WSL is the part
most likely to go wrong. A few hundred thousand cubes take tens of
milliseconds, so they are drawn again only when the cubes or the view
change; the drone's marks go on a copy every frame.

The camera orbits a point (the replica's middle, or the drone): drag to turn
it, scroll to come closer. Coordinates are the brain's: x and y on the
floor, z up from it.
"""

import collections
import copy
import math
import threading
import time

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
        self.follow_step = 0.5                # ...re-centring only every this many metres
        self._cached = None                   # (view, picture of the cubes)

    # -----------------------------------------------------------------
    # Steering
    # -----------------------------------------------------------------

    def drag(self, dx, dy):
        """A mouse drag of (dx, dy) pixels turns the view."""
        self.yaw -= dx * 0.008
        self.pitch = min(math.radians(89.0), max(math.radians(-10.0), self.pitch + dy * 0.008))

    def zoom(self, steps):
        """Scroll: closer (positive) or further."""
        self.distance = min(80.0, max(1.5, self.distance * (0.88 ** steps)))

    def look_at(self, point):
        """Centre on a point -- once it is follow_step away from the
        centre: re-centring on every 5 cm the drone moves meant drawing
        every cube again on every frame."""
        point = np.asarray(point, np.float32)

        if np.linalg.norm(point - self.target) >= self.follow_step:
            self.target = point

    # -----------------------------------------------------------------
    # Drawing
    # -----------------------------------------------------------------

    def render(self, centres, colours, size, voxel=0.05, markers=(), version=None):
        """The picture, (height, width, 3) uint8 RGB.

        `version` names the cubes: the same version, size and view give the
        same picture, so the cubes are drawn only when one of those changed.
        `markers` -- (xyz, rgb, radius px), the drone where it believes it
        is and where it really is -- are drawn on a copy every time.
        """
        view = (version, tuple(size), self.yaw, self.pitch, self.distance, tuple(self.target.tolist()))

        if self._cached is None or self._cached[0] != view:
            self._cached = (view, self._draw_cubes(centres, colours, size, voxel))

        # Without marks, the cached picture itself -- the same object until
        # it is drawn again, which tells the caller nothing changed.
        if not markers:
            return self._cached[1]

        image = self._cached[1].copy()

        for xyz, rgb, radius in markers:
            point = self.project(xyz, size)

            if point is not None:
                cv2.circle(image, point, radius, tuple(int(c) for c in rgb), 2, cv2.LINE_AA)

        return image

    def project(self, xyz, size):
        """Where a point lands in a picture of `size`, or None if it is
        behind the camera."""
        width, height = size
        eye, rotation = self._camera()
        focal = (width / 2.0) / math.tan(FIELD_OF_VIEW / 2.0)
        camera = (np.asarray(xyz, np.float32) - eye) @ rotation.T

        if camera[2] <= 0.2:
            return None

        return (int(focal * camera[0] / camera[2] + width / 2.0),
                int(focal * camera[1] / camera[2] + height / 2.0))

    def _draw_cubes(self, centres, colours, size, voxel):
        width, height = size
        image = np.zeros((height, width, 3), np.uint8)
        image[:] = (18, 20, 26)

        eye, rotation = self._camera()
        focal = (width / 2.0) / math.tan(FIELD_OF_VIEW / 2.0)

        self._floor_grid(image, eye, rotation, focal)

        if not len(centres):
            return image

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

        if not len(u):
            return image

        # Furthest first -- which also puts the radii in rising order, as a
        # nearer cube is a bigger square: the cubes with a square at least r
        # wide are then simply the ones from first[r] on.
        order = np.argsort(-depth)
        u, v, depth, radius, shade = u[order], v[order], depth[order], radius[order], shade[order]
        first = np.searchsorted(radius, np.arange(4))

        # One pass per pixel of the square, each pixel kept only where it is
        # nearer than what is there already. Within a pass the nearest is
        # written last, so it is the one left.
        zbuffer = np.full(width * height, np.inf, np.float32)
        flat = image.reshape(-1, 3)

        for du in range(-3, 4):
            for dv in range(-3, 4):
                start = first[max(abs(du), abs(dv))]

                if start >= len(u):
                    continue

                x, y = u[start:] + du, v[start:] + dv
                ok = (x >= 0) & (x < width) & (y >= 0) & (y < height)
                pixel = (y * width + x)[ok]
                d = depth[start:][ok]
                nearer = d < zbuffer[pixel]
                pixel, d = pixel[nearer], d[nearer]
                zbuffer[pixel] = d
                flat[pixel] = shade[start:][ok][nearer]

        return image

    def snapshot(self):
        """A copy of the view as it is now, to draw from while this one
        goes on being turned."""
        view = copy.copy(self)
        view.target = self.target.copy()
        view._cached = None
        return view

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


class Painter:
    """Draws an OrbitView's pictures on a thread of its own.

    A few hundred thousand cubes take 100-150 ms to draw at 4K, and the view
    changes every half metre the drone moves when it follows it: drawn in
    the window's own loop, that took the whole simulation down with it. Here
    the window asks for a picture whenever the cubes or the view changed,
    and always shows the newest one finished; asked again before one is
    done, only the newest request is drawn next. NumPy lets go of Python's
    lock for the heavy parts, so the two really do run side by side.
    """

    def __init__(self):
        self._asked = None             # what the newest request was for
        self._job = None               # (view, centres, colours, size, voxel)
        self._done = None              # (view it was drawn from, picture)
        self._wake = threading.Condition()
        # (ms drawing it, ms from asked to ready) per picture, for --profile:
        # the second is what a drag waits for, queueing behind the picture
        # already being drawn included.
        self.timings = collections.deque(maxlen=500)
        threading.Thread(target=self._run, daemon=True).start()

    def ask(self, view, centres, colours, size, voxel=0.05, version=None):
        key = (version, tuple(size), view.yaw, view.pitch, view.distance, tuple(view.target.tolist()))

        if key == self._asked:
            return

        self._asked = key

        with self._wake:
            self._job = (view.snapshot(), centres, colours, tuple(size), voxel, time.perf_counter())
            self._wake.notify()

    def latest(self):
        """(the view it was drawn from, the picture), or None before the
        first is done."""
        return self._done

    def _run(self):
        while True:
            with self._wake:
                while self._job is None:
                    self._wake.wait()

                (view, centres, colours, size, voxel, asked), self._job = self._job, None

            started = time.perf_counter()
            self._done = (view, view._draw_cubes(centres, colours, size, voxel))
            ready = time.perf_counter()
            self.timings.append((1000.0 * (ready - started), 1000.0 * (ready - asked)))
