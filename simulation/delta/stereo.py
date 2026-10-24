"""Two pictures in, metres out. The measuring half of the perception stack.

Extracted from the glasses project's `hazard.py`, which did two jobs: it
measured the scene with stereo, and then it decided when to beep at a blind
wearer. DELTA needs the measuring and not the beeping, so only the measuring
came across:

    left + right (RGB)
        -> Rectifier      line the rows up, shrink, grayscale
        -> StereoDepth    find how far each pixel slid between the pictures
        -> depth          metres = fx * baseline / slide
        -> depth_to_points   one (x, y, z) per pixel, in the left camera's frame
        -> FloorFinder    how far below the camera the floor is

Why this is the ruler and not the map: stereo only has an answer where it
finds a match, so a blank wall comes back empty. But where it does answer, it
answers in real metres -- measured at about 1% error on this rig. Depth
Anything is the opposite: an answer for every pixel, at about 8% error and a
scale that cannot be trusted. `depth_model.StereoFill` puts the two together,
which is the only reason the map is in metres at all.

Camera frame here is OpenCV's: x right, y DOWN, z forward. Habitat's is
different (y up, -z forward). The conversion is not needed because nothing in
this file touches Habitat -- it only ever sees images.
"""

import json
import time
from pathlib import Path

import cv2
import numpy as np


CONFIG_PATH = Path(__file__).resolve().parents[1] / "config" / "stereo.json"


def load_config(path=None):
    return json.loads(Path(path or CONFIG_PATH).read_text())


# ---------------------------------------------------------------------
# Rectify
# ---------------------------------------------------------------------


class Rectifier:
    """Makes the same real-world point land on the same row in both images.

    Built once from the rig. In simulation the cameras are parallel and the
    lenses are perfect, so this changes almost nothing -- it runs anyway so
    that the real camera, which is never perfectly parallel, goes through the
    same code and nothing downstream has to learn a second path.

    After construction:
        fx, cx, cy   the camera numbers of the rectified, process-size image
        baseline     metres between the cameras
    """

    def __init__(self, rig):
        left, right = rig.stereo_pair

        self.size = tuple(rig.process)
        self.baseline = float(rig.baseline)

        k_left = rig.intrinsics(left, "process")
        k_right = rig.intrinsics(right, "process")

        d_left = np.asarray(rig.distortion.get(left, [0.0] * 5), dtype=np.float64)
        d_right = np.asarray(rig.distortion.get(right, [0.0] * 5), dtype=np.float64)

        # OpenCV: X_right = R * X_left + T. The right camera sits +baseline
        # to the right, so a point in the left camera's frame is baseline
        # further left in the right camera's.
        r1, r2, p1, p2, _q, _roi1, _roi2 = cv2.stereoRectify(
            k_left,
            d_left,
            k_right,
            d_right,
            self.size,
            np.eye(3),
            np.array([-self.baseline, 0.0, 0.0]),
            flags=cv2.CALIB_ZERO_DISPARITY,
            alpha=0,
        )

        self._maps = (
            cv2.initUndistortRectifyMap(k_left, d_left, r1, p1, self.size, cv2.CV_32FC1),
            cv2.initUndistortRectifyMap(k_right, d_right, r2, p2, self.size, cv2.CV_32FC1),
        )

        self.fx = float(p1[0, 0])
        self.cx = float(p1[0, 2])
        self.cy = float(p1[1, 2])

        self.left_rgb = None
        self.right_rgb = None
        self.left_gray = None

    def __call__(self, left_rgb, right_rgb):
        """Two RGB images in, two rectified gray images out, at process size.

        The rectified LEFT colour image is kept as `self.left_rgb`: Depth
        Anything looks at that one, so its pixels line up with stereo's and
        the same camera numbers (fx, cx, cy) fit both.
        """
        pair = []

        for image, (map_x, map_y) in zip((left_rgb, right_rgb), self._maps):
            image = np.ascontiguousarray(image[:, :, :3])
            small = cv2.resize(image, self.size, interpolation=cv2.INTER_AREA)
            flat = cv2.remap(small, map_x, map_y, cv2.INTER_LINEAR)
            pair.append(cv2.cvtColor(flat, cv2.COLOR_RGB2GRAY))

            if len(pair) == 1:
                self.left_rgb = flat
                self.left_gray = pair[0]
            else:
                self.right_rgb = flat

        return pair


# ---------------------------------------------------------------------
# Depth and points
# ---------------------------------------------------------------------


class StereoDepth:
    """SGBM: for each left pixel, how far sideways its twin sits in the right.

    OpenCV hands the slide back multiplied by 16 and as a whole number, and
    marks "no match found" as a negative number. Both are handled here so
    nothing downstream has to remember.
    """

    def __init__(self, rectifier, config):
        s = config["sgbm"]
        block = s["block_size"]

        self.rectifier = rectifier
        self.block = block
        self.min_texture = s.get("min_texture", 0.0)
        self.matcher = cv2.StereoSGBM_create(
            minDisparity=0,
            numDisparities=s["num_disparities"],
            blockSize=block,
            P1=8 * block * block,
            P2=32 * block * block,
            disp12MaxDiff=s["disp12_max_diff"],
            uniquenessRatio=s["uniqueness_ratio"],
            speckleWindowSize=s["speckle_window"],
            speckleRange=s["speckle_range"],
            preFilterCap=s["prefilter_cap"],
            mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY,
        )

    def __call__(self, left_gray, right_gray):
        """Depth in metres along the optical axis; NaN where there is no match."""
        slide = self.matcher.compute(left_gray, right_gray).astype(np.float32) / 16.0

        depth = np.full(slide.shape, np.nan, dtype=np.float32)
        found = (slide >= 1.0) & self._textured(left_gray)

        depth[found] = self.rectifier.fx * self.rectifier.baseline / slide[found]

        return depth

    def _textured(self, gray):
        """Where the picture has enough left-right detail to match at all.

        On a plain wall every patch looks like every other, and SGBM will
        still pick one -- a guess that can read as an object 40 cm away.
        Average the sideways brightness change over the matching window;
        below `min_texture`, call it "no match" rather than trust the guess.
        """
        if self.min_texture <= 0.0:
            return np.ones(gray.shape, dtype=bool)

        window = self.block * 3
        sideways = np.abs(cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3))

        return cv2.blur(sideways, (window, window)) >= self.min_texture


def depth_to_points(depth, fx, cx, cy, max_range):
    """One (x, y, z) per valid pixel. x right, y down, z forward, metres."""
    rows, cols = np.nonzero(
        np.isfinite(depth) & (depth > 0.0) & (depth <= max_range)
    )

    z = depth[rows, cols]
    x = (cols - cx) * z / fx
    y = (rows - cy) * z / fx

    return np.stack([x, y, z], axis=1)


# ---------------------------------------------------------------------
# Floor
# ---------------------------------------------------------------------


def estimate_floor(points, config):
    """How far below the camera the floor is, in metres, or None.

    With a level camera every floor point has the same y, so the floor shows
    up as one tall spike in a histogram of y. Several flat things can spike
    (a tabletop does too); the floor is the lowest spike that is big enough.

    This is why the gimbal matters. A camera that has tipped forward three
    degrees smears the spike across the histogram and the floor stops being
    findable at all.
    """
    f = config["floor"]

    chosen = (
        (points[:, 2] >= f["min_range"])
        & (points[:, 2] <= f["max_range"])
        & (points[:, 1] >= f["min_drop"])
    )

    y = points[chosen, 1]

    if len(y) < f["min_points"]:
        return None

    edges = np.arange(f["min_drop"], y.max() + 2 * f["bin"], f["bin"])
    counts, edges = np.histogram(y, bins=edges)

    # Both a share of the points and a real count: when the floor is out of
    # view, a few wrong matches can make a "spike" out of very little.
    need = max(f["min_fraction"] * len(y), f["min_spike"])
    strong = np.nonzero(counts >= need)[0]

    if len(strong) == 0:
        return None

    lowest = strong[-1]
    centre = 0.5 * (edges[lowest] + edges[lowest + 1])

    return float(np.median(y[np.abs(y - centre) <= f["bin"]]))


class FloorFinder:
    """Keeps the best recent answer to "how high is the camera?".

    The floor is only visible from a metre or so out, so a drone nose-up
    against a wall cannot see it at all. When it is out of view the last good
    height is reused, because a map that stops recording whenever the floor
    goes out of frame records almost nothing.

    The floor is the LOWEST flat surface, so the TALLEST camera height seen
    lately wins: a kitchen counter, seen while the real floor is hidden
    behind it, can never override a floor seen moments ago.
    """

    def __init__(self, config, default_height=None):
        self.config = config

        # Never None. Starting up facing a wall must not leave the map blind.
        self.camera_height = float(
            default_height
            if default_height is not None
            else config.get("default_camera_height", 0.15)
        )
        self.measured = False

        self._sightings = []

    def update(self, points, now):
        """Returns True when this frame actually saw the floor."""
        height = estimate_floor(points, self.config)

        if height is None:
            return False

        memory = self.config["floor"].get("memory_s", 10.0)

        self._sightings = [
            (t, h) for t, h in self._sightings if now - t <= memory
        ]
        self._sightings.append((now, height))

        self.camera_height = max(h for _, h in self._sightings)
        self.measured = True

        return True


# ---------------------------------------------------------------------
# The whole measuring stage
# ---------------------------------------------------------------------


class StereoPipeline:
    """Rig in, metres out. What `HazardPipeline` was, minus the beeping."""

    def __init__(self, rig, config=None, default_camera_height=None):
        self.config = config or load_config()
        self.rectify = Rectifier(rig)
        self.match = StereoDepth(self.rectify, self.config)
        self.floor = FloorFinder(self.config, default_camera_height)

        self._frames = 0

    @property
    def camera_height(self):
        return self.floor.camera_height

    @property
    def floor_measured(self):
        return self.floor.measured

    def step(self, left_rgb, right_rgb, now=None):
        """Returns (depth, points, milliseconds)."""
        now = self._frames * 0.1 if now is None else now
        self._frames += 1

        t0 = time.perf_counter()
        left, right = self.rectify(left_rgb, right_rgb)

        t1 = time.perf_counter()
        depth = self.match(left, right)

        t2 = time.perf_counter()
        points = depth_to_points(
            depth,
            self.rectify.fx,
            self.rectify.cx,
            self.rectify.cy,
            self.config["floor"]["max_range"],
        )

        t3 = time.perf_counter()
        self.floor.update(points, now)

        milliseconds = {
            "rectify": (t1 - t0) * 1000.0,
            "sgbm": (t2 - t1) * 1000.0,
            "points": (t3 - t2) * 1000.0,
            "floor": (time.perf_counter() - t3) * 1000.0,
        }

        return depth, points, milliseconds
