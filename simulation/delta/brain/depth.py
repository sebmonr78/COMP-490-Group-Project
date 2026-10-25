"""Estimate depth -- Depth Anything 3 on the slide.

How far away is every pixel of the left image, in metres? Two answers behind
one interface:

    DA3Depth      the real thing: both images of the pair, with the rig's
                  intrinsics and baseline, through DA3 (the #11 setup).
    OracleDepth   Habitat's own depth. For measuring the other boxes with
                  depth error taken out.

Then one correction for both, FloorScale, which is the reason the map works
at all. #11 found two-view DA3 reads about 12% short, the same factor on
every pixel. The floor turns that into a measurement: the lens sits a known
height above it, so the depth DA3 reports for the floor says exactly how
short it is reading, on this frame, with no ground truth. Left uncorrected,
the floor of a 12%-short map floats 20 cm up and turns into an obstacle.

Depth here is planar -- distance along the optical axis, not along the ray --
which is what Habitat renders and what #11 found DA3 produces.
"""

import os
from collections import deque
from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class DepthMap:
    depth: np.ndarray       # h x w float32, metres, planar; 0 = no estimate
    K: np.ndarray           # intrinsics at this resolution
    scale: float = 1.0      # the floor correction already applied
    source: str = ""


def scaled_intrinsics(K, from_size, to_size):
    """K for an image resized from (w, h) to (w', h'), each axis on its own.

    DA3 rounds both sides to multiples of 14, so its output is very slightly
    stretched (504 x 322 from 960 x 600) and x and y scale differently.
    """
    sx = to_size[0] / from_size[0]
    sy = to_size[1] / from_size[1]

    return np.diag([sx, sy, 1.0]) @ np.asarray(K, dtype=float)


def drop_depth_edges(depth, ratio=0.08):
    """Zero out pixels on depth discontinuities.

    At the silhouette of a chair, a learned depth model blends chair and
    wall into pixels that belong to neither: points floating in mid-air
    that the map would mark as obstacles. A jump of more than `ratio` of
    the depth between neighbours marks one.
    """
    padded = np.pad(depth, 1, mode="edge")

    jumps = np.maximum.reduce(
        [
            np.abs(padded[1:-1, 2:] - depth),
            np.abs(padded[1:-1, :-2] - depth),
            np.abs(padded[2:, 1:-1] - depth),
            np.abs(padded[:-2, 1:-1] - depth),
        ]
    )

    cleaned = depth.copy()
    cleaned[jumps > ratio * np.maximum(depth, 1e-3)] = 0.0

    return cleaned


class DA3Depth:
    """Two-view Depth Anything 3, metric from the rig's baseline."""

    name = "da3"

    def __init__(self, settings, baseline):
        # DA3 logs three INFO lines per call; at five calls a second that
        # buries everything else.
        os.environ.setdefault("DA3_LOG_LEVEL", "WARN")

        import torch
        import depth_anything_3.api as api

        _patch_two_view_alignment(api)

        self.torch = torch
        self.process_res = int(settings.process_res)
        self.model = api.DepthAnything3.from_pretrained(settings.model).to("cuda").eval()

        # World-to-camera, OpenCV axes, the left camera as the world. The
        # cameras are parallel, so the right one is the left one moved
        # +baseline along x, and world-to-camera undoes that. Exactly as in
        # da3_sandbox/run_da3.py, where #11 measured it.
        self.extrinsics = np.tile(np.eye(4, dtype=np.float32), (2, 1, 1))
        self.extrinsics[1, 0, 3] = -float(baseline)

    def estimate(self, pair, frame=None):
        height, width = pair.left.shape[:2]
        K = np.asarray(pair.K, dtype=np.float32)

        with self.torch.inference_mode():
            prediction = self.model.inference(
                image=[pair.left, pair.right],
                extrinsics=self.extrinsics,
                intrinsics=np.stack([K, K]),
                process_res=self.process_res,
            )

        depth = np.asarray(prediction.depth[0], dtype=np.float32)

        return DepthMap(
            depth=drop_depth_edges(depth),
            K=scaled_intrinsics(K, (width, height), (depth.shape[1], depth.shape[0])),
            source=self.name,
        )


class OracleDepth:
    """Habitat's depth, at the resolution DA3 would have produced."""

    name = "oracle"

    def __init__(self, width=504):
        self.width = width

    def estimate(self, pair, frame=None):
        if frame is None or frame.true_depth is None:
            raise RuntimeError(
                "The oracle depth needs the sim to send true depth "
                "(run the sim with --depth oracle)."
            )

        height, width = frame.true_depth.shape
        size = (self.width, int(round(height * self.width / width)))

        # Nearest neighbour: no pixel may be a blend of a wall and the room
        # behind it.
        depth = cv2.resize(frame.true_depth, size, interpolation=cv2.INTER_NEAREST)

        return DepthMap(
            depth=depth.astype(np.float32),
            K=scaled_intrinsics(pair.K, (width, height), size),
            source=self.name,
        )


class FloorScale:
    """Find the floor in a depth map, and from it the missing scale.

    The floor is the lowest large flat surface below the lens. For every
    pixel looking downward, the depth and the camera's tilt give how far
    below the lens that point is. Floor pixels all agree on that number --
    they form the strongest low peak in a histogram of it -- and the known
    lens height divided by it is the scale.
    """

    # DA3's scale stays within this range on Habitat frames (0.74-1.31 seen),
    # and moves by about 15% between neighbouring frames. A measurement
    # outside the range, or 25% off the recent median, is a tabletop or a
    # bed mistaken for the floor -- in a cramped bedroom one read 1.97 and
    # doubled every depth, painting free floor through the walls.
    PLAUSIBLE = (0.6, 1.6)
    TOLERANCE = 0.25

    def __init__(self, lens_height, enabled=True):
        self.lens_height = float(lens_height)
        self.enabled = enabled
        self.scale = 1.0
        self.recent = deque(maxlen=9)
        self._rays = None

    def update(self, depth_map, world_from_optical):
        """This frame's scale, or None if this frame shows too little floor.

        Per frame, never smoothed over frames: DA3's scale jumps by +-15%
        from one frame to the next. A running average lags those jumps, and
        a scale 10% low lifts the whole visible floor 17 cm -- into the
        obstacle band. The first version of this smoothed, and its map was
        solid white.
        """
        if not self.enabled:
            return 1.0

        drop = self._vertical_drop(depth_map, world_from_optical[:3, :3])

        if drop is None:
            return None

        measured = self.lens_height / drop

        if not self.PLAUSIBLE[0] < measured < self.PLAUSIBLE[1]:
            return None

        if len(self.recent) >= 3:
            median = float(np.median(self.recent))

            if abs(measured - median) > self.TOLERANCE * median:
                return None

        self.recent.append(measured)
        self.scale = measured

        return measured

    def fallback(self):
        """The recent typical scale, for a frame that shows no floor.

        Good enough for what is close -- 15% of a wall 30 cm away is 5 cm --
        and a wall faced from up close is exactly what such a frame shows.
        Skipping those frames left walls out of the map, and routes ran
        through them.
        """
        return float(np.median(self.recent)) if len(self.recent) >= 3 else None

    def _vertical_rays(self, depth_map, rotation):
        """Height change per unit of depth, for every pixel.

        The world-z component of each pixel's ray: -1 is straight down, +1
        straight up. Cached, since it only changes when the camera tilts.
        """
        height, width = depth_map.depth.shape

        key = (height, width, tuple(np.round(rotation[2], 6)),
               tuple(np.round(depth_map.K.ravel(), 3)))

        if self._rays is None or self._rays[0] != key:
            u, v = np.meshgrid(np.arange(width) + 0.5, np.arange(height) + 0.5)
            rays = np.linalg.inv(depth_map.K) @ np.stack([u.ravel(), v.ravel(), np.ones(u.size)])
            self._rays = (key, (rotation[2] @ rays).reshape(height, width))

        return self._rays[1]

    def _vertical_drop(self, depth_map, rotation):
        """How far below the lens the floor is, in this map's own units."""
        depth = depth_map.depth
        height, width = depth.shape

        vertical = self._vertical_rays(depth_map, rotation)

        # Downward-looking pixels in the lower half, with a depth.
        lower = np.zeros_like(depth, dtype=bool)
        lower[height // 2:] = True

        mask = lower & (vertical < -0.15) & (depth > 0.2)

        if mask.sum() < 300:
            return None

        return floor_drop(-(depth[mask] * vertical[mask]), self.lens_height)

    def ceiling(self, depth_map, world_from_optical, low=0.3, high=4.0,
                support=0.04):
        """How far above the lens the ceiling is, in METRES, or None for none.

        Searched from 0.3 m above the lens. The glasses searched from 0.8,
        which suits a head 0.8 m under a 2.4 m ceiling; a drone flying 1.1 m
        up under a 2 m ceiling has it 0.7 m overhead, found nothing, and
        mapped the room it was flying in as outdoors.

        The mirror of the floor finder, and it answers the one question the
        map could not: is this spot indoors?

        It works because the lens is wide. At 120 degrees the top of the frame
        looks about 47 degrees upward, so a ceiling 2.4 m above the lens comes
        into view roughly 2 m ahead -- a real surface at a constant height,
        which is a sharp peak in a histogram of how far above the lens each
        upward-looking pixel is. Under open sky there is no such surface:
        Depth Anything will still return numbers for the sky, but they do not
        agree on a height, so no peak clears the threshold and the answer is
        None. That difference is the whole indoor/outdoor test.

        Call it with depth already in metres, unlike the floor finder, which
        runs before the scale is known and therefore works in whatever units
        the depth arrived in.
        """
        depth = depth_map.depth
        height, width = depth.shape

        vertical = self._vertical_rays(depth_map, world_from_optical[:3, :3])

        # Upward-looking pixels in the upper half, with a depth.
        upper = np.zeros_like(depth, dtype=bool)
        upper[: height // 2] = True

        mask = upper & (vertical > 0.15) & (depth > 0.2)

        if mask.sum() < 300:
            return None

        rises = depth[mask] * vertical[mask]
        rises = rises[(rises > low) & (rises < high)]

        if rises.size < 300:
            return None

        bins = np.linspace(low, high, 161)       # about 2 cm each
        histogram, edges = np.histogram(rises, bins=bins)
        histogram = np.convolve(histogram, np.ones(5) / 5.0, mode="same")

        peak = int(np.argmax(histogram))
        centre = 0.5 * (edges[peak] + edges[peak + 1])
        near = np.abs(rises - centre) < 0.10

        # A ceiling is a large flat thing. A doorframe, a lamp or a patch of
        # sky that happens to agree is not, so a real share of the upward
        # view has to land on it.
        if near.sum() < max(300, support * mask.sum()):
            return None

        return float(np.median(rises[near]))


def floor_drop(drops, expected=1.741):
    """The floor's distance below the lens, from many points' drops, or None.

    `drops` are how far below the lens each downward-looking point is. With
    DA3 they are in DA3's units, so the answer gives its scale; with metric
    stereo (SGBM) they are metres, so the answer IS the lens height.

    `expected` is roughly how far below the lens the floor should be -- the
    lens height. Everything here is measured in multiples of it, and that is
    not tidiness, it is the difference between working and not:

        glasses   lenses 1.74 m up, so the floor is 1.74 m below
        drone     lenses 0.15 m up, so the floor is 0.15 m below

    This used to search a fixed 0.3 m to 4.0 m window, which is right for a
    standing adult and throws away EVERY floor pixel for a robot 15 cm off
    the ground. The symptom was not an error: the brain simply reported "no
    floor" on every frame, never resolved Depth Anything's scale, and built
    no map at all.

    The window is 0.4x to 2.4x `expected`, which covers the whole range of
    scales DA3 is allowed to have (FloorScale.PLAUSIBLE, 0.6 to 1.6) with
    room to spare. At the glasses' height that is 0.70 m to 4.18 m -- within
    a few centimetres of the fixed window it replaces, so their numbers
    stand.
    """
    low, high = 0.4 * expected, 2.4 * expected

    drops = drops[(drops > low) & (drops < high)]

    if drops.size < 300:
        return None

    # Bins fine enough to separate a floor from a rug, at any height: 370
    # of them across the window, as the fixed 1 cm bins gave a standing adult.
    bins = np.linspace(low, high, 371)
    histogram, edges = np.histogram(drops, bins=bins)
    histogram = np.convolve(histogram, np.ones(5) / 5.0, mode="same")

    # The floor is the LOWEST strong peak: furniture tops are higher up,
    # nothing in a house is below the floor.
    strong = histogram >= 0.5 * histogram.max()
    peak = np.nonzero(strong)[0].max()

    # Then the strongest bin of that peak, and the points close to it.
    while peak > 0 and histogram[peak - 1] >= histogram[peak]:
        peak -= 1

    centre = 0.5 * (edges[peak] + edges[peak + 1])

    # "Close to it" also has to scale: 3 cm of a 1.74 m drop is 1.7%, and 3 cm
    # of a 15 cm drop is a fifth of the whole measurement.
    tolerance = max(0.017 * expected, 1.5 * float(edges[1] - edges[0]))
    near = np.abs(drops - centre) < tolerance

    # Enough of the view must agree on it to call it the floor.
    if near.sum() < max(300, 0.1 * drops.size):
        return None

    return float(np.median(drops[near]))


def _patch_two_view_alignment(api):
    """Let DA3 metric-scale a two-view prediction (from #11's run_da3.py).

    DA3 fits a similarity transform from the given camera centres to its
    predicted ones and divides depth by the fitted scale. With exactly two
    centres the rotation about the line through them is undetermined and
    the fit raises. The scale is not undetermined: it is the ratio of the
    two baselines, which is what the fit would return.
    """
    if getattr(api, "_two_view_patched", False):
        return

    original = api.align_poses_umeyama

    def camera_centres(extrinsics):
        rotation = extrinsics[:, :3, :3]
        translation = extrinsics[:, :3, 3]

        return -np.einsum("nji,nj->ni", rotation, translation)

    def align(ext_ref, ext_est, **kwargs):
        if len(ext_est) != 2:
            return original(ext_ref, ext_est, **kwargs)

        predicted = camera_centres(np.asarray(ext_ref, dtype=np.float64))
        given = camera_centres(np.asarray(ext_est, dtype=np.float64))

        scale = np.linalg.norm(predicted[1] - predicted[0]) / np.linalg.norm(given[1] - given[0])

        return np.eye(3), np.zeros(3), float(scale), np.asarray(ext_ref)

    api.align_poses_umeyama = align
    api._two_view_patched = True
