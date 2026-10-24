"""The SIMPLE brain: stereo, ORB odometry and the gap map, in one process.

Superseded by the `delta/brain/` package, which is the Spatial Memory
Glasses brain ported across: RTAB-Map with loop closure in place of this
file's frame-to-frame odometry, and an occupancy grid that redraws itself
when a loop closes.

Kept because it needs no second process and no second conda env, which
makes it the quick way to check that the drone, the cameras and the window
still work without starting the whole stack.

Inputs, and the ONLY inputs:
    two camera images, one IMU reading, the time

Nothing in here may import or read Habitat. The simulator renders the
pictures and nothing else. The moment this file reads a true position or a
navmesh, every number the project reports stops meaning anything -- a map
that is right because it was told the answer is not a result.

The stages, in the order the proposal lists them:

    sensors       stereo pair -> depth in real metres, and the floor height
    perception    Depth Anything -> a depth for every pixel, scaled by stereo
    localization  corners + depth + IMU -> where am I  (SLAM, drifting)
    mapping       all of the above -> a grid of vertical columns
    navigation    that grid -> a route, in MapPlanner

Built so far: those five. Not built: object detection, change detection,
exploration, the database. Those come one at a time, and each one hangs off
the map this file maintains.
"""

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

from delta import stereo as stereo_module
from delta.tracker import StereoTracker

CONFIG = Path(__file__).resolve().parents[1] / "config"


def load_position_config():
    return json.loads((CONFIG / "position.json").read_text())


def load_map_config():
    return json.loads((CONFIG / "map.json").read_text())


@dataclass
class Thought:
    """One frame's worth of everything the drone believes."""

    depth: object            # stereo depth, metres (NaN where no match)
    dense: object            # stereo with its holes filled by DA3; None when skipped
    da3: object              # DA3 alone, rescaled to metres (for scoring)
    points: object           # (N, 3) in the camera's frame
    camera_height: float     # measured floor distance, or the fallback
    floor_measured: bool
    track: object            # the tracker's answer: pose, source, inliers
    mapped: bool             # did this frame go into the map
    milliseconds: dict = field(default_factory=dict)


class Brain:
    def __init__(self, rig, with_map=True, depth_model=None, overrides=None):
        """`depth_model`: pass an already-loaded DA3 to share it. Loading
        takes several seconds and about 1.4 GB of GPU memory, so a tool that
        makes two brains should make one model."""
        self.stereo = stereo_module.StereoPipeline(rig)

        r = self.stereo.rectify
        self.tracker = StereoTracker(
            r.fx, r.cx, r.cy, load_position_config()["tracker"]
        )

        self.map_config = load_map_config()

        for dotted, value in (overrides or {}).items():
            node = self.map_config
            *path, last = dotted.split(".")

            for key in path:
                node = node[key]

            node[last] = value

        self.map = None
        self.depth_model = None
        self.frames = 0
        self.last_raw = None

        if with_map:
            from delta.depth_model import DepthModel, StereoFill

            self.depth_model = depth_model or DepthModel(
                self.map_config["model"],
                process_res=self.map_config.get("process_res", 504),
            )
            self.ruler = StereoFill(self.map_config["scale"])

    # -----------------------------------------------------------------

    def step(self, left_rgb, right_rgb, imu, now, dt):
        ms = {}

        depth, points, stereo_ms = self.stereo.step(left_rgb, right_rgb, now)
        ms.update(stereo_ms)

        t = time.perf_counter()
        # The tracker finds its corners in the rectified LEFT gray image,
        # which is the same picture the depth came from -- so a corner at
        # (u, v) and the depth at (u, v) are the same piece of the world.
        track = self.tracker.step(
            self.stereo.rectify.left_gray, depth, imu, dt
        )
        ms["position"] = _since(t)

        dense = da3 = None
        mapped = False

        if (
            self.depth_model is not None
            and self.frames % self.map_config["every_n_frames"] == 0
        ):
            t = time.perf_counter()
            dense, da3, mapped = self._map(depth, track.pose)
            ms["map"] = _since(t)

        self.frames += 1

        return Thought(
            depth=depth,
            dense=dense,
            da3=da3,
            points=points,
            camera_height=self.stereo.camera_height,
            floor_measured=self.stereo.floor_measured,
            track=track,
            mapped=mapped,
            milliseconds=ms,
        )

    def _map(self, stereo_depth, pose):
        """Depth Anything, scaled to metres by stereo, into the map."""
        right = (
            self.stereo.rectify.right_rgb
            if self.map_config.get("views", 1) == 2
            else None
        )

        raw = self.depth_model(self.stereo.rectify.left_rgb, right)
        self.last_raw = raw     # DA3 in its own units, kept for scoring

        dense, da3 = self.ruler(raw, stereo_depth)

        if dense is None:
            return None, None, False

        if self.map is None:
            # The map's floor datum has to be measured, not the fallback
            # guess, or every height in it is wrong by the difference.
            if not self.stereo.floor_measured:
                return dense, da3, False

            from delta.gap_map import GapMap

            self.map = GapMap(self.map_config, self.stereo.camera_height)

        r = self.stereo.rectify

        self.map.add(
            dense,
            r.fx,
            r.fx,
            r.cx,
            r.cy,
            pose,
            self.stereo.camera_height,
        )

        return dense, da3, True

    # -----------------------------------------------------------------

    def planner(self, diameter, costs=None):
        """A planner on the map as it stands now, or None if nothing is mapped."""
        if self.map is None:
            return None

        from delta.map_navigation import MapPlanner

        return MapPlanner(self.map, diameter, costs)


def _since(t):
    return (time.perf_counter() - t) * 1000.0
