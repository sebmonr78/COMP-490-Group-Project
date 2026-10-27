"""Score the brain's map against the truth, on a recorded walk.

Rebuilds the occupancy grid from a walk saved by tools/record_walk.py -- the
real depth and scale code, at Habitat's true poses so that only the map is
being judged -- and compares it with the navmesh:

    phantom      walkable floor the map calls solid. Routes go round it, or
                 cannot be found at all.
    unsafe       floor the map calls free that is inside furniture or walls.
                 The dangerous kind of mistake.
    routes       of 21 trips between points on the walk, how many it can plan,
                 and how much longer than a straight line they come out.
    no floor     frames with no floor in view to fix the scale: their
                 obstacles are mapped at the recent scale, their free space not.

This is what the occupancy numbers in config/brain.json were tuned with.

Run (brain env; the first run spends a few minutes on DA3, then caches it):

    python tools/check_map.py                         # outputs/walks/default
    python tools/check_map.py --depth oracle          # Habitat's depth: the ceiling
    python tools/check_map.py --set hit=0.85 --set occupied_above=0.7
"""

import argparse
import json
import math
import sys
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from delta import frames  # noqa: E402
from delta import rig as rig_module  # noqa: E402
from delta.brain import config, navmap, occupancy, planner  # noqa: E402
from delta.brain import depth as depth_module  # noqa: E402
from delta.brain.cameras import StereoFrame  # noqa: E402
from delta.brain.rectify import Rectifier  # noqa: E402

FOOTPRINT = (-0.33, -0.06, 0.25)
TRIPS = 25


def depths(walk, settings, rig, source):
    """(depth, K) per frame, DA3's cached after the first run."""
    files = sorted(walk.glob("frame_*.npz"))
    first = np.load(files[0])
    size = (first["left"].shape[1], first["left"].shape[0])
    rectifier = Rectifier(rig, size)

    if source == "oracle":
        oracle = depth_module.OracleDepth()

        for path in files:
            data = np.load(path)
            frame = StereoFrame(0, 0.0, data["left"], data["right"],
                                true_depth=data["true_depth"].astype(np.float32))
            result = oracle.estimate(rectifier(frame), frame)
            yield path, result.depth, result.K

        return

    cache = walk / "da3"
    cache.mkdir(exist_ok=True)
    model = None

    for path in files:
        cached = cache / path.name

        if not cached.exists():
            if model is None:
                model = depth_module.DA3Depth(settings.depth, rectifier.baseline)

            data = np.load(path)
            frame = StereoFrame(0, 0.0, data["left"], data["right"])
            result = model.estimate(rectifier(frame), frame)
            np.savez_compressed(cached, depth=result.depth.astype(np.float16), K=result.K)

        stored = np.load(cached)
        yield path, stored["depth"].astype(np.float32), stored["K"]


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--walk", default="default")
    parser.add_argument("--depth", choices=["da3", "oracle"], default="da3")
    parser.add_argument("--set", action="append", default=[],
                        help="override an occupancy setting, key=value (JSON value)")
    args = parser.parse_args()

    settings = config.load()
    rig = rig_module.load()
    lens = float(settings.drone.lens_height)

    walk = PROJECT_ROOT / "outputs" / "walks" / args.walk

    if not (walk / "navmesh.npz").exists():
        sys.exit(f"No walk at {walk}. Record one first: python tools/record_walk.py")

    occupancy_settings = config.Section(dict(settings.occupancy))

    for pair in args.set:
        key, value = pair.split("=", 1)
        occupancy_settings[key] = json.loads(value)

    floor = depth_module.FloorScale(lens)
    poses, grid, skipped, borrowed = [], None, 0, 0

    for path, depth, K in depths(walk, settings, rig, args.depth):
        data = np.load(path)
        pose = frames.world_from_habitat_camera(data["position"].tolist(), data["rotation"].tolist())
        poses.append(pose)

        if grid is None:
            grid = occupancy.OccupancyGrid(occupancy_settings, pose[:2, 3])

        # Exactly the brain's rule: no floor in view, borrow the recent scale
        # for the obstacles and declare nothing free.
        scale = floor.update(depth_module.DepthMap(depth, K), frames.world_from_optical(pose))
        measured = scale is not None or args.depth == "oracle"

        if scale is None:
            scale = 1.0 if args.depth == "oracle" else floor.fallback()

        if scale is None:
            skipped += 1
            continue

        if not measured:
            borrowed += 1

        scan = occupancy.make_scan(depth * scale, K, lens, occupancy_settings, FOOTPRINT,
                                   free_space=measured)
        grid.add(scan, pose)

    phantom, unsafe, free_area = score(grid, walk / "navmesh.npz")
    found, tried, detour = routes(grid, settings, [p[:2, 3] for p in poses])

    print(f"Walk          {args.walk}: {len(poses)} frames, depth {args.depth}")
    print(f"Phantom       {phantom:.1%} of walkable floor seen is marked solid")
    print(f"Unsafe        {unsafe:.1%} of free cells are inside furniture or walls")
    print(f"Free floor    {free_area:.0f} m2 mapped")
    print(f"Routes        {found} of {tried} planned, median {detour:.2f}x the straight line")
    print(f"No floor      {borrowed} frames mapped obstacles only, {skipped} skipped")


def score(grid, navmesh_path):
    truth = np.load(navmesh_path)
    navigable = truth["navigable"].astype(np.uint8)
    lower = truth["lower"]
    resolution = float(truth["resolution"])

    def disc(metres):
        size = 2 * int(round(metres / resolution)) + 1
        return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))

    # Well inside where a person can stand: should be clear. Well outside
    # (past a person's half-width and some): furniture and walls.
    clear = cv2.erode(navigable, disc(0.10)).astype(bool)
    floor = cv2.dilate(navigable, disc(0.35)).astype(bool)

    classes = grid.classes()
    rows, cols = np.nonzero(classes != occupancy.UNKNOWN)
    xy = grid.to_world(np.stack([cols, rows], axis=1))

    # Brain world (x east, y north) -> navmesh pixel (x, z = -north).
    col = np.round((xy[:, 0] - lower[0]) / resolution).astype(int)
    row = np.round((-xy[:, 1] - lower[1]) / resolution).astype(int)
    inside = (row >= 0) & (row < navigable.shape[0]) & (col >= 0) & (col < navigable.shape[1])

    in_clear = np.zeros(len(xy), bool)
    in_floor = np.zeros(len(xy), bool)
    in_clear[inside] = clear[row[inside], col[inside]]
    in_floor[inside] = floor[row[inside], col[inside]]

    solid = classes[rows, cols] == occupancy.OCCUPIED
    free = classes[rows, cols] == occupancy.FREE

    phantom = (solid & in_clear).sum() / max(in_clear.sum(), 1)
    unsafe = (free & ~in_floor).sum() / max(free.sum(), 1)

    return phantom, unsafe, (free & in_clear).sum() * grid.resolution ** 2


def routes(grid, settings, track):
    nav = navmap.build(grid, settings.navigation)
    rng = np.random.default_rng(0)
    found, tried, detours = 0, 0, []

    for _ in range(TRIPS):
        a, b = track[rng.integers(len(track))], track[rng.integers(len(track))]

        if math.dist(a, b) < 3.0:
            continue

        tried += 1
        points = planner.plan(nav, a, b, near=(0.0, 0.5))

        if points:
            found += 1
            detours.append(planner.length(points) / math.dist(a, b))

    return found, tried, float(np.median(detours)) if detours else float("nan")


if __name__ == "__main__":
    main()
