"""What is really above the cells the map wrongly calls solid?

    python tools/explain_phantoms.py outputs/diagnose/D_both_true_pictures/diagnose_end.png

Takes a score picture (written by the "score" script step), finds its orange
cells -- marked solid, but the drone's navmesh says the floor there is clear --
and, for each one, looks straight up from just above the floor to see what the
map was looking at. Sim side only: this is an examiner, not part of the brain.

    nothing overhead       the map invented it: a depth or scan artefact
    under 0.35 m           low enough to stop a rolling drone; the navmesh
                           disagrees only because of a thin gap
    0.35 m and up          a roof to the drone -- a table top, a chair seat, a
                           bed -- that a person's map calls a wall
"""

import argparse
import csv
import sys
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import habitat_sim
import magnum as mn
import numpy as np

from delta.sim import World, resolve_dataset_paths

ORANGE = np.array([0, 150, 255])        # BGR, as score_map draws phantoms
RESOLUTION = 0.05


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("picture")
    parser.add_argument("--samples", type=int, default=1500)
    args = parser.parse_args()

    picture = cv2.imread(args.picture)
    rows, cols = np.nonzero(np.all(picture == ORANGE, axis=2))

    if len(rows) == 0:
        print("no phantom cells in that picture")
        return 0

    world = World()
    sim = world.simulator
    lower, _ = world.pathfinder.get_bounds()
    floor_y = float(world.drone.floor_position.y)

    root, _, _, _ = resolve_dataset_paths(world.config)

    with open(root / "semantics" / "objects.csv") as f:
        label = {row["id"]: (row.get("main_category") or row.get("name") or "?") for row in csv.DictReader(f)}

    manager = sim.get_rigid_object_manager()
    names = {
        manager.get_object_by_handle(h).object_id: label.get(h.split("_:")[0], "?")
        for h in manager.get_object_handles()
    }
    body = world.drone.object.object_id

    pick = np.random.default_rng(0).choice(len(rows), min(args.samples, len(rows)), replace=False)

    bands = Counter()
    things = Counter()
    overhead = []

    for i in pick:
        x = lower.x + (cols[i] + 0.5) * RESOLUTION
        z = lower.z + (rows[i] + 0.5) * RESOLUTION

        origin = mn.Vector3(x, floor_y + 0.09, z)
        hits = sim.cast_ray(habitat_sim.geo.Ray(origin, mn.Vector3(0.0, 1.0, 0.0)), 2.2)
        first = next((h for h in hits.hits if h.object_id != body), None)

        if first is None:
            bands["nothing overhead"] += 1
            continue

        height = 0.09 + first.ray_distance
        overhead.append(height)
        what = "the building" if first.object_id == 0 else names.get(first.object_id, "?")

        if height < 0.35:
            bands["under 0.35 m (blocks rolling)"] += 1
        elif height < 1.9:
            bands["0.35-1.9 m (a roof to the drone)"] += 1
        else:
            bands["1.9 m and up (ceiling)"] += 1

        things[f"{what} ({height:.1f} m)" if height < 1.9 else f"{what}"] += 1

    total = sum(bands.values())

    print()
    print(f"{len(rows)} phantom cells in the picture; looked above {total} of them")
    print()

    for band, count in bands.most_common():
        print(f"  {count / total * 100:5.1f}%  {band}")

    if overhead:
        values = np.array(overhead)
        print()
        print("  how high it is overhead (above the floor):")
        for low, high in ((0, 1.85), (1.85, 1.9), (1.9, 1.95), (1.95, 2.0), (2.0, 2.05), (2.05, 2.2), (2.2, 9)):
            share = ((values >= low) & (values < high)).mean() * 100
            if share:
                print(f"    {share:5.1f}%  {low:.2f} - {high:.2f} m")

    print()
    print("  most common things overhead:")

    for thing, count in things.most_common(10):
        print(f"    {count / total * 100:5.1f}%  {thing}")

    world.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
