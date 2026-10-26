"""Can the drone roll onto every rug from open floor, without flying?

    conda activate habitat
    cd ~/COMP-490-Group-Project/simulation
    python tests/probe_rug.py

For each rug in the house and each of its four sides: start on open floor a
little way outside the edge, drive straight in, and check -- by looking down
at what is really under the drone -- whether it ended up on the rug. Done
twice, with the navmesh built two ways:

    old   Habitat's default 0.20 m voxels
    new   the 0.02 m voxels in config/project.json

No brain and no window: whether the drone can roll somewhere is decided by the
World alone. A side that starts against furniture or a wall is skipped, since
that measures the furniture, not the rug -- the first version of this probe
drove into a dining chair and blamed the carpet.
"""

import csv
import json
import math
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import habitat_sim
import magnum as mn

from delta.sim import World, navmesh_settings_for, resolve_dataset_paths


STOPS = []
PROFILES = []
ROUTES = []
BLAME_NEW = []
GOAL = [0.0, 0.0]
NAMES = {}


def find_rugs(world):
    """(object ids of every rug, and their centre and size in Habitat x/z)."""
    root, _, scene_file, _ = resolve_dataset_paths(world.config)

    category, dims = {}, {}

    with open(root / "semantics" / "objects.csv") as f:
        for row in csv.DictReader(f):
            category[row["id"]] = row.get("main_category", "")

            if row.get("aligned.dims"):
                x, _, z = (float(v) for v in row["aligned.dims"].split(","))
                dims[row["id"]] = (x, z)

    manager = world.simulator.get_rigid_object_manager()

    with open(root / "semantics" / "objects.csv") as f:
        label = {row["id"]: (row.get("name") or row.get("main_category") or "?") for row in csv.DictReader(f)}

    for handle in manager.get_object_handles():
        NAMES[manager.get_object_by_handle(handle).object_id] = label.get(handle.split("_:")[0], "?")[:40]

    ids = {
        manager.get_object_by_handle(handle).object_id
        for handle in manager.get_object_handles()
        if category.get(handle.split("_:")[0]) == "carpet"
    }

    rugs = []

    for inst in json.loads(scene_file.read_text())["object_instances"]:
        if category.get(inst["template_name"]) != "carpet":
            continue

        # Turned a quarter turn, a rug's long side runs the other way. The
        # first version ignored rotation and tested one rug from 0.6 m too
        # far out on two sides and from on top of it on the other two.
        w, _, y, _ = inst.get("rotation", [1.0, 0.0, 0.0, 0.0])
        yaw = abs(math.degrees(2.0 * math.atan2(y, w))) % 180.0
        size = dims[inst["template_name"]]

        if 45.0 < yaw < 135.0:
            size = (size[1], size[0])

        rugs.append(((inst["translation"][0], inst["translation"][2]), size))

    return ids, rugs


def under(world, x, z):
    """The object id directly beneath (x, z), looking down from 1.2 m.

    Skips the drone's own body. Looking down from above the drone, the first
    thing the ray meets is the drone -- the first version of this check never
    saw a rug for exactly that reason, and blamed every rug in the house.
    """
    hit = world.simulator.cast_ray(
        habitat_sim.geo.Ray(mn.Vector3(x, 1.2, z), mn.Vector3(0.0, -1.0, 0.0)), 4.0
    )
    body = world.drone.object.object_id

    for each in hit.hits:
        if each.object_id != body:
            return each.object_id

    return None


def blocker(world, rug_ids):
    """What the drone is up against: the nearest thing ahead of it, looked
    for at its own height, or None if the way ahead is open."""
    drone = world.drone
    ahead = world.forward
    body = drone.object.object_id

    nearest = None

    for lift in (0.03, drone.radius, 2 * drone.radius - 0.02):
        origin = mn.Vector3(drone.floor_position.x, drone.floor_position.y + lift, drone.floor_position.z)
        hit = world.simulator.cast_ray(habitat_sim.geo.Ray(origin, ahead), 1.0)

        for each in hit.hits:
            if each.object_id == body:
                continue

            if nearest is None or each.ray_distance < nearest[0]:
                nearest = (each.ray_distance, each.object_id, lift)

            break

    # The floor ahead, every 4 cm: how high it is and what it is made of.
    profile = []
    for k in range(0, 9):
        point = drone.floor_position + ahead * (0.04 * k)
        hit = world.simulator.cast_ray(
            habitat_sim.geo.Ray(mn.Vector3(point.x, 1.2, point.z), mn.Vector3(0.0, -1.0, 0.0)), 4.0)
        top = next((h for h in hit.hits if h.object_id != body), None)
        if top is not None:
            what = "r" if top.object_id in rug_ids else ("s" if top.object_id == 0 else "o")
            profile.append(f"{1.2 - top.ray_distance:.3f}{what}")
    PROFILES.append(" ".join(profile))

    if nearest is None or nearest[0] > drone.radius + 0.10:
        return None

    distance, object_id, lift = nearest
    kind = "the rug's edge" if object_id in rug_ids else ("a wall" if object_id == 0 else NAMES.get(object_id, f"object {object_id}"))

    return f"{kind} {distance:.2f} m ahead at {lift:.2f} m up"


def attempt(world, rug_ids, start, heading, distance):
    """Roll from `start` along `heading`. Returns what is under the drone at
    the end, and how far it actually got."""
    pathfinder = world.pathfinder
    snapped = pathfinder.snap_point(mn.Vector3(start[0], 0.0, start[1]))

    if not all(v == v for v in snapped):
        return "no floor to start on", 0.0

    if (mn.Vector3(snapped) - mn.Vector3(start[0], snapped[1], start[1])).length() > 0.15:
        return "start is not open floor", 0.0

    world.stand(snapped, heading)
    origin = world.drone.floor_position

    reached_rug = False

    for _ in range(int(distance / 0.05)):
        world.walk(0.05)
        here = world.drone.floor_position
        reached_rug = reached_rug or under(world, here.x, here.z) in rug_ids

    travelled = (world.drone.floor_position - origin).length()

    if world.drone.flying:
        return "it flew", travelled

    if not reached_rug:
        STOPS.append(blocker(world, rug_ids))

        request = habitat_sim.ShortestPath()
        request.requested_start = world.drone.floor_position
        request.requested_end = pathfinder.snap_point(mn.Vector3(GOAL[0], 0.0, GOAL[1]))
        ROUTES.append(request.geodesic_distance if pathfinder.find_path(request) else None)

    return ("rolled onto the rug" if reached_rug else "stopped before the rug"), travelled


def survey(world, rug_ids, rugs, label):
    import math

    print(f"--- {label} ---")

    results = {"rolled onto the rug": 0, "stopped before the rug": 0}

    for (cx, cz), (sx, sz) in rugs:
        line = []
        why = []
        GOAL[0], GOAL[1] = cx, cz

        # Start 0.5 m outside the middle of each side, facing inward.
        for name, (dx, dz) in (("W", (-1, 0)), ("E", (1, 0)), ("N", (0, -1)), ("S", (0, 1))):
            half = sx / 2 if dx else sz / 2
            start = (cx + dx * (half + 0.5), cz + dz * (half + 0.5))

            # Facing inward: a yaw of t sends (0,0,-1) to (-sin t, 0, -cos t).
            heading = math.atan2(dx, dz)

            outcome, travelled = attempt(world, rug_ids, start, heading, 1.2)

            if outcome in results:
                results[outcome] += 1
                mark = "ok" if outcome.startswith("rolled") else "BLOCKED"
            else:
                mark = "skip"

            line.append(f"{name}:{mark}({travelled:.2f}m)" if mark != "skip" else f"{name}:skip")

            if mark == "BLOCKED":
                why.append(f"{name}: {STOPS[-1] or 'nothing solid ahead -- the navmesh ended'}")
                why.append(f"   floor ahead (m; r rug, s building, o object): {PROFILES[-1]}")
                route = ROUTES[-1]
                why.append("   the rug's middle by another route: " + (f"yes, {route:.1f} m" if route is not None else "NO ROUTE AT ALL"))

        print(f"  rug at ({cx:6.2f}, {cz:6.2f})   " + "  ".join(line))

        for reason in why:
            print(f"        {reason}")

    total = sum(results.values())
    print(f"  rolled onto a rug from {results['rolled onto the rug']} of {total} clear sides")
    print()

    return results["rolled onto the rug"], total


def main():
    world = World()
    rug_ids, rugs = find_rugs(world)

    print()
    print(f"{len(rugs)} rugs in the house")
    print()

    old = navmesh_settings_for(world.config["drone"])
    old.cell_size, old.cell_height = 0.05, 0.20

    world.simulator.recompute_navmesh(world.pathfinder, old)
    before = survey(world, rug_ids, rugs, "old navmesh, 0.20 m voxels")

    new = navmesh_settings_for(world.config["drone"])

    world.simulator.recompute_navmesh(world.pathfinder, new)
    del STOPS[:]
    after = survey(world, rug_ids, rugs, f"new navmesh, {new.cell_height:.2f} m voxels")
    BLAME_NEW.extend(STOPS)

    world.close()

    print(f"old: {before[0]}/{before[1]} sides    new: {after[0]}/{after[1]} sides")

    # Pass or fail on WHY a side was blocked, not on how many were. A wall or
    # a chair beside a rug is a real obstacle and the drone is right to stop.
    # The rug itself stopping it -- the navmesh ending at its edge, or its
    # edge being the thing in front -- is the bug this probe exists to catch.
    rug_caused = [
        reason for reason in BLAME_NEW
        if reason is None or "rug's edge" in reason
    ]

    print()

    if rug_caused:
        print(f"FAIL  {len(rug_caused)} side(s) blocked by the rug itself")
        return 1

    print("PASS  every rug can be rolled onto; the sides that cannot are blocked by furniture or walls")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
