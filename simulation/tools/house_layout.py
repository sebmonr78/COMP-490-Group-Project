"""Draw where the drone can roll, which parts are indoors, and a metre grid.

    python tools/house_layout.py

Writes outputs/house_layout.png: the drone's navmesh seen from above, north up,
with indoor floor (a ceiling within 5 m overhead) and outdoor floor in
different colours, each navmesh island outlined, a grid line every 2 m labelled
in Habitat x and z, and the start marked. For planning test routes -- which
corridors are narrow, where the doors to outside are -- and nothing else.
Sim side only.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import habitat_sim
import magnum as mn
import numpy as np
from habitat.utils.visualizations import maps

from delta.sim import World

SCALE = 4            # output pixels per 5 cm cell
RES = 0.05


def main():
    world = World()
    pathfinder = world.pathfinder
    sim = world.simulator

    floor_y = float(world.drone.floor_position.y)
    navigable = maps.get_topdown_map(pathfinder, floor_y, meters_per_pixel=RES, draw_border=False) == 1
    lower, upper = pathfinder.get_bounds()

    rows, cols = navigable.shape
    image = np.full((rows, cols, 3), 25, np.uint8)

    # Indoors = something within 5 m overhead. Sampled every 3rd cell.
    roofed = np.zeros_like(navigable)
    rr, cc = np.nonzero(navigable)

    for r, c in zip(rr[::3], cc[::3]):
        x, z = lower.x + (c + 0.5) * RES, lower.z + (r + 0.5) * RES
        hit = sim.cast_ray(habitat_sim.geo.Ray(mn.Vector3(x, floor_y + 0.2, z), mn.Vector3(0, 1, 0)), 5.0)
        roofed[max(0, r - 1):r + 2, max(0, c - 1):c + 2] = hit.has_hits()

    image[navigable & roofed] = (200, 170, 120)       # indoor floor, blue-grey (BGR)
    image[navigable & ~roofed] = (90, 160, 90)        # outdoor floor, green

    # Narrow places: navigable cells whose clearance to the edge is small.
    clearance = cv2.distanceTransform(navigable.astype(np.uint8), cv2.DIST_L2, 5) * RES
    narrow = navigable & (clearance < 0.12)
    image[narrow] = (60, 200, 255)                    # amber: tight squeeze

    big = cv2.resize(image, (cols * SCALE, rows * SCALE), interpolation=cv2.INTER_NEAREST)

    # Metre grid every 2 m, labelled in Habitat coordinates.
    for x in np.arange(np.ceil(lower.x / 2) * 2, upper.x, 2.0):
        px = int((x - lower.x) / RES * SCALE)
        cv2.line(big, (px, 0), (px, rows * SCALE), (70, 70, 70), 1)
        cv2.putText(big, f"x{x:.0f}", (px + 2, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (220, 220, 220), 1)

    for z in np.arange(np.ceil(lower.z / 2) * 2, upper.z, 2.0):
        pz = int((z - lower.z) / RES * SCALE)
        cv2.line(big, (0, pz), (cols * SCALE, pz), (70, 70, 70), 1)
        cv2.putText(big, f"z{z:.0f}", (2, pz - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (220, 220, 220), 1)

    start = world.drone.floor_position
    sx, sz = int((start.x - lower.x) / RES * SCALE), int((start.z - lower.z) / RES * SCALE)
    cv2.circle(big, (sx, sz), 9, (0, 0, 255), -1)
    cv2.putText(big, "start", (sx + 10, sz), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

    out = PROJECT_ROOT / "outputs" / "house_layout.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), big)

    print()
    print(f"bounds x {lower.x:.1f} .. {upper.x:.1f}, z {lower.z:.1f} .. {upper.z:.1f}")
    print(f"start at x {start.x:.2f}, z {start.z:.2f}")
    print(f"islands: {pathfinder.num_islands}")

    for i in sorted(range(pathfinder.num_islands), key=pathfinder.island_area, reverse=True)[:6]:
        print(f"  island {i}: {pathfinder.island_area(i):7.1f} m2")

    print(f"wrote {out}")
    world.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
