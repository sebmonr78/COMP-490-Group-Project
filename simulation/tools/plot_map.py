"""Save a saved map as a picture you can look at.

    conda activate habitat
    cd ~/COMP-490-Group-Project/simulation
    python tools/plot_map.py data/maps/first.npz

Writes a PNG beside the map. Colour says what the drone can do in each
square, which is the only question the map is really for:

    dark      never seen
    grey      plain floor -- roll across it
    green     something overhead, and room to roll UNDER it
    blue      something in the way, and room to fly OVER it
    slate     the air is clear but the ground was never seen -- flyable,
              but the drone does not know it could land there
    orange    in the way with no gap either side, darkening with height

`--column ROW COL` prints one vertical column as text instead, layer by
layer, which is how to check a square the picture makes you suspicious of.
"""

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

from delta.gap_map import FREE, SOLID, UNKNOWN, GapMap


def crop_to_content(image, seen, margin=8):
    """Trim the empty border off. A 40 m map of a 6 m drive is mostly nothing."""
    rows, cols = np.nonzero(seen)

    if len(rows) == 0:
        return image

    top = max(0, rows.min() - margin)
    bottom = min(image.shape[0], rows.max() + margin + 1)
    left = max(0, cols.min() - margin)
    right = min(image.shape[1], cols.max() + margin + 1)

    return image[top:bottom, left:right]


def print_column(gap_map, row, col, diameter):
    """One square, read from the ceiling down."""
    grid = gap_map.states()
    fits = gap_map.passable(diameter, grid)

    words = {UNKNOWN: "unknown", FREE: "free   ", SOLID: "SOLID  "}

    print()
    print(f"Column at row {row}, column {col}")

    across, along = gap_map.to_world(row, col)
    print(f"  {across:+.2f} m across, {along:+.2f} m along, "
          f"in the drone's start frame")

    print(f"  ground layer  {int(gap_map.ground_level(grid)[row, col])}")
    print()
    print("  layer   height      state     drone fits here")

    for level in range(gap_map.levels - 1, -1, -1):
        height = level * gap_map.layer

        print(
            f"   {level:3d}    {height:4.2f} m   "
            f"{words[int(grid[row, col, level])]}"
            f"   {'yes' if fits[row, col, level] else ''}"
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("map", help="a .npz written by run_mapping.py --save")
    parser.add_argument("--out", default=None, help="where to write the PNG")
    parser.add_argument("--diameter", type=float, default=0.30)
    parser.add_argument("--scale", type=int, default=3,
                        help="pixels per grid square")
    parser.add_argument("--column", nargs=2, type=int, metavar=("ROW", "COL"),
                        help="print one column as text instead of drawing")
    args = parser.parse_args()

    path = Path(args.map)

    if not path.is_absolute():
        path = PROJECT_ROOT / path

    gap_map = GapMap.load(path)

    numbers = gap_map.coverage()

    print(f"Map from {path.name}")
    print(f"  built from {numbers['frames']} frames")
    print(f"  {numbers['columns_seen']} squares seen "
          f"({numbers['floor_area_m2']:.1f} m2)")
    print(f"  grid {gap_map.n} x {gap_map.n} squares of {gap_map.cell:.2f} m, "
          f"{gap_map.levels} layers of {gap_map.layer:.2f} m")

    if args.column:
        print_column(gap_map, args.column[0], args.column[1], args.diameter)
        return 0

    import cv2

    image = gap_map.picture(args.diameter)
    seen = (gap_map.states() != UNKNOWN).any(axis=2)

    image = crop_to_content(image, seen)

    if args.scale > 1:
        image = np.repeat(np.repeat(image, args.scale, axis=0), args.scale, axis=1)

    out = Path(args.out) if args.out else path.with_suffix(".png")

    if not out.is_absolute():
        out = PROJECT_ROOT / out

    # The picture is RGB; OpenCV writes BGR.
    cv2.imwrite(str(out), image[:, :, ::-1])

    print()
    print(f"Wrote {out}  ({image.shape[1]} x {image.shape[0]} pixels)")
    print()
    print("  dark   never seen      grey   roll across")
    print("  green  roll UNDER      blue   fly OVER")
    print("  slate  air is clear, but the ground was never seen")
    print("  orange no way past, darkening with height")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
