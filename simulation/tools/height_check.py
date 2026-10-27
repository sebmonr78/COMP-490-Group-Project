"""How well does the brain know its height? Motion by motion, from traces.

    python3 tools/height_check.py outputs/repeat/height/run*_trace.jsonl

For each {"label": ...} of the script (tests/scripts/height.json):

    altitude err   the brain's altitude minus the true one: median, worst
    over things    frames with something under the drone (a true surface
                   more than 10 cm up), and how far the brain's idea of that
                   surface's height was from the truth

Ground truth is only the examiner here. Stdlib only.
"""

import json
import statistics
import sys


def rows(path):
    out = []
    for line in open(path, errors="replace"):
        line = line.replace("\x00", "").strip()
        if line:
            try:
                out.append(json.loads(line))
            except ValueError:
                pass
    return out


def main(paths):
    by_label = {}
    order = []

    for path in paths:
        for row in rows(path):
            label = row.get("label")

            if label is None or row.get("altitude_brain") is None or row.get("altitude_true") is None:
                continue

            if label not in by_label:
                by_label[label] = []
                order.append(label)

            by_label[label].append(row)

    print(f"{len(paths)} run(s).\n")
    print(f"  {'motion':22} {'frames':>6} {'altitude err med':>16} {'worst':>7}"
          f"   {'over things':>11} {'their height err':>16}")

    for label in order:
        part = by_label[label]
        errors = [r["altitude_brain"] - r["altitude_true"] for r in part]
        over = [r for r in part if (r.get("surface_true") or 0.0) > 0.10]
        surface = [abs((r.get("surface_brain") or 0.0) - r["surface_true"]) for r in over]

        print(f"  {label:22} {len(part):6} {statistics.median(errors):+15.3f}m {max(errors, key=abs):+6.3f}m"
              f"   {len(over):11} "
              + (f"{statistics.median(surface):15.3f}m" if surface else f"{'-':>16}"))


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        raise SystemExit(1)

    main(sys.argv[1:])
