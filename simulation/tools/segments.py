"""Localization, motion by motion: which movement costs the brain its position?

    python3 tools/segments.py outputs/repeat/motions/run*_trace.jsonl

Reads traces written with --trace from a script whose steps carry labels
({"label": "climb"}, see tests/scripts/motions.json), and for each label
says, per run and on average:

    lost      frames RTAB-Map could not track during it (blind)
    inliers   median and worst feature matches -- how well the cameras saw
    error     position error at its start and end, and the most it reached
    grew      how much worse the position got during it (end - start)
    turn err  heading error at its end

Ground truth here is only the examiner: it is what the error is measured
against, nothing more. Stdlib only, so it runs from any environment.
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


def segments(trace):
    """{label: [rows]} in order of first appearance."""
    out = {}
    for row in trace:
        if row.get("label") and row.get("error_m") is not None:
            out.setdefault(row["label"], []).append(row)
    return out


def describe(part, before):
    """Numbers for one label in one run. `before` is the row just before it."""
    first = before or part[0]
    last = part[-1]
    inliers = [r["inliers"] for r in part if r.get("inliers") is not None]

    return {
        "lost": (last.get("lost") or 0) - (first.get("lost") or 0),
        "inliers_med": statistics.median(inliers) if inliers else float("nan"),
        "inliers_min": min(inliers) if inliers else float("nan"),
        "start": first["error_m"],
        "end": last["error_m"],
        "max": max(r["error_m"] for r in part),
        "grew": last["error_m"] - first["error_m"],
        "deg": last.get("error_deg") or 0.0,
    }


def main(paths):
    runs = []
    order = []

    for path in paths:
        trace = [r for r in rows(path) if r.get("error_m") is not None]
        parts = segments(trace)
        found = {}

        for label, part in parts.items():
            index = trace.index(part[0])
            found[label] = describe(part, trace[index - 1] if index > 0 else None)

            if label not in order:
                order.append(label)

        runs.append(found)

    print(f"{len(runs)} run(s). Per motion, averaged over runs (worst run in brackets):\n")
    print(f"  {'motion':16} {'lost':>5} {'inliers med':>11} {'min':>5} "
          f"{'error start':>11} {'end':>6} {'max':>6} {'grew':>13} {'turn err':>9}")

    for label in order:
        found = [run[label] for run in runs if label in run]

        def mean(key):
            return statistics.mean(f[key] for f in found)

        worst = max(f["grew"] for f in found)

        print(f"  {label:16} {mean('lost'):5.1f} {mean('inliers_med'):11.0f} {mean('inliers_min'):5.0f} "
              f"{mean('start'):10.2f}m {mean('end'):5.2f}m {mean('max'):5.2f}m "
              f"{mean('grew'):+6.2f}m ({worst:+.2f}) {mean('deg'):8.1f}d")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        raise SystemExit(1)

    main(sys.argv[1:])
