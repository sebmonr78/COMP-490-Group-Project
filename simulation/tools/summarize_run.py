"""Summarize one scripted run: how well the drone knew where it was, and how
good the map it built is.

    python tools/summarize_run.py outputs/eval/check.jsonl

Reads the --trace file (four lines a second: the true position, the brain's
pose error, RTAB-Map's health) and the map scores the script wrote to
outputs/eval/live_map.jsonl, and prints the numbers every change to the
mapping or localization is judged by.

Every number here is measured against Habitat's truth AFTER the fact. None of
it is ever fed back to the brain -- this file is the examiner, not a hint.

Standard library only, on purpose, so it runs in whatever env the terminal
happens to be in -- including (base), which has no numpy.
"""

import argparse
import json
import statistics
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_lines(path):
    """Every line that parses, and how many did not.

    A run that ended in a freeze or a power cut leaves its last line half
    written and padded with null bytes. That is precisely the run you most
    want a summary of, so a damaged line is skipped and counted, not fatal.
    """
    path = Path(path)

    if not path.is_file():
        return [], 0

    rows, damaged = [], 0

    for line in path.read_text(errors="replace").splitlines():
        line = line.replace("\x00", "").strip()

        if not line:
            continue

        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            damaged += 1

    return rows, damaged


def percentile(values, share):
    """The value below which `share` percent of the values fall."""
    ordered = sorted(values)
    index = round(share / 100.0 * (len(ordered) - 1))

    return ordered[min(len(ordered) - 1, max(0, index))]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("trace", help="the file passed to --trace")
    parser.add_argument(
        "--scores",
        default=str(PROJECT_ROOT / "outputs" / "eval" / "live_map.jsonl"),
    )
    parser.add_argument("--last", type=int, default=2, help="how many map scores to show")
    args = parser.parse_args()

    trace, damaged = load_lines(args.trace)

    if not trace:
        print(f"No readable trace in {args.trace}")
        return 1

    errors = [row["error_m"] for row in trace if row.get("error_m") is not None]
    headings = [row["error_deg"] for row in trace if row.get("error_deg") is not None]

    last = trace[-1]

    print()
    print("Localization")
    print(f"  duration            {last['t']:.0f} s, {len(trace)} trace lines")

    if damaged:
        print(f"  damaged lines       {damaged} skipped (a run that did not end cleanly)")

    print(f"  travelled           {last['walked']:.1f} m")

    if errors:
        walked = max(float(last["walked"]), 1e-6)

        print(
            f"  position error      median {statistics.median(errors):.2f} m   "
            f"p90 {percentile(errors, 90):.2f} m   max {max(errors):.2f} m   "
            f"final {errors[-1]:.2f} m"
        )
        print(f"  final error         {errors[-1] / walked * 100:.1f}% of distance travelled")

    if headings:
        print(
            f"  heading error       median {statistics.median(headings):.1f} deg   "
            f"max {max(headings):.1f} deg"
        )

    print(
        f"  RTAB-Map            lost {last.get('lost', 0)} frames, "
        f"loops {last.get('loops', 0)} kept / {last.get('rejected', 0)} rejected"
    )

    altitudes = [
        (row["altitude_true"], row["altitude_brain"])
        for row in trace
        if row.get("altitude_true") is not None and row.get("altitude_brain") is not None
    ]

    if altitudes and max(true for true, _ in altitudes) > 0.05:
        gaps = [abs(true - brain) for true, brain in altitudes]

        print(
            f"  altitude error      median {statistics.median(gaps):.2f} m   "
            f"max {max(gaps):.2f} m   (highest true altitude "
            f"{max(true for true, _ in altitudes):.2f} m)"
        )

    scores, _ = load_lines(args.scores)
    scores = scores[-args.last:]

    if scores:
        print()
        print("Map (against the drone's navmesh)")
        print(f"  {'when':<24}{'coverage':>9}{'phantom':>9}{'unsafe':>8}{'free m2':>9}{'pose err':>10}")

        for row in scores:
            pose = row.get("pose_error_m")
            pose_text = f"{pose:.2f} m" if pose is not None else "-"

            print(
                f"  {row['name']:<24}"
                f"{row['coverage'] * 100:>8.1f}%"
                f"{row['phantom'] * 100:>8.1f}%"
                f"{row['unsafe'] * 100:>7.1f}%"
                f"{row['free_m2']:>9.1f}"
                f"{pose_text:>10}"
            )

        print()
        print("  coverage  how much of the true floor the map has as free")
        print("  phantom   clear floor the map calls solid (routes go round nothing)")
        print("  unsafe    floor the map calls free that is really inside something")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
