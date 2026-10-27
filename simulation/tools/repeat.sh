#!/usr/bin/env bash
# Run the same scripted drive several times and summarize each.
#
#     bash tools/repeat.sh NAME COUNT [script.json] [extra run_drone.sh args...]
#     bash tools/repeat.sh guard 3
#     bash tools/repeat.sh perfect 2 tests/scripts/diagnose.json --pose oracle --depth oracle
#
# Why repeat: localization is not deterministic. Threads, frame timing and
# which frames RTAB-Map happens to skip differ between runs, and the same
# drive has ended 0.07 m off one time and 4.8 m off the next. A change judged
# on one run can look like a fix or a regression by luck alone.
#
# Headless. Results in outputs/repeat/NAME/, one line per run at the end.

set -uo pipefail

cd "$(dirname "$0")/.."

name="${1:?name}"; count="${2:?count}"; script="${3:-tests/scripts/diagnose.json}"
shift 3 2>/dev/null || shift $#

# Not while someone else is driving. Each run below deletes outputs/memory
# first, which is the map database of a run already in progress -- and that
# run's brain owns the port anyway, so these would all be refused after the
# damage was done. Checked once, up front, before anything is touched.
if pgrep -f "python -u -m delta.brain|python demo/run_drone.py" > /dev/null; then
    echo "DELTA is already running (a brain or a window) -- close it first. Nothing was changed:" >&2
    pgrep -af "python -u -m delta.brain|python demo/run_drone.py" >&2
    exit 1
fi

out="outputs/repeat/${name}"
rm -rf "${out}"
mkdir -p "${out}"

for i in $(seq 1 "${count}"); do
    rm -rf outputs/memory outputs/eval/live_map
    rm -f outputs/eval/live_map.jsonl

    bash run_drone.sh --headless --fresh "$@" --script "${script}" \
        --trace "${out}/run${i}_trace.jsonl" > "${out}/run${i}_sim.log" 2>&1

    cp outputs/eval/live_map.jsonl "${out}/run${i}_scores.jsonl" 2>/dev/null
    # Kept per run: the next run overwrites it.
    cp outputs/logs/brain.log "${out}/run${i}_brain.log" 2>/dev/null
    cp -r outputs/eval/live_map "${out}/run${i}_pictures" 2>/dev/null

    python3 tools/summarize_run.py "${out}/run${i}_trace.jsonl" \
        --scores "${out}/run${i}_scores.jsonl" > "${out}/run${i}_summary.txt" 2>&1
done

echo
echo "${name}: ${count} runs of ${script} $*"
echo "  run   median err   max err   final err   closures kept/rejected   phantom   unsafe"

for i in $(seq 1 "${count}"); do
    python3 - "${out}/run${i}_trace.jsonl" "${out}/run${i}_scores.jsonl" "${i}" <<'EOF'
import json, statistics, sys

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

trace = rows(sys.argv[1])
scores = rows(sys.argv[2]) if sys.argv[2] else []
errors = [r["error_m"] for r in trace if r.get("error_m") is not None]
last = trace[-1] if trace else {}
score = scores[-1] if scores else {}

print(f"  {sys.argv[3]:>3}   {statistics.median(errors) if errors else float('nan'):9.2f} m"
      f"  {max(errors) if errors else float('nan'):6.2f} m  {errors[-1] if errors else float('nan'):8.2f} m"
      f"          {last.get('loops')!s:>4} / {last.get('rejected')!s:<4}"
      f"            {score.get('phantom', float('nan')) * 100:5.1f}%  {score.get('unsafe', float('nan')) * 100:5.1f}%")
EOF
done
