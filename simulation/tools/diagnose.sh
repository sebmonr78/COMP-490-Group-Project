#!/usr/bin/env bash
# Which half is failing, mapping or localization? Run the same drive four
# ways, swapping the truth in for one stage at a time, and compare.
#
#     bash tools/diagnose.sh
#
#     A   RTAB-Map pose   Depth Anything depth    what you normally get
#     B   TRUE pose       Depth Anything depth    still bad -> the depth
#     C   RTAB-Map pose   TRUE depth              still bad -> localization
#     D   TRUE pose       TRUE depth              the best the map code can do
#
# Spatial-agent-12 built the two oracle switches for exactly this question.
# Using the truth here is not cheating: it is the examiner, swapped in one
# stage at a time and only to find out which stage to fix.
#
# Headless, one run after another. Results land in outputs/diagnose/.

set -uo pipefail

cd "$(dirname "$0")/.."

out=outputs/diagnose
rm -rf "${out}"
mkdir -p "${out}"

run() {
    local name="$1"; shift

    echo "=== ${name}: $* ==="

    rm -rf outputs/memory
    rm -f outputs/eval/live_map.jsonl
    rm -rf outputs/eval/live_map

    bash run_drone.sh --headless --fresh "$@" \
        --script tests/scripts/diagnose.json \
        --trace "${out}/${name}_trace.jsonl" > "${out}/${name}_sim.log" 2>&1

    cp outputs/eval/live_map.jsonl "${out}/${name}_scores.jsonl" 2>/dev/null
    cp -r outputs/eval/live_map "${out}/${name}_pictures" 2>/dev/null

    python3 tools/summarize_run.py "${out}/${name}_trace.jsonl" \
        --scores "${out}/${name}_scores.jsonl" > "${out}/${name}_summary.txt" 2>&1

    cat "${out}/${name}_summary.txt"
    echo
}

run A_normal
run B_true_pose  --pose oracle
run C_true_depth --depth oracle
run D_both_true  --pose oracle --depth oracle
