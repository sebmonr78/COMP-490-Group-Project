#!/usr/bin/env bash
# Same route, two stereo baselines: does the 8 cm pair cost tracking?
#
#     bash tools/compare_baseline.sh [COUNT] [script.json]
#
# Runs tools/repeat.sh COUNT times at the baseline in config/rig.json (8 cm),
# then COUNT times with the pair 12 cm apart (the glasses' spacing), then puts
# rig.json back exactly as it was. Results in outputs/repeat/b08 and b12.
# Headless; one run at a time. Don't start DELTA while it runs.

set -uo pipefail

cd "$(dirname "$0")/.."

count="${1:-3}"
script="${2:-tests/scripts/diagnose.json}"

cp config/rig.json config/rig.json.keep
trap 'mv -f config/rig.json.keep config/rig.json' EXIT

bash tools/repeat.sh b08 "${count}" "${script}"

sed -i 's/"baseline": 0\.08,/"baseline": 0.12,/' config/rig.json
grep -q '"baseline": 0.12,' config/rig.json || { echo "could not set the 12 cm baseline" >&2; exit 1; }

bash tools/repeat.sh b12 "${count}" "${script}"
