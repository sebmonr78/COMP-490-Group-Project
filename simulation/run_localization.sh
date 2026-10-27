#!/usr/bin/env bash
# DELTA with only LOCALIZATION real -- for checking that one thing.
#
#     bash run_localization.sh
#
# The brain works out where the drone is from its own sensors: the stereo
# cameras (RTAB-Map), the IMU's heading, and the cage's axle encoder. Depth is
# Habitat's truth, so the map is drawn from perfect depth at the brain's
# position: walls drawn twice, or out of place against the grey truth, are
# localization errors and nothing else.
#
# Drive it yourself: arrow keys roll and turn, 0 and 1 fly and land.
# Everything else is the same as run_drone.sh, and options pass through:
#
#     bash run_localization.sh --no-odometer   # without the axle encoder
#     bash run_localization.sh --trace outputs/my_drive.jsonl
#     python3 tools/segments.py outputs/my_drive.jsonl

exec bash "$(dirname "$0")/run_drone.sh" --fresh --depth oracle "$@"
