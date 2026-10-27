#!/usr/bin/env bash
# Start DELTA: the brain and the sim window, each in its own conda env.
#
#     bash run_drone.sh
#     bash run_drone.sh --pose oracle    # Habitat's true pose instead of RTAB-Map
#     bash run_drone.sh --depth oracle   # Habitat's true depth instead of DA3
#     bash run_drone.sh --fresh          # forget last session's map
#     bash run_drone.sh --no-brain       # just the sim, nothing to talk to
#
# Two processes, because they need two different Pythons:
#
#     the brain       conda env `brain`    (3.10, torch, Depth Anything 3)
#                     log: outputs/logs/brain.log
#     the sim window  conda env `habitat`  (3.9, habitat-sim)
#
# They talk over a local TCP socket -- which is also how the real drone will
# reach the laptop. Every option goes to the window (demo/run_drone.py, code
# in delta/window/); the brain is told what to load by the window when it
# connects.
#
# To watch the brain think, in another terminal:
#
#     tail -f outputs/logs/brain.log
#
# RTAB-Map needs its bridge compiled once:
#
#     bash setup/setup_rtabmap.sh

set -euo pipefail

cd "$(dirname "$0")"
mkdir -p outputs/logs

# Rendering on the GPU comes from ~/.bashrc (GALLIUM_DRIVER=d3d12), exactly as
# for spatial-agent-12, whose launcher sets nothing either. Kept that way on
# purpose: the two should start identically, so that any difference in how they
# behave is a difference in the code and not in how they were started.

eval "$(conda shell.bash hook)"

BRAIN_PID=""

# One DELTA at a time. Two runs at once share one port and one map database:
# the second brain cannot bind the port and dies, the second window connects
# to nobody and shows the house with no map and no localization -- and says
# nothing about why. Refuse up front instead, and say what is running.
if [[ " $* " != *" --no-brain "* ]]; then
    if pgrep -f "python -u -m delta.brain" > /dev/null; then
        echo "Another DELTA brain is already running:" >&2
        pgrep -af "python -u -m delta.brain" >&2
        echo "Close that run first (its window, or Ctrl+C in its terminal)." >&2
        exit 1
    fi

    # Each run its own log, so a second run cannot overwrite the first's --
    # a brain that died in one run was once unexplainable because a test run
    # straight after had written over its log. brain.log always points at the
    # newest.
    LOG="outputs/logs/brain-$(date +%Y%m%d-%H%M%S).log"
    ln -sf "$(basename "${LOG}")" outputs/logs/brain.log

    # In its own process group, so that the cleanup below can stop exactly
    # this brain and everything it started -- the RTAB-Map bridge included --
    # without touching anyone else's.
    setsid conda run -n brain --no-capture-output \
        python -u -m delta.brain > "${LOG}" 2>&1 &
    BRAIN_PID=$!
fi

finish() {
    if [ -z "${BRAIN_PID}" ]; then
        return
    fi

    # The brain saves its map when the sim disconnects; give it time.
    for _ in $(seq 1 60); do
        kill -0 "${BRAIN_PID}" 2> /dev/null || break
        sleep 0.5
    done

    # Then stop this run's whole process group: conda run, the brain, and
    # the bridge it started. Only this run's -- an earlier version stopped
    # every DELTA brain on the machine, including someone else's.
    kill -TERM -- "-${BRAIN_PID}" 2> /dev/null || true
}

trap finish EXIT

# Habitat's and RoboStack's activation scripts read unset variables.
set +u
conda activate habitat
set -u

python demo/run_drone.py "$@"
