#!/usr/bin/env bash
# Start DELTA: the brain and the sim window, each in its own conda env.
#
#     bash run_drone.sh
#     bash run_drone.sh --pose oracle    # Habitat's true pose instead of RTAB-Map
#     bash run_drone.sh --depth oracle   # Habitat's true depth instead of DA3
#     bash run_drone.sh --fresh          # forget last session's map
#     bash run_drone.sh --no-brain       # just the sim, nothing to talk to
#     bash run_drone.sh --run NAME       # a run of its own, next to others
#
# Two processes, because they need two different Pythons:
#
#     the brain       conda env `brain`    (3.10, torch, Depth Anything 3)
#     the sim window  conda env `habitat`  (3.9, habitat-sim)
#
# They talk over a local TCP socket -- which is also how the real drone will
# reach the laptop. Every option except --run goes to the window
# (demo/run_drone.py, code in delta/window/); the brain is told what to load
# by the window when it connects.
#
# Several at once. Each DELTA needs its own socket port and its own map
# folder: two brains cannot listen on one port, and two writing one map
# would wreck it. So:
#
#     without --run   the map is outputs/memory, the log outputs/logs/,
#                     and the port the first free one from brain.json's
#                     (47569) up to 47599 -- as before, for one person at a desk
#     --run NAME      the map is outputs/runs/NAME/memory and the log
#                     outputs/runs/NAME/brain.log
#     --port N        this port instead of the first free one (the website
#                     gives each of its runs its own)
#
# A second run on a map that is already in use is refused, and says which
# --run to add. To watch a brain think, in another terminal:
#
#     tail -f outputs/logs/brain.log            (or outputs/runs/NAME/brain.log)
#
# RTAB-Map needs its bridge compiled once:
#
#     bash setup/setup_rtabmap.sh

set -euo pipefail

cd "$(dirname "$0")"

# Rendering on the GPU comes from ~/.bashrc (GALLIUM_DRIVER=d3d12), exactly as
# for spatial-agent-12, whose launcher sets nothing either. Kept that way on
# purpose: the two should start identically, so that any difference in how they
# behave is a difference in the code and not in how they were started.

# --- Options this script keeps for itself, and the port ----------------------

RUN_NAME="${DELTA_RUN:-}"
PORT=""
NO_BRAIN=0
WINDOW_ARGS=()

while [ $# -gt 0 ]; do
    case "$1" in
        --run)       RUN_NAME="${2:?--run needs a name}"; shift 2 ;;
        --run=*)     RUN_NAME="${1#--run=}"; shift ;;
        --port)      PORT="${2:?--port needs a number}"; shift 2 ;;
        --port=*)    PORT="${1#--port=}"; shift ;;
        --no-brain)  NO_BRAIN=1; WINDOW_ARGS+=("$1"); shift ;;
        *)           WINDOW_ARGS+=("$1"); shift ;;
    esac
done

if [ -n "${RUN_NAME}" ]; then
    if [[ ! "${RUN_NAME}" =~ ^[A-Za-z0-9._-]+$ ]]; then
        echo "--run takes letters, digits, '.', '_' and '-' only: ${RUN_NAME}" >&2
        exit 2
    fi
    RUN_DIR="outputs/runs/${RUN_NAME}"
    MEMORY="${RUN_DIR}/memory"
    LOG_DIR="${RUN_DIR}"
else
    MEMORY="outputs/memory"
    LOG_DIR="outputs/logs"
fi
mkdir -p "${MEMORY}" "${LOG_DIR}"

# A port is free if it can be bound the way the brain binds it (with
# SO_REUSEADDR, so a just-finished run's port is free again at once). Not "if
# nothing answers": a brain takes the first connection it gets as its window,
# so knocking on another run's port to see if anyone is home would end it.
port_free() {
    python3 -c 'import socket, sys
s = socket.socket()
s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
try:
    s.bind(("127.0.0.1", int(sys.argv[1])))
except OSError:
    sys.exit(1)' "$1"
}

# --- One run per map ----------------------------------------------------------
#
# Two brains on one map folder would each overwrite the other's map and
# RTAB-Map database. The lock is held for as long as this script -- and the
# brain, which inherits it -- is running, and is let go of by the system
# however they end, so a crash never leaves a stale lock behind.

if [ "${NO_BRAIN}" = 0 ]; then
    exec 9> "${MEMORY}/.in-use"
    if ! flock -n 9; then
        echo "Another DELTA is already using the map in ${MEMORY}." >&2
        if [ -z "${RUN_NAME}" ]; then
            echo "To run a second one beside it: bash run_drone.sh --run NAME ..." >&2
        else
            echo "Close that run first, or pick another name for --run." >&2
        fi
        exit 1
    fi
fi

eval "$(conda shell.bash hook)"

BRAIN_PID=""

if [ "${NO_BRAIN}" = 0 ]; then
    # Choosing a port and the brain taking it happen under one lock, so two
    # runs started together cannot both pick the same free port. Without
    # --port: brain.json's port (47569) or the next free one up to 47599.
    # (The website gives its runs ports from 47600 up, out of this range.)
    exec 8> outputs/.choosing-port
    flock 8

    if [ -z "${PORT}" ]; then
        BASE_PORT="$(sed -n 's/.*"port": *\([0-9][0-9]*\).*/\1/p' config/brain.json | head -n 1)"
        PORT="${BASE_PORT:-47569}"
        while ! port_free "${PORT}"; do
            PORT=$((PORT + 1))
            if [ "${PORT}" -gt 47599 ]; then
                echo "No free port for the brain between ${BASE_PORT:-47569} and 47599." >&2
                exit 1
            fi
        done
    elif ! port_free "${PORT}"; then
        echo "Port ${PORT} is already in use (another DELTA, or something else)." >&2
        exit 1
    fi

    # Each run its own log, so a second run cannot overwrite the first's --
    # a brain that died in one run was once unexplainable because a test run
    # straight after had written over its log. brain.log always points at the
    # newest in its folder.
    LOG="${LOG_DIR}/brain-$(date +%Y%m%d-%H%M%S)-$$.log"
    ln -sf "$(basename "${LOG}")" "${LOG_DIR}/brain.log"
    export DELTA_BRAIN_LOG="${LOG_DIR}/brain.log"

    echo "DELTA run: ${RUN_NAME:-(default)}  port ${PORT}  map ${MEMORY}  log ${LOG}"

    # In its own process group, so that the cleanup below can stop exactly
    # this brain and everything it started -- the RTAB-Map bridge included --
    # without touching anyone else's. 8>&-: the brain must not keep the
    # port lock (it does keep the map lock, 9, on purpose).
    setsid conda run -n brain --no-capture-output \
        python -u -m delta.brain --port "${PORT}" --memory "${MEMORY}" > "${LOG}" 2>&1 8>&- &
    BRAIN_PID=$!

    # Let the next run choose once this brain holds its port (or is gone).
    for _ in $(seq 1 240); do
        port_free "${PORT}" || break
        kill -0 "${BRAIN_PID}" 2> /dev/null || break
        sleep 0.5
    done
    exec 8>&-
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
    # the bridge it started. Only this run's.
    kill -TERM -- "-${BRAIN_PID}" 2> /dev/null || true
}

trap finish EXIT

# Habitat's and RoboStack's activation scripts read unset variables.
set +u
conda activate habitat
set -u

python demo/run_drone.py --port "${PORT}" ${WINDOW_ARGS[@]+"${WINDOW_ARGS[@]}"}
