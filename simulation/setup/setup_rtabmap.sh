#!/usr/bin/env bash
# Builds the `rtabmap` conda env and compiles the bridge the brain talks to
# ("Estimate wearer position" on the slide).
#
#     bash setup/setup_rtabmap.sh
#
# RTAB-Map is a C++ library with no Python bindings, and the usual way in is
# ROS. Instead this takes RoboStack's conda build of the library (no sudo, no
# system ROS) and compiles one small program against it:
# spatial_agent/brain/rtabmap_bridge/bridge.cpp. The brain starts it as a
# subprocess, writes stereo frames to its stdin and reads poses back.
#
# Safe to re-run: if the env exists, only the bridge is rebuilt.
# To start over:
#
#     conda env remove -n rtabmap -y && bash setup/setup_rtabmap.sh

set -euo pipefail

ENV_NAME="rtabmap"
RTABMAP="0.22.1"

REPO="$(cd "$(dirname "$0")/.." && pwd)"
SOURCE="${REPO}/delta/brain/rtabmap_bridge"
BUILD="${REPO}/build/rtabmap_bridge"


if ! command -v conda > /dev/null; then
    echo "conda not found. Install Miniconda first (README section 3)." >&2
    exit 1
fi

eval "$(conda shell.bash hook)"

if conda env list | grep -qE "^${ENV_NAME}[[:space:]]"; then
    echo "==> The '${ENV_NAME}' env exists; rebuilding the bridge only"
else
    echo "==> Creating the '${ENV_NAME}' env (RTAB-Map ${RTABMAP})"

    # robostack-staging first: it is where RTAB-Map lives, and with strict
    # channel priority conda then takes everything else from conda-forge.
    conda create -n "${ENV_NAME}" \
        -c robostack-staging -c conda-forge --override-channels \
        "ros-humble-rtabmap=${RTABMAP}" \
        cxx-compiler cmake make pkg-config -y
fi

# RoboStack's activation scripts read variables that may be unset, which
# `set -u` would turn into a hard failure.
set +u
conda activate "${ENV_NAME}"
set -u

echo "==> Compiling the bridge"

cmake -S "${SOURCE}" -B "${BUILD}" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_PREFIX_PATH="${CONDA_PREFIX}"

cmake --build "${BUILD}" -j "$(nproc)"

echo "==> Checking it starts"

"${BUILD}/rtabmap_bridge" --version

echo
echo "Done. The brain finds the bridge at build/rtabmap_bridge/rtabmap_bridge."
