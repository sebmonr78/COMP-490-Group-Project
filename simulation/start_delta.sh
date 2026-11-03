#!/usr/bin/env bash
# One command from a fresh clone to DELTA running.
#
#     bash start_delta.sh            the full brain: Depth Anything 3 + RTAB-Map
#     bash start_delta.sh --light    the sim's true depth instead of Depth
#                                    Anything (lighter on the GPU)
#     bash start_delta.sh --check    only look: install nothing, start nothing
#
# Anything else goes on to run_drone.sh (--replica later, --profile, ...).
#
# It checks every piece the README sets up, in the README's order, and for
# each one that is missing says what it is and how big, and asks before
# installing it. The installs are the README's own steps (sections 3-9) and
# the two setup scripts; nothing is downloaded without a yes.

set -uo pipefail
cd "$(dirname "$0")"

LIGHT=0
CHECK_ONLY=0
PASS_ON=()

for arg in "$@"; do
    case "${arg}" in
        --light) LIGHT=1 ;;
        --check) CHECK_ONLY=1 ;;
        *) PASS_ON+=("${arg}") ;;
    esac
done

ok()   { printf '  \033[32mOK\033[0m       %s\n' "$*"; }
bad()  { printf '  \033[31mMISSING\033[0m  %s\n' "$*"; }
warn() { printf '  \033[33mWARN\033[0m     %s\n' "$*"; }
say()  { printf '           %s\n' "$*"; }

# yes or no from the keyboard; always no with --check.
ask() {
    [ "${CHECK_ONLY}" = 1 ] && return 1
    local answer
    read -r -p "           $1 [y/N] " answer < /dev/tty
    [[ "${answer}" =~ ^[Yy] ]]
}

stop() {
    echo
    echo "Stopped: $*"
    echo "Run 'bash start_delta.sh' again once it is done; finished steps are skipped."
    exit 1
}

# A missing piece: with --check, note it and go on to list the rest.
NOT_READY=0
missing_piece() {
    if [ "${CHECK_ONLY}" = 1 ]; then
        NOT_READY=1
        return
    fi
    stop "$@"
}

echo
echo "DELTA: checking this machine"
echo

# ------------------------------------------------------------------ Linux
if [ "$(uname -s)" != "Linux" ]; then
    stop "this needs Linux. On Windows, install WSL2 first (README section 2)."
fi

if [ -e /dev/dxg ] || grep -qi microsoft /proc/version 2> /dev/null; then
    WSL=1
    ok "Linux (WSL2)"
else
    WSL=0
    ok "Linux"
fi

# ------------------------------------------------------------------ GPU
if command -v nvidia-smi > /dev/null && nvidia-smi > /dev/null 2>&1; then
    ok "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1)"
else
    warn "no NVIDIA GPU found. Depth Anything needs one; try --light."
fi

# ------------------------------------------------------------------ conda
if ! command -v conda > /dev/null && [ -x "${HOME}/miniconda3/bin/conda" ]; then
    export PATH="${HOME}/miniconda3/bin:${PATH}"
fi

if command -v conda > /dev/null; then
    ok "conda $(conda --version | cut -d' ' -f2)"
else
    bad "conda (Miniconda), which holds every environment below"
    say "About 150 MB download, a few minutes. README section 3."

    if ask "Download and install Miniconda into ~/miniconda3?"; then
        wget -O /tmp/miniconda.sh https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
            || stop "could not download Miniconda."
        bash /tmp/miniconda.sh -b -p "${HOME}/miniconda3" || stop "the Miniconda installer failed."
        "${HOME}/miniconda3/bin/conda" init bash > /dev/null
        export PATH="${HOME}/miniconda3/bin:${PATH}"
        ok "conda installed"
    else
        stop "conda is needed for everything else."
    fi
fi

eval "$(conda shell.bash hook)"

has_env() { conda env list | grep -qE "^$1[[:space:]]"; }

# ------------------------------------------------------------------ habitat
if has_env habitat; then
    versions=$(conda run -n habitat python -c "
import habitat_sim, habitat
print(habitat_sim.__version__, habitat.__version__, habitat_sim.built_with_bullet)" 2> /dev/null | tail -1)

    if [ "${versions}" = "0.3.3 0.3.3 True" ]; then
        ok "habitat env: habitat-sim 0.3.3 with Bullet, habitat-lab 0.3.3"
    else
        bad "habitat env exists but is not right: '${versions:-imports fail}'"
        say "Needed: habitat-sim 0.3.3 built with Bullet, habitat-lab 0.3.3 (README 5-6)."
        missing_piece "fix the habitat env, or remove it (conda env remove -n habitat -y) and re-run."
    fi
else
    bad "habitat env: the simulator (habitat-sim 0.3.3 + habitat-lab 0.3.3)"
    say "About 3 GB, 10-20 minutes. README sections 4-6."

    if ask "Create it now?"; then
        conda create -n habitat python=3.9 cmake=3.14.0 -y || stop "could not create the habitat env."
        conda install -n habitat habitat-sim withbullet -c conda-forge -c aihabitat -y \
            || stop "could not install habitat-sim."

        if [ ! -d "${HOME}/habitat-lab" ]; then
            git clone --branch stable https://github.com/facebookresearch/habitat-lab.git "${HOME}/habitat-lab" \
                || stop "could not clone habitat-lab."
        fi

        git -C "${HOME}/habitat-lab" checkout v0.3.3 || stop "could not check out habitat-lab v0.3.3."

        for package in habitat-lab habitat-baselines habitat-hitl; do
            conda run -n habitat --cwd "${HOME}/habitat-lab" pip install -e "${package}" \
                || stop "could not install ${package}."
        done

        ok "habitat env created"
    else
        missing_piece "the habitat env is the simulator itself."
    fi
fi

# What the window itself imports, on top of Habitat. Marked lines only:
# pygame prints a greeting of its own when imported.
missing=$(conda run -n habitat python -c "
import importlib
for module, package in [('pygame', 'pygame'), ('psutil', 'psutil'), ('cv2', 'opencv-python')]:
    try:
        importlib.import_module(module)
    except ImportError:
        print('NEED', package)" 2> /dev/null | sed -n 's/^NEED //p' | tr '\n' ' ')

if [ -z "${missing// }" ]; then
    ok "window packages: pygame, psutil, OpenCV"
else
    bad "window packages: ${missing}"
    say "Small, under 100 MB."

    if ask "Install them into the habitat env?"; then
        conda run -n habitat pip install ${missing} || stop "could not install ${missing}."
        ok "window packages installed"
    else
        missing_piece "the window cannot open without them."
    fi
fi

# ------------------------------------------------------------------ HSSD
DATA="${HABITAT_DATA_PATH:-${HOME}/habitat-data/versioned_data/hssd-hab}"
SCENE="${DATA}/scenes-uncluttered/102816036.scene_instance.json"

if [ -f "${SCENE}" ]; then
    ok "HSSD houses: ${DATA}"
else
    bad "HSSD houses (the 3D homes DELTA drives in), not at ${DATA}"
    say "About 17 GB, an hour or more. README section 7."
    say "First, once, in a browser (no script can do this for you):"
    say "  1. make an account at https://huggingface.co"
    say "  2. accept the terms at https://huggingface.co/datasets/hssd/hssd-hab"
    say "  3. make a read token: Settings > Access Tokens > New token"

    if [ "${CHECK_ONLY}" = 1 ]; then
        :
    elif ! conda run -n habitat huggingface-cli whoami > /dev/null 2>&1; then
        conda run -n habitat python -c "import huggingface_hub" 2> /dev/null \
            || conda run -n habitat pip install huggingface_hub > /dev/null

        if ask "Log in to Hugging Face now? (paste the token; the terminal shows nothing while you paste)"; then
            conda activate habitat
            huggingface-cli login || stop "the Hugging Face login failed."
            conda deactivate
        else
            missing_piece "HSSD needs a Hugging Face login."
        fi
    fi

    if ask "Download HSSD into ~/habitat-data now (17 GB)?"; then
        conda run -n habitat --no-capture-output python -m habitat_sim.utils.datasets_download \
            --uids hssd-hab --data-path "${HOME}/habitat-data" || stop "the HSSD download failed."
        DATA="${HOME}/habitat-data/versioned_data/hssd-hab"
        export HABITAT_DATA_PATH="${DATA}"
        [ -f "${DATA}/scenes-uncluttered/102816036.scene_instance.json" ] \
            || stop "HSSD downloaded, but house 102816036 is not in it."
        ok "HSSD downloaded"
    else
        missing_piece "DELTA has no house to drive in without HSSD."
    fi
fi

# ------------------------------------------------------------------ brain
if has_env brain; then
    ok "brain env (PyTorch, Depth Anything 3)"
else
    bad "brain env: the drone's own computer (PyTorch with CUDA, Depth Anything 3)"
    say "About 6 GB, 20-30 minutes. setup/setup_brain.sh"

    if ask "Build it now?"; then
        bash setup/setup_brain.sh || stop "setup/setup_brain.sh failed."
        ok "brain env built"
    else
        missing_piece "the brain env runs the mapping and RTAB-Map's side of it."
    fi
fi

# ------------------------------------------------------------------ RTAB-Map
if has_env rtabmap && [ -x build/rtabmap_bridge/rtabmap_bridge ]; then
    ok "RTAB-Map (where the drone thinks it is), bridge compiled"
else
    bad "RTAB-Map: the env and the small C++ bridge the brain talks to"
    say "About 2 GB, about 10 minutes. setup/setup_rtabmap.sh"

    if ask "Build it now?"; then
        bash setup/setup_rtabmap.sh || stop "setup/setup_rtabmap.sh failed."
        ok "RTAB-Map built"
    else
        missing_piece "the brain cannot tell where it is without RTAB-Map."
    fi
fi

# ------------------------------------------------------------------ WSL drawing
if [ "${WSL}" = 1 ]; then
    # Without this Mesa draws on the CPU and everything runs at a few frames
    # a second (README section 9). Set here for this run, so it works without
    # editing ~/.bashrc.
    export GALLIUM_DRIVER=d3d12 MESA_LOADER_DRIVER_OVERRIDE=d3d12
    ok "drawing on the GPU (GALLIUM_DRIVER=d3d12)"

    # WSLg sometimes starts without its shared memory and then copies every
    # frame to Windows instead ("[WARN:COPY MODE]" in the window's title).
    # Measured: keys took 0.7 s to show and a drag 1.6 s. It comes and goes
    # between WSL starts (microsoft/wslg#972).
    if grep -q "rdp_allocate_shared_memory: Failed" /mnt/wslg/weston.log 2> /dev/null; then
        warn "WSL's window display is in COPY MODE: DELTA will lag badly (keys ~0.7 s late)."
        say "Fix: close Linux windows, then in Windows PowerShell (not Ubuntu) run:"
        say "  wsl -d ${WSL_DISTRO_NAME:-Ubuntu} --system /bin/bash -c 'pgrep weston | xargs kill -9'"
        say "and run this script again. If it comes back: wsl --shutdown, then try again."

        if [ "${CHECK_ONLY}" = 0 ] && ! ask "Start anyway, lag and all?"; then
            missing_piece "fix the window display first."
        fi
    else
        ok "WSL window display: shared memory (no COPY MODE lag)"
    fi
fi

# ------------------------------------------------------------------ start
echo

if [ "${CHECK_ONLY}" = 1 ]; then
    if [ "${NOT_READY}" = 1 ]; then
        echo "Not ready: run 'bash start_delta.sh' to install what is MISSING (it asks first)."
    else
        echo "Ready. Nothing was installed or started; 'bash start_delta.sh' runs it."
    fi
    exit "${NOT_READY}"
fi

if [ "${LIGHT}" = 1 ]; then
    DEPTH=(--depth oracle)
    echo "Starting DELTA, light: the sim's true depth instead of Depth Anything."
else
    DEPTH=()
    echo "Starting DELTA with the full brain: Depth Anything 3 + RTAB-Map."
    echo "The first start downloads Depth Anything's weights (about 1.4 GB) and builds"
    echo "the drone's navmesh, so it is slow once."
fi

echo
echo "  Up/Down roll   Left/Right turn   1 take off   0 land   Shift faster"
echo "  drag/scroll the 3D map to turn/zoom it   [ ] cut the roof off   M true layout"
echo "  Esc quits and saves the map"
echo

exec bash run_drone.sh --fresh "${DEPTH[@]}" "${PASS_ON[@]}"
