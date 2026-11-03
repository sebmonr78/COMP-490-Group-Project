#!/usr/bin/env bash
# Checks that everything the brain needs is installed, and installs whatever
# is missing or the wrong version. Safe to run as often as you like: what is
# already right is left alone.
#
#     bash setup/check_brain.sh            check, then install what's missing
#     bash setup/check_brain.sh --check    only check and report, install nothing
#
# What it checks, in order:
#
#     1. conda                         can't be installed from here: says how
#     2. the `brain` conda env         Python 3.10, made if it doesn't exist
#     3. its Python packages           the same pinned versions as setup_brain.sh
#     4. that each one imports         a package can be installed and still broken
#     5. the GPU                       torch must see CUDA (driver 570+); reported only
#     6. the model files               DA3, YOLOE, the Piper voice
#     7. RTAB-Map                      the `rtabmap` env and the compiled bridge,
#                                      built by setup/setup_rtabmap.sh if missing
#
# setup_brain.sh builds the env from nothing and stops if it already exists.
# This script is for an env that exists but may be incomplete or out of date.
# The versions below must match setup_brain.sh; if you change one there,
# change it here too.
#
# No `set -e` on purpose: one thing failing must not stop the rest from being
# checked. Every step reports what it found, and the summary at the end lists
# everything that is still not right.

set -uo pipefail

# ---- What the brain needs (keep in step with setup_brain.sh) ---------------

ENV_NAME="brain"
PYTHON_VERSION="3.10"

TORCH="2.10.0"
TORCHVISION="0.25.0"
TORCH_INDEX="https://download.pytorch.org/whl/cu128"
XFORMERS="0.0.35"

DA3_COMMIT="3d835ec1a5802d64a8b8b15f817a1ab54809bfe4"
DA3_URL="git+https://github.com/ByteDance-Seed/Depth-Anything-3@${DA3_COMMIT}"
DA3_MODEL="depth-anything/DA3-LARGE-1.1"

ULTRALYTICS="8.4.177"
YOLOE_MODEL="yoloe-11l-seg.pt"

PIPER="1.8.0"
PIPER_VOICE="en_US-lessac-medium"

RTABMAP_ENV="rtabmap"

REPO="$(cd "$(dirname "$0")/.." && pwd)"
WEIGHTS="${REPO}/weights"
BRIDGE="${REPO}/build/rtabmap_bridge/rtabmap_bridge"

# --check: report only
CHECK_ONLY=0
if [ "${1:-}" = "--check" ]; then
    CHECK_ONLY=1
fi

# ---- Output helpers ---------------------------------------------------------

PROBLEMS=()     # everything still not right at the end, for the summary
INSTALLED=()    # everything this run installed or fixed

ok()      { printf '  [ ok ]      %s\n' "$1"; }
missing() { printf '  [missing]   %s\n' "$1"; }
wrong()   { printf '  [wrong]     %s\n' "$1"; }
fixing()  { printf '  [install]   %s\n' "$1"; }
failed()  { printf '  [FAILED]    %s\n' "$1"; PROBLEMS+=("$1"); }
section() { printf '\n==> %s\n' "$1"; }

# In --check mode, note the problem instead of fixing it
cannot_fix_in_check_mode() {
    PROBLEMS+=("$1")
}

# Runs Python inside the brain env, wherever the current shell is.
# PYTHONNOUSERSITE=1: ignore ~/.local, as the env itself is set to.
brain_python() {
    PYTHONNOUSERSITE=1 conda run -n "${ENV_NAME}" --no-capture-output python "$@"
}

brain_pip() {
    brain_python -m pip install "$@"
}

# ---- 1. conda ---------------------------------------------------------------

section "conda"

if ! command -v conda > /dev/null; then
    failed "conda not found. Install Miniconda first (simulation/README.md, section 3), then run this again."
    printf '\nNothing else can be checked without conda.\n'
    exit 1
fi
ok "conda $(conda --version 2> /dev/null | awk '{print $2}')"

eval "$(conda shell.bash hook)"

# Not `conda env list | grep -q`: grep -q stops reading at the first match,
# conda then fails writing the rest, and with pipefail that counts as "no
# such env" -- sometimes, depending on timing. Read the whole list first.
env_exists() {
    local envs
    envs="$(conda env list 2> /dev/null)"
    grep -qE "^$1[[:space:]]" <<< "${envs}"
}

# ---- 2. the brain env -------------------------------------------------------

section "the '${ENV_NAME}' env"

if env_exists "${ENV_NAME}"; then
    have_python="$(brain_python -c 'import sys; print("%d.%d" % sys.version_info[:2])' 2> /dev/null)"
    if [ "${have_python}" = "${PYTHON_VERSION}" ]; then
        ok "exists, Python ${have_python}"
    else
        # The pinned packages are built for one Python, so installing them
        # into another would fail or break. A rebuild is the only fix.
        failed "the '${ENV_NAME}' env has Python ${have_python:-?}, not ${PYTHON_VERSION}. Rebuild it: conda env remove -n ${ENV_NAME} -y && bash setup/setup_brain.sh"
        WRONG_PYTHON=1
    fi
elif [ "${CHECK_ONLY}" = 1 ]; then
    missing "does not exist"
    cannot_fix_in_check_mode "the '${ENV_NAME}' env does not exist"
else
    missing "does not exist"
    fixing "creating it (Python ${PYTHON_VERSION})"
    # pip listed explicitly: without it, pip falls through to the system pip
    # and installs into ~/.local, outside the env
    if conda create -n "${ENV_NAME}" -c conda-forge --override-channels \
            "python=${PYTHON_VERSION}" pip -y > /dev/null; then
        conda env config vars set -n "${ENV_NAME}" PYTHONNOUSERSITE=1 > /dev/null
        INSTALLED+=("the '${ENV_NAME}' env")
        have_python="$(brain_python -c 'import sys; print("%d.%d" % sys.version_info[:2])' 2> /dev/null)"
        if [ "${have_python}" = "${PYTHON_VERSION}" ]; then
            ok "created, Python ${have_python}"
        else
            failed "created, but it has Python ${have_python:-?}, not ${PYTHON_VERSION}"
            WRONG_PYTHON=1
        fi
    else
        failed "could not create the '${ENV_NAME}' env (see the conda output above)"
    fi
fi

# Without the env, or with the wrong Python in it, the packages can't be
# checked or installed: skip to RTAB-Map, which lives in its own env
SKIP_BRAIN=0
if ! env_exists "${ENV_NAME}"; then
    printf '\n  The packages cannot be checked without the env.\n'
    SKIP_BRAIN=1
elif [ "${WRONG_PYTHON:-0}" = 1 ]; then
    printf '\n  Packages not checked: rebuild the env first (see above).\n'
    SKIP_BRAIN=1
fi

# ---- 3 and 4. packages, and that they import --------------------------------

# Prints the installed version of a pip package, or nothing if not installed
installed_version() {
    brain_python - "$1" 2> /dev/null <<'EOF'
import sys
from importlib import metadata
try:
    print(metadata.version(sys.argv[1]))
except metadata.PackageNotFoundError:
    pass
EOF
}

# Prints the git commit DA3 was installed from, or nothing
da3_commit() {
    brain_python 2> /dev/null <<'EOF'
import json
from importlib import metadata
try:
    info = json.loads(metadata.distribution("depth-anything-3").read_text("direct_url.json") or "{}")
    print(info.get("vcs_info", {}).get("commit_id", ""))
except metadata.PackageNotFoundError:
    pass
EOF
}

# version_ok WANTED HAVE: torch reports e.g. "2.10.0+cu128", so compare the
# part before "+". An empty WANTED means any version will do.
version_ok() {
    local wanted="$1" have="${2%%+*}"
    [ -n "$2" ] && { [ -z "${wanted}" ] || [ "${wanted}" = "${have}" ]; }
}

# check_package NAME WANTED_VERSION PIP_SPEC [needs-torch]
# Checks one package; installs PIP_SPEC if it is missing or the wrong version.
# "needs-torch": the package is built against torch, so it is only installed
# once the right torch is in. Otherwise pip would fetch a different torch
# from the ordinary package index to satisfy it (several GB, and not the
# CUDA 12.8 build).
check_package() {
    local name="$1" wanted="$2" spec="$3" needs="${4:-}"
    local have
    have="$(installed_version "${name}")"

    if version_ok "${wanted}" "${have}"; then
        ok "${name} ${have}"
        return
    fi

    if [ -z "${have}" ]; then
        missing "${name}${wanted:+ ${wanted}}"
    else
        wrong "${name} ${have}, needs ${wanted}"
    fi

    if [ "${CHECK_ONLY}" = 1 ]; then
        cannot_fix_in_check_mode "${name}${wanted:+ ${wanted}} (have: ${have:-not installed})"
        return
    fi

    if [ "${needs}" = "needs-torch" ] && [ "${TORCH_OK}" = 0 ]; then
        failed "${name} not installed: it needs torch ${TORCH} first (see above)"
        return
    fi

    fixing "${spec}"
    # Once torch is in, it is repeated in every install so pip can't
    # "upgrade" it while it resolves something else (setup_brain.sh does the
    # same). Without torch, repeating it would make pip download one.
    local pins=()
    if [ "${TORCH_OK}" = 1 ]; then
        pins=("torch==${TORCH}" "torchvision==${TORCHVISION}")
    fi
    if brain_pip "${spec}" "${pins[@]}"; then
        INSTALLED+=("${name}")
        ok "${name} $(installed_version "${name}")"
    else
        failed "could not install ${spec} (see the pip output above)"
    fi
}

if [ "${SKIP_BRAIN}" = 0 ]; then
    section "Python packages in '${ENV_NAME}'"

    # torch first, from the CUDA 12.8 index: everything else builds on it
    have_torch="$(installed_version torch)"
    have_vision="$(installed_version torchvision)"
    TORCH_OK=0  # whether the pinned torch is in; the packages built on it wait for it
    if version_ok "${TORCH}" "${have_torch}" && version_ok "${TORCHVISION}" "${have_vision}"; then
        ok "torch ${have_torch}, torchvision ${have_vision}"
        TORCH_OK=1
    elif [ "${CHECK_ONLY}" = 1 ]; then
        wrong "torch ${have_torch:-not installed}, torchvision ${have_vision:-not installed} (needs ${TORCH} / ${TORCHVISION})"
        cannot_fix_in_check_mode "torch ${TORCH} + torchvision ${TORCHVISION}"
    else
        wrong "torch ${have_torch:-not installed}, torchvision ${have_vision:-not installed} (needs ${TORCH} / ${TORCHVISION})"
        fixing "torch ${TORCH} + torchvision ${TORCHVISION} (CUDA 12.8, a few GB)"
        if brain_pip "torch==${TORCH}" "torchvision==${TORCHVISION}" --index-url "${TORCH_INDEX}"; then
            INSTALLED+=("torch ${TORCH}")
            ok "torch $(installed_version torch)"
            TORCH_OK=1
        else
            failed "could not install torch ${TORCH} (see the pip output above)"
        fi
    fi

    # torch must be right before these: they are built against it. (Their
    # installs repeat torch's pins, so a fixed torch is never "upgraded".)
    check_package xformers    "${XFORMERS}"    "xformers==${XFORMERS}"       needs-torch
    # addict: DA3 imports it but forgets to list it
    check_package addict      ""               "addict"
    check_package ultralytics "${ULTRALYTICS}" "ultralytics==${ULTRALYTICS}" needs-torch
    check_package piper-tts   "${PIPER}"       "piper-tts==${PIPER}"
    # These usually arrive with the packages above; checked anyway because
    # the brain imports them directly (cv2 in rectify/occupancy, psutil in
    # resources)
    check_package numpy       ""               "numpy"
    check_package psutil      ""               "psutil"

    # OpenCV can come as opencv-python or opencv-python-headless; any one
    # gives cv2, so check the import instead of a package name
    if brain_python -c "import cv2" 2> /dev/null; then
        ok "opencv (cv2 $(brain_python -c 'import cv2; print(cv2.__version__)' 2> /dev/null))"
    elif [ "${CHECK_ONLY}" = 1 ]; then
        missing "opencv (cv2)"
        cannot_fix_in_check_mode "opencv (cv2)"
    else
        missing "opencv (cv2)"
        fixing "opencv-python"
        if brain_pip opencv-python; then
            INSTALLED+=("opencv-python")
            ok "opencv"
        else
            failed "could not install opencv-python"
        fi
    fi

    # DA3 comes from git at one commit, so check the commit, not a version
    have_da3="$(installed_version depth-anything-3)"
    have_commit="$(da3_commit)"
    if [ -n "${have_da3}" ] && [ "${have_commit}" = "${DA3_COMMIT}" ]; then
        ok "depth-anything-3 (commit ${DA3_COMMIT:0:7})"
    else
        if [ -z "${have_da3}" ]; then
            missing "depth-anything-3"
        else
            wrong "depth-anything-3 from commit ${have_commit:-unknown}, needs ${DA3_COMMIT:0:7}"
        fi
        if [ "${CHECK_ONLY}" = 1 ]; then
            cannot_fix_in_check_mode "depth-anything-3 (commit ${DA3_COMMIT:0:7})"
        elif [ "${TORCH_OK}" = 0 ]; then
            failed "depth-anything-3 not installed: it needs torch ${TORCH} first (see above)"
        else
            fixing "depth-anything-3 @ ${DA3_COMMIT:0:7} (needs git)"
            if brain_pip "depth-anything-3 @ ${DA3_URL}" "torch==${TORCH}" "torchvision==${TORCHVISION}" \
                    "xformers==${XFORMERS}" addict; then
                INSTALLED+=("depth-anything-3")
                ok "depth-anything-3"
            else
                failed "could not install depth-anything-3 (is git installed? see the pip output above)"
            fi
        fi
    fi

    # 4. Installed is not the same as working: import each one the brain
    # uses. Only those that are installed -- the missing ones are already
    # on the list above.
    section "imports"
    for pair in numpy:numpy cv2: psutil:psutil torch:torch torchvision:torchvision \
                xformers:xformers depth_anything_3.api:depth-anything-3 \
                ultralytics:ultralytics piper:piper-tts; do
        module="${pair%%:*}" package="${pair#*:}"
        # cv2 has no single package name (opencv-python, -headless, ...)
        if [ -n "${package}" ] && [ -z "$(installed_version "${package}")" ]; then
            continue
        fi
        if error="$(brain_python -c "import ${module}" 2>&1 > /dev/null)"; then
            ok "import ${module}"
        elif [ -n "${package}" ]; then
            failed "${package} is installed but 'import ${module}' fails: $(printf '%s' "${error}" | tail -1)"
        fi
    done

    # ---- 5. the GPU ---------------------------------------------------------

    section "GPU"
    if [ -z "$(installed_version torch)" ]; then
        printf '  [skipped]   torch is not installed\n'
        gpu="skipped"
    else
        gpu="$(brain_python -c '
import torch
print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "")
' 2> /dev/null)"
    fi
    if [ "${gpu}" = "skipped" ]; then
        :
    elif [ -n "${gpu}" ]; then
        ok "torch sees ${gpu}"
    else
        # A driver is a system install (and needs a reboot): not done from here
        failed "torch can't see a CUDA GPU. Check nvidia-smi: these CUDA 12.8 builds need NVIDIA driver 570 or newer."
    fi

    # ---- 6. the model files -------------------------------------------------

    section "model files"
    mkdir -p "${WEIGHTS}/piper"

    # The Piper voice: a file in weights/piper
    voice="${WEIGHTS}/piper/${PIPER_VOICE}.onnx"
    if [ -f "${voice}" ]; then
        ok "Piper voice ${PIPER_VOICE}"
    elif [ "${CHECK_ONLY}" = 1 ]; then
        missing "Piper voice ${PIPER_VOICE}"
        cannot_fix_in_check_mode "Piper voice ${PIPER_VOICE}"
    elif [ -z "$(installed_version piper-tts)" ]; then
        missing "Piper voice ${PIPER_VOICE}"
        failed "Piper voice not downloaded: it needs piper-tts first (see above)"
    else
        missing "Piper voice ${PIPER_VOICE}"
        fixing "downloading it"
        if brain_python -m piper.download_voices "${PIPER_VOICE}" --data-dir "${WEIGHTS}/piper" && [ -f "${voice}" ]; then
            INSTALLED+=("Piper voice")
            ok "Piper voice"
        else
            failed "could not download the Piper voice ${PIPER_VOICE}"
        fi
    fi

    # YOLOE: a file in weights/
    if [ -f "${WEIGHTS}/${YOLOE_MODEL}" ]; then
        ok "YOLOE ${YOLOE_MODEL}"
    elif [ "${CHECK_ONLY}" = 1 ]; then
        missing "YOLOE ${YOLOE_MODEL}"
        cannot_fix_in_check_mode "YOLOE ${YOLOE_MODEL}"
    elif [ -z "$(installed_version ultralytics)" ]; then
        missing "YOLOE ${YOLOE_MODEL}"
        failed "YOLOE ${YOLOE_MODEL} not downloaded: it needs ultralytics first (see above)"
    else
        missing "YOLOE ${YOLOE_MODEL}"
        fixing "downloading it"
        # Run from weights/: ultralytics saves what it downloads in the
        # current folder
        if (cd "${WEIGHTS}" && brain_python -c "from ultralytics import YOLOE; YOLOE('${YOLOE_MODEL}')") \
                && [ -f "${WEIGHTS}/${YOLOE_MODEL}" ]; then
            INSTALLED+=("YOLOE model")
            ok "YOLOE ${YOLOE_MODEL}"
        else
            failed "could not download ${YOLOE_MODEL}"
        fi
    fi

    # DA3: lives in the Hugging Face cache (~/.cache/huggingface), not weights/
    if brain_python -c "
from huggingface_hub import snapshot_download
snapshot_download('${DA3_MODEL}', local_files_only=True)
" > /dev/null 2>&1; then
        ok "DA3 model ${DA3_MODEL}"
    elif [ "${CHECK_ONLY}" = 1 ]; then
        missing "DA3 model ${DA3_MODEL}"
        cannot_fix_in_check_mode "DA3 model ${DA3_MODEL}"
    elif [ -z "$(installed_version depth-anything-3)" ]; then
        missing "DA3 model ${DA3_MODEL}"
        failed "DA3 model not downloaded: it needs depth-anything-3 first (see above)"
    else
        missing "DA3 model ${DA3_MODEL}"
        fixing "downloading it (about 1.5 GB)"
        if brain_python -c "
from huggingface_hub import snapshot_download
snapshot_download('${DA3_MODEL}')
"; then
            INSTALLED+=("DA3 model")
            ok "DA3 model ${DA3_MODEL}"
        else
            failed "could not download ${DA3_MODEL}"
        fi
    fi
fi

# ---- 7. RTAB-Map ------------------------------------------------------------

section "RTAB-Map (the '${RTABMAP_ENV}' env and the bridge)"

# The bridge is a compiled program; --version checks both that it is there
# and that it can find RTAB-Map's libraries in the env
bridge_works() {
    env_exists "${RTABMAP_ENV}" && [ -x "${BRIDGE}" ] && "${BRIDGE}" --version > /dev/null 2>&1
}

if bridge_works; then
    ok "bridge works: $("${BRIDGE}" --version 2>&1 | head -1)"
else
    # Say exactly which part is wrong
    env_exists "${RTABMAP_ENV}" || missing "the '${RTABMAP_ENV}' env"
    if [ ! -x "${BRIDGE}" ]; then
        missing "the bridge (build/rtabmap_bridge/rtabmap_bridge)"
    elif env_exists "${RTABMAP_ENV}"; then
        # Built, but won't start: usually built against an env that has
        # since been removed or remade
        wrong "the bridge is there but won't start: $("${BRIDGE}" --version 2>&1 | tail -1)"
    fi

    if [ "${CHECK_ONLY}" = 1 ]; then
        cannot_fix_in_check_mode "RTAB-Map env or bridge"
    else
        # setup_rtabmap.sh is safe to re-run: it makes the env only if it is
        # missing, and always recompiles the bridge
        fixing "running setup/setup_rtabmap.sh (makes the env if needed, then compiles the bridge)"
        if bash "${REPO}/setup/setup_rtabmap.sh" && bridge_works; then
            INSTALLED+=("RTAB-Map bridge")
            ok "bridge built and works"
        else
            failed "setup/setup_rtabmap.sh failed (see its output above)"
        fi
    fi
fi

# ---- Summary ----------------------------------------------------------------

printf '\n==> Summary\n'

if [ "${#INSTALLED[@]}" -gt 0 ]; then
    printf '  Installed or fixed: %s\n' "$(IFS=,; echo "${INSTALLED[*]}" | sed 's/,/, /g')"
fi

if [ "${#PROBLEMS[@]}" -eq 0 ]; then
    printf '  Everything the brain needs is in place.\n'
    exit 0
fi

if [ "${CHECK_ONLY}" = 1 ]; then
    printf '  Missing or wrong (run without --check to install them):\n'
else
    printf '  Still not right:\n'
fi
for problem in "${PROBLEMS[@]}"; do
    printf '    - %s\n' "${problem}"
done
exit 1
