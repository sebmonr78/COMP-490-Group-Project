#!/usr/bin/env bash
# Builds the `brain` conda env: everything the glasses' computer runs, as
# opposed to the `habitat` env, which only runs the simulated world.
#
#     bash setup/setup_brain.sh
#
# What goes in, one line per box on the slide:
#
#     Estimate depth               Depth Anything 3      (same pins as #11)
#     Detect and segment objects   YOLOE, via ultralytics
#     Voice and audio guidance     Piper TTS
#     Rectify, grids, maps, paths  OpenCV and NumPy      (come with the above)
#
# RTAB-Map and Gemma have their own setup scripts, because neither is a
# Python package: setup/setup_rtabmap.sh and setup/setup_ollama.sh.
#
# Every version is pinned so both machines run the same stack. To change one,
# change it here and rebuild:
#
#     conda env remove -n brain -y && bash setup/setup_brain.sh

set -euo pipefail

ENV_NAME="brain"
PYTHON_VERSION="3.10"

# CUDA 12.8 builds need NVIDIA driver >= 570. Same choice as the da3 env.
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

REPO="$(cd "$(dirname "$0")/.." && pwd)"
WEIGHTS="${REPO}/weights"


if ! command -v conda > /dev/null; then
    echo "conda not found. Install Miniconda first (README section 3)." >&2
    exit 1
fi

eval "$(conda shell.bash hook)"

if conda env list | grep -qE "^${ENV_NAME}[[:space:]]"; then
    echo "The '${ENV_NAME}' env already exists. To rebuild it:" >&2
    echo "    conda env remove -n ${ENV_NAME} -y && bash $0" >&2
    exit 1
fi

echo "==> Creating the '${ENV_NAME}' env (Python ${PYTHON_VERSION})"

# pip is listed explicitly: without it, a plain `pip install` falls through
# to the system pip and lands in ~/.local, outside the env entirely.
conda create -n "${ENV_NAME}" -c conda-forge --override-channels \
    "python=${PYTHON_VERSION}" pip -y

# Ignore ~/.local, which is read BEFORE the env's own packages.
conda env config vars set -n "${ENV_NAME}" PYTHONNOUSERSITE=1

conda activate "${ENV_NAME}"
export PYTHONNOUSERSITE=1

echo "==> Installing torch ${TORCH} (CUDA 12.8)"

python -m pip install \
    "torch==${TORCH}" "torchvision==${TORCHVISION}" \
    --index-url "${TORCH_INDEX}"

echo "==> Installing Depth Anything 3, YOLOE and Piper"

# torch is repeated so pip cannot "upgrade" it while resolving the rest.
# addict is imported by DA3 but missing from its dependency list.
python -m pip install \
    "torch==${TORCH}" "torchvision==${TORCHVISION}" \
    "xformers==${XFORMERS}" \
    addict \
    "depth-anything-3 @ ${DA3_URL}" \
    "ultralytics==${ULTRALYTICS}" \
    "piper-tts==${PIPER}"

echo "==> Downloading the Piper voice"

mkdir -p "${WEIGHTS}/piper"
python -m piper.download_voices "${PIPER_VOICE}" --data-dir "${WEIGHTS}/piper"

echo "==> Downloading the models and loading each one once"

# Run from weights/, because ultralytics drops whatever it downloads into
# the current directory.
cd "${WEIGHTS}"

python - "${DA3_MODEL}" "${YOLOE_MODEL}" "${WEIGHTS}/piper/${PIPER_VOICE}.onnx" <<'EOF'
import io
import sys
import wave
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import torch

da3_name, yoloe_name, voice_path = sys.argv[1:4]

print(f"torch {torch.__version__} from {torch.__file__}")

if not torch.cuda.is_available():
    sys.exit(
        "CUDA is not available. Check `nvidia-smi`: the driver must be "
        "570 or newer for these CUDA 12.8 builds."
    )

print(f"GPU: {torch.cuda.get_device_name(0)}")

from depth_anything_3.api import DepthAnything3

DepthAnything3.from_pretrained(da3_name).to("cuda")
print(f"{da3_name} loaded")

from ultralytics import YOLOE

model = YOLOE(yoloe_name)
names = ["chair", "door"]
model.set_classes(names, model.get_text_pe(names))
model.predict(np.zeros((320, 512, 3), dtype=np.uint8), verbose=False)
print(f"{yoloe_name} loaded, text prompts work")

from piper import PiperVoice

voice = PiperVoice.load(voice_path)
buffer = io.BytesIO()

with wave.open(buffer, "wb") as wav:
    voice.synthesize_wav("Ready.", wav)

print(f"Piper voice loaded, {len(buffer.getvalue())} bytes of speech")
EOF

echo
echo "Done. Every new terminal:"
echo "    conda activate ${ENV_NAME}"
