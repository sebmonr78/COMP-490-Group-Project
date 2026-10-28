"""Run the brain: `python -m delta.brain`, in the `brain` env.

It waits for the sim (demo/run_glasses.py) to connect, loads what the sim's
hello asks for -- RTAB-Map or the true pose, DA3 or the true depth -- and runs
until the sim disconnects, then saves its memory and exits.

Normally `bash run_glasses.sh` starts it for you. Running it by hand, in a
second terminal, shows its log as it goes:

    conda activate brain
    python -m delta.brain
"""

import argparse
import os
import sys

# Fewer, growable blocks: on an 8 GB card shared with Gemma, fragmentation
# alone can turn a fitting model into an out-of-memory error.
os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")

from delta.brain import config

# How many threads the brain's number crunching may use: brain.json `threads`.
# Unset, PyTorch, NumPy's BLAS and OpenCV each start one worker per hardware
# thread -- 20 on the i9 -- and idle workers spin instead of sleeping.
# Measured with Depth Anything: twenty threads at a steady 25% each, five cores
# busy waiting, taken from the window beside it. Each library reads these once,
# when it loads, so they are set before anything below imports one (config
# reads JSON and nothing else). A value already in the environment wins.
THREADS = str(int(config.load().threads))

for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OPENCV_FOR_THREADS_NUM"):
    os.environ.setdefault(_name, THREADS)

os.environ.setdefault("OMP_WAIT_POLICY", "PASSIVE")

from delta import rig as rig_module
from delta import wire
from delta.brain.loop import Brain


def main():
    parser = argparse.ArgumentParser(description="The glasses' brain.")
    parser.add_argument("--port", type=int, default=None)
    args = parser.parse_args()

    settings = config.load()
    rig = rig_module.load()

    left = rig.camera(rig.stereo_pair[0])

    if left.pitch != 0.0:
        sys.exit(
            "The brain assumes level lenses (it reads heights straight off the "
            f"image rows), but config/rig.json pitches the stereo pair {left.pitch} deg."
        )

    host = settings.link.host
    port = args.port or int(settings.link.port)

    print(f"Brain waiting for the sim on {host}:{port}", flush=True)

    link = wire.serve(host, port)
    hello = link.receive()

    if hello.kind != "hello":
        sys.exit(f"Expected a hello from the sim, got {hello.kind!r}")

    print(f"Sim connected: {hello.fields}", flush=True)

    Brain(link, hello.fields, settings, rig).run()


if __name__ == "__main__":
    main()
