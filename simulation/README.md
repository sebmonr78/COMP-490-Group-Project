# DELTA — Simulation

**D**etecting **E**nvironmental **L**ayout and **T**racking **A**lterations.
COMP 490 group project, Y.A.K.S.

A rolling and flying drone that explores an environment, maps it, remembers
what it saw, and returns later to work out what has changed. This folder is
the simulation side of it, built on [Habitat](https://aihabitat.org/). The
physical prototype and the Isaac Sim work come later; everything here exists
so that navigation, mapping and perception can be developed and tested
safely first.

This README takes a machine with nothing on it to a running 3D house with a
drone in it. Set aside about **2 hours**, most of it downloading.

---

## 1. What you are installing, and why

| Component | What it is | Why we need it |
|---|---|---|
| **WSL2** | Real Linux running inside Windows | Habitat is Linux/macOS only. Windows users need this. |
| **Miniconda** | Environment manager | Habitat is distributed as conda packages. |
| **habitat-sim** | The simulator: 3D world, cameras, physics | Renders what the drone sees, and answers what it can hit. |
| **habitat-lab** | Tasks, agents, navigation helpers | We use its top-down map drawing. |
| **habitat-baselines** | Training code | Installed because habitat-lab expects it. |
| **habitat-hitl** | Human-in-the-loop tools | Installed for completeness. |
| **HSSD** | Scanned houses, ~17 GB on disk | The environments we map and navigate. |

All four habitat packages must be **version 0.3.3**. Mixing versions is the
single most common way to break this project.

---

## 2. Linux (Windows users only)

macOS and Linux users: skip to section 3.

Open **PowerShell as Administrator** and run:

```powershell
wsl --install -d Ubuntu
```

Restart the computer. Ubuntu opens and asks for a username and password —
this is a new Linux account, unrelated to your Windows login. Remember the
password; you need it for `sudo`.

After that, every command in this README goes in the **Ubuntu** terminal,
not PowerShell. Open it by typing `wsl` in PowerShell, or by launching Ubuntu
from the Start menu.

Check it worked:

```bash
uname -a
```

You should see `Linux` and `microsoft-standard-WSL2`.

### Important notes about WSL

- **Your files live in two places.** `/home/<you>/` is Linux. `/mnt/c/` is your
  Windows C: drive. Keep this project in your Linux home directory —
  working from `/mnt/c/` is much slower.
- **The GPU works for computing, but not for drawing.** WSL provides CUDA but
  not NVIDIA's OpenGL driver. Section 9 has the fix.

Install the basic tools:

```bash
sudo apt update
```

```bash
sudo apt install -y git git-lfs wget build-essential
```

```bash
git lfs install
```

`git-lfs` is not optional. The HSSD download fails without it.

---

## 3. Miniconda

If `conda --version` already prints something, skip ahead.

```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
```

```bash
bash Miniconda3-latest-Linux-x86_64.sh
```

Accept the licence, accept the default location, and answer **yes** when it
offers to initialise conda. Then close the terminal and open a new one.

```bash
conda --version
```

Your prompt now starts with `(base)`.

---

## 4. The conda environment

From the habitat-sim and habitat-lab READMEs — both give this exact command:

```bash
conda create -n habitat python=3.9 cmake=3.14.0
```

```bash
conda activate habitat
```

Your prompt must now read `(habitat)`.

> **Read this twice.** Every new terminal starts in `(base)`, not `(habitat)`.
> If anything in this project fails in a way that makes no sense, check your
> prompt first. This is the most frequent mistake by a wide margin.

To make it automatic:

```bash
echo "conda activate habitat" >> ~/.bashrc
```

---

## 5. habitat-sim

```bash
conda install habitat-sim withbullet -c conda-forge -c aihabitat
```

`withbullet` includes the Bullet physics engine. **It is not optional here.**
The drone's flight collision checks are Bullet calls — without it the drone
flies through walls. Verify:

```bash
python -c "import habitat_sim; print(habitat_sim.__version__)"
```

Must print **0.3.3**.

```bash
python -c "import habitat_sim; print(habitat_sim.built_with_bullet)"
```

Must print **True**.

---

## 6. habitat-lab, baselines and hitl

Clone into your home directory, **not** into this repository:

```bash
cd ~ && git clone --branch stable https://github.com/facebookresearch/habitat-lab.git
```

Make the clone match the version you installed:

```bash
cd ~/habitat-lab && git checkout v0.3.3
```

Then install all three:

```bash
pip install -e habitat-lab
```

```bash
pip install -e habitat-baselines
```

```bash
pip install -e habitat-hitl
```

`-e` means "editable" — it installs the folder you just cloned, so you can read
and change the source. Verify:

```bash
python -c "import habitat; print(habitat.__version__)"
```

Must print **0.3.3**.

---

## 7. The HSSD dataset (~17 GB)

### 7a. Hugging Face access

HSSD is hosted on Hugging Face and you must accept its terms **once, in a
browser**, with your own account. No script can do this for you.

1. Create an account at https://huggingface.co
2. Open https://huggingface.co/datasets/hssd/hssd-hab and accept the terms
3. Create an access token: Settings, then Access Tokens, then New token (read)

Log in from the terminal:

```bash
pip install huggingface_hub
```

```bash
huggingface-cli login
```

Paste the token when asked. **The terminal shows nothing while you paste —
that is normal.** Paste once and press Enter.

> Never put the token in a file, a commit, or a message to anyone. It is a
> password for your account.

### 7b. Download

```bash
python -m habitat_sim.utils.datasets_download --uids hssd-hab --data-path ~/habitat-data
```

This takes a long time. It creates:

```
~/habitat-data/versioned_data/hssd-hab/
```

`versioned_data` is made by the downloader, not by you — every dataset it
fetches is filed there.

---

## 8. This project

```bash
cd ~ && git clone https://github.com/sebmonr78/COMP-490-Group-Project.git
```

Everything in this README lives in the `simulation` folder:

```bash
cd ~/COMP-490-Group-Project/simulation
```

Tell it where your data is. If you used the exact path in section 7b this is
already the default and you can skip it. Otherwise:

```bash
echo 'export HABITAT_DATA_PATH=~/habitat-data/versioned_data/hssd-hab' >> ~/.bashrc
```

Then open a new terminal, or run `source ~/.bashrc`.

> The dataset is never committed to this repository — it is 17 GB, and GitHub
> rejects files over 100 MB. Everyone downloads their own copy and points this
> variable at it. It is the only machine-specific setting in the project.

---

## 9. Rendering on WSL

WSL gives you the GPU for computing but not for drawing. Without this, opening
a window fails with:

```
unable to find CUDA device 0 among 1 EGL devices
```

The fix routes rendering through Direct3D 12, which does reach the real GPU:

```bash
echo 'export GALLIUM_DRIVER=d3d12' >> ~/.bashrc
```

```bash
echo 'export MESA_LOADER_DRIVER_OVERRIDE=d3d12' >> ~/.bashrc
```

Open a new terminal afterwards. macOS and native Linux users do not need this.

Both lines matter. With only the second, a window opens but Mesa quietly
draws on the CPU (`llvmpipe`), and everything runs at a few frames a second.

---

## 10. Check it

```bash
conda activate habitat
```

```bash
cd ~/COMP-490-Group-Project/simulation && python config/check_setup.py
```

Expected:

```
PASS  Operating system          Linux x86_64
PASS  Conda environment         habitat
PASS  Python                    3.9.25
PASS  habitat-sim               version 0.3.3
PASS  habitat-lab               version 0.3.3
PASS  habitat-baselines         version 0.3.3
PASS  habitat-hitl              version 0.3.3
PASS  Habitat imports           habitat_sim and habitat imported
PASS  NVIDIA GPU                NVIDIA GeForce RTX 3080, 591.86
PASS  HSSD root                 /home/you/habitat-data/versioned_data/hssd-hab
PASS  HSSD configuration        ...
PASS  Selected scene            ...
PASS  Semantic configuration    ...
PASS  Stage model               ...
WARN  Drone navigation mesh     not created: ...
SETUP STATUS: READY WITH 1 WARNING(S)
```

That one warning is expected the first time. The drone's navmesh is not part
of HSSD — see section 12 — and `run_drone.sh` builds it on first run.

This checks **versions and files only**. It does not test rendering — that is
section 11.

Your Python patch number may differ (3.9.26 and so on). That is fine; only
3.9 is required.

---

## 11. Run it

The drone's brain runs in its own conda env, next to the simulator. Set it up
once:

```bash
bash setup/setup_brain.sh
```

```bash
bash setup/setup_rtabmap.sh
```

The first makes the `brain` env (PyTorch, Depth Anything 3). The second
installs RTAB-Map and compiles the small C++ bridge the brain talks to it
through. Then:

```bash
cd ~/COMP-490-Group-Project/simulation && bash run_drone.sh --fresh
```

Two processes start: the brain (env `brain`, its log in
`outputs/logs/brain.log`) and the sim window (env `habitat`). They talk over a
local socket, which is how the real drone will reach the laptop. The first run
builds the drone's navmesh and loads Depth Anything, so it is slow to start.

On the left is the house, seen from behind the drone. In the middle is the map
**the brain built for itself**, drawn over the true layout in grey. Where the
two disagree is the brain's error. On the right is the brain's 3D replica,
with the ceiling taken off to see into the rooms -- only in the picture: the
brain keeps the ceiling and knows how high it is (the height box shows it next
to the truth). Next to the height box, RESOURCES shows what it all costs,
every 2 s: memory and CPU of the window, the brain and RTAB-Map (CPU as `top`
shows it: 250% is two and a half cores), the GPU's load, memory and power,
and the size of the 2D map, the 3D replica and RTAB-Map's database.

| Key | Action |
|---|---|
| `↑` `↓` | roll forward, back |
| `←` `→` | turn |
| `1` | take off; hold to climb |
| `0` | hold to descend and land |
| `Shift` | faster |
| `V` | free camera: `WASD`, `IJKL`, `Z` `X` |
| `M` | show or hide the true layout |
| `N` | show or hide the rooms |
| `I` `K` `O` | tilt the cameras up, down, level |
| drag, scroll the 3D replica | turn it, zoom it; `F` follows the drone or stays put |
| `[` `]` | the replica without its ceiling (the start), a flat cut lower down, or everything |
| `B` | build the 3D replica now (with `--replica later`, it is built only then and when the map is saved) |
| click the map | plan a route **on the brain's own map** |
| `G` | follow it |
| `C` | clear it |
| `R` | roam to a random spot (a stand-in for a person at the keys) |
| `Tab` | save the map |
| `F12` | screenshot |
| `Esc` | quit; the brain saves its map |

Without `--fresh` the brain starts from the map it saved last time.

**One part at a time.** Each part of the brain can be swapped for the
simulator's ground truth, so that one part can be checked on its own:

```bash
bash run_localization.sh
```

Only localization is real: depth is the truth, so anything wrong in the map is
the position being wrong. The general switches, for `run_drone.sh` too:

| | |
|---|---|
| `--pose oracle` | the true position instead of RTAB-Map's |
| `--depth oracle` | the true depth instead of Depth Anything's |
| `--replica later` | keep the views, build the 3D replica only on `B` or at the end (`live`: as it goes; default from `config/brain.json`) |
| `--profile` | print every 10 s where the window's and the brain's time goes |
| `--ui 1.25` | make everything bigger or smaller (pixels per point; by default 1.5 on a 4K screen) |
| `--no-odometer` | take the cage's axle encoder out |
| `--script FILE` | drive from a script (`tests/scripts/`) |
| `--trace FILE` | log position error and RTAB-Map health |
| `--headless` | no window |

**This is the real test of the setup.** `check_setup.py` can report READY on
a machine that still cannot draw anything. If this window opens and the map
fills in as you drive, your setup is complete.

---

## 12. Rolling and flying

The drone is a 30 cm sphere. It has two ways of moving, and they are the
point of the design rather than two settings of one thing: rolling is cheap
and is how it should spend most of its life, and flying is what it does when
the ground runs out — stairs, obstacles, gaps, high shelves, blocked paths.

In the simulator that split is not cosmetic. The two modes ask the simulator
completely different questions:

| | Rolling | Flying |
|---|---|---|
| What decides where it can go | the **navmesh** | **Bullet** collision tests |
| The call | `pathfinder.try_step` | `perform_discrete_collision_detection` |
| Ceiling limit | not applicable | a ray cast straight up |

The reason is that **the navmesh is a map of the floor.** It knows every
surface the drone could rest on and has no opinion whatsoever about the air
above them. It is exactly the right tool on the ground and completely silent
once the drone leaves it, so flight has to ask the real geometry instead.

This is also why the project has **two navmeshes**. HSSD ships one built for
a walking human — roughly 1.6 m tall, 0.20 m wide, able to step up 0.20 m.
Ours describes a 30 cm ball, which fits under tables and through gaps a
person does not, and cannot climb a step at all. Different robots, different
maps of the same building. The drone's is generated on first run and saved
next to HSSD's as `..._drone.navmesh`.

Two limits apply while flying, both of them real geometry rather than
invented numbers:

- **it cannot rise through the ceiling** — a ray is cast straight up and the
  climb stops short of whatever it hits
- **it cannot pass through anything** — every candidate position is collision
  tested before the drone is moved there, and a blocked move is retried one
  axis at a time so it slides along a wall instead of stopping dead

One detail is worth knowing before you change any of this, because getting it
wrong fails **silently**. Bullet does not report collisions between two bodies
that are both non-dynamic, and every wall, floor and piece of furniture in an
HSSD scene is `STATIC`. So a drone marked `KINEMATIC` — which is what it
really is, since we move it ourselves and never step physics — is filtered
against the entire building and is told that everywhere is clear. It does not
crash. It flies through the walls.

The drone is therefore marked `DYNAMIC`, purely so that the collision reports
come back. Nothing steps physics, so it never falls. `tests/test_drone.py`
flies it at the house on purpose and fails if nothing stops it, because "the
drone kept going" is not something you would otherwise notice.

---

## 13. Seeing, knowing where it is, and remembering

Sections 11 and 12 are the drone's *body*. This is its *mind*, in
`delta/brain/`. It was ported from the group's other Habitat project, Spatial
Memory Glasses (`spatial-agent-12`), where each piece was built and measured
against ground truth. A stereo pair does not care whether it is on a head or
in a ball.

```
two camera images, IMU, axle encoder, rangefinder, climb rate
   -> loop.py        the brain's main loop: takes each frame through the rest
   -> localize.py    RTAB-Map: where am I; loop closures correct the past
   -> height.py      how high I am, and what is under me (rangefinder)
   -> depth.py       how far everything is: Depth Anything 3, scaled from the floor
   -> occupancy.py   the 2D map: rolling height, anything up to 1.8 m, wall height
   -> rooms.py       the floor plan split into rooms at the doors
   -> replica.py     the 3D replica in colour, and the ceiling's height
   -> planner.py     routes on the map
```

**Nothing in the brain may read Habitat.** The simulator sends what the
drone's own hardware would measure -- two pictures, the IMU, the encoder, the
rangefinder, the climb rate -- and nothing else. A map that is right because
it was told the answer is not a result. Ground truth is used only to *score*
the brain, or, with the oracle switches (`--pose oracle`, `--depth oracle`),
to stand in for one part while another is checked: `run_localization.sh`
gives the brain the true depth, so that only its localization is tested.

### Localization

RTAB-Map tracks features between the two cameras and from frame to frame,
and recognises places it has seen before (a loop closure), which pulls the
accumulated drift back out. The IMU's heading carries it through turns in
front of blank walls.

When the cameras see nothing at all -- pressed against a wall, closer than
the 17 cm stereo can measure -- the cage's axle encoder says how far the ball
actually rolled. Without it, the brain had to assume the drone kept moving at
its last speed, and against a wall that put it 0.75 m past where it
really was.

Measured motion by motion (`tests/scripts/motions.json`, three runs): turning,
even at 120 degrees a second, taking off, climbing and flying add a few
centimetres at most. Descending to land is the weakest motion.

### The map

A grid of 5 cm squares, built only from what the drone has seen. An obstacle
is anything 8 to 38 cm above the floor -- the drone's own height, plus a
margin -- because this is the map of where it can **roll**. Free floor is
floor it actually saw. A second layer records what is wall-high (0.9 to
1.8 m up) -- walls and tall cupboards, not furniture -- and `rooms.py` splits
that floor plan into rooms at the doors.

### Measuring it

```bash
bash tools/repeat.sh NAME 3 tests/scripts/motions.json --depth oracle
```

```bash
python3 tools/segments.py outputs/repeat/NAME/run*_trace.jsonl
```

`repeat.sh` runs a script several times, because localization is not
deterministic: the same drive can end a few centimetres or a metre off.
`segments.py` breaks the result down by motion. `tools/diagnose.sh` runs the
same drive with each part real and true in turn; `tools/summarize_run.py`
summarizes one run. `HSSD_SCENE=<scene id>` runs everything in another house.

---

## 14. What is in this folder

Four parts, as on the real drone: the world (here the simulator; later the
real house and the real drone), the cable, the brain, and a window to watch
it with.

```
THE WORLD -- knows the truth
delta/sim.py                the house, the navmesh, and the sensors: cameras,
                            IMU, axle encoder, rangefinder, climb rate
delta/drone.py              the drone: a sphere that rolls, flies and collides
delta/rig.py                the cameras, as hardware: where they are, what they see

THE CABLE
delta/wire.py               the socket: sensor readings to the brain, its state back

THE BRAIN -- never sees the truth (section 13)
delta/brain/

THE WINDOW -- the simulated drone, and what you see
demo/run_drone.py           starts it (run_drone.sh starts the brain too)
delta/window/app.py         drives the drone, sends its sensors, draws the panels
delta/window/examiner.py    the ONLY window code that reads the truth: the grey
                            true map, the green ring, the truth column, the scores
delta/window/link.py        the window's end of the cable
delta/window/script.py      test drives from tests/scripts/*.json
delta/window/view3d.py      draws the 3D replica panel

THE FIRST DEMO -- the web app's "Simple drive" runs it
demo/open_environment.py    drive the drone next to the true layout; no brain
delta/navigation.py         routes on the true navmesh, for that demo

config/project.json         env name, versions, data paths, scene, drone size
config/rig.json             the cameras, as hardware: baseline, lens, mounting
config/brain.json           the brain's settings, each with why
config/check_setup.py       verifies you have all of it

setup/                      the brain env, and RTAB-Map with its bridge
tests/test_drone.py         flies the drone at things, checks something stopped it
tests/test_brain.py         the brain's pieces, and that it reads no truth
tests/scripts/              scripted drives for the tools below
tools/                      measuring the brain against the truth
```

The line that matters most: **only the world and the examiner read the
truth**, and the brain gets only what real sensors would give.
`tests/test_brain.py` checks that: the brain cannot import Habitat, the
window sends only sensor fields, and true values only when an oracle switch
asks for them.

---

## 15. Warnings you can ignore

These appear and are harmless:

- `Gym has been unmaintained since 2022` — habitat-baselines imports Gym.
- `pkg_resources is deprecated` — from an old dependency.
- The pybullet startup banner.

Anything mentioning **EGL**, **CUDA**, or a **version mismatch** is not
harmless.

---

## 16. When something fails

| Symptom | Cause | Fix |
|---|---|---|
| `FAIL Conda environment` | prompt says `(base)` | `conda activate habitat` |
| `FAIL habitat-sim not installed` | wrong env, or section 5 skipped | check the prompt, then section 5 |
| `FAIL ... required 0.3.3, found 0.3.4` | clone and install disagree | `cd ~/habitat-lab && git checkout v0.3.3` |
| `built without Bullet` | installed without `withbullet` | redo section 5 with the full command |
| `FAIL HSSD root` | data missing, or variable wrong | `echo $HABITAT_DATA_PATH`, compare with section 7b |
| `unable to find CUDA device 0 among 1 EGL devices` | WSL rendering | section 9 |
| `ModuleNotFoundError: habitat` | installed into the wrong env | check the prompt, redo section 6 |
| `ModuleNotFoundError: delta` | run from the wrong folder | `cd ~/COMP-490-Group-Project/simulation` first |
| Window opens black, or everything is slow | rendering on the CPU | `echo $GALLIUM_DRIVER` must say `d3d12` (section 9) |
| Drone will not take off | no room overhead | drive it out from under the furniture and press `1` again |
| "Waiting for the brain..." never ends | the brain failed to start | read `outputs/logs/brain.log` |
| `Another DELTA brain is already running` | a second run was started | close the other window first; two runs share one port |
| `No RTAB-Map bridge` | the bridge was never compiled | `bash setup/setup_rtabmap.sh` |
| No map appears | the floor was never seen, so depth had no scale | drive where the floor is in view |
| Camera height reads far too low | the cameras are not level | `measured.pitch` in `config/rig.json` must be 0 -- read the note there |
| `ModuleNotFoundError: depth_anything_3` | the brain env is missing DA3 | `bash setup/setup_brain.sh` |

If something fails and this table does not cover it, paste the whole output.
The checker is written to say exactly what it found.
