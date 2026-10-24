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
echo 'export MESA_LOADER_DRIVER_OVERRIDE=d3d12' >> ~/.bashrc
```

Open a new terminal afterwards. macOS and native Linux users do not need this.

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
of HSSD — see section 12 — and `demo/run_mapping.py` builds it on first run.

This checks **versions and files only**. It does not test rendering — that is
section 11.

Your Python patch number may differ (3.9.26 and so on). That is fine; only
3.9 is required.

---

## 11. Run it

```bash
python demo/run_mapping.py
```

The first run pauses to build the drone navmesh and to load Depth Anything.
Later runs start faster.

Five panels:

| | |
|---|---|
| **Behind the drone** | a camera following it, so you can see where it is |
| **What the drone believes** | the map it has built FOR ITSELF, growing as it drives. Nothing in it came from the simulator. |
| **Left lens** | the raw picture the stereo pair is looking at |
| **Depth the drone is using** | stereo's metres, with Depth Anything filling the gaps stereo could not match |
| **Status** | what each stage costs, and how far the drone's idea of where it is has drifted from the truth |

The map is not a floor plan. Every square is a stack of 10 cm layers, and the
colour says what the drone could **do** there — see section 13.

| Key | Action |
|---|---|
| `↑` `↓` `←` `→` | drive the drone, by compass |
| `1` | take off, hold to climb |
| `0` | descend, hold to land |
| `V` | free camera: `WASD` to fly it, `IJKL` to look |
| `M` | show the true layout behind the drone's map |
| click the map | plan a route **on the drone's own map** to that spot |
| `Enter` | follow it |
| `C` | clear it |
| `Tab` | save the map to `data/maps/live.npz` |
| `P` | print believed and true position |
| `Shift` | move faster |
| `Esc` | quit |

Useful flags:

```bash
python demo/run_mapping.py --no-map
```

Stereo and localization only, no Depth Anything — about three times faster,
for when you are working on the driving rather than the map.

```bash
python demo/run_mapping.py --frames 120
```

Drives itself and quits, printing what it mapped. For checking the whole
thing still runs without anyone at the keyboard.

```bash
python demo/run_mapping.py --load data/maps/live.npz
```

Carries on from a map you saved earlier.

**This is the real test.** `check_setup.py` can report READY on a machine that
still cannot draw anything. If this window opens and the right-hand panel
starts filling in as you drive, your setup is genuinely complete.

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

Sections 11 and 12 are the drone's *body*. This is its *mind*, and almost all
of it was ported from the group's other Habitat project, Spatial Memory
Glasses, where each piece was built and measured against ground truth. A
stereo pair does not care whether it is on a head or in a ball.

Nothing in this stack may read Habitat. The simulator renders two pictures
and nothing else. The moment any of it reads a true position or the navmesh,
every number the project reports stops meaning anything — a map that is right
because it was told the answer is not a result.

```
two camera images  ->  stereo        metres, where it can find a match
                   ->  Depth Anything a depth for every pixel, rescaled by stereo
                   ->  tracker       where am I (SLAM, and it drifts)
                   ->  GapMap        a grid of vertical columns
                   ->  MapPlanner    a route through that grid
```

### Why there are two depth estimates

Neither one is enough on its own, and the way they fail is opposite:

| | covers | error | scale |
|---|---|---|---|
| **Stereo (SGBM)** | only where it finds a match — a blank wall comes back empty | ~1% | real metres |
| **Depth Anything 3** | every pixel | ~8% | not trustworthy, and drifts across the frame |

So stereo is used as a *ruler*: each frame, `StereoFill` fits a smooth scale
surface over the picture from the pixels where both have an answer, and
rescales Depth Anything into stereo's units. The result is dense like DA3 and
measured like stereo, and it is the only reason the map is in metres at all.

### The map, and the thing it had to learn

The glasses stored one number per floor square: the tallest thing in it. That
is right for a walking person, whose only answer to an obstacle is "go
around". A drone has three answers, and none of them can be read off a single
height:

| | height | what is around it | answer |
|---|---|---|---|
| a sofa | 0.9 m | clear above | fly **over** |
| a dining table | 0.75 m | clear below | roll **under** |
| a bookcase | 2.0 m | solid to the ceiling | go **around** |

The sofa and the table are nearly the same square to a height map. The
difference is not how tall they are, it is **where the empty space is**. So
`delta/gap_map.py` stores the whole vertical column: every floor square is
split into 10 cm layers, and each layer is `SOLID`, `FREE`, or `UNKNOWN`.

`FREE` is the half that does not come for free. A depth pixel at 3 m says two
things — there is something at 3 m, *and there is nothing in the 3 m in
between*. Without the second half, the air above the sofa is merely `UNKNOWN`,
indistinguishable from the inside of the bookcase, and the drone can never
justify flying over anything. So every sampled pixel **carves its own ray
clear** along its length. That is the only reason this map can say "go over"
rather than "nothing has contradicted going over".

`GapMap.classify()` reduces a column to what the drone can do there, and the
awkward case is worth knowing about:

- **the ceiling.** "Something solid overhead with room underneath" describes a
  dining table — and also every square of floor in a building. The first
  version of this coloured almost an entire house as roll-under. What
  separates a table from a ceiling is not height (a mezzanine is high, a
  crawlspace is low); it is that **a table has free space above it and a
  ceiling does not**. That is the test, and it needs no threshold to tune.
- **unmapped ground.** A square where the air is clear but the floor was never
  seen is flyable and not rollable — exactly like a square with a sofa in it.
  Reporting both as "fly over" would claim a measured obstacle where the truth
  is an unanswered question, so they are separate: `OVER` and `AIRBORNE`.

### Navigating on it

`delta/map_navigation.py` plans A\* over `(row, column, level)` — a floor
square and a height — using nothing but what the drone has mapped. It is the
same job `navigation.py` does on Habitat's navmesh, done honestly, and three
things follow:

- **it can be wrong.** The map has holes, so a plan can fail. That is real
  information: "not through anything I have seen" is a different answer from
  "there is no way", and the cure is to go and look.
- **it knows about height**, so a route can go over and under things.
- **it chooses how to move.** Rolling costs 1 per metre, flying 4, climbing 6
  (`DEFAULT_COSTS`). Nothing anywhere decides between rolling and flying — the
  cheapest route is a rolling route wherever one exists, and flight appears
  exactly where the proposal says it should. That is feature 4, and it falls
  out of A\* rather than being ruled on.

### Run it

Drive the drone, build a map from its own cameras, plan on it, save it:

```bash
python tools/run_mapping.py --steps 80 --speed 0.10 --save data/maps/first.npz
```

Look at the map it built:

```bash
python tools/plot_map.py data/maps/first.npz --scale 4
```

Inspect one square, layer by layer, when the picture makes you suspicious:

```bash
python tools/plot_map.py data/maps/first.npz --column 200 200
```

Measured on the default house, 80 frames at 1 m/s: **1.0% drift** over 5 m
with 78 of 80 frames tracked visually, 55.8 m² of floor mapped, and a 6.2 m
rolling route planned on the drone's own map in 16 ms.

### Where localization stands, honestly

The tracker is visual-inertial odometry with keyframes, and **it has no loop
closure**: error only grows, and returning to a room it has already mapped
corrects nothing. Over a short, gentle drive it holds to about 1% of the
distance travelled. It comes apart in three situations, all of them measured:

| | why |
|---|---|
| nose-up against a blank wall | nothing to track, and the floor is too close to give depth |
| turning fast | a view that swings far between frames leaves ORB nothing to match |
| **a slow frame rate** | the drone covers more ground between frames than the matcher can bridge |

The third is the one to watch, because it is not really the tracker's fault:
at 2 frames a second the drone travels 0.4 m and swings 30 degrees between
frames, and tracking fails outright. Keep the window fast, drive gently, and
watch the drift figure in the status panel. If it climbs, slow down.

Two things make the window slow, and both have been dealt with once already:
classifying the whole map twice per screen frame (now done once per *map*
frame), and **software rendering** — if the status shows a few frames a
second, check `echo $MESA_LOADER_DRIVER_OVERRIDE` says `d3d12`, because
without it Mesa falls back to `llvmpipe` on the CPU and everything else
is beside the point.

The real fix for localization is **RTAB-Map**, which the group's other
project already runs: it adds loop closure, so revisiting a room corrects the
map behind you instead of leaving two copies of each wall. That is a port
worth doing, and it is the next thing this stack needs.

Two numbers that will bite you if you change them:

- **drive slower than 3 m/s.** `position.json` rejects a visual match that
  implies a bigger jump, as an impossible one. Driving the drone at 3.5 m/s
  sent 14 of 40 frames to the IMU fallback and the drift to 34%.
- **keep the cameras level.** `config/rig.json` explains at length why
  `measured.pitch` is 0 and what has to change before it can be anything else.

---

## 14. What is in this folder

Kept deliberately small. Every file here is on the path from two camera
images to a route the drone can follow; nothing else is.

```
config/project.json         every setting: env name, versions, data paths, scene, drone size
config/rig.json             the cameras, as hardware: baseline, lens, mounting
config/stereo.json          SGBM settings and the floor finder
config/position.json        the tracker, its keyframes, and the simulated IMU
config/map.json             the grid, and how depth becomes map
config/check_setup.py       verifies you have all of it

delta/drone.py              the drone: a sphere that rolls, flies and collides
delta/sim.py                WORLD SIDE: opens the house, builds the navmesh,
                            moves the sensor head, renders
delta/rig.py                where the lenses are and what they can see
delta/stereo.py             rectify, SGBM, points, floor height
delta/depth_model.py        Depth Anything 3, rescaled into stereo's metres
delta/tracker.py            localization: corners + depth + IMU -> where am I
delta/imu.py                a simulated IMU chip
delta/gap_map.py            the map: vertical columns, and where the gaps are
delta/map_navigation.py     routes on the map the drone built itself
delta/world_frame.py        joining the map's frame to Habitat's
delta/brain.py              wires the five stages together

demo/run_mapping.py         THE demo: drive the drone and watch it map
tools/plot_map.py           draw a saved map, or print one column of it
tests/test_drone.py         flies the drone at things, checks something stopped it
tests/test_mapping.py       checks the map understands gaps and routes use them
.gitignore                  keeps the 17 GB of data, and saved maps, out of git
```

The one line that matters most: **`delta/sim.py` is the only file allowed to
read Habitat.** `delta/brain.py` and everything it imports get two images and
an IMU reading, and nothing else. A brain that can reach the pathfinder will
eventually be "helped" with it, and then none of the project's numbers mean
anything.

Three things were deliberately removed rather than kept "just in case":

- **a second demo.** `demo/run_mapping.py` drives the drone *and* shows the
  map, so a drive-only window was two places to fix the same bug. Its scene
  loading moved into `delta/sim.py`, where the tests can reach it too.
- **routes on Habitat's navmesh.** The drone plans on its own map now. The
  navmesh is still there — it is what `drone.drive` rolls on — but planning
  with it was planning with the answer key.
- **a headless copy of the demo.** `--frames` does the same run in the same
  file.

Run both test suites any time with:

```bash
python tests/test_drone.py && python tests/test_mapping.py
```

`tests/test_mapping.py` needs no scene, no GPU and no Depth Anything — it
builds maps by hand, one obstacle at a time, so every check has exactly one
right answer. It is the fast one to run while editing.

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
| Window opens black | GPU not reached | `echo $MESA_LOADER_DRIVER_OVERRIDE` must say `d3d12` |
| Drone will not take off | no room overhead | drive it out from under the furniture and press `1` again |
| Drone will not follow a route | it is airborne | press and hold `0` to land, then `Enter` |
| `No map was built` | the floor was never measured, so the map had no datum | check `config/stereo.json`; the floor must come into view at least once |
| Tracker falls back to the IMU constantly | driving faster than `max_speed` | slow down, or raise it in `config/position.json` |
| Camera height reads far too low | the cameras are not level | `measured.pitch` in `config/rig.json` must be 0 -- read the note there |
| `NO ROUTE` on the drone's own map | the mapped floor is in disconnected pieces | drive further; `run_mapping.py` prints how many regions it found |
| `ModuleNotFoundError: depth_anything_3` | DA3 is not installed in this env | run with `--no-map` for stereo and SLAM only |

If something fails and this table does not cover it, paste the whole output.
The checker is written to say exactly what it found.
