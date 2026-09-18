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
of HSSD — see section 12 — and `open_environment.py` builds it on first run.

This checks **versions and files only**. It does not test rendering — that is
section 11.

Your Python patch number may differ (3.9.26 and so on). That is fine; only
3.9 is required.

---

## 11. Run it

```bash
python demo/open_environment.py
```

The first run pauses for a minute or two to build the drone navmesh, then
saves it. Later runs load it and start immediately.

A window opens with two panels:

- **left** — what the camera sees inside the house
- **right** — the navigation mesh: the ground the drone can roll on.
  Grey is rollable, white is blocked, red is the camera, green is the drone.

| Key | Action |
|---|---|
| `W` `A` `S` `D` | move the camera |
| `I` `J` `K` `L` | look around |
| `Z` `X` | camera height |
| `↑` `↓` `←` `→` | drive the drone, by compass |
| `1` | take off, hold to climb higher |
| `0` | descend, hold to land |
| Click the map | plan a rolling route to that spot |
| `Enter` | follow the planned route |
| `C` | clear the route |
| `Shift` | move faster |
| `P` | print camera and drone positions |
| `Esc` | quit |

The camera and the drone move independently, at the same time. The drone
marker turns blue and hollow when it is airborne, and a line inside the house
joins it to its shadow on the floor so you can see how high it is.

Neither `1` nor `0` moves the drone directly. They set the altitude it is
aiming for, and it flies there over the next second or so, easing off as it
arrives — so a take-off is a climb and a landing is a settle, not a jump in
either direction.

**This is the real test.** `check_setup.py` can report READY on a machine that
still cannot draw anything. If this window opens and you can drive the drone
around, your setup is genuinely complete.

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

## 13. What is in this folder

```
config/project.json         every setting: env name, versions, data paths, scene, drone size
config/check_setup.py       verifies you have all of it
delta/drone.py              the drone: a sphere that rolls, flies and collides
delta/navigation.py         planning a rolling route and following it
demo/open_environment.py    opens the house and lets you drive the drone through it
tests/test_drone.py         flies the drone at things and checks something stopped it
.gitignore                  keeps the 17 GB of data out of git
```

Run the checks any time with:

```bash
python tests/test_drone.py
```

`project.json` is the single source of truth. Every script reads it, so they
can never disagree about which house to open, which version to expect, or how
big the drone is.

Nothing in `delta/` is allowed to know which house it is in. That rule is what
will let this code survive the move to Isaac Sim: a module that only works in
one scene has to be rewritten, and a module that asks the simulator questions
does not.

---

## 14. Warnings you can ignore

These appear and are harmless:

- `Gym has been unmaintained since 2022` — habitat-baselines imports Gym.
- `pkg_resources is deprecated` — from an old dependency.
- The pybullet startup banner.

Anything mentioning **EGL**, **CUDA**, or a **version mismatch** is not
harmless.

---

## 15. When something fails

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

If something fails and this table does not cover it, paste the whole output.
The checker is written to say exactly what it found.
