# DELTA

**Detecting Environmental Layout and Tracking Alterations** -- COMP 490, team
Y.A.K.S. A drone in a 45 cm ball-shaped cage that rolls and flies through a
home, maps it on its own, and notices what changed since last time. Built in
simulation first, hardware later.

This page tells you **what each part is, which one to run, and the exact
steps for each**. Every folder has its own README with the details.

---

## 1. The parts

There are four parts, and each one is set up separately. Two of them are
quick to set up and two take a while:

| Folder | What it is | What you need | Setup time |
|---|---|---|---|
| [`physical-drone/`](physical-drone/README.md) | **The physics twin.** The real drone's hardware (weight, motors, flight controller) simulated in MuJoCo. Answers "how long does it fly? does it hover steadily?" You can fly it yourself. | Python 3.10+ | ~5 min |
| [`database/`](database/README.md) | **The database.** Users, drones, scans, maps, objects, what changed. | Python (any 3.x) | ~1 min |
| [`app/`](app/README.md) | **The website.** Log in, see your drones and what changed, and Run buttons that start the simulations. | Python 3.10+, the database | ~5 min |
| [`simulation/`](simulation/README.md) | **DELTA itself, in a simulated house (Habitat).** The drone's *world* (house, sensors), its *brain* (where am I, the 2D map, rooms, 3D replica) and a *window* to watch it in. | Linux or WSL2, conda, an NVIDIA GPU, 17 GB of house scans | ~2 hours |

**The twin and the Habitat simulation are two different simulations**, and
they don't share anything yet:

- The **twin** (`physical-drone/`) is about the *body*: physics, motors,
  battery, flight. It has no house and no mapping.
- The **Habitat simulation** (`simulation/`) is about the *mind*: seeing,
  knowing where it is, building the map. Its drone is a simple ball that
  moves where you tell it.

---

## 2. Which one should I run?

- **"Run the simulation" for a class demo or a quick look:** the physics
  twin, section 4. It works on any laptop in a few minutes.
- **Working on the website or the database:** sections 5 and 6.
- **Working on mapping, localization or the brain:** the Habitat
  simulation, section 7. Set aside the 2 hours.
- **Running simulations from the website, for several people at once:**
  everything above, then section 8.

---

## 3. Python environments: keep them apart

Most "it doesn't work" problems come from running something in the wrong
Python environment. Each part has its own, and **they cannot be shared**:

| Environment | Python | Used by | Make it with | Your prompt shows |
|---|---|---|---|---|
| `physical-drone/venv` (or conda env `twin`) | **3.10+** | the physics twin | section 4 | `(venv)` or `(twin)` |
| `app/venv` | 3.10+ | the website | section 6 | `(venv)` |
| conda env `habitat` | **3.9 exactly** | the Habitat simulation's window | `simulation/README.md` §4 | `(habitat)` |
| conda env `brain` | 3.10 | DELTA's brain | `simulation/README.md` §11 | (started for you by `run_drone.sh`) |

Before you run anything, **check your prompt**. Two common mix-ups:

- If `pip install` for the twin fails with `requires a different python
  version ... mujoco==3.17.0`, you made its venv while `(habitat)` was active,
  which is Python 3.9. Run `conda deactivate`, delete the `venv` folder, and
  make it again (section 4).
- If you see `ModuleNotFoundError: habitat`, you are not in `(habitat)`. Run
  `conda activate habitat`.

---

## 4. The physics twin

**Setup (once).** From the repository folder:

```bash
cd physical-drone
conda deactivate                 # only if your prompt shows (habitat) or (base)
python3 --version                # must say 3.10 or newer
python3 -m venv venv
source venv/bin/activate         # Windows PowerShell: venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

If `python3 --version` says 3.9 or older, make a conda env instead:
`conda create -n twin python=3.11 -y`, then `conda activate twin`, then
`pip install -r requirements.txt`.

**Run it.** From `physical-drone/`, with `(venv)` or `(twin)` active:

```bash
python -m twin.play
```

This opens a window to fly it yourself. Press `1` to take off, use the
arrows to fly, `G` for a gust, `L` to land and `Esc` to quit. Click the
window first so it gets the keys.

```bash
python -m twin.fly twin/flights/hover.json --out ~/delta-twin-out/hover
```

This flies a fixed script: take off, hover, fly 2 m, climb, turn, a gust,
then land. With no window, it writes `report.json` (power, hover time,
overshoot) and `flight.png` (a plot of the whole flight) to the `--out`
folder. It takes 10–20 seconds.

Always run it as `python -m twin.…` from inside `physical-drone/`, not as
`python twin/play.py`. Otherwise you get `No module named twin`.

---

## 5. The database

**Setup (once)**:

```bash
cd database
python3 create_db.py --seed
```

This makes `delta.db` with sample data. The demo login is
`demo@example.com` / `demo1234`. To start over, run
`python3 create_db.py --reset --seed`.

---

## 6. The website

You need the database from section 5 first.

**Setup (once)**:

```bash
cd app
python3 -m venv venv
source venv/bin/activate         # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

**Run it** from `app/`, with `(venv)` active:

```bash
uvicorn main:app
```

Then open http://127.0.0.1:8000 and log in as `demo@example.com` /
`demo1234`. Press `Ctrl+C` to stop it.

The dashboard shows your drones, what changed in the last scan, and a
**Simulations** list with Run buttons. A button works only if what it starts
is set up on this machine:

| Run button | Needs |
|---|---|
| Physics twin: scripted flight / fly it yourself | section 4 (the website finds `physical-drone/venv` or the conda env `twin` by itself) |
| Drive the drone: the "(new)" buttons start **DELTA**, the plain ones the first demo | section 7, and uvicorn started from a terminal where `conda activate habitat` works |
| Brain checks, Drone tests, Check setup | the same |
| "Run in browser" (the window inside the page) | `sudo apt install xvfb x11vnc novnc websockify` on the server |

The scripted twin flight shows its hover time, energy and the flight plot
right on the page when it finishes.

---

## 7. DELTA in the Habitat simulation

This is the long one. Follow
[`simulation/README.md`](simulation/README.md) sections 1–11 in order. They
cover WSL, Miniconda, the `habitat` env, habitat-sim and habitat-lab 0.3.3,
the HSSD houses (17 GB, and you need a Hugging Face account), GPU rendering
on WSL, and the brain's env and RTAB-Map.

Once that is done:

```bash
conda activate habitat
cd simulation
python config/check_setup.py          # every line PASS, maybe one WARN about the navmesh
bash run_drone.sh --fresh             # the drone, its brain, the map -- drive with the arrows
```

| Command (from `simulation/`) | What it runs |
|---|---|
| `bash run_drone.sh --fresh` | DELTA with everything real; `--fresh` starts a new map |
| `bash run_localization.sh` | DELTA with only localization real (true depth), to check that one thing |
| `bash run_drone.sh --run NAME ...` | a run with its own map and log, next to others (section 8) |
| `python demo/open_environment.py` | the first demo: drive the drone next to the true layout, no brain |
| `python tests/test_brain.py` | the brain's 21 checks |
| `python tests/test_drone.py` | the drone rolls, flies, crashes and lands; no window |

The keys (arrows to drive, `1`/`0` to fly and land, `M` for the true map,
and so on) are in `simulation/README.md` §11.

---

## 8. Several at once

**From a terminal.** Each DELTA run needs its own brain port and its own map,
because two brains cannot share either. `run_drone.sh` handles both:

```bash
bash run_drone.sh --fresh                    # the default map, outputs/memory
bash run_drone.sh --run alice --fresh        # its own map, outputs/runs/alice/memory
bash run_drone.sh --run bob --fresh          # and another
```

Each run picks a free brain port by itself, starting from 47569. If you
start a second run on a map that is already in use, it is refused and tells
you to add `--run NAME`. Each run's brain log is `outputs/logs/brain.log`
(for the default) or `outputs/runs/NAME/brain.log`.

**From the website.** Every login session gets its own run. The same account
on two devices counts as two sessions. The server runs at most **3** at once
(`DELTA_MAX_SIMULATIONS`), because the Habitat ones share one GPU. Each run
gets a numbered slot, and the slot decides everything it must not share with
the others:

| Slot | Viewer port (Run in browser) | DELTA brain port | DELTA map | Twin results |
|---|---|---|---|---|
| 1 | 6080 | 47600 | `simulation/outputs/runs/web-1/` | `physical-drone/outputs/web-1/` |
| 2 | 6081 | 47601 | `…/web-2/` | `…/web-2/` |
| 3 | 6082 | 47602 | `…/web-3/` | `…/web-3/` |

Nobody can see or stop anyone else's run. Logging out, the login running out
(after 2 hours), or a password reset stops that session's run. To let other
computers in, start uvicorn with `--host 0.0.0.0` and open ports
8000 and 6080–6082 (`app/README.md` has the details).

---

## 9. Checks

Run these after every merge and before every push:

```bash
cd app && python test_app.py                 # 29 tests: login, per-user data, lockout, reset, side-by-side runs
cd database && python3 test_db.py            # 31 tests
cd simulation && python tests/test_brain.py  # 21 checks, in (habitat)
```

---

## 10. How the parts fit together

```
  BROWSER                        SERVER (the PC running it)
 ┌────────────┐   /api/...   ┌──────────────┐  reads, writes  ┌────────────┐
 │ front end  │ ───────────► │   back end   │ ──────────────► │  database  │
 │ app/static │ ◄─────────── │ app/main.py  │ ◄────────────── │  delta.db  │
 └────────────┘     JSON     └──────┬───────┘                 └────────────┘
                                    │ Run button: starts it as its own program,
                                    │ one per login session, up to 3 at once
                  ┌─────────────────┴──────────────────┐
                  ▼                                    ▼
 ┌────────────────────────────────────┐   ┌──────────────────────────────┐
 │ simulation  (bash run_drone.sh)    │   │ physical-drone  (twin.fly,   │
 │   window ◄── socket ──► brain      │   │   twin.play): MuJoCo physics │
 │   (env habitat)        (env brain) │   │   (its own venv)             │
 └────────────────────────────────────┘   └──────────────────────────────┘
```

| | |
|---|---|
| Website ↔ database | ✅ log in, sign up, password reset, drones, scans, what changed |
| Website → simulations | ✅ DELTA, the physics twin, brain checks, the first demo, drone tests, setup check; several people at once |
| Simulation → database | ❌ not yet: the brain's maps stay in `simulation/outputs/`, so the dashboard shows sample data |
| Twin ↔ Habitat simulation | ❌ not yet: planned in `physical-drone/README.md` §5 |

---

## 11. When something fails

| You see | It means | Do this |
|---|---|---|
| `Could not find a version that satisfies the requirement mujoco==3.17.0` | the twin's venv is Python 3.9 (made inside `(habitat)`) | section 3, then redo section 4 |
| `No module named twin` | run from the wrong folder, or as a file | `cd physical-drone`, then `python -m twin.play` |
| `ModuleNotFoundError: habitat` | not in the habitat env | `conda activate habitat` |
| `ModuleNotFoundError: delta` | not in `simulation/` | `cd simulation` |
| `Another DELTA is already using the map in ...` | a second run on the same map | add `--run NAME` (section 8) |
| Website: "The server is already running 3 simulations" | all slots are taken | wait for one to finish, or raise `DELTA_MAX_SIMULATIONS` |
| Website: "The physics twin's Python was not found" | section 4 not done on the server | do section 4, or set `DELTA_TWIN_PYTHON` |
| Website: "Could not find conda" | uvicorn started where conda isn't set up | start uvicorn from a terminal where `conda activate habitat` works |
| `unable to find CUDA device 0 among 1 EGL devices` | WSL rendering | `simulation/README.md` §9 |
| Website shows "Server returned 404" | opened with Live Server | open http://127.0.0.1:8000 (uvicorn) instead |

---

## 12. Working on it

Work on a branch and merge into `main` through a pull request. Run the
checks (section 9) before you push. `simulation/ROADMAP.md` says what
comes next for the drone, and `physical-drone/README.md` §5 for the twin.
