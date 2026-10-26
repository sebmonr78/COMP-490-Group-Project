# DELTA

**Detecting Environmental Layout and Tracking Alterations** -- COMP 490, team
Y.A.K.S. A drone that rolls and flies through a home, maps it on its own, and
notices what changed since last time. Built in simulation first (Habitat),
hardware later.

## What is where

| Folder | What it is | Built with |
|---|---|---|
| [`simulation/`](simulation/README.md) | The drone in a simulated house: the **world** (house, drone, its sensors), the drone's **brain** (where am I, the 2D map, rooms, the 3D replica), and the **window** to watch it in | Python, Habitat, RTAB-Map |
| [`app/`](app/README.md) | The **website**: the back end (`main.py`, FastAPI) and the front end (`static/`: the log-in page and the dashboard) | Python, FastAPI, HTML |
| [`database/`](database/README.md) | The **database**: users, drones, scans, maps, objects seen, what changed | SQLite |

## How the parts fit together

```
  BROWSER                        SERVER (the PC running it)
 ┌────────────┐   /api/...   ┌──────────────┐  reads, writes  ┌────────────┐
 │ front end  │ ───────────► │   back end   │ ──────────────► │  database  │
 │ app/static │ ◄─────────── │ app/main.py  │ ◄────────────── │  delta.db  │
 └────────────┘     JSON     └──────┬───────┘                 └────────────┘
                                    │ Run button: starts it as its own program
                                    ▼
                    ┌────────────────────────────────────┐
                    │ simulation  (bash run_drone.sh)    │
                    │   window ◄── socket ──► brain      │
                    │   (env habitat)        (env brain) │
                    └────────────────────────────────────┘
```

- The **front end** is two pages in the browser: log in (or sign up, or
  recover an email or password), then the dashboard -- your drones, what
  changed in the latest scan, and Run buttons for the simulations.
- The **back end** answers the pages, checks logins against the database,
  and starts the simulations when asked, showing what they print. Their
  window opens on the server's screen, or inside the page (needs Xvfb,
  x11vnc and noVNC on the server; see `app/README.md`).
- The **database** keeps everything the website shows. Its tables follow the
  drone's work: a user has drones, a drone has scans (sessions), a scan has
  maps, objects and changes.
- The **simulation** is the drone itself: two programs talking over a local
  socket, exactly as the real drone will talk to the laptop -- the window
  (world and sensors) and the brain (which only ever gets what real sensors
  would give it).

## What works, and what is not connected yet

| | |
|---|---|
| Website ↔ database | ✅ log in, sign up, drones, scans, what changed |
| Website → simulation | ✅ Run: **DELTA** (the drone and its brain), brain checks, the first simple demo, drone tests, setup check |
| Simulation → database | ❌ not yet: the brain's maps, rooms, objects and changes stay in `simulation/outputs/`, so the dashboard shows sample data. This is the next bridge to build. |

## Run it

**The simulation** (setup once: `simulation/README.md`):

```bash
cd simulation && bash run_localization.sh
```

**The database** (once):

```bash
cd database && python3 create_db.py --seed
```

**The website** (setup once: `app/README.md`), then open http://127.0.0.1:8000
(demo log-in from the sample data: `demo@example.com` / `demo1234`):

```bash
cd app && venv/bin/uvicorn main:app
```

## Checks

```bash
cd simulation && python tests/test_brain.py     # the brain: 21 checks, including that it cannot cheat
cd database && python3 test_db.py               # the database: 31 tests
```

## Working on it

Work on a branch and merge into `main` through a pull request. Each folder's
README says what is in it; `simulation/ROADMAP.md` says what comes next for
the drone.
