# DELTA App

This is the web app part of DELTA. The backend (FastAPI) pulls data out of the SQLite database in `../database`, and a simple web page shows the drones and what changed between scans.

## Setting it up

Make sure the database exists first. If you haven't built it yet, `../database/README.md` has the details, but basically:

```
cd database
python3 create_db.py --seed
```

Then in the `app` folder, make a virtual environment and install the packages:

```
cd app
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

On Windows the activate line is `venv\Scripts\activate`. If it worked you'll see `(venv)` at the start of your terminal line.

## Running it

```
uvicorn main:app --reload
```

Go to http://127.0.0.1:8000 and you should see the page. The login page is at http://127.0.0.1:8000/login (demo login: `demo@example.com` / `demo1234`, phone `5555550100`). `Ctrl+C` stops it, and `--reload` restarts it on its own whenever you save a file. There's also an auto-made API page at http://127.0.0.1:8000/docs if you want to poke around.

## What's in here

- `main.py` is the backend (the endpoints and the database queries)
- `static/index.html` is the page
- `static/login.html` is the login page (log in, forgot email, forgot password, make an account)
- `simulations.py` starts the simulation scripts for the Run buttons on the page
- `virtual_screen.py` the invisible screen that lets the page show the simulation window
- `requirements.txt` is what pip installs

## Endpoints

- `/` the page
- `/api/health` just checks the backend is up
- `/api/drones` all the drones
- `/api/sessions` all the scans, newest first
- `/api/sessions/{id}/changes` what moved, appeared, or disappeared in that scan compared to the one before it
- `/login` the login page
- `/api/simulations` the simulations the page can run
- `POST /api/simulations/{id}/start?where=browser|server` runs one (`open-environment`, `drone-tests`, `check-setup`)
- `POST /api/simulations/stop` stops your running one
- `/api/simulations/run` your current or last run: status, exit code and output

The simulation endpoints need a login (401 without one).
- `POST /api/login` checks an email and password (401 if wrong)
- `POST /api/recover-email` looks up the email for a phone number (404 if not in the system)
- `POST /api/recover-password` checks an email + phone pair and "sends" a reset email (404 if no match)
- `POST /api/register` makes an account (400 not filled out / not valid, 409 already taken, 500 server error)

## Running the simulations from the page

The Simulations section on the page has a Run button for each of the scripts in `../simulation`:

- **Open environment** runs `demo/open_environment.py`, with two buttons:
  - **Run in browser** shows the window on the page, and your keyboard and mouse control it. Click inside it first.
  - **Run on server** opens it on the server's own monitor instead.
- **Drone tests** runs `tests/test_drone.py`.
- **Check setup** runs `config/check_setup.py`.

The app doesn't import Habitat. It runs each script as `conda run -n habitat python <script>` from the `simulation` folder, and the page shows what the script prints. So for the buttons to work:

- run uvicorn on the machine (or WSL) where the simulation already works (`simulation/README.md`)
- start it from a terminal where `conda activate habitat` works, so it can find conda and `HABITAT_DATA_PATH`

The app's venv is still separate from the habitat env; you don't install FastAPI into habitat.

### Run in browser

The server needs these once:

```
sudo apt install xvfb x11vnc novnc websockify
```

How it works: the window opens on an invisible screen on the server (Xvfb) instead of its monitor. x11vnc shares that screen, and websockify serves noVNC, a VNC viewer that runs in the page. Habitat still draws with the GPU.

### More than one person at a time

Every login session gets its own simulation: its own process, and for Run in browser its own invisible screen, viewer port and VNC password. The same account logged in on two devices counts as two sessions. Nobody sees or stops anyone else's run, and only the page that started a window gets its password.

- Each session runs one simulation at a time.
- The whole server runs at most **3** at once (they all load the house onto the same GPU). Change it with `DELTA_MAX_SIMULATIONS`, e.g. `DELTA_MAX_SIMULATIONS=2` if the GPU runs out of memory. When it's full, Run says so.
- Run in browser uses ports **6080, 6081, 6082** (one per allowed simulation, starting at `DELTA_VIEWER_PORT`).
- Logging out, or the login running out, stops that session's simulation.

To use it from another computer, start uvicorn with `--host 0.0.0.0` and open `http://<server-ip>:8000`. Open the ports: `sudo ufw allow 8000/tcp` and `sudo ufw allow 6080:6082/tcp` (widen the range if you raise `DELTA_MAX_SIMULATIONS`). The viewer ports aren't encrypted, so keep this to the lab network.

If the page says a viewer port is in use, an earlier run was left behind: `pkill websockify`.

Stop ends your run (pressing Esc in the window does too). Stopping uvicorn, or closing its terminal, stops everyone's.

If it says it can't find conda, point it at the habitat Python directly:

```
DELTA_SIM_PYTHON=~/miniconda3/envs/habitat/bin/python uvicorn main:app --reload
```

## Heads up

- Open the page through uvicorn (the link above), not Live Server. Live Server doesn't have our API, so you'll just get "Server returned 404."
- The login page works, but `/` doesn't check that you logged in yet, so it still shows every drone and always the newest scan.
- There's no email server, so "Forgot password" prints the reset link in the uvicorn terminal instead of emailing it. There's no reset-password page behind that link yet.
- Login added `first_name`, `last_name` and `phone_number` to the `User` table. If you made `delta.db` before that, rebuild it: `python3 create_db.py --reset --seed`.
- `venv` isn't in git, so everybody makes their own.
- The backend uses `database/db.py` to open the database, so keep the `database` folder next to `app`.

## Still to do

- make `/` require a login (keep the user in a session cookie)
- reset-password page and real email sending
- have the simulation save its sessions, maps and objects to the database, so the page shows real runs instead of the seed data
- map view
- let you pick which scan to look at instead of always the newest
