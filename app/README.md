# DELTA App

This is the web app part of DELTA. The backend (FastAPI) pulls data out of the SQLite database in `../database`, and a simple web page shows the drones and what changed between scans. You log in first, then you get to the dashboard.

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

If the venv comes out with no `pip` in it (Ubuntu without the `python3-venv`
package), make it with conda's Python instead: `~/miniconda3/bin/python3 -m venv venv`.

On Windows the activate line is `venv\Scripts\activate`. If it worked you'll see `(venv)` at the start of your terminal line.

## Running it

```
uvicorn main:app --reload
```

Go to http://127.0.0.1:8000 and you should see the login page. Log in with the demo account (`demo@example.com` / `demo1234`, phone `5555550100`) and you land on the dashboard at http://127.0.0.1:8000/dashboard. `Ctrl+C` stops it, and `--reload` restarts it on its own whenever you save a file. There's also an auto-made API page at http://127.0.0.1:8000/docs if you want to poke around.

## What's in here

- `main.py` is the backend (the endpoints and the database queries)
- `static/index.html` is the dashboard (drones, what changed, simulations)
- `static/login.html` is the login page (log in, forgot email, forgot password, make an account)
- `static/reset.html` is the page for choosing a new password
- `mailer.py` sends the reset email (only when the email settings below are set)
- `test_app.py` has the automated checks (login, per-user data, lockout, password reset, reset email, simulations side by side)
- `simulations.py` starts the simulation scripts for the Run buttons on the page
- `virtual_screen.py` the invisible screen that lets the page show the simulation window
- `requirements.txt` is what pip installs

## How login works

- When you log in, the server makes a random "ticket" and remembers which user it belongs to. Your browser keeps it as a cookie (`delta_session`) and sends it back on every request.
- Tickets last 2 hours (`SESSION_HOURS` in `main.py`), then you have to log in again. They only live in memory, so restarting the server logs everybody out too.
- `/dashboard`, the data endpoints and everything under `/api/simulations` need a ticket. No ticket and the page sends you back to the login page, and the API answers 401.
- Everybody only sees their own stuff. Drones are matched by `owner_id` and scans go through their drone. Asking for somebody else's scan gives 404 "Scan not found".
- Passwords are hashed (PBKDF2 with a random salt, see `database/passwords.py`), never stored as plain text.
- Forgot password: unless the email settings are on (see below), the reset link prints in the uvicorn terminal instead of being emailed. The link works once and lasts 15 minutes (`RESET_MINUTES`). It opens `/reset-password` where you pick a new password, and that also logs out any browser that was already signed in as you.
- Wrong passwords are counted per email. After 5 wrong tries that email is locked for 5 minutes, even for the right password (`MAX_LOGIN_FAILURES` and `LOGIN_LOCK_MINUTES` in `main.py`).

## Turning on reset emails

Reset emails are off by default, so the reset link just prints in the uvicorn terminal. To send real emails, set these before you start uvicorn:

```
DELTA_SMTP_HOST=smtp.gmail.com DELTA_SMTP_USER=you@gmail.com DELTA_SMTP_PASSWORD=your-app-password uvicorn main:app --reload
```

Optional extras: `DELTA_SMTP_PORT` (default 587), `DELTA_SMTP_FROM` (default is the user) and `DELTA_BASE_URL` (default http://127.0.0.1:8000, the address used in the link).

- Never put these values in a file that gets committed.
- Gmail needs 2-step verification turned on and an "app password". Your normal password won't work.
- If the mail server can't be reached, the page shows the Server Error box and the reset link doesn't work.
- It's only been tested with a fake mail server so far, not a real inbox.

## Running the tests

In the `app` folder, with the venv on:

```
pip install -r requirements.txt
python3 test_app.py
```

It should end with `OK`. The tests use a temporary database (your `delta.db` is not touched) and never send a real email. Run them after every merge and before every push. If one fails, its name says what broke, like `test_logged_out_data_and_simulation_endpoints_return_401`.

## Endpoints

- `/` the login page (`/login` is the same page)
- `/dashboard` the main page (needs a login, otherwise it sends you back to `/`)
- `/api/health` just checks the backend is up
- `/api/drones` your drones (needs a login)
- `/api/sessions` your scans, newest first (needs a login)
- `/api/sessions/{id}/changes` what moved, appeared, or disappeared in that scan compared to the one before it (needs a login, 404 if it's not your scan)
- `/api/simulations` the simulations the page can run (all the simulation endpoints need a login too)
- `POST /api/simulations/{id}/start?where=browser|server` runs one (`delta`, `twin-flight`, `twin-play`, `open-environment`, `brain-checks`, `drone-tests`, `check-setup`)
- `/api/simulations/run/picture?number=N` the physics twin's flight plot, for your own run
- `POST /api/simulations/stop` stops your running one (only yours: each login session has its own)
- `/api/simulations/run` your current or last run: status, exit code and output
- `POST /api/login` checks an email and password (401 if wrong) and gives you a ticket
- `POST /api/logout` throws away your ticket
- `POST /api/recover-email` looks up the email for a phone number (404 if not in the system)
- `POST /api/recover-password` checks an email + phone pair and "sends" a reset email (404 if no match)
- `/reset-password?token=...` the page for choosing a new password
- `POST /api/reset-password` sets a new password from a reset link (400 if the link is bad or expired, or the passwords don't match or are too short)
- `POST /api/register` makes an account (400 not filled out / not valid, 409 already taken, 500 server error)

## Running the simulations from the page

The Simulations section on the page has a Run button for each of these (the list is `SIMULATIONS` in `simulations.py`):

- **DELTA** -- the **"(new)"** buttons on the *Drive the drone* row -- runs `run_drone.sh --run web-<slot> --port <port> --fresh --depth oracle` in `../simulation`: the drone and its brain, the map, the rooms and the 3D replica, as you drive. Two buttons:
  - **Run in browser** shows the window on the page, and your keyboard and mouse control it. Click inside it first.
  - **Run on server** opens it on the server's own monitor instead.
  Stop shuts it down as Esc would: the brain saves its map first (it gets 45 seconds).
- **Physics twin: scripted flight** runs `python -m twin.fly twin/flights/hover.json` in `../physical-drone`. When it lands, the page shows the hover time, the energy used, each step, and the flight plot.
- **Physics twin: fly it yourself** runs `python -m twin.play`. Same two buttons as DELTA.
- **Brain checks** runs `tests/test_brain.py`: the brain's 21 checks, including that it cannot cheat.
- **The first demo** -- the plain buttons on the *Drive the drone* row, without "(new)" -- runs `demo/open_environment.py`: drive the drone next to the true layout, no brain.
- **Drone tests** runs `tests/test_drone.py`.
- **Check setup** runs `config/check_setup.py`.

The app doesn't import Habitat or MuJoCo. It starts each one as its own program, and the page shows what it prints:

- the Habitat scripts as `conda run -n habitat python <script>` from `../simulation`
- DELTA through its launcher, which starts the brain (conda env `brain`) and the window (env `habitat`) itself
- the physics twin with its own Python: `../physical-drone/venv` if it exists, otherwise the conda env `twin`, or whatever `DELTA_TWIN_PYTHON` points at

So for the buttons to work:

- run uvicorn on the machine (or WSL) where the simulation already works (`simulation/README.md`), and where the twin is set up (`physical-drone/README.md`, step 1)
- start it from a terminal where `conda activate habitat` works, so it can find conda and `HABITAT_DATA_PATH`

The app's venv is still separate from the habitat env and from the twin's; you don't install FastAPI into either.

### Run in browser

The server needs these once:

```
sudo apt install xvfb x11vnc novnc websockify
```

How it works: the window opens on an invisible screen on the server (Xvfb) instead of its monitor. x11vnc shares that screen, and websockify serves noVNC, a VNC viewer that runs in the page. Habitat still draws with the GPU.

### More than one person at a time

Every login session gets its own simulation: its own process, and for Run in browser its own invisible screen, viewer port and VNC password. The same account logged in on two devices counts as two sessions. Nobody sees or stops anyone else's run, and only the page that started a window gets its password.

- Each session runs one simulation at a time.
- The whole server runs at most **3** at once (the Habitat ones all load the house onto the same GPU). Change it with `DELTA_MAX_SIMULATIONS`, e.g. `DELTA_MAX_SIMULATIONS=2` if the GPU runs out of memory. When it's full, Run says so.
- Each run takes a numbered slot while it runs, and gets everything it must not share from it:

  | Slot | Viewer port | DELTA brain port | DELTA map and log | Twin results |
  |---|---|---|---|---|
  | 1 | 6080 | 47600 | `simulation/outputs/runs/web-1/` | `physical-drone/outputs/web-1/` |
  | 2 | 6081 | 47601 | `simulation/outputs/runs/web-2/` | `physical-drone/outputs/web-2/` |
  | 3 | 6082 | 47602 | `simulation/outputs/runs/web-3/` | `physical-drone/outputs/web-3/` |

  The viewer ports start at `DELTA_VIEWER_PORT` (6080). DELTA runs started from a terminal use brain ports 47569–47599, so they never meet the website's.
- Logging out, the login running out, or a password reset stops that session's simulation.

To use it from another computer, start uvicorn with `--host 0.0.0.0` and open `http://<server-ip>:8000`. Open the ports: `sudo ufw allow 8000/tcp` and `sudo ufw allow 6080:6082/tcp` (widen the range if you raise `DELTA_MAX_SIMULATIONS`). The viewer ports aren't encrypted, so keep this to the lab network.

If the page says a viewer port is in use, an earlier run was left behind: `pkill websockify`.

Stop ends your run (pressing Esc in the window does too). Stopping uvicorn, or closing its terminal, stops everyone's.

If it says the physics twin's Python was not found, set the twin up (`physical-drone/README.md`, step 1), or point at its Python: `DELTA_TWIN_PYTHON=~/miniconda3/envs/twin/bin/python`.

If it says it can't find conda, point it at the habitat Python directly:

```
DELTA_SIM_PYTHON=~/miniconda3/envs/habitat/bin/python uvicorn main:app --reload
```

## Heads up

- Open the page through uvicorn (the link above), not Live Server. Live Server doesn't have our API, so you'll just get "Server returned 404."
- Login added `first_name`, `last_name` and `phone_number` to the `User` table. If you made `delta.db` before that, rebuild it: `python3 create_db.py --reset --seed`.
- `venv` isn't in git, so everybody makes their own.
- The backend uses `database/db.py` to open the database, so keep the `database` folder next to `app`.
- The login lockout is basic. It counts per email, so someone can lock a person out for 5 minutes by typing wrong passwords for their email. The counts only live in memory, so restarting the server clears them, and old entries are never cleaned up. That's fine for the simulation phase but it would need fixing before the app goes on the internet.

## Still to do

- make the login lockout stronger (also track the computer's address, keep the counts in the database, clean up old entries; the lockout tests in `test_app.py` will need updating too)
- try the reset email with a real mail account (the code is ready, see Turning on reset emails)
- have the simulation save its sessions, maps and objects to the database
- map view and picking which scan to look at (on hold until the simulation settles)
