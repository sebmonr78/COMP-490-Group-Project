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
- `test_app.py` has the automated checks (login, per-user data, lockout, password reset, reset email)
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
- `POST /api/simulations/{id}/start?where=browser|server` runs one (`open-environment`, `drone-tests`, `check-setup`)
- `POST /api/simulations/stop` stops the one that's running
- `/api/simulations/run` the current or last run: status, exit code and output
- `POST /api/login` checks an email and password (401 if wrong) and gives you a ticket
- `POST /api/logout` throws away your ticket
- `POST /api/recover-email` looks up the email for a phone number (404 if not in the system)
- `POST /api/recover-password` checks an email + phone pair and "sends" a reset email (404 if no match)
- `/reset-password?token=...` the page for choosing a new password
- `POST /api/reset-password` sets a new password from a reset link (400 if the link is bad or expired, or the passwords don't match or are too short)
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

How it works: the window opens on an invisible screen on the server (Xvfb) instead of its monitor. x11vnc shares that screen, and websockify serves noVNC, a VNC viewer that runs in the page, on port **6080**. Habitat still draws with the GPU.

To use it from another computer, start uvicorn with `--host 0.0.0.0` and open `http://<server-ip>:8000`. Both ports have to be reachable: `sudo ufw allow 8000/tcp` and `sudo ufw allow 6080/tcp`. Nothing on either port asks for a password, so keep this to the lab network.

If the page says port 6080 is in use, an earlier run was left behind: `pkill websockify`. `DELTA_VIEWER_PORT` picks a different port.

Only one simulation runs at a time. Stop ends it (pressing Esc in the window does too). Stopping uvicorn, or closing its terminal, also stops it.

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
