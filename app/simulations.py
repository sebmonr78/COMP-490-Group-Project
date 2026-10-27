"""Running the simulation scripts from the web page.

The web app and the simulation live in different Pythons: the app runs in
its own venv, and Habitat only exists inside the `habitat` conda env (see
simulation/README.md). So the app never imports Habitat. It starts the
simulation scripts as separate processes inside that env, the same way you
would from a terminal:

    conda run -n habitat python demo/open_environment.py

and keeps their output so the page can show it.

Only the scripts listed in SIMULATIONS can be started. The page sends an id,
never a path or a command, so nobody can use this to run anything else.

Each login session gets its own run: its own process, and for "Run in
browser" its own invisible screen and viewer port. The same account logged in
on two devices counts as two sessions, so it gets two runs. A session has one
run at a time, and the whole server at most DELTA_MAX_SIMULATIONS, because
every run loads the house onto the same GPU.

Settings (environment variables, all optional):

    DELTA_CONDA_ENV         conda env to run in   (default: the one project.json asks for)
    DELTA_SIM_PYTHON        a python to use instead of conda run, e.g.
                            ~/miniconda3/envs/habitat/bin/python
    DELTA_MAX_SIMULATIONS   how many can run at once, across everyone (default 3)
"""

import atexit
import json
import os
import shutil
import signal
import subprocess
import sys
import threading
from collections import deque
from datetime import datetime, timezone
from pathlib import Path

from fastapi import APIRouter, HTTPException, Request

import virtual_screen

SIMULATION_DIR = Path(__file__).resolve().parents[1] / "simulation"
PROJECT_CONFIG = SIMULATION_DIR / "config" / "project.json"

# Keep the last this-many lines of output. The demo prints a line every time
# you press P, so an unlimited log would grow for as long as it runs.
MAX_LINES = 2000

# What the page can run. Paths are relative to the simulation folder.
SIMULATIONS = {
    "open-environment": {
        "title": "Open environment (interactive)",
        "script": "demo/open_environment.py",
        "description": "Opens the HSSD house and lets you drive the drone. "
                       "In the browser: shown on this page. On the server: opens "
                       "on the server's own monitor.",
        "interactive": True,
    },
    "drone-tests": {
        "title": "Drone tests",
        "script": "tests/test_drone.py",
        "description": "Rolls, flies, crashes and lands the drone with no window, "
                       "and checks each one worked.",
        "interactive": False,
    },
    "check-setup": {
        "title": "Check setup",
        "script": "config/check_setup.py",
        "description": "Checks the conda env, Habitat versions and HSSD data are in place.",
        "interactive": False,
    },
}

router = APIRouter(prefix="/api/simulations")

# The login cookie main.py hands out. Each value is one login session, and
# that is who a run belongs to. main.py only lets logged-in requests in here.
SESSION_COOKIE = "delta_session"

# Every session's current (or most recent) run: session token --> run.
# Guarded by _lock because the reader threads write to the runs while
# requests read from them.
_lock = threading.Lock()
_runs = {}
_run_count = 0  # numbers each run, so the page can tell runs apart


def max_running():
    try:
        return max(1, int(os.environ.get("DELTA_MAX_SIMULATIONS", "3")))
    except ValueError:
        return 3


def viewer_ports():
    """The ports "Run in browser" can use: one per simulation that can run at once."""
    first = virtual_screen.first_viewer_port()
    return range(first, first + max_running())


def _owner(request: Request):
    token = request.cookies.get(SESSION_COOKIE)
    if not token:
        raise HTTPException(status_code=401, detail="Log in first")
    return token


def _now():
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")


def conda_env_name():
    if os.environ.get("DELTA_CONDA_ENV"):
        return os.environ["DELTA_CONDA_ENV"]
    try:
        with PROJECT_CONFIG.open(encoding="utf-8") as f:
            return json.load(f)["required_conda_environment"]
    except (OSError, ValueError, KeyError):
        return "habitat"


def python_command():
    """The command that starts Python inside the Habitat env, or None."""
    custom = os.environ.get("DELTA_SIM_PYTHON")
    if custom:
        return [os.path.expanduser(custom), "-u"]

    # conda sets CONDA_EXE when it is initialised in your shell.
    conda = os.environ.get("CONDA_EXE") or shutil.which("conda")
    if conda:
        # --no-capture-output streams the output as it happens instead of
        # holding it all until the script exits. conda run also activates
        # the env properly, which check_setup.py looks for.
        return [conda, "run", "--no-capture-output", "-n", conda_env_name(), "python", "-u"]

    return None


def simulation_environment():
    """The environment variables the simulation runs with.

    uvicorn runs inside the app's venv, and an activated venv puts its own
    bin folder at the front of PATH. Passed on as-is, `python` inside
    `conda run -n habitat` would still find the venv's Python first -- the
    wrong version, with no Habitat in it. So the venv is taken back out.
    """
    env = dict(os.environ, PYTHONUNBUFFERED="1")

    venv_dirs = set()
    if env.get("VIRTUAL_ENV"):
        venv_dirs.add(os.path.realpath(env.pop("VIRTUAL_ENV")))
    if sys.prefix != sys.base_prefix:  # running in a venv even if not "activated"
        venv_dirs.add(os.path.realpath(sys.prefix))

    def in_venv(folder):
        folder = os.path.realpath(folder)
        return any(folder == v or folder.startswith(v + os.sep) for v in venv_dirs)

    env["PATH"] = os.pathsep.join(
        folder for folder in env.get("PATH", "").split(os.pathsep)
        if folder and not in_venv(folder)
    )
    env.pop("PYTHONHOME", None)
    return env


def _read_output(run):
    """Runs in a background thread: copies the process output into the log."""
    for line in run["process"].stdout:
        with _lock:
            run["lines"].append(line.rstrip("\n"))
            run["line_count"] += 1

    code = run["process"].wait()

    # The window is gone, so the screen it was on can go too
    if run["screen"]:
        run["screen"].stop()

    with _lock:
        run["exit_code"] = code
        run["ended_at"] = _now()
        if run["status"] == "stopping":
            run["status"] = "stopped"
        else:
            run["status"] = "finished" if code == 0 else "failed"
        # Its owner logged out while it ran: nobody is left to show it to
        token = run.get("owner_gone")
        if token and _runs.get(token) is run:
            del _runs[token]


def _is_running(run):
    return run is not None and run["status"] in ("starting", "running", "stopping")


def _running_count():
    return sum(1 for run in _runs.values() if _is_running(run))


def _summary(run, since=0):
    if run is None:
        return None
    lines = list(run["lines"])
    dropped = run["line_count"] - len(lines)  # lines that fell off the front
    start = max(since - dropped, 0)
    return {
        "number": run["number"],
        "id": run["id"],
        "title": SIMULATIONS[run["id"]]["title"],
        "status": run["status"],
        "exit_code": run["exit_code"],
        "started_at": run["started_at"],
        "ended_at": run["ended_at"],
        "lines": lines[start:],
        "line_count": run["line_count"],
        # Where the browser can watch it, while it runs (None = not in the browser)
        "viewer": run["viewer"] if run["status"] in ("running", "stopping") else None,
        "server_running": _running_count(),
        "server_max": max_running(),
    }


@router.get("")
def list_simulations(request: Request):
    owner = _owner(request)
    with _lock:
        mine = _runs.get(owner)
        running = mine["id"] if _is_running(mine) else None
    missing = virtual_screen.missing_requirements()
    return [
        {"id": sim_id, "title": sim["title"], "description": sim["description"],
         "interactive": sim["interactive"], "running": sim_id == running,
         # Can it be shown in the browser? If not, why not.
         "browser_ready": sim["interactive"] and not missing,
         "browser_missing": missing if sim["interactive"] else []}
        for sim_id, sim in SIMULATIONS.items()
    ]


@router.get("/run")
def current_run(request: Request, since: int = 0):
    # The page polls this. `since` = how many lines it already has, so it
    # only gets the new ones. Each session only ever sees its own run.
    owner = _owner(request)
    with _lock:
        return {"run": _summary(_runs.get(owner), since)}


@router.post("/{sim_id}/start")
def start_simulation(sim_id: str, request: Request, where: str = "browser"):
    # where: "browser" = window on its own invisible screen, shown on the page
    #        "server"  = window on the server's own monitor
    # Only matters for interactive simulations; the others have no window.
    global _run_count
    owner = _owner(request)

    if sim_id not in SIMULATIONS:
        raise HTTPException(status_code=404, detail="No simulation with that id")
    if where not in ("browser", "server"):
        raise HTTPException(status_code=400, detail="where must be browser or server")

    command = python_command()
    if command is None:
        raise HTTPException(
            status_code=503,
            detail="Could not find conda. Start uvicorn from a terminal where "
                   "`conda activate habitat` works, or set DELTA_SIM_PYTHON.",
        )

    in_browser = SIMULATIONS[sim_id]["interactive"] and where == "browser"

    # Claim a place (and a viewer port) under the lock, then start things
    # without holding it: the screen takes a second or two, and everyone
    # else's page is polling in the meantime.
    with _lock:
        mine = _runs.get(owner)
        if _is_running(mine):
            raise HTTPException(
                status_code=409,
                detail=f"{SIMULATIONS[mine['id']]['title']} is already running. Stop it first.",
            )
        if _running_count() >= max_running():
            raise HTTPException(
                status_code=503,
                detail=f"The server is already running {max_running()} simulations, "
                       f"the most it allows at once. Try again when one finishes.",
            )

        port = None
        if in_browser:
            taken = {run["port"] for run in _runs.values() if _is_running(run) and run["port"]}
            port = next((p for p in viewer_ports() if p not in taken), None)
            if port is None:  # can't happen while ports == max_running, but be safe
                raise HTTPException(status_code=503, detail="No free viewer port. Try again shortly.")

        _run_count += 1
        run = {
            "number": _run_count,
            "id": sim_id,
            "process": None,
            "status": "starting",
            "exit_code": None,
            "started_at": _now(),
            "ended_at": None,
            "lines": deque(maxlen=MAX_LINES),
            "line_count": 0,
            "screen": None,
            "port": port,
            "viewer": None,
        }
        _runs[owner] = run

    def give_up(status_code, detail):
        with _lock:
            if _runs.get(owner) is run:
                del _runs[owner]
        raise HTTPException(status_code=status_code, detail=detail)

    env = simulation_environment()

    screen = None
    if in_browser:
        screen = virtual_screen.VirtualScreen(port)
        try:
            screen.start()
        except RuntimeError as error:
            give_up(503, f"Could not show it in the browser: {error}")
        env = screen.environment(env)

    try:
        process = subprocess.Popen(
            command + [SIMULATIONS[sim_id]["script"]],
            cwd=SIMULATION_DIR,
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            errors="replace",
            bufsize=1,
            # Own process group, so Stop also kills what `conda run` started
            start_new_session=(os.name == "posix"),
        )
    except OSError as error:
        if screen:
            screen.stop()
        give_up(500, f"Could not start the simulation: {error}")

    with _lock:
        run["process"] = process
        run["screen"] = screen
        run["viewer"] = {"port": port, "password": screen.password} if screen else None
        stopped_meanwhile = run["status"] == "stopping"  # logged out while starting
        if not stopped_meanwhile:
            run["status"] = "running"
    threading.Thread(target=_read_output, args=(run,), daemon=True).start()
    if stopped_meanwhile:
        _stop_run(run)

    with _lock:
        return _summary(run)


@router.post("/stop")
def stop_simulation(request: Request):
    owner = _owner(request)
    with _lock:
        run = _runs.get(owner)
        if not _is_running(run):
            raise HTTPException(status_code=409, detail="Nothing is running")
        if run["status"] == "starting":
            raise HTTPException(status_code=409, detail="It is still starting. Try again in a moment.")
        run["status"] = "stopping"
    _stop_run(run)
    return {"message": "Stopped"}


def _stop_run(run):
    process = run["process"]
    _kill(process, signal.SIGTERM)
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        _kill(process, signal.SIGKILL if os.name == "posix" else signal.SIGTERM)


def stop_for_session(token):
    """main.py calls this when someone logs out or their login runs out:
    their simulation stops, and its output is forgotten once it has."""
    with _lock:
        run = _runs.get(token)
        if run is None:
            return
        if not _is_running(run):
            del _runs[token]
            return
        # Stays in _runs until it has really stopped, so its viewer port and
        # its place on the GPU aren't handed to someone else too early.
        # _read_output removes it then.
        run["owner_gone"] = token
        status = run["status"]
        run["status"] = "stopping"
    if status == "running":
        threading.Thread(target=_stop_run, args=(run,), daemon=True).start()
    # "starting": start_simulation sees "stopping" and stops it once it can


def _kill(process, sig):
    try:
        if os.name == "posix":
            os.killpg(process.pid, sig)
        else:
            process.terminate()
    except (ProcessLookupError, PermissionError):
        pass  # already gone


# Closing the terminal sends SIGHUP, which ends Python without any cleanup,
# so a running simulation and its in-browser screen would be left behind
# (holding the viewer port). Turn it into a normal shutdown instead.
if os.name == "posix" and threading.current_thread() is threading.main_thread() \
        and signal.getsignal(signal.SIGHUP) == signal.SIG_DFL:
    signal.signal(signal.SIGHUP, lambda *_: os.kill(os.getpid(), signal.SIGTERM))


@atexit.register
def stop_on_shutdown():
    # main.py calls this when uvicorn shuts down (Ctrl+C, a --reload restart,
    # the terminal closing), so no simulation window is left running with
    # nothing attached to it. atexit is a backup for other ways Python exits.
    with _lock:
        runs = list(_runs.values())
    for run in runs:
        if run["process"] and run["process"].poll() is None:
            _kill(run["process"], signal.SIGTERM)
        if run["screen"]:
            run["screen"].stop()
