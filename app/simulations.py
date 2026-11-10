"""Simulation Manager: starts, tracks and stops the simulations for the web dashboard.

Module      : simulations.py   (Design: Simulation Manager)
Date        : Created 2026-10-05, last updated 2026-11-02
Programmer  : Sebastian

DESCRIPTION
    The web app and the simulations need different Pythons: the app runs in
    its own venv, Habitat exists only in the `habitat` conda env, and the
    physics twin needs Python 3.10+ with MuJoCo. So this module imports none
    of them. It starts each simulation as a separate process, the same way it
    would be started from a terminal, keeps its output for the page, and
    serves the endpoints under /api/simulations.

    Only the simulations listed in SIMULATIONS can be started. The page sends
    an id, never a path or a command, so nothing else can be run through it.

    Each login session (one account on one device) gets its own run. A
    session runs one simulation at a time, and the server runs at most
    DELTA_MAX_SIMULATIONS (default 3) across everyone, because the Habitat
    simulations share one GPU.

IMPORTANT FUNCTIONS
    start_simulation(sim_id, request, where="browser") -> dict
        POST /api/simulations/{sim_id}/start.
        In : simulation id, the caller's session cookie, and where its
             window opens ("browser" or "server").
        Out: summary of the new run (see _summary).
        Errors: 404 unknown id, 409 already running, 503 server full or
        something not installed.

    stop_simulation(request) -> dict
        POST /api/simulations/stop.
        In : the caller's session cookie.
        Out: {"message": "Stopping"}; the run stops in the background.

    current_run(request, since=0) -> dict
        GET /api/simulations/run, polled by the page once a second.
        In : session cookie; how many output lines the page already has.
        Out: {"run": summary with only the newer lines, or None}.

    list_simulations(request) -> list[dict]
        GET /api/simulations.
        Out: one entry per simulation: id, title, description, whether it is
             running, and whether the in-browser view is available.

    stop_for_session(token) -> None
        Called by main.py on logout or login expiry: stops that session's run.

    stop_on_shutdown() -> None
        Called when uvicorn shuts down: stops every run and its screen.

IMPORTANT DATA STRUCTURES
    SIMULATIONS : dict[str, dict]
        The catalogue of what can run: id -> kind ("habitat", "command" or
        "twin"), title, what to run, and options (interactive, stop_seconds,
        results).

    _runs : dict[str, dict]
        Session token -> that session's current or last run. A run holds the
        process, its status ("starting", "running", "stopping", "stopped",
        "finished" or "failed"), output, slot, virtual screen and viewer.
        Guarded by _lock, since output-reader threads and requests share it.

    run["lines"] : collections.deque(maxlen=MAX_LINES)
        The last 2000 lines of output; run["line_count"] counts all of them.

ALGORITHMS
    Slot allocation (first fit)
        Each run takes the lowest free slot number (1..DELTA_MAX_SIMULATIONS).
        Everything two runs must not share comes from it: viewer port
        6080 + slot - 1, DELTA's brain port 47600 + slot - 1, and the
        web-<slot> map and results folders. Chosen over asking the OS for
        random free ports because the ports stay predictable (a firewall
        rule can list them) and are reused, and the fixed pool doubles as
        the GPU limit.

    Output ring buffer with a cursor
        Output goes into a bounded deque, plus a total line count. The page
        asks for lines "since" the count it already has, so each poll only
        downloads new lines and memory stays fixed however long a run lasts.
        Chosen over WebSocket streaming because once-a-second HTTP polling is
        simpler and works on any network the page itself works on.

    Two-phase start
        A slot is claimed while holding the lock; the slow work (virtual
        screen, process start) happens after it is released, so other users'
        polling is never blocked while a run starts.

    Graceful stop
        SIGTERM to the run's whole process group, wait up to the
        simulation's stop_seconds, then SIGKILL. The group also catches what
        `conda run` started; the wait lets DELTA save its map first.

SETTINGS (environment variables, all optional)
    DELTA_CONDA_ENV         conda env for the Habitat scripts (default: from project.json)
    DELTA_SIM_PYTHON        a Python to use instead of `conda run`
    DELTA_MAX_SIMULATIONS   how many can run at once, across everyone (default 3)
    DELTA_TWIN_PYTHON       the physics twin's Python (default: physical-drone/venv,
                            else the conda env `twin`)
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
from fastapi.responses import FileResponse

import virtual_screen

REPOSITORY = Path(__file__).resolve().parents[1]
SIMULATION_DIR = REPOSITORY / "simulation"
TWIN_DIR = REPOSITORY / "physical-drone"
PROJECT_CONFIG = SIMULATION_DIR / "config" / "project.json"

# Keep the last this-many lines of output. The demo prints a line every time
# you press P, so an unlimited log would grow for as long as it runs.
MAX_LINES = 2000

# DELTA's brain listens on a port of its own in every run. A run started from
# a terminal picks one from 47569 up (run_drone.sh); the website's runs use
# these, so the two never meet.
FIRST_BRAIN_PORT = 47600

# What the page can run, in the order shown. Each has a `kind`:
#
#   habitat   `script` runs with the habitat env's Python, in simulation/
#   command   `command` runs as it is, in simulation/. {run}, {brain_port}
#             and {slot} are filled in from the run's slot. DELTA's launcher
#             starts its two processes, the brain (env `brain`) and the
#             window (env `habitat`), itself.
#   twin      `args` run with the physics twin's Python, in physical-drone/.
#             {out} is the run's results folder.
#
# `stop_seconds`: how long Stop waits for it to finish by itself before it
# is killed. DELTA needs time: the brain saves its map first.
SIMULATIONS = {
    "open-environment": {
        "kind": "habitat",
        "title": "Drive the drone (interactive)",
        "script": "demo/open_environment.py",
        # Its "(new)" buttons start this one instead: the page shows DELTA as
        # a second pair of buttons on this row, not as a row of its own.
        "newer": "delta",
        "description": "(new): DELTA as it is now -- the drone and its brain, every panel, the "
                       "2D map, the rooms and the 3D replica. Without (new): the first demo, the "
                       "drone next to the true layout, no brain. Click the window before using "
                       "the keys.",
        "interactive": True,
    },
    "delta": {
        "kind": "command",
        "title": "DELTA (interactive)",
        "command": ["bash", "run_drone.sh", "--run", "{run}", "--port", "{brain_port}",
                    "--fresh", "--depth", "oracle"],
        "needs_conda": True,
        "stop_seconds": 45,
        "description": "The drone and its brain: it finds where it is (RTAB-Map), builds the "
                       "2D map, the rooms and the 3D replica as you drive. Arrows drive, "
                       "1/0 fly and land, R roams. Depth is the simulator's, so what is "
                       "tested is the brain's own localization and mapping.",
        "interactive": True,
    },
    "twin-flight": {
        "kind": "twin",
        "title": "Physics twin: scripted flight",
        "args": ["-m", "twin.fly", "twin/flights/hover.json", "--out", "{out}"],
        "results": True,
        "description": "The real drone's hardware in MuJoCo, flying twin/flights/hover.json: "
                       "take off, hover, fly 2 m, climb, turn, ride a gust, land. Shows its "
                       "power, hover time and a plot of the flight. No Habitat or GPU needed.",
        "interactive": False,
    },
    "twin-play": {
        "kind": "twin",
        "title": "Physics twin: fly it yourself (interactive)",
        "args": ["-m", "twin.play"],
        "description": "The same drone in a room with a table, a sofa and a doorway. 1 takes "
                       "off, arrows fly, G is a gust, L lands. Click the window before using "
                       "the keys.",
        "interactive": True,
    },
    "brain-checks": {
        "kind": "habitat",
        "title": "Brain checks",
        "script": "tests/test_brain.py",
        "description": "The brain's 21 checks: that it cannot cheat (no Habitat, only what "
                       "real sensors measure), and its map, rooms, replica and height.",
        "interactive": False,
    },
    "drone-tests": {
        "kind": "habitat",
        "title": "Drone tests",
        "script": "tests/test_drone.py",
        "description": "Rolls, flies, crashes and lands the drone with no window, "
                       "and checks each one worked.",
        "interactive": False,
    },
    "check-setup": {
        "kind": "habitat",
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
    """How many simulations may run at once on the whole server.

    Returns:
        int: DELTA_MAX_SIMULATIONS, at least 1; 3 if it is unset or not a number.
    """
    try:
        return max(1, int(os.environ.get("DELTA_MAX_SIMULATIONS", "3")))
    except ValueError:
        return 3


def _owner(request: Request):
    """The login session a request belongs to.

    Args:
        request: the incoming request; its session cookie identifies the session.

    Returns:
        str: the session token, used as the key in _runs.

    Raises:
        HTTPException(401): the request has no session cookie.
    """
    token = request.cookies.get(SESSION_COOKIE)
    if not token:
        raise HTTPException(status_code=401, detail="Log in first")
    return token


def _now():
    """The current UTC time as "YYYY-MM-DD HH:MM:SS"."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")


# ---------------------------------------------------------------------------
# Finding the right Python for each kind
# ---------------------------------------------------------------------------

def conda_env_name():
    """The conda env the Habitat scripts run in.

    Returns:
        str: DELTA_CONDA_ENV if set, else the env simulation/config/project.json
        names, else "habitat".
    """
    if os.environ.get("DELTA_CONDA_ENV"):
        return os.environ["DELTA_CONDA_ENV"]
    try:
        with PROJECT_CONFIG.open(encoding="utf-8") as f:
            return json.load(f)["required_conda_environment"]
    except (OSError, ValueError, KeyError):
        return "habitat"


def conda_executable():
    """Path to the conda program, or None if conda cannot be found.

    conda sets CONDA_EXE when it is initialised in the shell; otherwise PATH
    is searched.
    """
    return os.environ.get("CONDA_EXE") or shutil.which("conda")


def python_command():
    """The command that starts Python inside the Habitat env.

    Returns:
        list[str] | None: DELTA_SIM_PYTHON if set, else `conda run` in the
        Habitat env, or None when neither is available.
    """
    custom = os.environ.get("DELTA_SIM_PYTHON")
    if custom:
        return [os.path.expanduser(custom), "-u"]

    conda = conda_executable()
    if conda:
        # --no-capture-output streams the output as it happens instead of
        # holding it all until the script exits. conda run also activates
        # the env properly, which check_setup.py looks for.
        return [conda, "run", "--no-capture-output", "-n", conda_env_name(), "python", "-u"]

    return None


def twin_python_command():
    """The command that starts the physics twin's Python.

    The twin needs Python 3.10+ and MuJoCo, which neither the app's venv nor
    the habitat env (3.9) has.

    Returns:
        list[str] | None: DELTA_TWIN_PYTHON if set, else physical-drone/venv,
        else the conda env `twin`, or None if none of them exists.
    """
    custom = os.environ.get("DELTA_TWIN_PYTHON")
    if custom:
        return [os.path.expanduser(custom), "-u"]

    for candidate in (TWIN_DIR / "venv" / "bin" / "python",
                      TWIN_DIR / "venv" / "Scripts" / "python.exe"):
        if candidate.exists():
            return [str(candidate), "-u"]

    conda = conda_executable()
    if conda:
        # conda lives in <root>/bin/conda (or <root>/condabin/conda)
        root = Path(conda).resolve().parents[1]
        if (root / "envs" / "twin").is_dir():
            return [conda, "run", "--no-capture-output", "-n", "twin", "python", "-u"]

    return None


def simulation_environment():
    """The environment variables a simulation runs with.

    uvicorn runs inside the app's venv, which puts its own bin folder at the
    front of PATH. Passed on as-is, `python` inside `conda run -n habitat`
    would find the venv's Python first: the wrong version, with no Habitat.
    So the venv's folders are taken out of PATH, and conda's folder is added.

    Returns:
        dict[str, str]: a copy of the server's environment, adjusted.
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

    folders = [folder for folder in env.get("PATH", "").split(os.pathsep)
               if folder and not in_venv(folder)]

    # run_drone.sh calls `conda` by name. Found only through CONDA_EXE, it
    # would not be on the PATH the script gets, so put it there.
    conda = conda_executable()
    if conda and os.path.dirname(conda) not in folders:
        folders.append(os.path.dirname(conda))

    env["PATH"] = os.pathsep.join(folders)
    env.pop("PYTHONHOME", None)
    return env


def _slot_values(slot):
    """Everything a run must not share with another, derived from its slot.

    Args:
        slot (int): the run's slot number, 1..max_running().

    Returns:
        dict[str, str]: "slot", "run" (web-<slot>), "brain_port" (47600 + slot - 1)
        and "out" (the physics twin's results folder).
    """
    return {
        "slot": str(slot),
        "run": f"web-{slot}",
        "brain_port": str(FIRST_BRAIN_PORT + slot - 1),
        "out": str(TWIN_DIR / "outputs" / f"web-{slot}"),
    }


def _fill(part, values):
    """Fills the {name} placeholders in one command-line argument.

    Args:
        part (str): one argument, e.g. "--run {run}".
        values (dict[str, str]): from _slot_values.

    Returns:
        str: e.g. "--run web-2". Only the names in values are replaced.
    """
    for name, value in values.items():
        part = part.replace("{" + name + "}", value)
    return part


def _command_for(sim_id, slot):
    """The command line and working folder for a simulation in a slot.

    Args:
        sim_id (str): a key of SIMULATIONS.
        slot (int): the run's slot number.

    Returns:
        tuple[list[str], Path]: the command to run, and the folder to run it in.

    Raises:
        HTTPException(503): the Python or conda it needs is not installed.
    """
    simulation = SIMULATIONS[sim_id]
    values = _slot_values(slot)
    kind = simulation["kind"]

    if kind == "habitat":
        command = python_command()
        if command is None:
            raise HTTPException(
                status_code=503,
                detail="Could not find conda. Start uvicorn from a terminal where "
                       "`conda activate habitat` works, or set DELTA_SIM_PYTHON.",
            )
        return command + [simulation["script"]], SIMULATION_DIR

    if kind == "command":
        if simulation.get("needs_conda") and conda_executable() is None:
            raise HTTPException(
                status_code=503,
                detail="Could not find conda, which DELTA's launcher needs. Start uvicorn "
                       "from a terminal where `conda activate habitat` works.",
            )
        argv = [_fill(part, values) for part in simulation["command"]]
        return argv, Path(simulation.get("cwd", SIMULATION_DIR))

    if kind == "twin":
        command = twin_python_command()
        if command is None:
            raise HTTPException(
                status_code=503,
                detail="The physics twin's Python was not found. Set it up once (Python 3.10+): "
                       "cd physical-drone && python3 -m venv venv && "
                       "venv/bin/pip install -r requirements.txt -- or set DELTA_TWIN_PYTHON.",
            )
        return command + [_fill(part, values) for part in simulation["args"]], TWIN_DIR

    raise HTTPException(status_code=500, detail=f"Unknown kind of simulation: {kind}")


# ---------------------------------------------------------------------------
# A run's life
# ---------------------------------------------------------------------------

def _read_output(run):
    """Copies a run's output into its log until the process ends.

    Runs in a background thread, one per run. When the process exits it
    stops the run's virtual screen, records the exit code and final status,
    and removes the run if its owner has logged out meanwhile.

    Args:
        run (dict): the run, as stored in _runs.
    """
    for line in run["process"].stdout:
        with _lock:
            run["lines"].append(line.rstrip("\n"))
            run["line_count"] += 1
    run["process"].stdout.close()

    code = run["process"].wait()

    # The window is gone, so the screen it was on can go too
    if run["screen"]:
        run["screen"].stop()

    results = _read_results(run) if code == 0 else None

    with _lock:
        run["exit_code"] = code
        run["ended_at"] = _now()
        run["results"] = results
        if run["status"] == "stopping":
            run["status"] = "stopped"
        else:
            run["status"] = "finished" if code == 0 else "failed"
        # Its owner logged out while it ran: nobody is left to show it to
        token = run.get("owner_gone")
        if token and _runs.get(token) is run:
            del _runs[token]


def _read_results(run):
    """The physics twin's flight results, for the page to show.

    twin.fly writes report.json and flight.png to the run's results folder.

    Args:
        run (dict): a finished run.

    Returns:
        dict | None: {"summary": [[label, value], ...], "steps": [...],
        "picture": bool}, or None if the simulation has no results or the
        report cannot be read.
    """
    if not SIMULATIONS[run["id"]].get("results"):
        return None
    folder = Path(_slot_values(run["slot"])["out"])
    try:
        report = json.loads((folder / "report.json").read_text())
    except (OSError, ValueError):
        return None

    def number(value, digits=1):
        return f"{value:.{digits}f}" if isinstance(value, (int, float)) else "-"

    summary = [
        ["Hover time on the 4S2P battery", number(report.get("hover_minutes_on_4s2p")) + " min"],
        ["Energy used in this flight", number(report.get("energy_wh"), 2) + " Wh"],
        ["Physics speed", number(report.get("real_time_factor")) + "x real time"],
    ]
    steps = [
        {"name": step.get("name", step.get("kind", "")),
         "seconds": number(step.get("seconds")),
         "mean_watts": number(step.get("mean_watts"), 0),
         "max_tilt_deg": number(step.get("max_tilt_deg"))}
        for step in report.get("steps", [])
    ]
    return {"summary": summary, "steps": steps,
            "picture": (folder / "flight.png").is_file()}


def _is_running(run):
    """True if the run exists and has not ended yet (starting, running or stopping)."""
    return run is not None and run["status"] in ("starting", "running", "stopping")


def _running_count():
    """How many runs are in progress across all sessions. Call with _lock held."""
    return sum(1 for run in _runs.values() if _is_running(run))


def _summary(run, since=0):
    """What the page is told about a run. Call with _lock held.

    Args:
        run (dict | None): the run, as stored in _runs.
        since (int): how many output lines the page already has.

    Returns:
        dict | None: the run's number, id, title, status, exit code, times,
        output lines after `since`, slot, viewer (port and password, while it
        runs in the browser), results, and how many slots are in use; None
        if there is no run.
    """
    if run is None:
        return None
    lines = list(run["lines"])
    dropped = run["line_count"] - len(lines)  # lines that fell off the front
    start = max(since - dropped, 0)
    results = run.get("results")
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
        "slot": run["slot"],
        # Where the browser can watch it, while it runs (None = not in the browser)
        "viewer": run["viewer"] if run["status"] in ("running", "stopping") else None,
        # The physics twin's numbers, once its flight has finished
        "results": None if results is None else {
            "summary": results["summary"],
            "steps": results["steps"],
            "picture": f"/api/simulations/run/picture?number={run['number']}"
                       if results["picture"] else None,
        },
        "server_running": _running_count(),
        "server_max": max_running(),
    }


@router.get("")
def list_simulations(request: Request):
    """GET /api/simulations: what the page can run.

    Args:
        request: the caller's request (its session cookie).

    Returns:
        list[dict]: per simulation: id, title, description, interactive,
        running (for this session), newer, browser_ready, browser_missing.
    """
    owner = _owner(request)
    with _lock:
        mine = _runs.get(owner)
        running = mine["id"] if _is_running(mine) else None
    missing = virtual_screen.missing_requirements()
    return [
        {"id": sim_id, "title": sim["title"], "description": sim["description"],
         "interactive": sim["interactive"], "running": sim_id == running,
         "newer": sim.get("newer"),
         # Can it be shown in the browser? If not, why not.
         "browser_ready": sim["interactive"] and not missing,
         "browser_missing": missing if sim["interactive"] else []}
        for sim_id, sim in SIMULATIONS.items()
    ]


@router.get("/run")
def current_run(request: Request, since: int = 0):
    """GET /api/simulations/run: this session's current or last run.

    The page polls this once a second. Each session only sees its own run.

    Args:
        request: the caller's request (its session cookie).
        since (int): how many output lines the page already has, so only
            newer ones are sent.

    Returns:
        dict: {"run": summary (see _summary) or None}.
    """
    owner = _owner(request)
    with _lock:
        return {"run": _summary(_runs.get(owner), since)}


@router.get("/run/picture")
def run_picture(request: Request, number: int):
    """GET /api/simulations/run/picture: the physics twin's flight plot.

    Args:
        request: the caller's request (its session cookie).
        number (int): the run number the page is showing.

    Returns:
        FileResponse: flight.png of this session's own run.

    Raises:
        HTTPException(404): no such run for this session, or it has no plot.
    """
    owner = _owner(request)
    with _lock:
        run = _runs.get(owner)
        results = run.get("results") if run else None
        ok = run is not None and run["number"] == number and results and results["picture"]
        slot = run["slot"] if ok else None
    if not ok:
        raise HTTPException(status_code=404, detail="No picture for that run")
    return FileResponse(Path(_slot_values(slot)["out"]) / "flight.png", media_type="image/png",
                        headers={"Cache-Control": "no-store"})


@router.post("/{sim_id}/start")
def start_simulation(sim_id: str, request: Request, where: str = "browser"):
    """POST /api/simulations/{sim_id}/start: starts a simulation for this session.

    Args:
        sim_id (str): a key of SIMULATIONS.
        request: the caller's request (its session cookie).
        where (str): "browser" opens the window on its own virtual screen,
            shown in the page; "server" opens it on the server's monitor.
            Only matters for interactive simulations.

    Returns:
        dict: the new run's summary (see _summary).

    Raises:
        HTTPException: 404 unknown id, 400 bad `where`, 409 this session
        already has a run, 503 all slots taken or something not installed,
        500 the process could not be started.
    """
    global _run_count
    owner = _owner(request)

    if sim_id not in SIMULATIONS:
        raise HTTPException(status_code=404, detail="No simulation with that id")
    if where not in ("browser", "server"):
        raise HTTPException(status_code=400, detail="where must be browser or server")

    simulation = SIMULATIONS[sim_id]
    in_browser = simulation["interactive"] and where == "browser"

    # Claim a slot under the lock, then start things without holding it: the
    # screen takes a second or two, and everyone else's page is polling in
    # the meantime.
    with _lock:
        mine = _runs.get(owner)
        if _is_running(mine):
            raise HTTPException(
                status_code=409,
                detail=f"{SIMULATIONS[mine['id']]['title']} is already running. Stop it first.",
            )
        taken = {run["slot"] for run in _runs.values() if _is_running(run)}
        slot = next((s for s in range(1, max_running() + 1) if s not in taken), None)
        if slot is None:
            raise HTTPException(
                status_code=503,
                detail=f"The server is already running {max_running()} simulations, "
                       f"the most it allows at once. Try again when one finishes.",
            )

        # Checked before anything starts, so a missing install says so
        # straight away instead of half-starting a screen
        argv, folder = _command_for(sim_id, slot)

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
            "slot": slot,
            "viewer": None,
            "results": None,
        }
        _runs[owner] = run

    def give_up(status_code, detail):
        """Releases the claimed slot and reports the error to the page."""
        with _lock:
            if _runs.get(owner) is run:
                del _runs[owner]
        raise HTTPException(status_code=status_code, detail=detail)

    # The physics twin's results folder: last time's results in this slot
    # are someone else's, so they go first
    if simulation.get("results"):
        shutil.rmtree(_slot_values(slot)["out"], ignore_errors=True)

    env = simulation_environment()

    screen = None
    if in_browser:
        port = virtual_screen.first_viewer_port() + slot - 1
        screen = virtual_screen.VirtualScreen(port)
        try:
            screen.start()
        except RuntimeError as error:
            give_up(503, f"Could not show it in the browser: {error}")
        env = screen.environment(env)

    try:
        process = subprocess.Popen(
            argv,
            cwd=folder,
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
        run["viewer"] = {"port": screen.port, "password": screen.password} if screen else None
        stopped_meanwhile = run["status"] == "stopping"  # logged out while starting
        if not stopped_meanwhile:
            run["status"] = "running"
    threading.Thread(target=_read_output, args=(run,), daemon=True).start()
    if stopped_meanwhile:
        threading.Thread(target=_stop_run, args=(run,), daemon=True).start()

    with _lock:
        return _summary(run)


@router.post("/stop")
def stop_simulation(request: Request):
    """POST /api/simulations/stop: stops this session's run.

    The stop happens in a background thread; the page sees "stopping", then
    "stopped", by polling current_run.

    Args:
        request: the caller's request (its session cookie).

    Returns:
        dict: {"message": "Stopping"} (or "Already stopping").

    Raises:
        HTTPException(409): nothing is running, or it is still starting.
    """
    owner = _owner(request)
    with _lock:
        run = _runs.get(owner)
        if not _is_running(run):
            raise HTTPException(status_code=409, detail="Nothing is running")
        if run["status"] == "starting":
            raise HTTPException(status_code=409, detail="It is still starting. Try again in a moment.")
        if run["status"] == "stopping":
            return {"message": "Already stopping"}
        run["status"] = "stopping"
    # In the background: DELTA can take a while to save its map, and the page
    # sees "stopping", then "stopped", by polling as usual
    threading.Thread(target=_stop_run, args=(run,), daemon=True).start()
    return {"message": "Stopping"}


def _stop_run(run):
    """Ends a run: SIGTERM, wait up to its stop_seconds (default 5), then SIGKILL.

    Args:
        run (dict): the run, with its process started.
    """
    process = run["process"]
    _kill(process, signal.SIGTERM)
    try:
        process.wait(timeout=SIMULATIONS[run["id"]].get("stop_seconds", 5))
    except subprocess.TimeoutExpired:
        _kill(process, signal.SIGKILL if os.name == "posix" else signal.SIGTERM)


def stop_for_session(token):
    """Stops a session's run when it logs out or its login expires.

    Called by main.py. The run stays in _runs until it has really ended, so
    its slot (ports, map folder, GPU share) is not handed out too early.

    Args:
        token (str): the session token that was logged out.
    """
    with _lock:
        run = _runs.get(token)
        if run is None:
            return
        if not _is_running(run):
            del _runs[token]
            return
        # Stays in _runs until it has really stopped, so its slot (its ports,
        # its map folder, its place on the GPU) isn't handed to someone else
        # too early. _read_output removes it then.
        run["owner_gone"] = token
        status = run["status"]
        run["status"] = "stopping"
    if status == "running":
        threading.Thread(target=_stop_run, args=(run,), daemon=True).start()
    # "starting": start_simulation sees "stopping" and stops it once it can


def _kill(process, sig):
    """Sends a signal to a run's whole process group (terminate() on Windows).

    Args:
        process (subprocess.Popen): the run's process.
        sig (int): the signal, e.g. signal.SIGTERM.
    """
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
    """Stops every run and its virtual screen when the server shuts down.

    Called by main.py when uvicorn stops (Ctrl+C, a --reload restart, the
    terminal closing); registered with atexit as a backup for other exits.
    """
    with _lock:
        runs = list(_runs.values())
    for run in runs:
        if run["process"] and run["process"].poll() is None:
            _kill(run["process"], signal.SIGTERM)
        if run["screen"]:
            run["screen"].stop()
