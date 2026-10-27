"""An invisible screen on the server that the browser can look at.

When "Open environment" runs in the browser, its window must not open on the
server's own monitor. So we start a screen that only exists in memory, open
the window there, and let the browser watch and control it:

    Xvfb        the invisible screen (a display like :99, the size of the window)
    x11vnc      shares that screen over VNC, on this machine only
    websockify  serves noVNC, the in-browser VNC viewer, and carries the VNC
                traffic over a WebSocket. This is the port the browser uses.

The simulation still renders with the GPU (Habitat uses EGL, not the
screen); the screen only receives the finished pictures, which is cheap.

Each run gets its own screen, its own VNC password and its own viewer port,
so several people can each have their own window at the same time. The
viewer ports are DELTA_VIEWER_PORT and the ones right after it (6080, 6081,
6082, ... one per simulation that can run at once).

Needs, on the server:

    sudo apt install xvfb x11vnc novnc websockify

Settings (environment variables, all optional):

    DELTA_VIEWER_PORT   first port the browser connects to   (default 6080)
    DELTA_NOVNC_DIR     where noVNC's files are        (default /usr/share/novnc)
"""

import os
import secrets
import shutil
import signal
import socket
import subprocess
import tempfile
import threading
import time
from pathlib import Path

# The demo's window is 1600 x 800 (WINDOW_WIDTH/HEIGHT in open_environment.py).
# Make the screen exactly that size so the window fills it with no border.
SCREEN_SIZE = "1600x800x24"

REQUIRED_PROGRAMS = ["Xvfb", "x11vnc", "websockify"]
INSTALL_HINT = "On the server run: sudo apt install xvfb x11vnc novnc websockify"


def first_viewer_port():
    return int(os.environ.get("DELTA_VIEWER_PORT", "6080"))


# Two screens starting at the same moment could both pick the same free
# display number before either Xvfb has taken it, so they take turns.
_start_lock = threading.Lock()


def novnc_dir():
    return Path(os.environ.get("DELTA_NOVNC_DIR", "/usr/share/novnc"))


def missing_requirements():
    """What is not installed, as a list of names. Empty means ready."""
    missing = [name for name in REQUIRED_PROGRAMS if shutil.which(name) is None]
    if not (novnc_dir() / "vnc.html").is_file():
        missing.append("novnc")
    return missing


def _free_display():
    # :99 and up are rarely used by real screens. A lock file means taken.
    for number in range(99, 140):
        if not Path(f"/tmp/.X{number}-lock").exists() and \
           not Path(f"/tmp/.X11-unix/X{number}").exists():
            return number
    raise RuntimeError("No free display number between :99 and :139")


def _port_in_use(port):
    with socket.socket() as s:
        return s.connect_ex(("127.0.0.1", port)) == 0


def _read_port(path):
    # x11vnc -flag writes "PORT=5901" to the file once it is listening
    try:
        text = path.read_text().strip()
    except OSError:
        return None
    if text.startswith("PORT=") and text[5:].isdigit():
        return int(text[5:])
    return None


def _wait_for(check, seconds, what):
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        if check():
            return
        time.sleep(0.1)
    raise RuntimeError(f"{what} did not start within {seconds} seconds")


class VirtualScreen:
    def __init__(self, port):
        self.port = port      # the viewer port the browser connects to
        self.display = None
        self.processes = []
        # Only the person who started this run is given the password, so
        # nobody else can open their window by trying the other ports.
        # (VNC only uses the first 8 characters of a password.)
        self.password = secrets.token_urlsafe(6)[:8]
        # Each helper's error output goes here, so a failure can say why
        self.log_dir = Path(tempfile.mkdtemp(prefix="delta-screen-"))

    def start(self):
        """Start all three programs. Raises RuntimeError if any fails."""
        missing = missing_requirements()
        if missing:
            raise RuntimeError(f"Missing {', '.join(missing)}. {INSTALL_HINT}")

        port = self.port
        if _port_in_use(port):
            raise RuntimeError(
                f"Port {port} is already in use. If an earlier run was left "
                f"behind, run `pkill websockify` on the server. Otherwise set "
                f"DELTA_VIEWER_PORT to a different first port.")

        # x11vnc reads the password from this file and deletes it ("rm:"),
        # so it never shows up in `ps` and is not left on disk
        password_file = self.log_dir / "passwd"
        password_file.touch(mode=0o600)
        password_file.write_text(self.password + "\n")

        try:
            # One screen starts at a time: otherwise two could pick the same
            # free display or VNC port before either has actually taken it
            with _start_lock:
                number = _free_display()
                self.display = f":{number}"
                self._launch(["Xvfb", self.display, "-screen", "0", SCREEN_SIZE,
                              "-nolisten", "tcp"])
                _wait_for(lambda: Path(f"/tmp/.X11-unix/X{number}").exists(), 10, "Xvfb")

                # -localhost: VNC itself is only reachable through websockify.
                # x11vnc picks a free VNC port itself (-autoport) and writes
                # it to vnc_port_file once it is listening (-flag).
                vnc_port_file = self.log_dir / "vncport"
                self._launch(["x11vnc", "-display", self.display, "-autoport", "5900",
                              "-localhost", "-forever", "-shared",
                              "-passwdfile", f"rm:{password_file}", "-quiet",
                              "-noxdamage", "-flag", str(vnc_port_file)])
                _wait_for(lambda: _read_port(vnc_port_file) is not None, 10, "x11vnc")
                vnc_port = _read_port(vnc_port_file)

            self._launch(["websockify", "--web", str(novnc_dir()),
                          f"0.0.0.0:{port}", f"127.0.0.1:{vnc_port}"])
            _wait_for(lambda: _port_in_use(port), 10, "websockify")
        except Exception:
            self.stop()
            raise

    def _helper_environment(self):
        # The server is usually logged in to a Wayland desktop (Ubuntu's
        # default). x11vnc sees WAYLAND_DISPLAY and refuses to start, even
        # though the screen it is given is an X screen. So hide it.
        env = dict(os.environ)
        env.pop("WAYLAND_DISPLAY", None)
        env.pop("XAUTHORITY", None)
        env["XDG_SESSION_TYPE"] = "x11"
        if self.display:
            env["DISPLAY"] = self.display
        return env

    def _launch(self, command):
        log_path = self.log_dir / f"{command[0]}.log"
        with log_path.open("w") as log:
            process = subprocess.Popen(
                command,
                env=self._helper_environment(),
                stdin=subprocess.DEVNULL,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        self.processes.append(process)
        # If it died straight away (bad option, missing library), say so now
        time.sleep(0.3)
        if process.poll() is not None:
            raise RuntimeError(
                f"{command[0]} exited right away (code {process.returncode}): "
                + self._last_lines(log_path))

    @staticmethod
    def _last_lines(log_path, count=3):
        try:
            lines = [l.strip() for l in log_path.read_text(errors="replace").splitlines() if l.strip()]
        except OSError:
            return "no output"
        return " | ".join(lines[-count:]) or "no output"

    def environment(self, env):
        """Changes env so a window opens on this screen instead of the monitor."""
        env = dict(env)
        env["DISPLAY"] = self.display
        env["SDL_VIDEODRIVER"] = "x11"     # pygame: use the X screen, not Wayland
        env["XDG_SESSION_TYPE"] = "x11"
        env.pop("WAYLAND_DISPLAY", None)
        env.pop("XAUTHORITY", None)        # Xvfb has no login cookie
        return env

    def stop(self):
        # Last started, first stopped: the viewer, then VNC, then the screen
        for process in reversed(self.processes):
            if process.poll() is None:
                try:
                    os.killpg(process.pid, signal.SIGTERM)
                    process.wait(timeout=3)
                except (ProcessLookupError, PermissionError):
                    pass
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
        self.processes = []
        shutil.rmtree(self.log_dir, ignore_errors=True)
