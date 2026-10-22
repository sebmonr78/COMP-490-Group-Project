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

Needs, on the server:

    sudo apt install xvfb x11vnc novnc websockify

Settings (environment variables, all optional):

    DELTA_VIEWER_PORT   port the browser connects to   (default 6080)
    DELTA_NOVNC_DIR     where noVNC's files are        (default /usr/share/novnc)
"""

import os
import shutil
import signal
import socket
import subprocess
import tempfile
import time
from pathlib import Path

# The demo's window is 1600 x 800 (WINDOW_WIDTH/HEIGHT in open_environment.py).
# Make the screen exactly that size so the window fills it with no border.
SCREEN_SIZE = "1600x800x24"

REQUIRED_PROGRAMS = ["Xvfb", "x11vnc", "websockify"]
INSTALL_HINT = "On the server run: sudo apt install xvfb x11vnc novnc websockify"


def viewer_port():
    return int(os.environ.get("DELTA_VIEWER_PORT", "6080"))


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


def _free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _port_in_use(port):
    with socket.socket() as s:
        return s.connect_ex(("127.0.0.1", port)) == 0


def _wait_for(check, seconds, what):
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        if check():
            return
        time.sleep(0.1)
    raise RuntimeError(f"{what} did not start within {seconds} seconds")


class VirtualScreen:
    def __init__(self):
        self.display = None
        self.processes = []
        # Each helper's error output goes here, so a failure can say why
        self.log_dir = Path(tempfile.mkdtemp(prefix="delta-screen-"))

    def start(self):
        """Start all three programs. Raises RuntimeError if any fails."""
        missing = missing_requirements()
        if missing:
            raise RuntimeError(f"Missing {', '.join(missing)}. {INSTALL_HINT}")

        port = viewer_port()
        if _port_in_use(port):
            raise RuntimeError(
                f"Port {port} is already in use. If an earlier run was left "
                f"behind, run `pkill websockify` on the server. Otherwise set "
                f"DELTA_VIEWER_PORT to another port.")

        number = _free_display()
        self.display = f":{number}"
        vnc_port = _free_port()

        try:
            self._launch(["Xvfb", self.display, "-screen", "0", SCREEN_SIZE,
                          "-nolisten", "tcp"])
            _wait_for(lambda: Path(f"/tmp/.X11-unix/X{number}").exists(), 10, "Xvfb")

            # -localhost: VNC itself is only reachable through websockify
            self._launch(["x11vnc", "-display", self.display, "-rfbport", str(vnc_port),
                          "-localhost", "-forever", "-shared", "-nopw", "-quiet",
                          "-noxdamage"])
            _wait_for(lambda: _port_in_use(vnc_port), 10, "x11vnc")

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
