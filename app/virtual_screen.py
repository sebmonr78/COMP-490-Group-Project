"""Virtual Screen: an invisible screen on the server that a browser can watch and control.

Module      : virtual_screen.py   (Design: Virtual Screen)
Class       : VirtualScreen
Date        : Created 2026-10-21, last updated 2026-10-26
Programmer  : Sebastian

DESCRIPTION
    When a simulation runs "in the browser", its window must not open on the
    server's own monitor. Instead it opens on a screen that only exists in
    memory, and the browser watches and controls it:

        Xvfb        the invisible screen (a display such as :99)
        x11vnc      shares that screen over VNC, on this machine only
        websockify  serves noVNC, the in-browser VNC viewer, and carries the
                    VNC traffic over a WebSocket; this is the browser's port

    The simulation still renders with the GPU; the screen only receives the
    finished pictures. Each run gets its own screen, VNC password and viewer
    port, so several people can each have their own window at once.

    Needs on the server: sudo apt install xvfb x11vnc novnc websockify

IMPORTANT FUNCTIONS
    VirtualScreen(port)
        In : the viewer port the browser will connect to (6080, 6081, ...).
        Out: a screen object, not started yet; .password is set.
    VirtualScreen.start() -> None
        Starts Xvfb, x11vnc and websockify. Raises RuntimeError (with the
        reason) if a program is missing, the port is taken or one fails.
    VirtualScreen.environment(env) -> dict
        In : environment variables. Out: a copy that opens windows on this
        screen (DISPLAY, SDL_VIDEODRIVER=x11, Wayland variables removed).
    VirtualScreen.stop() -> None
        Stops the three programs and deletes the screen's temporary files.
    missing_requirements() -> list[str]
        Out: names of the programs not installed; empty means ready.

IMPORTANT DATA STRUCTURES
    VirtualScreen.processes : list[subprocess.Popen]
        The helper programs, in the order they were started.
    VirtualScreen.log_dir : Path
        A temporary folder with each helper's log, the VNC password file
        (deleted by x11vnc once read) and the file x11vnc writes its port to.

ALGORITHMS
    Display number: linear search from :99 upward for the first number with
    no X lock file or socket, done under a lock so two screens starting at
    the same moment cannot take the same one.

    VNC port: x11vnc picks a free port itself (-autoport) and reports it
    through a file (-flag), which is then read back. Chosen over finding a
    free port first and handing it to x11vnc, because another program could
    take that port in between.

    Access control: an 8-character random password per screen (secrets
    module; VNC uses at most 8 characters), given only to the session that
    started the run, so nobody can open another person's window.

    Shutdown in reverse start order (viewer, VNC, screen), so nothing is left
    attached to a screen that is already gone.

    Why this approach: Xvfb + x11vnc + noVNC lets an unchanged desktop
    program appear in any browser with nothing to install on the client.
    Remote-desktop software would need an install on every client, and
    rewriting the simulation to draw in the browser would mean a new renderer.

SETTINGS (environment variables, all optional)
    DELTA_VIEWER_PORT   first viewer port   (default 6080)
    DELTA_NOVNC_DIR     where noVNC's files are   (default /usr/share/novnc)
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
    """The viewer port of slot 1 (DELTA_VIEWER_PORT, default 6080); slot n uses this + n - 1."""
    return int(os.environ.get("DELTA_VIEWER_PORT", "6080"))


# Two screens starting at the same moment could both pick the same free
# display number before either Xvfb has taken it, so they take turns.
_start_lock = threading.Lock()


def novnc_dir():
    """The folder holding noVNC's web files (DELTA_NOVNC_DIR, default /usr/share/novnc)."""
    return Path(os.environ.get("DELTA_NOVNC_DIR", "/usr/share/novnc"))


def missing_requirements():
    """What the in-browser view needs but is not installed.

    Returns:
        list[str]: missing program names ("Xvfb", "x11vnc", "websockify",
        "novnc"); empty means ready.
    """
    missing = [name for name in REQUIRED_PROGRAMS if shutil.which(name) is None]
    if not (novnc_dir() / "vnc.html").is_file():
        missing.append("novnc")
    return missing


def _free_display():
    """The first free X display number from 99 to 139.

    :99 and up are rarely used by real screens; a lock file or socket means
    the number is taken.

    Returns:
        int: the display number.

    Raises:
        RuntimeError: all of them are taken.
    """
    for number in range(99, 140):
        if not Path(f"/tmp/.X{number}-lock").exists() and \
           not Path(f"/tmp/.X11-unix/X{number}").exists():
            return number
    raise RuntimeError("No free display number between :99 and :139")


def _port_in_use(port):
    """True if something on this machine is listening on the TCP port."""
    with socket.socket() as s:
        return s.connect_ex(("127.0.0.1", port)) == 0


def _read_port(path):
    """The VNC port x11vnc reported, or None if it has not written it yet.

    Args:
        path (Path): the file given to x11vnc -flag; it holds "PORT=5901"
            once x11vnc is listening.

    Returns:
        int | None: the port.
    """
    try:
        text = path.read_text().strip()
    except OSError:
        return None
    if text.startswith("PORT=") and text[5:].isdigit():
        return int(text[5:])
    return None


def _wait_for(check, seconds, what):
    """Polls check() every 0.1 s until it returns True.

    Args:
        check (callable): returns True once the thing is ready.
        seconds (float): how long to wait.
        what (str): the name used in the error message.

    Raises:
        RuntimeError: check() was still False after `seconds`.
    """
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        if check():
            return
        time.sleep(0.1)
    raise RuntimeError(f"{what} did not start within {seconds} seconds")


class VirtualScreen:
    """One invisible screen, shared with one browser through its own port.

    Attributes:
        port (int): the viewer port the browser connects to.
        password (str): random 8-character VNC password for this screen.
        display (str | None): the X display, e.g. ":99", once started.
        processes (list[subprocess.Popen]): the helpers, in start order.
        log_dir (Path): temporary folder for the helpers' logs and files.
    """

    def __init__(self, port):
        """Prepares a screen for the given viewer port; start() starts it.

        Args:
            port (int): the viewer port the browser will connect to.
        """
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
        """Starts Xvfb, x11vnc and websockify, in that order.

        Raises:
            RuntimeError: a program is missing, the viewer port is in use, or
            a program failed to start (the message includes its last output).
            Whatever had already started is stopped first.
        """
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
        """Environment for the helper programs, pointed at this screen.

        Ubuntu's desktop is usually Wayland, and x11vnc refuses to start when
        it sees WAYLAND_DISPLAY, even for an X screen, so that is removed.

        Returns:
            dict[str, str]: a copy of the server's environment, adjusted.
        """
        env = dict(os.environ)
        env.pop("WAYLAND_DISPLAY", None)
        env.pop("XAUTHORITY", None)
        env["XDG_SESSION_TYPE"] = "x11"
        if self.display:
            env["DISPLAY"] = self.display
        return env

    def _launch(self, command):
        """Starts one helper program, logging its output to log_dir.

        Args:
            command (list[str]): the program and its arguments.

        Raises:
            RuntimeError: it exited within 0.3 s (bad option, missing library);
            the message includes its last output.
        """
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
        """The last `count` non-empty lines of a log, joined with " | "."""
        try:
            lines = [l.strip() for l in log_path.read_text(errors="replace").splitlines() if l.strip()]
        except OSError:
            return "no output"
        return " | ".join(lines[-count:]) or "no output"

    def environment(self, env):
        """Environment for the simulation, so its window opens on this screen.

        Args:
            env (dict[str, str]): the simulation's environment variables.

        Returns:
            dict[str, str]: a copy with DISPLAY set to this screen,
            SDL_VIDEODRIVER=x11 for pygame, and the Wayland variables removed.
        """
        env = dict(env)
        env["DISPLAY"] = self.display
        env["SDL_VIDEODRIVER"] = "x11"     # pygame: use the X screen, not Wayland
        env["XDG_SESSION_TYPE"] = "x11"
        env.pop("WAYLAND_DISPLAY", None)
        env.pop("XAUTHORITY", None)        # Xvfb has no login cookie
        return env

    def stop(self):
        """Stops the helpers in reverse start order and deletes log_dir.

        Each gets SIGTERM and 3 seconds, then SIGKILL. Safe to call twice.
        """
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
