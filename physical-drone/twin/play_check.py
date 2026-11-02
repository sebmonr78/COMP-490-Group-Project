"""Check the play window end to end, the way a person would use it.

    python -m twin.play_check [--out ~/delta-twin-out/play-check]

Opens the window and drives it with real key presses and mouse drags,
sent through the X server (XTEST) as if typed -- so they go the whole way:
X, GLFW, the window's handlers, the pilot, the flight brain, the physics.
If X won't deliver them (the window never got the focus), it says so and
calls the handlers directly instead, which checks everything but X.

Prints PASS or FAIL per check, and saves a picture at the main ones.
"""

import argparse
import ctypes
import json
import math
import time
from pathlib import Path

import glfw
import mujoco
import numpy as np

from twin.flight import heading
from twin.play import App

# X keysyms (X11/keysymdef.h).
XK = {"Up": 0xFF52, "Down": 0xFF54, "Left": 0xFF51, "Right": 0xFF53, "1": 0x31, "0": 0x30,
      "space": 0x20, "l": 0x6C, "g": 0x67, "r": 0x72, "f": 0x66, "c": 0x63, "h": 0x68,
      "Escape": 0xFF1B, "Shift": 0xFFE1, "Ctrl": 0xFFE3}
GLFW_KEY = {"Up": glfw.KEY_UP, "Down": glfw.KEY_DOWN, "Left": glfw.KEY_LEFT, "Right": glfw.KEY_RIGHT,
            "1": glfw.KEY_1, "0": glfw.KEY_0, "space": glfw.KEY_SPACE, "l": glfw.KEY_L, "g": glfw.KEY_G,
            "r": glfw.KEY_R, "f": glfw.KEY_F, "c": glfw.KEY_C, "h": glfw.KEY_H, "Escape": glfw.KEY_ESCAPE,
            "Shift": glfw.KEY_LEFT_SHIFT, "Ctrl": glfw.KEY_LEFT_CONTROL}
LEFT, RIGHT, SCROLL_UP, SCROLL_DOWN = 1, 3, 4, 5


class XInput:
    """Real input events through the X server, like a keyboard and mouse."""

    def __init__(self, window):
        self.window = window
        self.x11 = ctypes.CDLL("libX11.so.6")
        self.tst = ctypes.CDLL("libXtst.so.6")
        self.x11.XOpenDisplay.restype = ctypes.c_void_p
        self.x11.XKeysymToKeycode.argtypes = [ctypes.c_void_p, ctypes.c_ulong]
        self.x11.XKeysymToKeycode.restype = ctypes.c_ubyte
        self.x11.XFlush.argtypes = [ctypes.c_void_p]
        self.tst.XTestFakeKeyEvent.argtypes = [ctypes.c_void_p, ctypes.c_uint, ctypes.c_int, ctypes.c_ulong]
        self.tst.XTestFakeButtonEvent.argtypes = [ctypes.c_void_p, ctypes.c_uint, ctypes.c_int, ctypes.c_ulong]
        self.tst.XTestFakeMotionEvent.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_int,
                                                  ctypes.c_ulong]
        self.display = self.x11.XOpenDisplay(None)

    def key(self, name, down):
        code = self.x11.XKeysymToKeycode(self.display, XK[name])
        self.tst.XTestFakeKeyEvent(self.display, code, int(down), 0)
        self.x11.XFlush(self.display)

    def move(self, x, y):
        wx, wy = glfw.get_window_pos(self.window)
        self.tst.XTestFakeMotionEvent(self.display, -1, int(wx + x), int(wy + y), 0)
        self.x11.XFlush(self.display)

    def button(self, n, down):
        self.tst.XTestFakeButtonEvent(self.display, n, int(down), 0)
        self.x11.XFlush(self.display)


class DirectInput:
    """The same, straight into the window's handlers (no X)."""

    def __init__(self, app):
        self.app = app
        self.pos = (0.0, 0.0)
        self.down = set()
        app.where = lambda window: self.pos

    def _mods(self):
        return (glfw.MOD_CONTROL if "Ctrl" in self.down else 0) | (glfw.MOD_SHIFT if "Shift" in self.down else 0)

    def key(self, name, down):
        (self.down.add if down else self.down.discard)(name)
        self.app.on_key(self.app.window, GLFW_KEY[name], 0, glfw.PRESS if down else glfw.RELEASE, self._mods())

    def move(self, x, y):
        self.pos = (float(x), float(y))
        self.app.on_cursor(self.app.window, float(x), float(y))

    def button(self, n, down):
        if n in (SCROLL_UP, SCROLL_DOWN):
            if down:
                self.app.on_scroll(self.app.window, 0.0, 1.0 if n == SCROLL_UP else -1.0)
            return
        button = glfw.MOUSE_BUTTON_LEFT if n == LEFT else glfw.MOUSE_BUTTON_RIGHT
        self.app.on_button(self.app.window, button, glfw.PRESS if down else glfw.RELEASE, self._mods())


def script(app, out, results):
    """The checks, as a generator: each yield is seconds to wait."""
    d, twin = app.twin.data, app.twin
    width, height = glfw.get_window_size(app.window)
    cx, cy = width // 2, height // 2
    glfw.focus_window(app.window)
    yield 1.0

    # Does X deliver to the window? H hides the help.
    inp = XInput(app.window)
    inp.key("h", True)
    yield 0.05
    inp.key("h", False)
    yield 0.4
    via_x = app.help is False
    if not via_x:
        inp = DirectInput(app)
        inp.key("h", True)
        inp.key("h", False)
    app.help = True
    results.append(("input path", True, "real X events reach the window" if via_x
                    else "X did not deliver (no focus?): handlers called directly"))

    def check(name, ok, detail):
        results.append((name, bool(ok), detail))

    def hold(keys, seconds):
        for k in keys:
            inp.key(k, True)
        yield seconds
        for k in reversed(keys):
            inp.key(k, False)

    def drag(button, dx, dy, steps=20, seconds=0.6, x0=None, y0=None, keys=()):
        x0 = cx if x0 is None else x0
        y0 = cy if y0 is None else y0
        for k in keys:
            inp.key(k, True)
        inp.move(x0, y0)
        yield 0.05
        inp.button(button, True)
        yield 0.05
        for i in range(1, steps + 1):
            inp.move(x0 + dx * i / steps, y0 + dy * i / steps)
            yield seconds / steps
        inp.button(button, False)
        for k in keys:
            inp.key(k, False)
        yield 0.1

    # 1. Take off: hold 1.
    yield from hold(["1"], 1.6)
    yield 1.0
    check("1 takes off", app.pilot.mode == "flying" and d.qpos[2] - 0.225 > 0.4,
          f"{app.pilot.mode}, {d.qpos[2] - 0.225:.2f} m up")
    app.snapshot(out / "1_takeoff.png")

    # 2. Forward: hold Up, let go, it stops.
    start = d.qpos[0:3].copy()
    yield from hold(["Up"], 2.0)
    at_release = d.qpos[0:3].copy()
    yield 1.5
    moved = d.qpos[0:3] - start
    check("Up flies forward", moved[0] > 0.9 and abs(moved[1]) < 0.1, f"moved {moved[0]:.2f} m forward")
    check("letting go stops it", np.linalg.norm(d.qvel[0:3]) < 0.1,
          f"speed {np.linalg.norm(d.qvel[0:3]):.2f} m/s, {np.linalg.norm(d.qpos[0:3] - at_release):.2f} m after letting go")
    app.snapshot(out / "2_forward.png")

    # 3. Turn: hold Left 1 s at 45 deg/s.
    h0 = heading(d.qpos[3:7])
    yield from hold(["Left"], 1.0)
    yield 1.0
    turned = math.degrees(math.remainder(heading(d.qpos[3:7]) - h0, 2 * math.pi))
    check("Left turns left", 35 < turned < 55, f"turned {turned:.0f} deg")

    # 4. Back, with Shift: faster.
    top = 0.0
    inp.key("Shift", True)
    inp.key("Down", True)
    for _ in range(20):
        yield 0.05
        top = max(top, np.linalg.norm(d.qvel[0:2]))
    inp.key("Down", False)
    inp.key("Shift", False)
    yield 0.1
    inp.key("space", True)
    inp.key("space", False)
    yield 1.5
    check("Shift + Down: back, faster", top > 0.9, f"top speed {top:.2f} m/s")
    check("Space stops", np.linalg.norm(d.qvel[0:3]) < 0.1, f"speed {np.linalg.norm(d.qvel[0:3]):.2f} m/s")

    # 5. Up and down: 1 and 0.
    z0 = d.qpos[2]
    yield from hold(["1"], 1.0)
    yield 0.5
    z1 = d.qpos[2]
    yield from hold(["0"], 1.0)
    yield 0.5
    check("1 climbs, 0 descends", z1 - z0 > 0.3 and z1 - d.qpos[2] > 0.3,
          f"+{z1 - z0:.2f} m, then -{z1 - d.qpos[2]:.2f} m")

    # 6. Gust: G.
    p0 = d.qpos[0:3].copy()
    inp.key("g", True)
    inp.key("g", False)
    yield 0.05
    pushed = np.linalg.norm(twin.push) > 0
    yield 0.8
    check("G gusts", pushed and np.linalg.norm(d.qpos[0:2] - p0[0:2]) > 0.05,
          f"pushed {100 * np.linalg.norm(d.qpos[0:2] - p0[0:2]):.0f} cm")
    yield 2.5

    # 7. The mouse: orbit, zoom, slide -- all around the drone.
    az0, dist0 = app.camera.azimuth, app.camera.distance
    yield from drag(LEFT, 250, 0)
    check("left drag turns the view", abs(app.camera.azimuth - az0) > 20,
          f"azimuth {az0:.0f} -> {app.camera.azimuth:.0f} deg")
    check("... still on the drone", np.linalg.norm(app.camera.lookat - d.qpos[0:3]) < 0.05,
          f"centre {100 * np.linalg.norm(app.camera.lookat - d.qpos[0:3]):.1f} cm from it")
    app.snapshot(out / "3_orbit.png")

    for _ in range(4):
        inp.button(SCROLL_UP, True)
        inp.button(SCROLL_UP, False)
        yield 0.05
    yield 0.2
    check("scroll zooms in", app.camera.distance < dist0 * 0.95, f"{dist0:.2f} -> {app.camera.distance:.2f} m")

    yield from drag(RIGHT, 150, 0)
    offset = app.offset.copy()
    check("right drag slides the view", np.linalg.norm(offset) > 0.1, f"slid {np.linalg.norm(offset):.2f} m")
    yield from hold(["Up"], 1.0)
    yield 1.0
    kept = np.linalg.norm((app.camera.lookat - d.qpos[0:3]) - offset)
    check("... and the slide moves with the drone", kept < 0.02, f"off by {100 * kept:.1f} cm after flying 0.8 m")
    app.snapshot(out / "4_slid.png")
    inp.key("c", True)
    inp.key("c", False)
    yield 0.2
    check("C puts the view back on the drone", np.linalg.norm(app.offset) < 1e-6, "")

    # 8. Land: L.
    inp.key("l", True)
    inp.key("l", False)
    for _ in range(100):
        yield 0.1
        if app.pilot.mode == "landed":
            break
    cage = twin.body.cage
    under = sorted({app.names[g] for c in d.contact[:d.ncon]
                    if cage in (twin.model.geom_bodyid[c.geom1], twin.model.geom_bodyid[c.geom2])
                    for g in (c.geom1, c.geom2) if twin.model.geom_bodyid[g] != cage})
    check("L lands, motors off", app.pilot.mode == "landed" and not twin.motors[0].armed,
          f"{app.pilot.mode} at {d.qpos[0]:.2f}, {d.qpos[1]:.2f}, {d.qpos[2] - 0.225:.2f} m up, on {under}")

    # 9. Pick a box and push it: look at box 1, double-click it, Ctrl + right drag.
    box = mujoco.mj_name2id(twin.model, mujoco.mjtObj.mjOBJ_BODY, "box 1")
    inp.key("f", True)
    inp.key("f", False)
    yield 0.1
    app.camera.lookat[:] = d.xpos[box]
    app.camera.distance, app.camera.azimuth, app.camera.elevation = 2.5, 180.0, -25.0
    yield 0.3
    for _ in range(2):
        inp.move(cx, cy)
        inp.button(LEFT, True)
        inp.button(LEFT, False)
        yield 0.08
    yield 0.2
    check("double-click picks the box", app.perturb.select == box,
          f"picked body {app.perturb.select} ({mujoco.mj_id2name(twin.model, mujoco.mjtObj.mjOBJ_BODY, app.perturb.select)})")
    # Sideways: dragged up it would lift, then drop back where it was.
    b0 = d.xpos[box].copy()
    yield from drag(RIGHT, 150, 0, steps=30, seconds=1.2, keys=["Ctrl"])
    yield 0.5
    check("Ctrl + right drag pushes it", np.linalg.norm(d.xpos[box][:2] - b0[:2]) > 0.1,
          f"the box slid {100 * np.linalg.norm(d.xpos[box][:2] - b0[:2]):.0f} cm")
    check("... and the view stayed", np.linalg.norm(app.camera.lookat - b0) < 0.05, "")
    app.snapshot(out / "5_pushed.png")

    # 10. Start again: R. Quit: Esc.
    inp.key("r", True)
    inp.key("r", False)
    yield 0.3
    check("R starts again", np.linalg.norm(d.qpos[0:2]) < 0.01 and app.pilot.mode == "on the floor",
          f"at {d.qpos[0]:.2f}, {d.qpos[1]:.2f}; {app.pilot.mode}")
    check("no key stuck down", not app.held, f"held: {sorted(app.held)}")
    inp.key("Escape", True)
    inp.key("Escape", False)
    yield 0.5
    check("Esc quits", not app.running, "")
    app.running = False


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", default="~/delta-twin-out/play-check")
    args = parser.parse_args()
    out = Path(args.out).expanduser()
    out.mkdir(parents=True, exist_ok=True)

    app = App()
    results = []

    # Keep the raw pointer events, to see what X really delivers.
    seen = []
    handler = app.on_cursor

    def on_cursor(window, x, y):
        seen.append((round(time.perf_counter(), 3), round(x, 1), round(y, 1), sorted(app.buttons)))
        handler(window, x, y)

    glfw.set_cursor_pos_callback(app.window, on_cursor)

    # And the keys and focus changes, to see what arrives and when.
    events = []
    key_handler, focus_handler, button_handler = app.on_key, app.on_focus, app.on_button

    def on_key(window, key, scancode, action, mods):
        events.append((round(time.perf_counter(), 3), "key", glfw.get_key_name(key, scancode) or key, action, mods))
        key_handler(window, key, scancode, action, mods)

    def on_focus(window, focused):
        events.append((round(time.perf_counter(), 3), "focus", focused))
        focus_handler(window, focused)

    def on_button(window, button, action, mods):
        events.append((round(time.perf_counter(), 3), "button", button, action, mods))
        button_handler(window, button, action, mods)

    glfw.set_key_callback(app.window, on_key)
    glfw.set_window_focus_callback(app.window, on_focus)
    glfw.set_mouse_button_callback(app.window, on_button)
    steps = script(app, out, results)
    due = [time.perf_counter()]

    def hook(app):
        if time.perf_counter() >= due[0]:
            try:
                due[0] = time.perf_counter() + next(steps)
            except StopIteration:
                app.running = False

    glfw_pos = (glfw.get_window_pos(app.window), glfw.get_window_size(app.window))
    app.run(hook)
    if results[-1][0] == "no key stuck down":          # Esc closes the window before its check can run
        results.append(("Esc quits", not app.running, ""))
    width = max(len(r[0]) for r in results)
    for name, ok, detail in results:
        print(f"{'PASS' if ok else 'FAIL'}  {name:<{width}}  {detail}")
    (out / "results.json").write_text(json.dumps(results, indent=1))
    (out / "pointer.json").write_text(json.dumps(seen))
    (out / "events.json").write_text(json.dumps(events))
    print("last events:", events[-12:])
    print(f"window at {glfw_pos[0]}, size {glfw_pos[1]}; {len(seen)} pointer events kept")


if __name__ == "__main__":
    main()
