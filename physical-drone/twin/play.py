"""Fly the twin yourself.

    python -m twin.play

One window, like the DELTA window: the drone in a room, its numbers in the
corner, the keys and the mouse.

    keys (the DELTA window's)            mouse
    Up / Down     forward / back         left drag     turn the view around the drone
    Left / Right  turn                   right drag    slide the view
    1 / 0         up / down              scroll        zoom
      (also Page Up / Page Down;         double-click  pick a thing (again: let go)
       1 on the floor takes off)         Ctrl + right drag   push what is picked
    Shift         faster                 Ctrl + left drag    twist it
    Space         stop, hold here
    L             land                   F    the view follows the drone, or stays
    G             a gust from the side   C    view back on the drone
    R             start again            H    hide / show this help
    Esc           quit

Like PX4's Position mode, a key asks for a speed and letting go stops the
drone where it is. The view moves with the drone: what you turn, zoom or
slide is kept relative to it.

Why not MuJoCo's own viewer: its keys are taken (0-5 hide geom groups,
Space pauses, G, R, F... switch drawing options), it never says when a key
is let go, and its tracking camera snaps back when slid.
"""

import math
import os
import platform
import time

# Under WSL the window draws on the CPU (Mesa's llvmpipe), not through WSL's
# D3D12 bridge to the GPU: MuJoCo draws geom by geom, and the bridge charges
# each draw ~0.3-1 ms. Measured on the twin in its room: 9 ms a frame on the
# CPU, 56 ms through the bridge. Habitat is the opposite -- few, big draws --
# so it keeps d3d12. Anywhere else (Windows, a Mac, Linux on its own
# graphics card) the window gets the GPU as usual.
if "microsoft" in platform.uname().release.lower() or "WSL_DISTRO_NAME" in os.environ:
    os.environ["GALLIUM_DRIVER"] = os.environ.get("TWIN_GALLIUM_DRIVER", "llvmpipe")
    os.environ.pop("MESA_LOADER_DRIVER_OVERRIDE", None)

import glfw  # noqa: E402
import mujoco  # noqa: E402
import numpy as np  # noqa: E402

from twin.fly import SIM_HZ, Twin  # noqa: E402
from twin.flight import Setpoint, heading, tilt  # noqa: E402

# A room to fly in: 8 x 6 m, 2.5 m walls (see-through, so the view can
# look in), a wall across it with a doorway, furniture that stays put, and
# boxes and a ball that don't.
GLASS = 1                     # the walls' geom group: drawn, but clicks pass through
WALL = f'rgba="0.85 0.82 0.78 0.25" group="{GLASS}"'
FIXED = 'rgba="0.55 0.45 0.35 1"'
ROOM = f"""
    <geom name="wall north" type="box" pos="1 3.05 1.25" size="4.1 0.05 1.25" {WALL}/>
    <geom name="wall south" type="box" pos="1 -3.05 1.25" size="4.1 0.05 1.25" {WALL}/>
    <geom name="wall west" type="box" pos="-3.05 0 1.25" size="0.05 3 1.25" {WALL}/>
    <geom name="wall east" type="box" pos="5.05 0 1.25" size="0.05 3 1.25" {WALL}/>
    <geom name="wall across" type="box" pos="2.5 -1.05 1.25" size="0.05 1.95 1.25" {WALL}/>
    <geom name="wall across 2" type="box" pos="2.5 2.55 1.25" size="0.05 0.45 1.25" {WALL}/>
    <geom name="above the door" type="box" pos="2.5 1.5 2.15" size="0.05 0.6 0.35" {WALL}/>
    <geom name="table" type="box" pos="0.5 -2 0.73" size="0.6 0.35 0.02" {FIXED}/>
    <geom name="table leg 1" type="capsule" fromto="-0.05 -2.3 0 -0.05 -2.3 0.71" size="0.02" {FIXED}/>
    <geom name="table leg 2" type="capsule" fromto="1.05 -2.3 0 1.05 -2.3 0.71" size="0.02" {FIXED}/>
    <geom name="table leg 3" type="capsule" fromto="-0.05 -1.7 0 -0.05 -1.7 0.71" size="0.02" {FIXED}/>
    <geom name="table leg 4" type="capsule" fromto="1.05 -1.7 0 1.05 -1.7 0.71" size="0.02" {FIXED}/>
    <geom name="sofa" type="box" pos="-2.5 0.5 0.22" size="0.45 1.0 0.22" rgba="0.3 0.4 0.55 1"/>
    <geom name="sofa back" type="box" pos="-2.85 0.5 0.45" size="0.1 1.0 0.45" rgba="0.3 0.4 0.55 1"/>
    <geom name="shelf" type="box" pos="4.75 -2.2 0.9" size="0.2 0.5 0.9" {FIXED}/>
    <geom name="step" type="box" pos="4 1.5 0.075" size="0.6 0.6 0.075" rgba="0.6 0.6 0.6 1"/>
    <body name="box 1" pos="1.5 0.8 0.15"><freejoint/><geom name="box 1" type="box" size="0.15 0.15 0.15" mass="0.4" rgba="0.8 0.65 0.4 1"/></body>
    <body name="box 2" pos="1.5 1.25 0.15"><freejoint/><geom name="box 2" type="box" size="0.15 0.15 0.15" mass="0.4" rgba="0.8 0.65 0.4 1"/></body>
    <body name="box 3" pos="1.5 1.0 0.45"><freejoint/><geom name="box 3" type="box" size="0.15 0.15 0.15" mass="0.4" rgba="0.8 0.65 0.4 1"/></body>
    <body name="ball" pos="0.8 1.6 0.12"><freejoint/><geom name="ball" type="sphere" size="0.12" mass="0.15" rgba="0.85 0.25 0.2 1"/></body>
"""

# Stick speeds (and with Shift), PX4 Position-mode style.
FORWARD = (0.8, 1.5)          # m/s
TURN = (45.0, 90.0)           # degrees/s
UP = (0.6, 1.0)               # m/s
DOWN = (0.5, 0.7)
ACCEL = 2.0                   # m/s^2: how fast the asked speed changes
LEASH = 0.3                   # m: the target never runs further than this ahead

K = glfw
FORWARD_KEYS, BACK_KEYS = {K.KEY_UP}, {K.KEY_DOWN}
LEFT_KEYS, RIGHT_KEYS = {K.KEY_LEFT}, {K.KEY_RIGHT}
UP_KEYS = {K.KEY_1, K.KEY_KP_1, K.KEY_PAGE_UP}
DOWN_KEYS = {K.KEY_0, K.KEY_KP_0, K.KEY_PAGE_DOWN}
SHIFT_KEYS = {K.KEY_LEFT_SHIFT, K.KEY_RIGHT_SHIFT}
CTRL_KEYS = {K.KEY_LEFT_CONTROL, K.KEY_RIGHT_CONTROL}

FRAME_RATE = 60
DOUBLE_CLICK = 0.35           # seconds


class Pilot:
    """Keys -> where the flight brain should be: a moving target, like PX4's
    Position mode. Holding a key moves the target; letting go leaves it, and
    the drone stops on it."""

    def __init__(self, twin):
        self.twin = twin
        self.reset()

    def reset(self):
        d = self.twin.data
        self.sp = Setpoint(position=d.qpos[0:3].copy(), yaw=heading(d.qpos[3:7]))
        self.speed = np.zeros(3)                 # world m/s, after the ACCEL limit
        self.mode = "on the floor"
        self.lift_from = None
        self.quiet = 0.0
        self.gust_left = 0.0
        self.braking = False                     # Space: stop, until the keys are let go

    def step(self, forward, turn, climb, fast, dt):
        """forward, turn, climb: -1, 0 or +1 from the keys; fast: Shift."""
        twin, d = self.twin, self.twin.data
        fast = int(bool(fast))

        if self.gust_left > 0:
            self.gust_left -= dt
            if self.gust_left <= 0:
                twin.push = np.zeros(3)

        if self.mode in ("on the floor", "landed"):
            if climb > 0:
                twin.arm(True)
                self.sp.position = d.qpos[0:3].copy()
                self.speed[:] = 0.0
                self.lift_from = d.qpos[2]
                self.mode = "taking off"
            else:
                return

        if self.mode == "taking off" and d.qpos[2] > self.lift_from + 0.03:
            twin.brain.airborne = True
            self.mode = "flying"

        if self.mode == "landing":
            forward, turn, climb, fast = 0, 0, -1, 0

        if self.braking:
            if forward or turn or climb:
                forward, turn, climb = 0, 0, 0
            elif np.linalg.norm(self.speed) < 1e-3:
                self.braking = False

        # The speed asked for, in the world, eased toward at ACCEL.
        yaw = self.sp.yaw
        want = np.array([forward * FORWARD[fast] * math.cos(yaw), forward * FORWARD[fast] * math.sin(yaw),
                         climb * (UP[fast] if climb > 0 else DOWN[fast])])
        change = want - self.speed
        size = np.linalg.norm(change)
        if size > ACCEL * dt:
            change *= ACCEL * dt / size
        self.speed += change

        self.sp.yaw += math.radians(turn * TURN[fast]) * dt
        self.sp.position = self.sp.position + self.speed * dt
        self.sp.velocity = self.speed.copy()
        self.sp.acceleration = change / dt

        # On a leash: pressed against a wall, the target must not run off.
        gap = self.sp.position - d.qpos[0:3]
        if np.linalg.norm(gap) > LEASH:
            self.sp.position = d.qpos[0:3] + gap * LEASH / np.linalg.norm(gap)

        # Sitting on something with the motors low: landed. Motors off.
        hover = twin.body.mass * 9.81
        descending = climb < 0 or self.mode == "landing"
        if descending and twin.brain.thrust < 0.6 * hover and abs(d.qvel[2]) < 0.1:
            self.quiet += dt
        else:
            self.quiet = 0.0
        if self.quiet > 0.5:
            twin.arm(False)
            self.speed[:] = 0.0
            self.mode = "landed"

    def land(self):
        if self.mode in ("flying", "taking off"):
            self.mode = "landing"

    def hold(self):
        """Brake: the target starts where the drone is, moving as it moves,
        and slows to a stop at ACCEL -- snapping it to a halt at 1.7 m/s
        overshot by half a metre and swung back."""
        d = self.twin.data
        self.sp.position = d.qpos[0:3].copy()
        self.speed = d.qvel[0:3].copy()
        self.braking = True

    def gust(self):
        side = np.random.uniform(0, 2 * math.pi)
        self.twin.push = 3.0 * np.array([math.cos(side), math.sin(side), 0.0])
        self.gust_left = 0.5


class App:
    """The window: draws the twin, reads the keys and the mouse, runs the
    physics in step with the clock."""

    def __init__(self, width=1280, height=800):
        self.twin = Twin(world=ROOM)
        self.pilot = Pilot(self.twin)
        m = self.twin.model

        glfw.init()
        self.window = glfw.create_window(width, height, "DELTA twin", None, None)
        glfw.make_context_current(self.window)
        glfw.swap_interval(0)

        self.scene = mujoco.MjvScene(m, maxgeom=4000)
        self.context = mujoco.MjrContext(m, mujoco.mjtFontScale.mjFONTSCALE_100)
        self.option = mujoco.MjvOption()
        # Clicks go through the see-through walls (geom group 1) to what is
        # behind them, as it looks like they should.
        self.pick_option = mujoco.MjvOption()
        self.pick_option.geomgroup[GLASS] = 0
        self.camera = mujoco.MjvCamera()
        self.camera.distance, self.camera.azimuth, self.camera.elevation = 2.4, 200.0, -18.0
        self.perturb = mujoco.MjvPerturb()
        self.twin.perturb = self.perturb
        # The mouse's spring, per kg of what is pushed: MuJoCo's 100 N/m let
        # the cursor run 25 cm ahead of a box with another on top before it
        # moved; at 400 a box follows the drag.
        m.vis.map.stiffness = 400.0

        self.follow = True
        self.offset = np.zeros(3)               # the view's centre, from the drone
        self.help = True
        self.held = set()                       # keys down now
        self.buttons = set()                    # mouse buttons down now
        self.cursor = (0.0, 0.0)
        self.last_click = (-1.0, 0.0, 0.0)      # time, x, y of the last left press
        self.drag = None                        # "camera" or "push" while a button is down
        self.message, self.message_until = "", 0.0
        self.energy_wh = 0.0
        self.names = {i: mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, i) or "" for i in range(m.ngeom)}
        self.clock = (time.perf_counter(), self.twin.data.time)
        self.pace = (time.perf_counter(), self.twin.data.time, 1.0)    # for the sim speed readout
        self.running = True
        self.where = glfw.get_cursor_pos        # the pointer, when a button goes down

        glfw.set_key_callback(self.window, self.on_key)
        glfw.set_mouse_button_callback(self.window, self.on_button)
        glfw.set_cursor_pos_callback(self.window, self.on_cursor)
        glfw.set_scroll_callback(self.window, self.on_scroll)
        glfw.set_window_focus_callback(self.window, self.on_focus)
        self._centre()

    # -- input ---------------------------------------------------------------

    def on_key(self, window, key, scancode, action, mods):
        if action == glfw.PRESS:
            self.held.add(key)
            self.tap(key)
        elif action == glfw.RELEASE:
            self.held.discard(key)

    def tap(self, key):
        """A key's one-off action, on its press."""
        actions = {
            K.KEY_SPACE: (self.pilot.hold, "Stop: holding here."),
            K.KEY_L: (self.pilot.land, "Landing."),
            K.KEY_G: (self.pilot.gust, "A gust: 3 N from the side."),
            K.KEY_R: (self.restart, "Started again."),
            K.KEY_F: (self.toggle_follow, None),
            K.KEY_C: (self._centre, "View back on the drone."),
            K.KEY_H: (self.toggle_help, None),
            K.KEY_ESCAPE: (self.quit, None),
        }
        if key in actions:
            action, said = actions[key]
            action()
            if said:
                self.say(said)

    def on_focus(self, window, focused):
        if not focused:                          # a key let go elsewhere would stay down
            self.held.clear()
            self.buttons.clear()
            self._end_drag()

    def on_button(self, window, button, action, mods):
        x, y = self.where(window)
        self.cursor = (x, y)

        if action == glfw.PRESS:
            self.buttons.add(button)
            if button == glfw.MOUSE_BUTTON_LEFT:
                t, x0, y0 = self.last_click
                if time.perf_counter() - t < DOUBLE_CLICK and math.hypot(x - x0, y - y0) < 6:
                    self.pick(x, y)
                    self.last_click = (-1.0, 0.0, 0.0)
                    return
                self.last_click = (time.perf_counter(), x, y)

            ctrl = bool(mods & glfw.MOD_CONTROL)
            if ctrl and self.perturb.select > 0:
                self.drag = "push"
                mujoco.mjv_initPerturb(self.twin.model, self.twin.data, self.scene, self.perturb)
                self.perturb.active = (mujoco.mjtPertBit.mjPERT_TRANSLATE if button == glfw.MOUSE_BUTTON_RIGHT
                                       else mujoco.mjtPertBit.mjPERT_ROTATE)
            else:
                self.drag = "camera"
        else:
            self.buttons.discard(button)
            if not self.buttons:
                self._end_drag()

    def _end_drag(self):
        self.drag = None
        self.perturb.active = 0

    def on_cursor(self, window, x, y):
        dx, dy = x - self.cursor[0], y - self.cursor[1]
        self.cursor = (x, y)
        if not self.buttons or self.drag is None:
            return

        _, height = glfw.get_window_size(window)
        shift = bool(self.held & SHIFT_KEYS)
        if glfw.MOUSE_BUTTON_RIGHT in self.buttons:
            how = mujoco.mjtMouse.mjMOUSE_MOVE_H if shift else mujoco.mjtMouse.mjMOUSE_MOVE_V
        elif glfw.MOUSE_BUTTON_LEFT in self.buttons:
            how = mujoco.mjtMouse.mjMOUSE_ROTATE_H if shift else mujoco.mjtMouse.mjMOUSE_ROTATE_V
        else:
            how = mujoco.mjtMouse.mjMOUSE_ZOOM

        if self.drag == "push":
            mujoco.mjv_movePerturb(self.twin.model, self.twin.data, how, dx / height, dy / height,
                                   self.scene, self.perturb)
        else:
            mujoco.mjv_moveCamera(self.twin.model, how, dx / height, dy / height, self.camera)
            self.offset = self.camera.lookat - self._drone()

    def on_scroll(self, window, dx, dy):
        """Wheel up (away from you) zooms in."""
        mujoco.mjv_moveCamera(self.twin.model, mujoco.mjtMouse.mjMOUSE_ZOOM, 0.0, 0.05 * dy, self.camera)

    def pick(self, x, y):
        """Double-click: pick the thing under the cursor (or let go of it)."""
        m, d = self.twin.model, self.twin.data
        width, height = glfw.get_window_size(self.window)
        point = np.zeros(3)
        geom, flex, skin = np.zeros(1, np.int32), np.zeros(1, np.int32), np.zeros(1, np.int32)
        body = mujoco.mjv_select(m, d, self.pick_option, width / height, x / width, (height - y) / height,
                                 self.scene, point, geom, flex, skin)

        if body <= 0 or body == self.perturb.select:
            self.perturb.select = 0
            self.say("Let go.")
            return

        self.perturb.select = body
        self.perturb.skinselect = -1
        local = d.xmat[body].reshape(3, 3).T @ (point - d.xpos[body])
        self.perturb.localpos[:] = local
        name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, body)
        self.say(f"Picked: {name}. Ctrl + right drag to push it.")

    # -- view ----------------------------------------------------------------

    def _drone(self):
        return self.twin.data.qpos[0:3].copy()

    def _centre(self):
        self.follow = True
        self.offset = np.zeros(3)
        self.camera.lookat[:] = self._drone()

    def toggle_follow(self):
        self.follow = not self.follow
        if self.follow:
            self.offset = self.camera.lookat - self._drone()
        self.say("The view follows the drone." if self.follow else "The view stays put.")

    def toggle_help(self):
        self.help = not self.help

    def say(self, text, seconds=3.0):
        self.message, self.message_until = text, time.perf_counter() + seconds

    # -- running -------------------------------------------------------------

    def restart(self):
        self.twin.reset()
        self.pilot.reset()
        self.perturb.select = 0
        self.perturb.active = 0
        self.energy_wh = 0.0
        self.clock = (time.perf_counter(), self.twin.data.time)
        self._centre()

    def quit(self):
        self.running = False

    def sticks(self):
        h = self.held
        return (bool(h & FORWARD_KEYS) - bool(h & BACK_KEYS), bool(h & LEFT_KEYS) - bool(h & RIGHT_KEYS),
                bool(h & UP_KEYS) - bool(h & DOWN_KEYS), bool(h & SHIFT_KEYS))

    def advance(self):
        """Run the physics up to the clock (at most 0.1 s a frame: slow motion
        beats a frozen window)."""
        twin, d = self.twin, self.twin.data
        wall0, sim0 = self.clock
        wanted = sim0 + (time.perf_counter() - wall0)
        budget = 100
        forward, turn, climb, fast = self.sticks()

        while d.time < wanted and budget > 0:
            if twin.steps % 2 == 0:
                self.pilot.step(forward, turn, climb, fast, 2.0 / SIM_HZ)
            twin.step(self.pilot.sp)
            self.energy_wh += twin.watts() / SIM_HZ / 3600.0
            budget -= 1

        if budget == 0:
            self.clock = (time.perf_counter(), d.time)

    def draw(self):
        m, d = self.twin.model, self.twin.data
        if self.follow:
            self.camera.lookat[:] = self._drone() + self.offset

        width, height = glfw.get_framebuffer_size(self.window)
        viewport = mujoco.MjrRect(0, 0, width, height)
        mujoco.mjv_updateScene(m, d, self.option, self.perturb, self.camera,
                               mujoco.mjtCatBit.mjCAT_ALL, self.scene)
        mujoco.mjr_render(viewport, self.scene, self.context)

        labels, values = self._readout()
        mujoco.mjr_overlay(mujoco.mjtFont.mjFONT_NORMAL, mujoco.mjtGridPos.mjGRID_TOPLEFT, viewport,
                           labels, values, self.context)
        if self.help:
            mujoco.mjr_overlay(mujoco.mjtFont.mjFONT_NORMAL, mujoco.mjtGridPos.mjGRID_BOTTOMRIGHT, viewport,
                               HELP_KEYS, HELP_DOES, self.context)
        if self.message and time.perf_counter() < self.message_until:
            mujoco.mjr_overlay(mujoco.mjtFont.mjFONT_BIG, mujoco.mjtGridPos.mjGRID_TOP, viewport,
                               self.message, "", self.context)

    def _readout(self):
        twin, d = self.twin, self.twin.data
        floor = twin.body.spec["cage"]["radius"]
        watts = twin.watts()
        usable = 121.0 * 0.8
        left = max(usable - self.energy_wh, 0.0)
        share = max(mo.thrust for mo in twin.motors) / twin.body.motor_model.max_thrust
        cage = twin.body.cage
        touching = sorted({self.names[g] for c in d.contact[:d.ncon]
                           if cage in (twin.model.geom_bodyid[c.geom1], twin.model.geom_bodyid[c.geom2])
                           for g in (c.geom1, c.geom2)
                           if twin.model.geom_bodyid[g] != cage and self.names[g] not in ("", "floor")})

        now = time.perf_counter()
        wall, sim, speed = self.pace
        if now - wall > 1.0:
            speed = (d.time - sim) / (now - wall)
            self.pace = (now, d.time, speed)

        rows = [
            ("", self.pilot.mode.upper()),
            ("height", f"{d.qpos[2] - floor:.2f} m"),
            ("speed", f"{np.linalg.norm(d.qvel[0:3]):.2f} m/s"),
            ("lean", f"{math.degrees(tilt(d.qpos[3:7])):.1f} deg"),
            ("heading", f"{math.degrees(heading(d.qpos[3:7])):.0f} deg"),
            ("power", f"{watts:.0f} W"),
            ("top motor", f"{100 * share:.0f} %"),
            ("battery", f"{100 * left / usable:.1f} % ({self.energy_wh:.2f} Wh used)"),
            ("time left", f"{60 * left / max(watts, 1e-6):.1f} min at this power"),
            ("touching", ", ".join(touching) or "-"),
            ("keys", " ".join(KEY_NAMES.get(k, "?") for k in sorted(self.held)) or "-"),
            ("sim speed", f"{speed:.2f} x real time"),
        ]
        return "\n".join(r[0] for r in rows), "\n".join(r[1] for r in rows)

    def run(self, hook=None):
        """The loop: physics, picture, input, ~60 times a second. `hook(app)`
        runs each frame (the self-test drives the window through it)."""
        frame = 1.0 / FRAME_RATE
        while self.running and not glfw.window_should_close(self.window):
            start = time.perf_counter()
            self.advance()
            self.draw()
            glfw.swap_buffers(self.window)
            glfw.poll_events()
            if hook is not None:
                hook(self)
            spare = frame - (time.perf_counter() - start)
            if spare > 0:
                time.sleep(spare)
        self.context.free()
        glfw.terminate()

    def snapshot(self, path):
        """What the window shows now, saved as a picture."""
        from PIL import Image

        self.draw()                              # the back buffer, before any swap
        width, height = glfw.get_framebuffer_size(self.window)
        rgb = np.zeros((height, width, 3), np.uint8)
        mujoco.mjr_readPixels(rgb, None, mujoco.MjrRect(0, 0, width, height), self.context)
        Image.fromarray(np.flipud(rgb)).save(path)


KEY_NAMES = {K.KEY_UP: "Up", K.KEY_DOWN: "Down", K.KEY_LEFT: "Left", K.KEY_RIGHT: "Right",
             K.KEY_1: "1", K.KEY_0: "0", K.KEY_KP_1: "1", K.KEY_KP_0: "0",
             K.KEY_PAGE_UP: "PgUp", K.KEY_PAGE_DOWN: "PgDn",
             K.KEY_LEFT_SHIFT: "Shift", K.KEY_RIGHT_SHIFT: "Shift",
             K.KEY_LEFT_CONTROL: "Ctrl", K.KEY_RIGHT_CONTROL: "Ctrl"}

HELP_KEYS = "\n".join(["Up / Down", "Left / Right", "1 / 0", "Shift", "Space   L   G   R",
                       "F   C   H   Esc", "", "left drag", "right drag", "scroll",
                       "double-click", "Ctrl + right drag", "Ctrl + left drag"])
HELP_DOES = "\n".join(["forward / back", "turn", "up / down (1 on the floor: take off)", "faster",
                       "stop   land   gust   start again", "follow   centre view   help   quit", "",
                       "turn the view", "slide the view", "zoom", "pick a thing (again: let go)",
                       "push what is picked", "twist it"])


def main():
    App().run()


if __name__ == "__main__":
    main()
