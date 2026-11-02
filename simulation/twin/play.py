"""Fly the twin yourself.

    python -m twin.play

Two windows open:

    the 3D window (MuJoCo's viewer)   watch, and touch:
        left drag            turn the camera        right drag   move it
        scroll               zoom                   F (panel)    follow the drone or not
        double-click a thing, then Ctrl + right drag    push it (the drone too)
        Ctrl + left drag     twist it
        Tab / Shift+Tab      MuJoCo's own panels: show contacts, forces, centres
                             of mass, each motor's thrust...

    the control panel (the remote)    fly -- click it first, so it has the keys:
        Up / Down            forward / back         Left / Right   turn
        1 / 0                up / down (1 on the floor takes off)
        Shift                faster                 Space          stop, hold here
        L                    land                   G              a gust from the side
        R                    start again

The keys are the DELTA window's (delta/window/app.py). Like PX4's Position
mode: the keys ask for a speed, and letting go stops the drone where it is.
"""

import math
import os
import time
import tkinter as tk

# The 3D window draws on the CPU (Mesa's llvmpipe), not through WSL's D3D12
# bridge to the GPU: MuJoCo draws geom by geom, and the bridge charges each
# draw ~0.3-1 ms. Measured on the twin in its room: 9 ms a frame on the CPU,
# 56 ms through the bridge (and 900 ms before the cage became one mesh).
# Habitat is the opposite -- few, big draws -- so it keeps d3d12.
os.environ["GALLIUM_DRIVER"] = os.environ.get("TWIN_GALLIUM_DRIVER", "llvmpipe")
os.environ.pop("MESA_LOADER_DRIVER_OVERRIDE", None)         # ~/.bashrc sets d3d12 for Habitat

import mujoco  # noqa: E402
import mujoco.viewer  # noqa: E402
import numpy as np  # noqa: E402

from twin.fly import SIM_HZ, Twin  # noqa: E402
from twin.flight import Setpoint, heading, tilt  # noqa: E402

# A room to fly in: 8 x 6 m, 2.5 m walls (see-through, so the camera can
# look in), a wall across it with a doorway, furniture that stays put, and
# boxes and a ball that don't.
WALL = 'rgba="0.85 0.82 0.78 0.25"'
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
# 1 and 0, on the row and the keypad, Num Lock on or off; with Shift they
# arrive as ! and ).
UP_KEYS = {"1", "KP_1", "KP_End", "exclam"}
DOWN_KEYS = {"0", "KP_0", "KP_Insert", "parenright"}


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

    def step(self, keys, dt):
        twin, d = self.twin, self.twin.data
        fast = 1 if ("Shift_L" in keys or "Shift_R" in keys) else 0
        forward = ("Up" in keys) - ("Down" in keys)
        turn = ("Left" in keys) - ("Right" in keys)
        climb = bool(keys & UP_KEYS) - bool(keys & DOWN_KEYS)

        if self.mode in ("on the floor", "landed"):
            if climb > 0:
                twin.arm(True)
                self.sp.position = d.qpos[0:3].copy()
                self.lift_from = d.qpos[2]
                self.mode = "taking off"
            else:
                return

        if self.mode == "taking off" and d.qpos[2] > self.lift_from + 0.03:
            twin.brain.airborne = True
            self.mode = "flying"

        if self.mode == "landing":
            forward, turn, climb = 0, 0, -1
            fast = 0

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

        # Sitting on the floor with the motors low: landed. Motors off.
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

        if self.gust_left > 0:
            self.gust_left -= dt
            if self.gust_left <= 0:
                twin.push = np.zeros(3)

    def land(self):
        if self.mode in ("flying", "taking off"):
            self.mode = "landing"

    def hold(self):
        self.speed[:] = 0.0
        self.sp.position = self.twin.data.qpos[0:3].copy()

    def gust(self):
        side = np.random.uniform(0, 2 * math.pi)
        self.twin.push = 3.0 * np.array([math.cos(side), math.sin(side), 0.0])
        self.gust_left = 0.5


class Panel:
    """The remote: the readouts, the buttons, and the keyboard."""

    def __init__(self, root, twin, pilot, viewer):
        self.root, self.twin, self.pilot, self.viewer = root, twin, pilot, viewer
        self.keys = set()
        self._release = {}
        self.follow = True
        self.energy_wh = 0.0
        self.started = time.perf_counter()
        self.sim_started = twin.data.time
        self.last_sync = 0.0
        self.last_text = 0.0
        self.pace = (time.perf_counter(), twin.data.time)     # wall, sim: for the sim speed readout
        self.sim_speed = 1.0
        self.names = {i: mujoco.mj_id2name(twin.model, mujoco.mjtObj.mjOBJ_GEOM, i) or ""
                      for i in range(twin.model.ngeom)}

        root.title("DELTA twin - remote")
        root.configure(bg="#f4f4f2")
        font = ("DejaVu Sans Mono", 11)
        self.readout = tk.Label(root, font=font, justify="left", anchor="nw", bg="#f4f4f2", width=46, height=13)
        self.readout.pack(padx=12, pady=(12, 6), fill="x")

        row = tk.Frame(root, bg="#f4f4f2")
        row.pack(padx=12, pady=4)
        for text, action in [("Take off (1)", self._take_off), ("Land (L)", pilot.land),
                             ("Stop (Space)", pilot.hold), ("Gust (G)", pilot.gust),
                             ("Follow (F)", self._toggle_follow), ("Start again (R)", self._reset)]:
            tk.Button(row, text=text, command=action, width=13).pack(side="left", padx=2)

        tk.Label(root, font=("DejaVu Sans", 10), justify="left", bg="#f4f4f2", fg="#444", text=(
            "Click this window first: it has the keys.\n"
            "Up/Down forward/back   Left/Right turn   1/0 up/down   Shift faster\n"
            "3D window: drag to look, scroll to zoom; double-click a thing,\n"
            "then Ctrl + right drag to push it (the drone too).")).pack(padx=12, pady=(6, 12))

        root.bind("<KeyPress>", self._press)
        root.bind("<KeyRelease>", self._unpress)
        root.protocol("WM_DELETE_WINDOW", self._close)
        self._set_follow()

    # X11 repeats a held key as release + press pairs: a release only counts
    # if no press follows it within 40 ms.
    def _press(self, event):
        key = event.keysym
        pending = self._release.pop(key, None)
        if pending is not None:
            self.root.after_cancel(pending)
        if key not in self.keys:
            self.keys.add(key)
            self._tap(key)

    def _unpress(self, event):
        key = event.keysym
        self._release[key] = self.root.after(40, lambda: (self.keys.discard(key), self._release.pop(key, None)))

    def _tap(self, key):
        actions = {"l": self.pilot.land, "L": self.pilot.land, "space": self.pilot.hold,
                   "g": self.pilot.gust, "G": self.pilot.gust, "f": self._toggle_follow, "F": self._toggle_follow,
                   "r": self._reset, "R": self._reset, "Escape": self._close}
        if key in actions:
            actions[key]()

    def _take_off(self):
        self.keys.add("1")
        self.root.after(1200, lambda: self.keys.discard("1"))

    def _toggle_follow(self):
        self.follow = not self.follow
        self._set_follow()

    def _set_follow(self):
        with self.viewer.lock():
            cam = self.viewer.cam
            if self.follow:
                cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
                cam.trackbodyid = self.twin.body.frame
                cam.distance = 2.2
                cam.elevation = -18.0
            else:
                cam.type = mujoco.mjtCamera.mjCAMERA_FREE

    def _reset(self):
        self.twin.reset()
        self.pilot.reset()
        self.energy_wh = 0.0
        self.started = time.perf_counter()
        self.sim_started = self.twin.data.time
        self.viewer.sync()

    def _close(self):
        self.viewer.close()
        self.root.destroy()

    def tick(self):
        """Run the physics up to the wall clock, show it, and come back."""
        if not self.viewer.is_running():
            self.root.destroy()
            return

        twin, d = self.twin, self.twin.data
        wanted = self.sim_started + (time.perf_counter() - self.started)
        budget = 40                                   # steps a tick at most: slow motion beats freezing
        while d.time < wanted and budget > 0:
            if twin.steps % 2 == 0:
                self.pilot.step(self.keys, 2.0 / SIM_HZ)
            twin.step(self.pilot.sp)
            self.energy_wh += twin.watts() / SIM_HZ / 3600.0
            budget -= 1
        if budget == 0:                               # behind: let the clock slip instead
            self.started += wanted - d.time

        now = time.perf_counter()
        if now - self.last_sync > 1 / 60:
            self.viewer.sync()
            self.last_sync = now
        if now - self.last_text > 0.1:
            wall, sim = self.pace
            if now - wall > 1.0:
                self.sim_speed = (d.time - sim) / (now - wall)
                self.pace = (now, d.time)
            self._show()
            self.last_text = now

        self.root.after(2, self.tick)

    def _show(self):
        twin, d = self.twin, self.twin.data
        floor = twin.body.spec["cage"]["radius"]
        watts = twin.watts()
        usable = 121.0 * 0.8
        left = max(usable - self.energy_wh, 0.0)
        share = max(m.thrust for m in twin.motors) / twin.body.motor_model.max_thrust
        touching = sorted({self.names[g] for c in d.contact[:d.ncon] for g in (c.geom1, c.geom2)
                           if twin.model.geom_bodyid[g] not in (twin.body.cage,)
                           and self.names[g] and self.names[g] != "floor"
                           and twin.body.cage in (twin.model.geom_bodyid[c.geom1], twin.model.geom_bodyid[c.geom2])})
        self.readout.configure(text=(
            f"{self.pilot.mode.upper():<16} time {d.time:6.1f} s\n"
            f"height     {d.qpos[2] - floor:5.2f} m above the floor\n"
            f"speed      {np.linalg.norm(d.qvel[0:3]):5.2f} m/s    lean {math.degrees(tilt(d.qpos[3:7])):4.1f} deg\n"
            f"heading    {math.degrees(heading(d.qpos[3:7])):6.1f} deg\n"
            f"power      {watts:5.0f} W     top motor {100 * share:3.0f}%\n"
            f"battery    {100 * left / usable:5.1f}% left  ({self.energy_wh:.2f} Wh used)\n"
            f"time left  {60 * left / max(watts, 1e-6):5.1f} min at this power\n"
            f"touching   {', '.join(touching) or '-'}\n"
            f"keys       {' '.join(sorted(self.keys)) or '-'}\n"
            f"sim speed  {self.sim_speed:4.2f} x real time"))


def main():
    twin = Twin(world=ROOM)
    pilot = Pilot(twin)
    viewer = mujoco.viewer.launch_passive(twin.model, twin.data, show_left_ui=False, show_right_ui=False)
    twin.perturb = viewer.perturb

    root = tk.Tk()
    panel = Panel(root, twin, pilot, viewer)
    root.after(50, panel.tick)

    if os.environ.get("TWIN_PLAY_DEMO"):
        demo(root, panel)

    root.mainloop()


def demo(root, panel):
    """Hold keys on a timer, as a pilot would, then close: a check that the
    remote, the brain and the windows work together (TWIN_PLAY_DEMO=1)."""
    t = 1000
    for keys, seconds in [({"1"}, 2.0), (set(), 1.5), ({"Up"}, 2.5), (set(), 1.5),
                          ({"Left"}, 2.0), ({"Up", "Shift_L"}, 1.5), (set(), 1.5), ({"0"}, 4.0)]:
        root.after(t, lambda k=keys: (panel.keys.clear(), panel.keys.update(k)))
        t += int(seconds * 1000)
    root.after(t, lambda: (panel.keys.clear(), print(panel.readout.cget("text"), flush=True)))
    root.after(t + 500, panel._close)


if __name__ == "__main__":
    main()
