"""Fly a script with the twin, and write down what happened.

    python -m twin.fly twin/flights/hover.json --out ~/delta-twin-out/hover [--film]

A script is a list of steps, each done before the next starts. Heights are
of the ball's centre above the floor (0.225 m when it sits on it).

    {"takeoff": 1.0}                    arm, climb there at 0.5 m/s
    {"hold": 10}                        stay put for 10 s
    {"goto": [x, y, z]}                 fly there (the brain caps the speed);
                                        done within 5 cm and slow, or 15 s
    {"yaw": 90}                         face 90 degrees left of the start
    {"push": [fx, fy, fz], "for": 0.5}  a gust: newtons on the cage
    {"land": true}                      down at 0.5 m/s, disarm on the floor

Any step can carry "name": the report gives numbers per named step.

Out: log.npz (100 Hz), report.json, flight.png, and with --film flight.mp4
plus a still of each named step.
"""

import argparse
import json
import math
import shutil
import subprocess
import time
from pathlib import Path

import mujoco
import numpy as np

from twin.body import Body
from twin.flight import FlightController, Setpoint, State, Trajectory, heading, tilt
from twin.motors import Motor

SIM_HZ = 1000
CONTROL_EVERY = 2          # the flight brain at 500 Hz
LOG_EVERY = 10             # the log at 100 Hz
FILM_FPS = 30


class Twin:
    """The body, its four motors and the flight brain, stepped together."""

    def __init__(self, world=""):
        self.body = Body(world=world)
        self.model, self.data = self.body.model, self.body.data
        air = self.body.spec["air"]
        self.drag = 0.5 * air["density"] * air["drag_area"]
        self.perturb = None            # a viewer's mouse push (MjvPerturb), if any
        self.reset()

    def reset(self):
        """Back on the floor where it started, motors off, brain fresh."""
        m, b = self.model, self.body
        mujoco.mj_resetData(m, self.data)
        mujoco.mj_forward(m, self.data)
        self.motors = [Motor(b.motor_model) for _ in range(4)]
        self.brain = FlightController(b.mass, b.inertia, b.rotors - b.centre_of_mass, b.spin,
                                      abs(float(m.actuator_gear[0][5])), b.motor_model.max_thrust)
        self.push = np.zeros(3)
        self.steps = 0
        self.thrust_cmd = np.zeros(4)

    @property
    def t(self):
        return self.data.time

    def state(self):
        """The truth, for step 1 (an oracle: step 2 estimates it instead)."""
        d = self.data
        return State(position=d.qpos[0:3].copy(), velocity=d.qvel[0:3].copy(),
                     attitude=d.qpos[3:7].copy(), rates=d.qvel[3:6].copy())

    def arm(self, on=True):
        for motor in self.motors:
            motor.armed = on
        if not on:
            self.brain.airborne = False
            self.brain.reset()

    def step(self, setpoint):
        m, d, b = self.model, self.data, self.body

        if self.steps % CONTROL_EVERY == 0 and self.motors[0].armed:
            self.thrust_cmd = self.brain.update(self.state(), setpoint, self.t, CONTROL_EVERY / SIM_HZ)

        twist = 0.0
        for i, motor in enumerate(self.motors):
            motor.command(self.thrust_cmd[i]) if motor.armed else motor.command(0.0)
            motor.step(1.0 / SIM_HZ)
            d.ctrl[i] = motor.thrust
            twist -= b.spin[i] * motor.model.rotor_inertia * motor.acceleration

        up = d.xmat[b.frame].reshape(3, 3)[:, 2]
        d.xfrc_applied[:] = 0.0
        d.xfrc_applied[b.frame, 3:6] = twist * up
        v = d.qvel[0:3]
        d.xfrc_applied[b.cage, 0:3] = -self.drag * np.linalg.norm(v) * v + self.push
        if self.perturb is not None:
            mujoco.mjv_applyPerturbForce(m, d, self.perturb)

        mujoco.mj_step(m, d)
        self.steps += 1

    def watts(self):
        return sum(motor.watts for motor in self.motors) + self.body.electronics_watts()


def run(script, film=False, out=None, width=640, height=360):
    twin = Twin()
    m, d, b = twin.model, twin.data, twin.body
    rows, marks, frames, stills = [], [], [], {}

    renderer = camera = None
    if film:
        renderer = mujoco.Renderer(m, height, width)
        camera = mujoco.MjvCamera()
        camera.distance, camera.azimuth, camera.elevation = 1.7, 215.0, -12.0
        camera.lookat[:] = d.qpos[0:3]

    start = d.qpos[0:3].copy()
    sp = Setpoint(position=start.copy(), yaw=0.0)
    wall = {"physics": 0.0, "film": 0.0}
    next_frame = 0.0

    def tick(label):
        nonlocal next_frame
        t0 = time.perf_counter()
        twin.step(sp)
        wall["physics"] += time.perf_counter() - t0

        if twin.steps % LOG_EVERY == 0:
            rows.append(_row(twin, sp, label))

        if film and d.time >= next_frame:
            t0 = time.perf_counter()
            camera.lookat[:] += 0.15 * (d.qpos[0:3] - camera.lookat)
            renderer.update_scene(d, camera=camera)
            frames.append(_caption(renderer.render(), twin, label))
            next_frame += 1.0 / FILM_FPS
            wall["film"] += time.perf_counter() - t0

    # Sit on the floor first: the cage settles, the frame hangs level.
    while d.time < 0.5:
        tick("on the floor")

    for step in script:
        label = step.get("name", next(k for k in step if k != "name"))
        used = sum(1 for mk in marks if mk["name"].split(" #")[0] == label)
        label = f"{label} #{used + 1}" if used else label
        began = d.time

        if "takeoff" in step:
            # Spool up, ask gently upward until the cage leaves the floor, then
            # fly the rest as a smooth line (PX4: MPC_TKO_RAMP_T, then MPC_TKO_SPEED).
            twin.arm(True)
            target = np.array([*d.qpos[0:2], float(step["takeoff"])])
            z0 = d.qpos[2]
            while d.qpos[2] < z0 + 0.03 and d.time - began < 5.0:
                sp.position[2] = z0 + 0.3 * (d.time - began)
                tick(label)
            twin.brain.airborne = True
            line = Trajectory(d.qpos[0:3].copy(), target, d.time, twin.brain.g)
            while d.time - began < line.duration + 10.0:
                sp.position, sp.velocity, sp.acceleration = line.sample(d.time)
                tick(label)
                if d.time > line.t0 + line.duration and abs(d.qpos[2] - target[2]) < 0.05:
                    break
            sp.position, sp.velocity, sp.acceleration = target, np.zeros(3), np.zeros(3)

        elif "hold" in step:
            while d.time - began < float(step["hold"]):
                tick(label)

        elif "goto" in step:
            line = Trajectory(sp.position, step["goto"], d.time, twin.brain.g)
            target = np.array(step["goto"], dtype=float)
            while d.time - began < line.duration + 10.0:
                sp.position, sp.velocity, sp.acceleration = line.sample(d.time)
                tick(label)
                if d.time - began > line.duration and np.linalg.norm(d.qpos[0:3] - target) < 0.05 \
                        and np.linalg.norm(d.qvel[0:3]) < 0.1:
                    break
            sp.position, sp.velocity, sp.acceleration = target, np.zeros(3), np.zeros(3)

        elif "yaw" in step:
            # Turned at PX4's auto yaw rate (MPC_YAWRAUTO_MAX), not all at once.
            target = math.radians(float(step["yaw"]))
            first = sp.yaw
            turn = math.remainder(target - first, 2 * math.pi)
            rate = math.radians(45.0)
            while d.time - began < abs(turn) / rate + 10.0:
                done = min(1.0, rate * (d.time - began) / max(abs(turn), 1e-9))
                sp.yaw = first + done * turn
                tick(label)
                if done < 1.0:
                    continue
                off = math.remainder(heading(d.qpos[3:7]) - sp.yaw, 2 * math.pi)
                if abs(off) < math.radians(2.0) and abs(d.qvel[5]) < 0.05:
                    break

        elif "push" in step:
            twin.push = np.array(step["push"], dtype=float)
            while d.time - began < float(step.get("for", 0.5)):
                tick(label)
            twin.push = np.zeros(3)

        elif "land" in step:
            # PX4's MPC_LAND_SPEED, then slower for the last 0.4 m (MPC_LAND_ALT2,
            # which needs the height above the floor: here the truth, in step 2
            # the TF-Luna's).
            quiet = 0.0
            floor = b.spec["cage"]["radius"]
            while d.time - began < 20.0:
                speed = 0.5 if d.qpos[2] - floor > 0.4 else 0.2
                sp.position[2] -= speed / SIM_HZ
                sp.velocity[2] = -speed
                tick(label)
                hover = twin.body.mass * 9.81
                quiet = quiet + 1.0 / SIM_HZ if (twin.brain.thrust < 0.6 * hover and abs(d.qvel[2]) < 0.1) else 0.0
                if quiet > 0.5:
                    break
            twin.arm(False)
            sp.velocity[:] = 0.0
            for _ in range(500):
                tick("landed")

        marks.append({"name": label, "kind": next(k for k in step if k != "name"),
                      "step": step, "start": began, "end": d.time})
        if film and frames:
            stills[label] = frames[-1]

    log = {k: np.array([r[k] for r in rows]) for k in rows[0] if k != "label"}
    log["label"] = np.array([r["label"] for r in rows])
    report = summarise(twin, log, marks, wall)

    if out is not None:
        out = Path(out).expanduser()
        out.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(out / "log.npz", **log)
        (out / "report.json").write_text(json.dumps(report, indent=1))
        (out / "model.xml").write_text(b.xml)
        plot(log, marks, out / "flight.png")
        if film:
            _write_video(frames, out / "flight.mp4", width, height)
            _write_stills(stills, out)

    return report, log, marks


def _row(twin, sp, label):
    d, b = twin.data, twin.body
    q = d.qpos[3:7]
    cage = np.zeros(6)
    mujoco.mj_objectVelocity(twin.model, d, mujoco.mjtObj.mjOBJ_BODY, b.cage, cage, 0)
    full = twin.body.motor_model.max_thrust
    return {
        "t": d.time, "label": label,
        "position": d.qpos[0:3].copy(), "velocity": d.qvel[0:3].copy(),
        "setpoint": sp.position.copy(), "yaw_sp": sp.yaw, "velocity_ff": sp.velocity.copy(),
        "velocity_sp": np.array(twin.brain.out.get("v_sp", np.zeros(3))),
        "accel_sp": np.array(twin.brain.out.get("accel_sp", np.zeros(3))),
        "heading": heading(q), "tilt": tilt(q), "tilt_sp": tilt(twin.brain.attitude_sp),
        "rates": d.qvel[3:6].copy(),
        "rpm": np.array([mo.speed * 60 / (2 * math.pi) for mo in twin.motors]),
        "thrust": np.array([mo.thrust for mo in twin.motors]),
        "motor_share": np.array([mo.thrust / full for mo in twin.motors]),
        "watts": twin.watts(),
        "motor_watts": np.array([mo.watts for mo in twin.motors]),
        "cage_spin": cage[0:3].copy(),
        "bearings": d.qpos[7:10].copy(),
        "airborne": float(twin.brain.airborne),
        "touching": float(any(b.cage in (twin.model.geom_bodyid[c.geom1], twin.model.geom_bodyid[c.geom2])
                              for c in d.contact[:d.ncon])),
    }


def summarise(twin, log, marks, wall):
    b, mm = twin.body, twin.body.motor_model
    t = log["t"]
    rpm_k = 60 / (2 * math.pi)
    hover_n = b.mass * 9.81 / 4
    report = {
        "body": b.report(),
        "motors": {**mm.fit_report(),
                   "hover_thrust_each_n": hover_n,
                   "hover_share_of_full": hover_n / mm.max_thrust,
                   "hover_rpm_predicted": math.sqrt(hover_n / mm.k) * rpm_k,
                   "hover_watts_predicted": 4 * mm.watts(hover_n) + b.electronics_watts()},
        "real_time_factor": float(t[-1] / wall["physics"]),
        "film_seconds": wall["film"],
        "steps": [],
    }
    energy_wh = float(np.trapezoid(log["watts"], t)) / 3600.0
    report["energy_wh"] = energy_wh

    for mk in marks:
        inside = (t >= mk["start"]) & (t <= mk["end"])
        if not inside.any():
            continue
        pos, sp = log["position"][inside], log["setpoint"][inside]
        entry = {"name": mk["name"], "kind": mk["kind"], "seconds": mk["end"] - mk["start"],
                 "mean_watts": float(log["watts"][inside].mean()),
                 "peak_watts": float(log["watts"][inside].max()),
                 "peak_motor_share": float(log["motor_share"][inside].max()),
                 "max_tilt_deg": float(np.degrees(log["tilt"][inside].max()))}

        if mk["kind"] == "hold":
            err = pos - sp
            entry.update(height_rms_cm=float(100 * np.sqrt(np.mean(err[:, 2] ** 2))),
                         drift_max_cm=float(100 * np.max(np.linalg.norm(err[:, :2], axis=1))),
                         mean_rpm=float(log["rpm"][inside].mean()),
                         mean_motor_share=float(log["motor_share"][inside].mean()))
        elif mk["kind"] == "goto":
            a = pos[0]
            target = np.array(mk["step"]["goto"], dtype=float)
            line = (target - a) / max(np.linalg.norm(target - a), 1e-9)
            along = (pos - a) @ line
            dist = float(np.linalg.norm(target - a))
            reached = np.nonzero(np.linalg.norm(pos - target, axis=1) < 0.05)[0]
            entry.update(distance_m=dist,
                         overshoot_cm=float(100 * max(along.max() - dist, 0.0)),
                         seconds_to_5cm=float(t[inside][reached[0]] - mk["start"]) if len(reached) else None,
                         max_speed=float(np.linalg.norm(log["velocity"][inside], axis=1).max()))
        elif mk["kind"] == "yaw":
            target = math.radians(float(mk["step"]["yaw"]))
            nxt = next((m2 for m2 in marks if m2["start"] >= mk["end"]), None)
            after = (t >= mk["start"]) & (t <= (nxt["end"] if nxt else mk["end"]))
            off = np.degrees(np.remainder(log["heading"][after] - target + math.pi, 2 * math.pi) - math.pi)
            way = 1.0 if off[0] < 0 else -1.0                  # turning toward +off
            entry.update(turned_deg=float(-off[0]),
                         overshoot_deg=float(max(0.0, (way * off).max())),
                         seconds=float(mk["end"] - mk["start"]),
                         cage_turned_deg=float(np.degrees(np.trapezoid(log["cage_spin"][inside][:, 2], t[inside]))))
        elif mk["kind"] == "push":
            after = (t >= mk["start"])
            nxt = next((m2 for m2 in marks if m2["start"] >= mk["end"]), None)
            if nxt is not None:
                after &= t <= nxt["end"]
            err = np.linalg.norm(log["position"][after] - log["setpoint"][after], axis=1)
            back = np.nonzero((t[after] > mk["end"]) & (err < 0.05))[0]
            entry.update(newtons=mk["step"]["push"], pushed_cm=float(100 * err.max()),
                         back_within_5cm_s=float(t[after][back[0]] - mk["end"]) if len(back) else None)
        elif mk["kind"] == "land":
            vz = log["velocity"][inside][:, 2]
            touch = np.nonzero(log["touching"][inside] > 0)[0]
            entry.update(fastest_descent=float(-vz.min()),
                         touchdown_speed=float(-vz[max(touch[0] - 1, 0)]) if len(touch) else None)
        elif mk["kind"] == "takeoff":
            entry.update(seconds=float(mk["end"] - mk["start"]))

        report["steps"].append(entry)

    hover = next((s for s in report["steps"] if s["kind"] == "hold"), None)
    if hover:
        usable = 121.0 * 0.8
        report["hover_minutes_on_4s2p"] = 60.0 * usable / hover["mean_watts"]

    return report


def plot(log, marks, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    t = log["t"]
    colours = ["#2a6fb0", "#d0702a", "#3b9a5a", "#8a5bb0"]
    fig, axes = plt.subplots(5, 1, figsize=(11, 12), sharex=True)
    ax = axes[0]
    ax.plot(t, log["position"][:, 2], color=colours[0], lw=1.6, label="height")
    ax.plot(t, log["setpoint"][:, 2], color="#888", lw=1.0, ls="--", label="asked")
    ax.set_ylabel("height of ball centre (m)")
    ax.legend(loc="upper right", frameon=False)

    ax = axes[1]
    for k, name in enumerate(["x (forward)", "y (left)"]):
        ax.plot(t, log["position"][:, k], color=colours[k], lw=1.6, label=name)
        ax.plot(t, log["setpoint"][:, k], color=colours[k], lw=1.0, ls="--", alpha=0.6)
    ax.set_ylabel("position (m)")
    ax.legend(loc="upper right", frameon=False)

    ax = axes[2]
    ax.plot(t, np.degrees(log["tilt"]), color=colours[0], lw=1.4, label="tilt")
    ax.plot(t, np.degrees(log["heading"]), color=colours[1], lw=1.4, label="heading")
    ax.plot(t, np.degrees(log["yaw_sp"]), color=colours[1], lw=1.0, ls="--", alpha=0.6)
    ax.set_ylabel("degrees")
    ax.legend(loc="upper right", frameon=False)

    ax = axes[3]
    for k, name in enumerate(["front right", "back left", "front left", "back right"]):
        ax.plot(t, 100 * log["motor_share"][:, k], color=colours[k], lw=1.0, label=name)
    ax.set_ylabel("motor thrust (% of full)")
    ax.set_ylim(0, max(35.0, 1.3 * 100 * log["motor_share"].max()))
    ax.legend(loc="upper right", frameon=False, ncol=4)

    ax = axes[4]
    ax.plot(t, log["watts"], color=colours[0], lw=1.4)
    ax.set_ylabel("power drawn (W)")
    ax.set_ylim(bottom=0)
    ax.set_xlabel("seconds")

    for a in axes:
        a.grid(alpha=0.25)
        a.spines[["top", "right"]].set_visible(False)
        for mk in marks:
            a.axvline(mk["start"], color="#bbb", lw=0.7)
    for mk in marks:
        if mk["kind"] != "hold":                         # the holds speak for themselves
            axes[0].annotate(mk["name"], (mk["start"], 1.01), xycoords=("data", "axes fraction"),
                             fontsize=8, rotation=35, va="bottom", ha="left", color="#555")
    axes[2].legend(loc="center right", frameon=False)

    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def _caption(image, twin, label):
    from PIL import Image, ImageDraw

    d = twin.data
    img = Image.fromarray(image)
    draw = ImageDraw.Draw(img)
    share = max(mo.thrust for mo in twin.motors) / twin.body.motor_model.max_thrust
    text = (f"{d.time:5.1f} s   {label}\nheight {d.qpos[2]:.2f} m   speed {np.linalg.norm(d.qvel[0:3]):.2f} m/s\n"
            f"power {twin.watts():4.0f} W   top motor {100 * share:3.0f}%   tilt {math.degrees(tilt(d.qpos[3:7])):4.1f} deg")
    draw.rectangle([0, 0, 330, 52], fill=(255, 255, 255))
    draw.multiline_text((8, 4), text, fill=(20, 20, 20), spacing=3)
    return np.asarray(img)


def _ffmpeg():
    """ffmpeg on the PATH, or the one another conda env brought along."""
    found = shutil.which("ffmpeg")
    if found:
        return found
    env = Path(mujoco.__file__).resolve().parents[4]           # envs/twin/lib/python3.12/site-packages/mujoco
    return next((str(p) for p in sorted(env.parent.glob("*/bin/ffmpeg"))), None)


def _write_video(frames, path, width, height):
    ffmpeg = _ffmpeg()
    if ffmpeg is None or not frames:
        return
    proc = subprocess.Popen([ffmpeg, "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
                             "-s", f"{width}x{height}", "-r", str(FILM_FPS), "-i", "-",
                             "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "22", str(path)],
                            stdin=subprocess.PIPE)
    for f in frames:
        proc.stdin.write(f.tobytes())
    proc.stdin.close()
    proc.wait()


def _write_stills(stills, out):
    from PIL import Image

    for i, (name, image) in enumerate(stills.items()):
        Image.fromarray(image).save(out / f"still_{i:02d}_{name.replace(' ', '_')}.png")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("script")
    parser.add_argument("--out", default="~/delta-twin-out/flight")
    parser.add_argument("--film", action="store_true")
    args = parser.parse_args()

    script = json.loads(Path(args.script).read_text())
    report, _, _ = run(script, film=args.film, out=args.out)
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
