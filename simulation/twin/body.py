"""config/body.json -> a MuJoCo model of the drone.

    frame (free to move: the whole drone inside)
      every part as a solid of its own mass, where it sits
      four rotor sites, each pushed by a motor actuator: thrust along the
        frame's +z, and the props' drag twisting the frame about it
      inner ring  -- bearing (left/right) --
        outer ring  -- bearing (front/back) --
          cage  -- bearing (top/bottom) --  the only thing that touches anything

Nothing inside the cage collides: the parts are packed inside it, and what
can hit what is checked by clearances() from the layout instead.
"""

import itertools
import json
import math
from pathlib import Path

import mujoco
import numpy as np

from twin.motors import MotorModel

CONFIG = Path(__file__).resolve().parent.parent / "config" / "body.json"

COLOURS = {
    "battery": "0.25 0.45 0.75 1",
    "motors": "0.55 0.55 0.6 1",
    "props": "0.15 0.15 0.15 0.35",
    "arms": "0.12 0.12 0.12 1",
    "frame plate": "0.12 0.12 0.12 1",
    "lidar": "0.85 0.45 0.15 1",
    "stereo pair": "0.2 0.2 0.2 1",
    "pi": "0.2 0.6 0.3 1",
    "flight controller": "0.75 0.2 0.2 1",
}
ELECTRONICS = "0.3 0.55 0.35 1"


def load_spec(path=CONFIG):
    with open(path) as f:
        return json.load(f)


def motor_positions(spec):
    """(x, y) of each motor, in PX4's quad X order: front right, back left,
    front left, back right."""
    a = spec["motors"]["wheelbase"] / 2.0 / math.sqrt(2.0)
    return [(a, -a), (-a, a), (a, a), (-a, -a)]


def placements(spec, part):
    """Centres of a part's pieces, each [x, y, z]."""
    at = part["at"]

    if at == "motors":
        below = part.get("below", 0.0)
        return [[x, y, -below] for x, y in motor_positions(spec)]

    return at if isinstance(at[0], list) else [at]


def _f(values):
    return " ".join(f"{v:.6g}" for v in values)


def _circle(radius, u, v, n=32, z=0.0):
    """n points around a circle spanned by unit vectors u and v."""
    u, v = np.asarray(u, float), np.asarray(v, float)
    return [radius * (math.cos(t) * u + math.sin(t) * v) + np.array([0, 0, z])
            for t in np.linspace(0, 2 * math.pi, n, endpoint=False)]


def _ring_geoms(points, tube, rgba, mass=None, name="ring"):
    """A closed loop of capsules through `points`, its mass shared out."""
    each = "" if mass is None else f' mass="{mass / len(points):.6g}"'
    out = []

    for i, (a, b) in enumerate(zip(points, points[1:] + points[:1])):
        out.append(f'<geom name="{name}_{i}" type="capsule" fromto="{_f(a)} {_f(b)}" size="{tube}"'
                   f' rgba="{rgba}" contype="0" conaffinity="0"{each}/>')

    return out


def _part_geoms(spec, part):
    grams = part["grams"]
    centres = placements(spec, part)
    mass = grams / 1000.0 / len(centres)
    rgba = COLOURS.get(part["name"], ELECTRONICS)
    name = part["name"].replace(" ", "_")
    out = []

    for i, c in enumerate(centres):
        tag = f'name="{name}_{i}" mass="{mass:.6g}" rgba="{rgba}" contype="0" conaffinity="0"'

        if part["shape"] == "box":
            half = [s / 2.0 for s in part["size"]]
            out.append(f'<geom {tag} type="box" pos="{_f(c)}" size="{_f(half)}"/>')
        elif part["shape"] == "cylinder":
            radius, length = part["size"]
            out.append(f'<geom {tag} type="cylinder" pos="{_f(c)}" size="{radius:.6g} {length / 2:.6g}"/>')
        elif part["shape"] == "arm":
            out.append(f'<geom {tag} type="capsule" fromto="0 0 0 {c[0]:.6g} {c[1]:.6g} 0"'
                       f' size="{part["size"][0]:.6g}"/>')

    return out


def mjcf(spec, motor_model):
    """The MuJoCo XML for the drone, standing on a floor."""
    cage, gimbal, floor = spec["cage"], spec["gimbal"], spec["floor"]
    radius = cage["radius"]
    props = next(p for p in spec["parts"] if p["name"] == "props")
    hover = sum(p["grams"] for p in spec["parts"]) + gimbal["inner"]["grams"] \
        + gimbal["outer"]["grams"] + cage["grams"]
    twist = motor_model.twist_ratio(hover / 1000.0 * 9.81 / 4.0)
    friction = f'{floor["friction"]} 0.005 {floor["rolling"]}'
    bearing = (f'damping="{gimbal["damping"]}" frictionloss="{gimbal["friction"]}"'
               f' armature="1e-6"')

    parts = list(itertools.chain.from_iterable(_part_geoms(spec, p) for p in spec["parts"]))
    rotors = [f'<site name="rotor{i}" pos="{x:.6g} {y:.6g} {-props["below"]:.6g}" size="0.005"/>'
              for i, (x, y) in enumerate(motor_positions(spec))]

    inner = gimbal["inner"]
    outer = gimbal["outer"]
    inner_ring = _ring_geoms(_circle(inner["radius"], (1, 0, 0), (0, 1, 0)), inner["tube"],
                             "0.75 0.75 0.78 1", inner["grams"] / 1000.0, "inner_ring")
    outer_ring = _ring_geoms(_circle(outer["radius"], (1, 0, 0), (0, 0, 1)), outer["tube"],
                             "0.75 0.75 0.78 1", outer["grams"] / 1000.0, "outer_ring")

    # The cage, to look at: four great circles through the poles and five
    # circles of latitude. What touches the floor is one sphere.
    bars = []

    for k, phi in enumerate(np.radians([0, 45, 90, 135])):
        bars += _ring_geoms(_circle(radius, (math.cos(phi), math.sin(phi), 0), (0, 0, 1)),
                            cage["bar"], "0.95 0.95 0.95 0.8", name=f"meridian{k}")

    for k, lat in enumerate(np.radians([-60, -30, 0, 30, 60])):
        bars += _ring_geoms(_circle(radius * math.cos(lat), (1, 0, 0), (0, 1, 0), z=radius * math.sin(lat)),
                            cage["bar"], "0.95 0.95 0.95 0.8", name=f"latitude{k}")

    shell = 2.0 / 3.0 * cage["grams"] / 1000.0 * radius ** 2
    actuators = [f'<motor name="motor{i}" site="rotor{i}" gear="0 0 1 0 0 {-s * twist:.6g}"'
                 f' ctrllimited="true" ctrlrange="0 {motor_model.max_thrust:.6g}"/>'
                 for i, s in enumerate(spec["motors"]["spin"])]

    nl = "\n"
    return f"""
<mujoco model="delta_twin">
  <compiler angle="radian" inertiafromgeom="auto"/>
  <option timestep="0.001" integrator="implicitfast"/>
  <visual>
    <global offwidth="1280" offheight="720"/>
    <quality shadowsize="4096"/>
    <headlight ambient="0.4 0.4 0.4" diffuse="0.5 0.5 0.5"/>
  </visual>
  <asset>
    <texture name="sky" type="skybox" builtin="gradient" rgb1="0.78 0.84 0.92" rgb2="0.97 0.97 0.98" width="256" height="256"/>
    <texture name="grid" type="2d" builtin="checker" rgb1="0.84 0.84 0.82" rgb2="0.74 0.74 0.72" width="512" height="512"/>
    <material name="floor" texture="grid" texrepeat="25 25" reflectance="0.03"/>
  </asset>
  <worldbody>
    <light pos="3 -3 6" dir="-0.4 0.4 -1" castshadow="true"/>
    <geom name="floor" type="plane" size="25 25 0.1" material="floor" friction="{friction}"/>
    <body name="frame" pos="0 0 {radius}">
      <freejoint name="free"/>
      {(nl + "      ").join(parts)}
      {(nl + "      ").join(rotors)}
      <site name="imu" pos="0 0 0.0115" size="0.004"/>
      <body name="inner_ring">
        <joint name="bearing_left_right" type="hinge" axis="0 1 0" {bearing}/>
        {(nl + "        ").join(inner_ring)}
        <body name="outer_ring">
          <joint name="bearing_front_back" type="hinge" axis="1 0 0" {bearing}/>
          {(nl + "          ").join(outer_ring)}
          <body name="cage">
            <joint name="bearing_top_bottom" type="hinge" axis="0 0 1" {bearing}/>
            <inertial pos="0 0 0" mass="{cage['grams'] / 1000.0}" diaginertia="{shell:.6g} {shell:.6g} {shell:.6g}"/>
            <geom name="cage" type="sphere" size="{radius}" rgba="0 0 0 0" condim="6" friction="{friction}"/>
            {(nl + "            ").join(bars)}
          </body>
        </body>
      </body>
    </body>
  </worldbody>
  <actuator>
    {(nl + "    ").join(actuators)}
  </actuator>
</mujoco>
"""


class Body:
    """The built drone: MuJoCo's model and data, and what the rest needs to
    know about it."""

    def __init__(self, path=CONFIG):
        self.spec = load_spec(path)
        self.motor_model = MotorModel(self.spec["motors"])
        self.xml = mjcf(self.spec, self.motor_model)
        self.model = mujoco.MjModel.from_xml_string(self.xml)
        self.data = mujoco.MjData(self.model)

        body = lambda name: mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)
        self.frame = body("frame")
        self.cage = body("cage")
        self.rings = (body("inner_ring"), body("outer_ring"))
        self.spin = np.array(self.spec["motors"]["spin"], dtype=float)
        self.rotors = np.array([[x, y, -self._props_below()] for x, y in motor_positions(self.spec)])
        mujoco.mj_forward(self.model, self.data)

    def _props_below(self):
        return next(p for p in self.spec["parts"] if p["name"] == "props")["below"]

    @property
    def mass(self):
        return float(self.model.body_subtreemass[self.frame])

    @property
    def frame_mass(self):
        return float(self.model.body_mass[self.frame])

    @property
    def centre_of_mass(self):
        """The frame's (everything inside the rings) in its own frame."""
        return self.model.body_ipos[self.frame].copy()

    @property
    def inertia(self):
        """The frame's inertia matrix about its centre of mass, in its own
        frame (MuJoCo keeps principal moments + their axes)."""
        R = np.zeros(9)
        mujoco.mju_quat2Mat(R, self.model.body_iquat[self.frame])
        R = R.reshape(3, 3)
        return R @ np.diag(self.model.body_inertia[self.frame]) @ R.T

    def electronics_watts(self):
        return sum(p.get("watts", 0.0) for p in self.spec["parts"]) / self.spec["converter"]

    def report(self):
        twist = self.model.actuator_gear[0][5]
        return {
            "mass_kg": self.mass,
            "inside_rings_kg": self.frame_mass,
            "centre_of_mass_m": self.centre_of_mass.round(4).tolist(),
            "inertia_diag_kg_m2": np.diag(self.inertia).round(6).tolist(),
            "twist_per_newton_m": abs(float(twist)),
            "electronics_w": self.electronics_watts(),
            **clearances(self.spec),
        }


def _corners(spec, part):
    """Points on the outside of a part, for clearances."""
    out = []

    for c in placements(spec, part):
        c = np.asarray(c, float)

        if part["shape"] == "box":
            half = np.asarray(part["size"]) / 2.0
            out += [c + half * np.array(s) for s in itertools.product((-1, 1), repeat=3)]
        elif part["shape"] == "cylinder":
            radius, length = part["size"]
            for t in np.linspace(0, 2 * math.pi, 24, endpoint=False):
                for dz in (-length / 2, length / 2):
                    out.append(c + np.array([radius * math.cos(t), radius * math.sin(t), dz]))
        elif part["shape"] == "arm":
            out.append(np.array([c[0], c[1], 0.0]))

    return out


def clearances(spec):
    """What can hit what, in metres (negative = they overlap):

    prop_gap     the tightest gap between a prop's disc and any part beside
                 it at the same height (the motors themselves excluded)
    ring_gap     between the farthest point of anything inside and the
                 inner ring, which sweeps a sphere around the pivot
    cage_gap     between the outer ring and the cage
    """
    props = next(p for p in spec["parts"] if p["name"] == "props")
    prop_r, prop_t = props["size"]
    prop_z = -props["below"]
    worst, worst_part = math.inf, None

    for part in spec["parts"]:
        if part["shape"] != "box":
            continue

        half = np.asarray(part["size"]) / 2.0

        for c in placements(spec, part):
            if abs(c[2] - prop_z) > half[2] + prop_t / 2:
                continue                                   # not at the props' height

            for x, y in motor_positions(spec):
                dx = max(abs(x - c[0]) - half[0], 0.0)
                dy = max(abs(y - c[1]) - half[1], 0.0)
                gap = math.hypot(dx, dy) - prop_r

                if gap < worst:
                    worst, worst_part = gap, part["name"]

    far = max(np.linalg.norm(p) for part in spec["parts"] for p in _corners(spec, part))
    inner, outer, cage = spec["gimbal"]["inner"], spec["gimbal"]["outer"], spec["cage"]

    return {
        "prop_gap_m": round(worst, 4),
        "prop_gap_part": worst_part,
        "farthest_inside_m": round(float(far), 4),
        "ring_gap_m": round(inner["radius"] - inner["tube"] - float(far), 4),
        "cage_gap_m": round(cage["radius"] - cage["bar"] - outer["radius"] - outer["tube"], 4),
    }
