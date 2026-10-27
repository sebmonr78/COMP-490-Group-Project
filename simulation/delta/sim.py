"""WORLD SIDE. Opening the house, and the two agents that look at it.

Everything in this file is allowed to read Habitat, because this file IS the
simulator side: it loads the scene, builds the navmesh the drone rolls on,
and renders the pictures. Nothing here is perception.

Keep the line in mind when adding to it. `delta/brain.py` and everything it
imports may never call into this module -- the brain gets two images and an
IMU reading, and that is all. A brain that can reach the pathfinder will
eventually be "helped" with it, and then none of the project's numbers mean
anything.

Two agents, and they do different jobs:

    SPECTATOR   a free camera for a person to watch with. Not a sensor.
    SENSORS     the drone's own stereo pair, moved to the drone every frame.
                This is the only thing the brain ever sees through.
"""

import json
import math
import os
from pathlib import Path

import habitat_sim
import magnum as mn
import numpy as np

from delta import rig as rig_module


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "config" / "project.json"

SPECTATOR, SENSORS, SCORER = 0, 1, 2

# The scoring camera's picture size: true depth, for judging the brain's 3D
# replica after a drive (World.true_depth_at). Never the brain's.
SCORER_SIZE = (320, 200)

# A navmesh point with something within this many metres overhead counts
# as being under a roof. Tall enough to clear a stairwell, short enough
# that open sky does not.
INDOOR_CEILING = 5.0


def fail(message):
    print(f"[FAIL] {message}")
    raise SystemExit(1)


# ---------------------------------------------------------------------
# Configuration and files
# ---------------------------------------------------------------------


def load_project_config():
    if not CONFIG_PATH.is_file():
        fail(f"Project configuration not found:\n{CONFIG_PATH}")

    try:
        return apply_scene(json.loads(CONFIG_PATH.read_text(encoding="utf-8")))

    except json.JSONDecodeError as error:
        fail(f"Invalid JSON in {CONFIG_PATH}:\n{error}")


def apply_scene(config, scene=None):
    """Point the dataset entries at another HSSD house, if one is asked for.

        HSSD_SCENE=103997424_171030444 bash run_drone.sh

    Every file name follows from the scene id, the way the dataset names
    them, and the drone's navmesh is generated for that house the first time
    (see load_or_generate_navmesh). Unset, the configuration is exactly what
    project.json says. Same switch as the glasses' bench-slam-depth-mapping
    branch, for testing that what works here is not one house's luck.
    """
    scene = scene or os.environ.get("HSSD_SCENE")

    if scene:
        dataset = config["dataset"]
        dataset["scene_id"] = scene
        dataset["scene_file"] = f"scenes-uncluttered/{scene}.scene_instance.json"
        dataset["semantic_file"] = f"semantics/scenes/{scene}.semantic_config.json"
        dataset["stage_file"] = f"stages/{scene}.glb"
        dataset["drone_navmesh_file"] = f"generated_navmeshes/{scene}_drone.navmesh"

    return config


def resolve_dataset_paths(config):
    """Absolute paths from project.json, with the data root from the
    environment if it has been set."""
    dataset = config["dataset"]

    root = Path(
        os.environ.get(
            dataset["environment_variable"],
            dataset["default_root"],
        )
    ).expanduser().resolve()

    return (
        root,
        (root / dataset["configuration_file"]).resolve(),
        (root / dataset["scene_file"]).resolve(),
        (root / dataset["drone_navmesh_file"]).resolve(),
    )


def verify_files(root, dataset_config, scene_file):
    if not root.is_dir():
        fail(f"HSSD dataset directory not found:\n{root}")

    if not dataset_config.is_file():
        fail(f"HSSD dataset configuration not found:\n{dataset_config}")

    if not scene_file.is_file():
        fail(f"HSSD scene file not found:\n{scene_file}")

    print("[PASS] HSSD dataset found")


def verify_physics():
    """Refuse to run without Bullet, instead of flying through walls.

    Without it, the drone's collision test has nothing to test against and
    quietly answers "nothing is touching" for every position in the
    building. A silent wrong answer is worse than a stop.
    """
    if not getattr(habitat_sim, "built_with_bullet", False):
        fail(
            "This habitat-sim was built without Bullet, so the drone has no\n"
            "way to detect collisions while flying. Reinstall with:\n"
            "    conda install habitat-sim withbullet -c conda-forge -c aihabitat"
        )

    print("[PASS] Bullet physics available")


# ---------------------------------------------------------------------
# The simulator
# ---------------------------------------------------------------------


def create_simulator(dataset_config, scene_file, rig,
                     spectator_size=(800, 600), stage="process",
                     viewer_hfov=90.0, true_depth=False):
    """The scene, with a spectator camera and the drone's sensor head."""
    backend = habitat_sim.SimulatorConfiguration()

    backend.scene_id = str(scene_file)
    backend.scene_dataset_config_file = str(dataset_config)

    # Physics is ON because the drone's flight collision test is a Bullet
    # call. Nothing ever steps physics -- see the note in drone.py about
    # why the drone is DYNAMIC and still does not fall.
    backend.enable_physics = True

    # Required by the working WSL setup.
    backend.gpu_device_id = -1

    spectator = habitat_sim.agent.AgentConfiguration()

    camera = habitat_sim.CameraSensorSpec()
    camera.uuid = "spectator_rgb"
    camera.sensor_type = habitat_sim.SensorType.COLOR
    camera.sensor_subtype = habitat_sim.SensorSubType.PINHOLE
    camera.resolution = [spectator_size[1], spectator_size[0]]
    camera.position = [0.0, 0.0, 0.0]
    camera.hfov = float(viewer_hfov)

    spectator.sensor_specifications = [camera]

    head = habitat_sim.agent.AgentConfiguration()
    head.sensor_specifications = rig_module.sensor_specs(
        rig,
        stage=stage,
        ground_truth_depth=true_depth,
    )

    # A third camera, for the examiner only: true depth from wherever it is
    # put, rendered only when asked (World.true_depth_at), never sent to the
    # brain. It is how the brain's 3D replica is scored.
    scorer = habitat_sim.agent.AgentConfiguration()
    depth = habitat_sim.CameraSensorSpec()
    depth.uuid = "scorer_depth"
    depth.sensor_type = habitat_sim.SensorType.DEPTH
    depth.sensor_subtype = habitat_sim.SensorSubType.PINHOLE
    depth.resolution = [SCORER_SIZE[1], SCORER_SIZE[0]]
    depth.position = [0.0, 0.0, 0.0]
    depth.hfov = float(rig.camera(rig.stereo_pair[0]).hfov)
    scorer.sensor_specifications = [depth]

    return habitat_sim.Simulator(
        habitat_sim.Configuration(backend, [spectator, head, scorer])
    )


def navmesh_settings_for(drone_config):
    """The navmesh that describes THIS drone, not a walking human.

    HSSD ships one built for a person: 1.6 m tall, 0.20 m wide, able to step
    up 0.20 m. The drone is a 0.30 m ball. Those are different maps of the
    same building, and the difference is not cosmetic -- the drone fits under
    tables and through gaps a person does not, and cannot climb a step at
    all. With a human navmesh it would refuse to roll under its own
    inspection targets, which is most of what it exists to do.
    """
    settings = habitat_sim.NavMeshSettings()
    settings.set_defaults()

    diameter = float(drone_config["diameter"])

    settings.agent_height = diameter
    settings.agent_radius = diameter / 2.0
    settings.agent_max_climb = float(drone_config["max_climb"])
    settings.agent_max_slope = float(drone_config["max_slope"])
    settings.include_static_objects = True

    # Fine voxels, because the robot is small. Recast measures climb and
    # clearance in whole voxels, and Habitat's default 0.20 m voxel turned the
    # 8 cm climb into zero and made a 2 cm rug a wall. See project.json.
    settings.cell_size = float(drone_config.get("navmesh_cell_size", settings.cell_size))
    settings.cell_height = float(drone_config.get("navmesh_cell_height", settings.cell_height))

    return settings


# The settings that decide where the drone can go. A saved navmesh built with
# different values is out of date, however recently it was written.
_DECISIVE = (
    "agent_height", "agent_radius", "agent_max_climb", "agent_max_slope",
    "cell_size", "cell_height", "include_static_objects",
)


def _same_settings(a, b):
    for name in _DECISIVE:
        x, y = getattr(a, name), getattr(b, name)

        if isinstance(x, bool) or isinstance(y, bool):
            if bool(x) != bool(y):
                return False
        elif abs(float(x) - float(y)) > 1e-4:
            return False

    return True


def load_or_generate_navmesh(simulator, navmesh_file, drone_config):
    """Load the drone's saved navmesh, or build one for this building.

    A saved navmesh is only reused if it was built with the settings asked
    for now. Without that check, changing the drone's size or the voxel
    height in project.json did nothing at all: the old file was found, loaded
    and trusted, and the change silently never happened.
    """
    wanted = navmesh_settings_for(drone_config)

    if navmesh_file.is_file():
        simulator.pathfinder.load_nav_mesh(str(navmesh_file))

        if simulator.pathfinder.is_loaded:
            saved = getattr(simulator.pathfinder, "nav_mesh_settings", None)

            if saved is None or _same_settings(saved, wanted):
                print(f"[PASS] Drone navmesh loaded: {navmesh_file.name}")
                return

            print(
                "Saved drone navmesh was built with different settings "
                f"(cell height {saved.cell_height:.2f} m, now "
                f"{wanted.cell_height:.2f} m); rebuilding it..."
            )
        else:
            print("[WARNING] Saved navmesh would not load; rebuilding")

    else:
        print("Building a navigation mesh for the drone (first run only)...")

    if not simulator.recompute_navmesh(simulator.pathfinder, wanted):
        fail("Habitat failed to generate a navigation mesh")

    print("[PASS] Drone navmesh generated")

    navmesh_file.parent.mkdir(parents=True, exist_ok=True)
    simulator.pathfinder.save_nav_mesh(str(navmesh_file))

    if navmesh_file.is_file():
        print(f"[PASS] Saved: {navmesh_file.name}")


def choose_start_position(simulator, seed=7):
    """The same deterministic starting point every run."""
    simulator.pathfinder.seed(seed)

    start = simulator.pathfinder.get_random_navigable_point()

    if np.isnan(np.array(start)).any():
        fail("Habitat could not find a navigable starting point")

    return start


# ---------------------------------------------------------------------
# The sensor head
# ---------------------------------------------------------------------


def mount_sensors(simulator, drone):
    """Put the sensor head where the drone is, looking where it looks.

    The heading comes from `drone.sensor_rotation`, which is the gimbal: the
    sphere's own tumble is deliberately not used, because cameras bolted to a
    rolling shell would spend half their time upside down.
    """
    node = simulator.agents[SENSORS].scene_node

    node.translation = drone.position
    node.rotation = drone.sensor_rotation


def read_cameras(simulator, rig):
    """The stereo pair, as the brain will get it: two RGB images."""
    observations = simulator.get_sensor_observations(agent_ids=SENSORS)

    left, right = rig.stereo_pair

    return (
        observations[rig.rgb_uuid(left)][:, :, :3],
        observations[rig.rgb_uuid(right)][:, :, :3],
    )


def camera_pose_in_world(simulator, rig):
    """ORACLE. The left camera's true pose, in OpenCV axes.

    The brain never sees this. Two world-side things need it: the simulated
    IMU, which plays the chip and so is allowed to read the truth, and
    anything scoring the tracker against it.
    """
    node = simulator.agents[SENSORS].scene_node

    rotation = np.array(node.rotation.to_matrix())
    position = np.array(node.translation)

    offset = rotation @ np.array(rig.camera(rig.stereo_pair[0]).position)

    # Habitat is +y up and -z forward; the IMU and tracker both work in
    # OpenCV axes, +y down and +z forward. Flipping y and z is that change
    # of basis.
    flip = np.diag([1.0, -1.0, -1.0])

    return rotation @ flip, position + offset


# ---------------------------------------------------------------------
# The spectator
# ---------------------------------------------------------------------


def place_spectator(simulator, drone, yaw, pitch, distance, height):
    """Put the free camera behind the drone, looking at it."""
    node = simulator.agents[SPECTATOR].scene_node

    yaw_rotation = mn.Quaternion.rotation(mn.Deg(yaw), mn.Vector3.y_axis())
    pitch_rotation = mn.Quaternion.rotation(mn.Deg(pitch), mn.Vector3.x_axis())

    node.rotation = yaw_rotation * pitch_rotation

    back = yaw_rotation.transform_vector(mn.Vector3(0.0, 0.0, 1.0))

    node.translation = (
        drone.position
        + back * distance
        + mn.Vector3(0.0, height, 0.0)
    )


def spectator_frame(simulator):
    observations = simulator.get_sensor_observations(agent_ids=SPECTATOR)

    return observations["spectator_rgb"][:, :, :3]


# ---------------------------------------------------------------------
# The world the drone lives in
# ---------------------------------------------------------------------
#
# This is `World` from the Spatial Memory Glasses project (spatial-agent-12),
# with the wearer taken out and the drone put in. The API is deliberately
# unchanged -- `heading`, `forward`, `walk`, `turn`, `stand`, `observe_rig`,
# `observe_viewer` -- because the viewer that drives it is that project's
# viewer, and a port that keeps the seam in the same place is a port that can
# be checked against the original.
#
# What differs is underneath:
#
#     glasses   an avatar that walks, with a visor on its face
#     drone     a sphere that rolls and flies, with the lenses at its centre
#               on a gimbal
#
# and one thing is new: `altitude`. The lenses rise with the drone, and every
# height the brain records is measured down from the lens, so the sim has to
# say how high it is or the whole map shifts.


class Imu:
    """The baseboard's IMU, reduced to what the brain uses: heading.

    The OAK-FFC-4P carries a BNO086, which fuses its gyro into an orientation
    on the chip. Without a magnetometer that heading has an arbitrary zero and
    drifts slowly, so this is the true turn since power-on plus a constant
    gyro bias and a little noise. Deliberately pessimistic next to the
    datasheet.

    A simulation of a real sensor, not an oracle: RTAB-Map gets no more from
    it than the hardware would give.
    """

    def __init__(self, seed=0, bias=0.1, noise=0.05):
        rng = np.random.default_rng(seed)

        self.zero = float(rng.uniform(-math.pi, math.pi))
        self.bias = math.radians(bias) * float(rng.choice([-1.0, 1.0]))
        self.noise = math.radians(noise)
        self.rng = rng
        self.start = None

    def read(self, heading, stamp):
        """Heading reported at `stamp` seconds, drone truly at `heading`."""
        if self.start is None:
            self.start = (heading, stamp)

        turned = heading - self.start[0]
        drift = self.bias * (stamp - self.start[1])

        return math.atan2(
            math.sin(self.zero + turned + drift + self.rng.normal(0.0, self.noise)),
            math.cos(self.zero + turned + drift + self.rng.normal(0.0, self.noise)),
        )


class Odometer:
    """The cage's axle encoder: how far the drone has rolled, in metres.

    The inner frame hangs in the cage on an axle. An encoder on that axle
    counts the cage's turns, and turns x circumference is distance rolled --
    a robot vacuum's wheel odometry, with the whole cage as the wheel. It is
    what the brain has to go on when the cameras see nothing: facing a blank
    wall, or pressed up against one closer than stereo can measure.

    Read as a running total, like a car's odometer, so a frame the brain
    drops loses no distance. Forward counts up, backward down.

    Errors a real one has: the rolling circumference is known to a couple of
    percent (the cage flexes; carpet lets it sink), and it slips a little on
    smooth floors. It counts only real rolling: pressed against a wall the
    cage stops, and in the air it does not roll at all.

    A simulation of a real sensor, not an oracle -- it reports distance
    rolled, never where the drone is.
    """

    def __init__(self, seed=0, scale_error=0.02, slip=0.01):
        self.rng = np.random.default_rng(seed)
        self.scale = 1.0 + scale_error
        self.slip = slip
        self.total = 0.0

    def roll(self, forward):
        """Add one step of rolling, `forward` metres along the heading."""
        if abs(forward) > 1e-4:
            self.total += forward * self.scale * (1.0 + self.slip * self.rng.normal())

        return self.total


class Rangefinder:
    """The down-facing distance sensor under the drone (config/rig.json).

    Reports how far below it the nearest surface is: the floor, or the top of
    whatever the drone is flying over. That is all it knows. It does not know
    whether it is looking at the floor or a table -- telling those apart is
    the brain's job (delta/brain/height.py).

    An ultrasonic sensor answers with the first echo to come back, so the
    reading is the NEAREST surface anywhere in its cone, not the one straight
    below: the edge of a table registers before the drone is over it. The cone
    is cast as seven rays, one straight down and six around its rim. Outside
    2 cm to 4 m there is no answer at all, and the reading is None.

    A simulation of a real sensor, not an oracle -- it reports a distance,
    never an altitude.
    """

    def __init__(self, spec, simulator, seed=0):
        self.below_lens = float(spec.get("below_lens", 0.10))
        self.min_range = float(spec.get("min_range", 0.02))
        self.max_range = float(spec.get("max_range", 4.0))
        self.noise = float(spec.get("noise", 0.005))
        self.simulator = simulator
        self.rng = np.random.default_rng(seed)

        half = math.radians(float(spec.get("cone_degrees", 15.0)) / 2.0)
        self.directions = [mn.Vector3(0.0, -1.0, 0.0)] + [
            mn.Vector3(math.sin(half) * math.cos(a), -math.cos(half), math.sin(half) * math.sin(a))
            for a in np.linspace(0.0, 2.0 * math.pi, 6, endpoint=False)
        ]

    def read(self, drone):
        """Metres from the sensor to the nearest surface below, or None."""
        origin = drone.object.translation - mn.Vector3(0.0, self.below_lens, 0.0)
        nearest = None

        for direction in self.directions:
            distance = first_hit(self.simulator, origin, direction, self.max_range, drone.object.object_id)

            if distance is not None and (nearest is None or distance < nearest):
                nearest = distance

        if nearest is None or nearest < self.min_range:
            return None

        return nearest + float(self.rng.normal(0.0, self.noise))


class ClimbRate:
    """The flight controller's vertical speed, m/s, up positive.

    A Pixhawk estimates it all the time, fusing its accelerometer with its
    barometer, and reports it to the companion computer -- it is how the
    drone holds an altitude at all. No new hardware.

    What it is for: the rangefinder alone cannot tell the drone climbing from
    the ground rising under it, and over stairs, climbing while crossing a
    step, the two cancel. Knowing how fast it is climbing separates them.

    Errors: a few cm/s of noise, and a bias that wanders slowly, as an
    accelerometer-and-barometer estimate does indoors.
    """

    def __init__(self, seed=0, noise=0.03, wander=0.01, settle=10.0):
        self.rng = np.random.default_rng(seed)
        self.noise = noise
        self.wander = wander
        self.settle = settle
        self.bias = 0.0
        self._last = None

    def read(self, height, stamp):
        """Vertical speed reported at `stamp`, the drone's centre at `height`."""
        if self._last is None or stamp <= self._last[1]:
            self._last = (height, stamp)
            return 0.0

        dt = stamp - self._last[1]
        true_rate = (height - self._last[0]) / dt
        self._last = (height, stamp)

        self.bias += -self.bias * dt / self.settle + self.wander * math.sqrt(2.0 * dt / self.settle) * self.rng.normal()

        return true_rate + self.bias + self.noise * self.rng.normal()


def first_hit(simulator, origin, direction, max_distance, ignore_object=None):
    """Distance along a ray to the first thing it hits, or None.

    Hits on `ignore_object` are skipped -- the drone's own sphere, which a
    ray starting inside it would otherwise find first.
    """
    result = simulator.cast_ray(habitat_sim.geo.Ray(origin, direction), max_distance=max_distance)

    for hit in result.hits if result.has_hits() else []:
        if hit.object_id != ignore_object:
            return float(hit.ray_distance) * direction.length()

    return None


class World:
    """Habitat with the drone in it, and the stereo rig on the drone.

    `heading` is the drone's yaw in Habitat's sense: radians about +Y, 0
    facing north (-Z), positive turning LEFT. Moving goes through `walk` and
    `turn`, so the navmesh -- or, in the air, Bullet -- decides what is solid,
    never this class.
    """

    def __init__(self, stage="process", true_depth=False,
                 viewer_size=(800, 540), viewer_hfov=80.0):
        from delta.drone import Drone, find_spot_near

        self.config = load_project_config()
        self.scene_id = self.config["dataset"]["scene_id"]

        root, dataset_config, scene_file, navmesh_file = resolve_dataset_paths(self.config)
        verify_files(root, dataset_config, scene_file)
        verify_physics()

        self.rig = rig_module.load()
        self.stage = stage
        self.true_depth = true_depth

        self.simulator = create_simulator(
            dataset_config,
            scene_file,
            self.rig,
            spectator_size=viewer_size,
            stage=stage,
            viewer_hfov=viewer_hfov,
            true_depth=true_depth,
        )

        load_or_generate_navmesh(self.simulator, navmesh_file, self.config["drone"])

        self.pathfinder = self.simulator.pathfinder

        self.drone = Drone(
            self.simulator, diameter=float(self.config["drone"]["diameter"])
        )
        self.drone.place_at(
            find_spot_near(self.pathfinder, self.start_position())
        )

        self.viewer_node = self.simulator.agents[SPECTATOR].scene_node
        self.blocked = False

        self.heading = 0.0
        self.face(0.0)

    # -----------------------------------------------------------------
    # Where it is
    # -----------------------------------------------------------------

    @property
    def floor_island(self):
        """The navmesh island that is the building's floor. INDOORS.

        The glasses took the biggest island and that was the house, because a
        1.70 m person with a 0.20 m radius cannot stand on most of a garden.
        A 30 cm ball can roll almost anywhere, so for this drone the biggest
        island is the LAWN -- and the drone started outside, nose against a
        fence, mapping grass. There is no door to open: it was never in the
        building.

        So "indoors" is tested rather than assumed, and the test is physical:
        indoors has a ceiling. Sample each sizeable island, cast a ray
        straight up from each sample, and keep the islands where most samples
        hit something within `INDOOR_CEILING` metres. The biggest of those is
        the house.

        This is also why it is worth doing properly rather than hard-coding a
        start point: the next scene will have a different layout, and a
        hard-coded point would silently be outside again.
        """
        if hasattr(self, "_floor_island"):
            return self._floor_island

        islands = sorted(
            range(self.pathfinder.num_islands),
            key=self.pathfinder.island_area,
            reverse=True,
        )

        best = islands[0] if islands else -1

        for island in islands[:8]:
            if self.pathfinder.island_area(island) < 4.0:
                break

            if self._roofed_fraction(island) >= 0.6:
                best = island
                break

        self._floor_island = best

        return self._floor_island

    def _roofed_fraction(self, island, samples=40):
        """How much of this island has something overhead: 0 outside, ~1 in."""
        import habitat_sim

        self.pathfinder.seed(11)

        roofed = 0
        taken = 0

        for _ in range(samples):
            point = self.pathfinder.get_random_navigable_point(island_index=island)

            if np.isnan(np.array(point)).any():
                continue

            taken += 1

            ray = habitat_sim.geo.Ray(
                mn.Vector3(point) + mn.Vector3(0.0, 0.2, 0.0),
                mn.Vector3(0.0, 1.0, 0.0),
            )

            if self.simulator.cast_ray(ray, max_distance=INDOOR_CEILING).has_hits():
                roofed += 1

        return roofed / taken if taken else 0.0

    def random_floor_point(self):
        return mn.Vector3(
            self.pathfinder.get_random_navigable_point(island_index=self.floor_island)
        )

    def start_position(self, seed=7):
        """The same starting spot every run, so a saved map lines up."""
        self.pathfinder.seed(seed)
        point = self.random_floor_point()

        if np.isnan(np.array(point)).any():
            raise RuntimeError("Habitat could not find a navigable starting point")

        return point

    @property
    def altitude(self):
        return self.drone.altitude

    @property
    def flying(self):
        return self.drone.flying

    def floor_position(self):
        return self.drone.floor_position

    # -----------------------------------------------------------------
    # Moving it
    # -----------------------------------------------------------------

    def stand(self, floor_point, heading):
        self.drone.place_at(mn.Vector3(floor_point))
        self.face(heading)

    def face(self, heading):
        """Point the gimballed sensor head. The shell's own tumble is not it."""
        self.heading = math.atan2(math.sin(heading), math.cos(heading))
        self.drone.heading = math.degrees(self.heading)

    def turn(self, radians):
        self.face(self.heading + radians)

    @property
    def forward(self):
        """Unit vector the sensor head faces, in Habitat coordinates."""
        return mn.Vector3(-math.sin(self.heading), 0.0, -math.cos(self.heading))

    def walk(self, distance):
        """Move forward (or back). Walls stop it: the navmesh on the ground,
        Bullet in the air. Returns how far it actually got."""
        before = self.drone.position

        # drive() would turn the head to face the direction of travel; here
        # the head is steered separately, so the heading is held.
        held = self.drone.heading
        self.drone.drive(self.forward * distance, self.pathfinder)
        self.drone.heading = held

        moved = (self.drone.position - before).length()
        self.blocked = moved < abs(distance) * 0.4

        return moved

    def climb(self, seconds):
        if not self.drone.flying:
            return self.drone.take_off()

        self.drone.climb(seconds)
        return True

    def descend(self, seconds):
        self.drone.descend(seconds)

    def settle(self, seconds):
        """One step of whatever the drone is doing vertically."""
        self.drone.update(seconds)

    def heading_towards(self, point):
        delta = mn.Vector3(point) - self.drone.floor_position

        return math.atan2(-delta.x, -delta.z)

    # -----------------------------------------------------------------
    # What the cameras see
    # -----------------------------------------------------------------

    def observe_rig(self):
        """Render the rig. Returns the stereo pair, and the oracle fields."""
        mount_sensors(self.simulator, self.drone)

        observations = self.simulator.get_sensor_observations(agent_ids=[SENSORS])
        observations = observations[SENSORS] if SENSORS in observations else observations

        left_name, right_name = self.rig.stereo_pair

        state = self.simulator.agents[SENSORS].get_state()
        lens = state.sensor_states[self.rig.rgb_uuid(left_name)]
        rotation = lens.rotation

        result = {
            "left": np.ascontiguousarray(
                observations[self.rig.rgb_uuid(left_name)][..., :3]
            ),
            "right": np.ascontiguousarray(
                observations[self.rig.rgb_uuid(right_name)][..., :3]
            ),
            "true_position": [float(value) for value in lens.position],
            "true_rotation": [
                float(rotation.w), float(rotation.x),
                float(rotation.y), float(rotation.z),
            ],
        }

        if self.true_depth:
            result["true_depth"] = np.squeeze(
                observations[self.rig.depth_uuid(left_name)]
            ).astype(np.float32)

        return result

    # -----------------------------------------------------------------
    # The rig, one picture at a time
    # -----------------------------------------------------------------

    def start_rig(self, draw=True):
        """Put the rig where the drone is now, for a frame drawn one picture
        at a time (render_rig_part) -- or, with `draw` False, drawn by the
        camera process (rig_process.py) from rig_pose(). Returns where the
        left lens is -- the frame's true pose -- before anything is drawn.

        observe_rig draws the three pictures in one go, ~40 ms under WSL:
        one loop in three took that much longer, and the view behind the
        drone stuttered. Drawn a picture a loop instead, the cost is even.
        The rig stays where this put it until the last picture is drawn, so
        all three are from the same instant, as a real stereo pair is.
        """
        mount_sensors(self.simulator, self.drone)

        left_name, right_name = self.rig.stereo_pair
        state = self.simulator.agents[SENSORS].get_state()
        lens = state.sensor_states[self.rig.rgb_uuid(left_name)]
        rotation = lens.rotation

        self._rig_parts = [(self.rig.rgb_uuid(left_name), "left"), (self.rig.rgb_uuid(right_name), "right")]

        if self.true_depth:
            self._rig_parts.append((self.rig.depth_uuid(left_name), "true_depth"))

        if not draw:
            self._rig_parts = []

        self._rig_seen = {
            "true_position": [float(value) for value in lens.position],
            "true_rotation": [float(rotation.w), float(rotation.x), float(rotation.y), float(rotation.z)],
        }

        return dict(self._rig_seen)

    def rig_pose(self):
        """Where start_rig put the rig: (translation, rotation wxyz)."""
        node = self.simulator.agents[SENSORS].scene_node
        rotation = node.rotation

        return ([float(v) for v in node.translation],
                [float(rotation.scalar)] + [float(v) for v in rotation.vector])

    @property
    def rig_busy(self):
        """A frame started and not all of its pictures drawn yet."""
        return bool(getattr(self, "_rig_parts", None))

    def render_rig_part(self):
        """Draw the next picture of the frame start_rig began. Returns the
        whole frame -- as observe_rig does -- after the last one, else None.

        Habitat's own get_sensor_observations draws every sensor of an agent
        and then reads each back; this does the same for one sensor. Its
        sensors are private to the Simulator (habitat-sim 0.3.3); without
        them, the whole frame is drawn at once, as before.
        """
        sensors = getattr(self.simulator, "_Simulator__sensors", None)

        if sensors is None:
            seen = self.observe_rig()
            self._rig_parts = []
            return seen

        uuid, key = self._rig_parts.pop(0)
        sensor = sensors[SENSORS][uuid]
        sensor.draw_observation()
        picture = sensor.get_observation()

        if key == "true_depth":
            self._rig_seen[key] = np.squeeze(picture).astype(np.float32)
        else:
            self._rig_seen[key] = np.ascontiguousarray(picture[..., :3])

        return None if self._rig_parts else self._rig_seen

    def true_depth_at(self, position, rotation_wxyz):
        """True planar depth from a camera at `position` facing `rotation`
        (as observe_rig reports the left lens), and its intrinsics K. For
        scoring only: this is the answer key, never the brain's input."""
        node = self.simulator.agents[SCORER].scene_node
        node.translation = mn.Vector3(*position)
        w, x, y, z = rotation_wxyz
        node.rotation = mn.Quaternion(mn.Vector3(x, y, z), w)

        observations = self.simulator.get_sensor_observations(agent_ids=[SCORER])
        observations = observations[SCORER] if SCORER in observations else observations
        depth = np.squeeze(observations["scorer_depth"]).astype(np.float32)

        width, height = SCORER_SIZE
        focal = (width / 2.0) / math.tan(math.radians(self.rig.camera(self.rig.stereo_pair[0]).hfov) / 2.0)
        K = np.array([[focal, 0.0, width / 2.0], [0.0, focal, height / 2.0], [0.0, 0.0, 1.0]])

        return depth, K

    def observe_viewer(self):
        observations = self.simulator.get_sensor_observations(agent_ids=[SPECTATOR])
        observations = observations[SPECTATOR] if SPECTATOR in observations else observations

        return observations["spectator_rgb"][..., :3]

    def close(self):
        self.simulator.close()
