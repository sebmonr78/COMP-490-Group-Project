"""Open the shared HSSD environment and drive the DELTA drone around it.

The window contains:

Left:
    Controllable RGB camera.

Right:
    Ground-truth 2D layout generated from Habitat's navmesh -- the ground
    the drone can actually roll on.

Controls:

    The camera -- flies anywhere, looks anywhere:
        W/S       Move forward/backward
        A/D       Move left/right
        I/K       Look up/down
        J/L       Look left/right
        Z/X       Move vertically
        Shift     Move faster

    The drone -- rolls the floor, by compass:
        Up        North (up on the layout map)
        Down      South
        Left      West
        Right     East

    The drone -- flying:
        1         Take off, and hold to climb
        0         Descend, and hold to land

    Planning a route:
        Click     Plan a rolling route to that spot on the layout
        Enter     Follow it
        C         Clear it

    P             Print camera and drone positions
    Escape        Exit

The camera and the drone move independently, at the same time.

Two ways of moving, and the difference is the point of the whole design:

    rolling   the navmesh decides what is legal, and it is cheap
    flying    the navmesh means nothing, so real collision decides

Run:
    conda activate habitat
    cd ~/COMP-490-Group-Project/simulation
    python demo/open_environment.py
"""

import json
import math
import os
import sys
from pathlib import Path

import habitat_sim
import magnum as mn
import numpy as np
import pygame
from habitat.utils.visualizations import maps


# ---------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "config" / "project.json"

# Let this script import the project's own package from one folder up.
sys.path.insert(0, str(PROJECT_ROOT))

from delta.drone import Drone, find_spot_near
from delta.navigation import Route


# ---------------------------------------------------------------------
# Window configuration
# ---------------------------------------------------------------------

WINDOW_WIDTH = 1600
WINDOW_HEIGHT = 800

VIEW_WIDTH = WINDOW_WIDTH // 2
VIEW_HEIGHT = WINDOW_HEIGHT

MAP_RESOLUTION = 1024

# Who is who on the layout map.
CAMERA_COLOR = (255, 60, 60)     # red    -- where you are looking from
ROLLING_COLOR = (60, 220, 90)    # green  -- the drone, on the ground
FLYING_COLOR = (80, 200, 255)    # blue   -- the drone, airborne
ROUTE_COLOR = (70, 150, 255)     # blue   -- the planned route
GOAL_COLOR = (255, 230, 80)      # amber  -- the destination
REJECT_COLOR = (255, 70, 70)     # red    -- you clicked somewhere unreachable

# The drone's outline inside the house, drawn with Habitat's debug lines.
# Debug lines are the one place Habitat gives us colour without a model file.
ROLLING_OUTLINE = mn.Color4(0.24, 0.86, 0.35, 1.0)
FLYING_OUTLINE = mn.Color4(0.31, 0.78, 1.0, 1.0)
SHADOW_OUTLINE = mn.Color4(0.31, 0.78, 1.0, 0.45)


# ---------------------------------------------------------------------
# Camera configuration
# ---------------------------------------------------------------------

CAMERA_HEIGHT = 1.6

NORMAL_MOVE_SPEED = 2.0
FAST_MOVE_SPEED = 7.0
LOOK_SPEED = 90.0

# The drone is a 30 cm ball, not a person. It should read as smaller and
# more nimble than the free camera, so it gets its own speeds.
DRONE_ROLL_SPEED = 1.4
DRONE_FLY_SPEED = 1.8

# Climb and descent speed is the drone's own business -- see CLIMB_SPEED in
# delta/drone.py. It is not set here, because the keys no longer move the
# drone themselves; they only say which way it should be heading.


def fail(message):
    """Print an error and stop the program."""

    print(f"[FAIL] {message}")
    raise SystemExit(1)


def load_project_config():
    """Load config/project.json."""

    if not CONFIG_PATH.is_file():
        fail(f"Project configuration not found:\n{CONFIG_PATH}")

    try:
        with CONFIG_PATH.open("r", encoding="utf-8") as config_file:
            return json.load(config_file)

    except json.JSONDecodeError as error:
        fail(f"Invalid JSON in {CONFIG_PATH}:\n{error}")


def resolve_dataset_paths(config):
    """Create absolute paths from project.json."""

    dataset = config["dataset"]

    dataset_root_value = os.environ.get(
        dataset["environment_variable"],
        dataset["default_root"],
    )

    dataset_root = Path(dataset_root_value).expanduser().resolve()

    dataset_config = (
        dataset_root / dataset["configuration_file"]
    ).resolve()

    scene_file = (
        dataset_root / dataset["scene_file"]
    ).resolve()

    # The drone's own navmesh, not the one HSSD ships. See
    # load_or_generate_navmesh for why they are different maps.
    navmesh_file = (
        dataset_root / dataset["drone_navmesh_file"]
    ).resolve()

    return (
        dataset_root,
        dataset_config,
        scene_file,
        navmesh_file,
    )


def verify_files(
    dataset_root,
    dataset_config,
    scene_file,
):
    """Check that the required HSSD files exist."""

    if not dataset_root.is_dir():
        fail(f"HSSD dataset directory not found:\n{dataset_root}")

    if not dataset_config.is_file():
        fail(
            "HSSD dataset configuration not found:\n"
            f"{dataset_config}"
        )

    if not scene_file.is_file():
        fail(f"HSSD scene file not found:\n{scene_file}")

    print("[PASS] HSSD dataset directory found")
    print("[PASS] HSSD dataset configuration found")
    print("[PASS] Selected HSSD scene found")


def create_simulator(dataset_config, scene_file):
    """Create Habitat-Sim with one invisible RGB camera."""

    backend = habitat_sim.SimulatorConfiguration()

    backend.scene_id = str(scene_file)
    backend.scene_dataset_config_file = str(dataset_config)

    # Physics is ON, which it was not for the glasses project, and the drone
    # does not work without it.
    #
    # The navmesh is a map of the FLOOR. It is what stops a ground robot
    # walking through a wall, and it has no opinion whatsoever about the air
    # above it. The moment the drone takes off, the navmesh stops being able
    # to answer "is this legal?" -- so the question has to go to the real
    # collision geometry instead, through simulator.contact_test. That call
    # is Bullet, and Bullet only exists here if physics is enabled.
    #
    # We still move the drone ourselves, every frame. Enabling physics buys
    # us the collision TEST, not a simulation that flies the drone for us.
    backend.enable_physics = True

    # This is required by the working WSL setup.
    backend.gpu_device_id = -1

    camera = habitat_sim.CameraSensorSpec()

    camera.uuid = "viewer_rgb"
    camera.sensor_type = habitat_sim.SensorType.COLOR
    camera.sensor_subtype = habitat_sim.SensorSubType.PINHOLE
    camera.resolution = [VIEW_HEIGHT, VIEW_WIDTH]
    camera.position = [0.0, 0.0, 0.0]
    camera.hfov = 90.0

    camera_holder = habitat_sim.agent.AgentConfiguration()

    camera_holder.sensor_specifications = [camera]

    simulator_configuration = habitat_sim.Configuration(
        backend,
        [camera_holder],
    )

    return habitat_sim.Simulator(simulator_configuration)


def verify_physics():
    """Refuse to run without Bullet, instead of flying through walls.

    Without Bullet, contact_test has nothing to test against and quietly
    answers "nothing is touching" for every position in the building. The
    drone would then fly through walls, and nothing on screen would say why.
    A silent wrong answer is worse than a stop, so this is a stop.
    """

    if not getattr(habitat_sim, "built_with_bullet", False):
        fail(
            "This habitat-sim was built without Bullet, so the drone has no\n"
            "way to detect collisions while flying.\n"
            "Reinstall with:\n"
            "    conda install habitat-sim withbullet -c conda-forge -c aihabitat"
        )

    print("[PASS] Bullet physics available")


def navmesh_settings_for(drone_config):
    """The navmesh settings that describe this drone, not a person.

    HSSD ships a navmesh built for a walking human: about 1.6 m tall and
    0.20 m wide, able to step up 0.20 m. Our drone is a 0.30 m ball. Those
    are different maps of the same building, and the difference is not
    cosmetic:

        it fits under tables, shelves and chairs that a person does not
        it fits through gaps a person's shoulders do not
        it cannot climb a 20 cm step -- it is 30 cm across with no legs

    So we build our own and save it beside theirs. Getting this wrong is
    subtle and nasty: with a human navmesh the drone would refuse to roll
    under its own inspection targets, which is most of what it exists to do.
    """

    settings = habitat_sim.NavMeshSettings()
    settings.set_defaults()

    diameter = float(drone_config["diameter"])

    settings.agent_height = diameter
    settings.agent_radius = diameter / 2.0
    settings.agent_max_climb = float(drone_config["max_climb"])
    settings.agent_max_slope = float(drone_config["max_slope"])

    # Furniture is part of the environment, not scenery to pass through.
    settings.include_static_objects = True

    return settings


def load_or_generate_navmesh(simulator, navmesh_file, drone_config):
    """Load the drone's saved navmesh, or build one for this building."""

    if navmesh_file.is_file():
        print(f"Loading drone navmesh:\n{navmesh_file}")

        simulator.pathfinder.load_nav_mesh(str(navmesh_file))

        if simulator.pathfinder.is_loaded:
            print("[PASS] Existing drone navigation mesh loaded")
            return

        print("[WARNING] Existing navmesh could not be loaded")
        print("Generating a replacement navmesh...")

    else:
        print("No saved drone navmesh found")
        print("Generating a navigation mesh for the drone...")

    success = simulator.recompute_navmesh(
        simulator.pathfinder,
        navmesh_settings_for(drone_config),
    )

    if not success:
        fail("Habitat failed to generate a navigation mesh")

    if not simulator.pathfinder.is_loaded:
        fail("The generated navigation mesh was not loaded")

    print("[PASS] Drone navigation mesh generated")

    navmesh_file.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    simulator.pathfinder.save_nav_mesh(
        str(navmesh_file)
    )

    if navmesh_file.is_file():
        print(f"[PASS] Drone navigation mesh saved:\n{navmesh_file}")

    else:
        print(
            "[WARNING] Navmesh works for this session "
            "but could not be saved"
        )


def choose_start_position(simulator):
    """Choose the same deterministic starting point each time."""

    simulator.pathfinder.seed(7)

    start_position = (
        simulator.pathfinder.get_random_navigable_point()
    )

    if np.isnan(np.array(start_position)).any():
        fail("Habitat could not find a navigable starting point")

    return start_position


def create_layout_map(simulator, floor_height):
    """Create the ground-truth 2D navigability map."""

    topdown_map = maps.get_topdown_map(
        simulator.pathfinder,
        floor_height,
        map_resolution=MAP_RESOLUTION,
        draw_border=True,
    )

    if topdown_map is None or topdown_map.size == 0:
        fail("Habitat could not create the top-down layout")

    colored_map = maps.colorize_topdown_map(
        topdown_map
    )

    return colored_map


def world_position_to_map(
    position,
    map_image,
    pathfinder,
):
    """Convert a 3D position into map pixels."""

    lower_bound, upper_bound = pathfinder.get_bounds()

    map_height, map_width = map_image.shape[:2]

    world_width = upper_bound.x - lower_bound.x
    world_depth = upper_bound.z - lower_bound.z

    if world_width <= 0 or world_depth <= 0:
        return None

    column = int(
        ((position.x - lower_bound.x) / world_width)
        * (map_width - 1)
    )

    row = int(
        ((position.z - lower_bound.z) / world_depth)
        * (map_height - 1)
    )

    column = max(
        0,
        min(map_width - 1, column),
    )

    row = max(
        0,
        min(map_height - 1, row),
    )

    return column, row


def map_click_to_world(
    mouse_x,
    mouse_y,
    layout_map,
    pathfinder,
    half_width,
    current_height,
):
    """Turn a click on the layout panel into a point in the building.

    This undoes, in order, the two steps that put the map on screen:
    stretching the map image to fill the right-hand panel, and turning world
    metres into map pixels. Returns None if the click was not on the panel.
    """

    if mouse_x < half_width or half_width <= 0 or current_height <= 0:
        return None

    map_height, map_width = layout_map.shape[:2]

    # Undo the stretch onto the panel.
    map_column = (mouse_x - half_width) / half_width * map_width
    map_row = mouse_y / current_height * map_height

    lower_bound, upper_bound = pathfinder.get_bounds()

    world_width = upper_bound.x - lower_bound.x
    world_depth = upper_bound.z - lower_bound.z

    if world_width <= 0 or world_depth <= 0:
        return None

    # Undo the metres-to-pixels conversion.
    world_x = lower_bound.x + (map_column / (map_width - 1)) * world_width
    world_z = lower_bound.z + (map_row / (map_height - 1)) * world_depth

    # The map is flat, so the click carries no height. Ask the navmesh which
    # floor is under that spot.
    snapped = pathfinder.snap_point(
        mn.Vector3(world_x, lower_bound.y, world_z)
    )

    if not all(value == value for value in snapped):
        return None

    return mn.Vector3(snapped)


def draw_route_on_layout(
    screen,
    route,
    layout_map,
    simulator,
    half_width,
    current_height,
):
    """Draw the planned route as a blue line across the layout panel."""

    if not route.is_planned:
        return

    pixels = []

    for point in route.points:
        map_position = world_position_to_map(
            point,
            layout_map,
            simulator.pathfinder,
        )

        if map_position is None:
            continue

        map_column, map_row = map_position

        pixels.append(
            (
                half_width
                + int(map_column / layout_map.shape[1] * half_width),
                int(map_row / layout_map.shape[0] * current_height),
            )
        )

    if len(pixels) >= 2:
        pygame.draw.lines(
            screen,
            ROUTE_COLOR,
            False,
            pixels,
            4,
        )

    if pixels:
        pygame.draw.circle(
            screen,
            GOAL_COLOR,
            pixels[-1],
            8,
        )

        pygame.draw.circle(
            screen,
            (255, 255, 255),
            pixels[-1],
            11,
            2,
        )


def numpy_to_surface(image):
    """Convert a Habitat or NumPy image to Pygame."""

    return pygame.surfarray.make_surface(
        np.transpose(image, (1, 0, 2))
    )


def draw_layout_marker(
    screen,
    position,
    layout_map,
    simulator,
    half_width,
    current_height,
    color=(255, 60, 60),
    hollow=False,
):
    """Draw one dot on the layout.

    Takes a plain 3D position rather than the drone, so the same function can
    mark the camera, the drone, or anything else we add later. The colour says
    which is which, and `hollow` says the drone is airborne -- an open ring
    reads as "above this spot" rather than "on it", which matters once the
    drone can be over a place it is not standing on.
    """

    map_position = world_position_to_map(
        position,
        layout_map,
        simulator.pathfinder,
    )

    if map_position is None:
        return

    map_column, map_row = map_position

    marker_x = half_width + int(
        map_column
        / layout_map.shape[1]
        * half_width
    )

    marker_y = int(
        map_row
        / layout_map.shape[0]
        * current_height
    )

    # Coloured centre, or an open ring when flying.
    pygame.draw.circle(
        screen,
        color,
        (marker_x, marker_y),
        9,
        3 if hollow else 0,
    )

    # White border.
    pygame.draw.circle(
        screen,
        (255, 255, 255),
        (marker_x, marker_y),
        12,
        2,
    )


def draw_drone_outline(debug_lines, drone):
    """Trace the drone inside the house, so you can find it against grey walls.

    Three parts, and the second two only when it is airborne:

        a ring around the ball itself, in green on the ground and blue in
        the air, so the mode is readable from the 3D view alone

        a ring on the floor underneath it -- its shadow

        a line joining the two, which is the only thing on screen that shows
        how high it actually is
    """

    centre = drone.position

    color = FLYING_OUTLINE if drone.flying else ROLLING_OUTLINE

    debug_lines.draw_circle(centre, drone.radius, color)

    if not drone.flying:
        return

    ground = drone.floor_position

    # Lift the shadow a hair off the floor, or it fights with the floor
    # surface for the same pixels and flickers.
    shadow = mn.Vector3(ground.x, ground.y + 0.01, ground.z)

    debug_lines.draw_circle(shadow, drone.radius, SHADOW_OUTLINE)

    debug_lines.draw_transformed_line(shadow, centre, SHADOW_OUTLINE)


def main():
    config = load_project_config()
    dataset = config["dataset"]
    drone_config = config["drone"]

    (
        dataset_root,
        dataset_config,
        scene_file,
        navmesh_file,
    ) = resolve_dataset_paths(config)

    print()
    print(config["project_name"])
    print(config["project_subtitle"])
    print("Shared HSSD Development Environment")
    print("-----------------------------------")
    print(f"Scene ID:       {dataset['scene_id']}")
    print(f"Dataset root:   {dataset_root}")
    print(f"Dataset config: {dataset_config}")
    print(f"Scene file:     {scene_file}")
    print(f"Navmesh file:   {navmesh_file}")
    print(f"Drone diameter: {drone_config['diameter']} m")
    print()

    verify_files(
        dataset_root,
        dataset_config,
        scene_file,
    )

    verify_physics()

    print()
    print("Loading the HSSD environment...")

    try:
        simulator = create_simulator(
            dataset_config,
            scene_file,
        )

    except Exception as error:
        fail(f"Could not load the HSSD environment:\n{error}")

    print("[PASS] HSSD environment loaded")
    print("[PASS] RGB camera created")

    load_or_generate_navmesh(
        simulator,
        navmesh_file,
        drone_config,
    )

    floor_position = choose_start_position(
        simulator
    )

    floor_height = float(floor_position.y)

    camera_node = simulator.agents[0].scene_node

    camera_node.translation = mn.Vector3(
        float(floor_position.x),
        floor_height + CAMERA_HEIGHT,
        float(floor_position.z),
    )

    print(
        "[PASS] Camera placed at: "
        f"{camera_node.translation}"
    )

    # Put the drone a short roll away, so the camera can see it rather than
    # start inside it.
    drone = Drone(
        simulator,
        diameter=float(drone_config["diameter"]),
    )

    drone.place_at(
        find_spot_near(
            simulator.pathfinder,
            floor_position,
        )
    )

    print(
        "[PASS] Drone resting at: "
        f"{drone.floor_position}"
    )

    print("Creating the ground-truth layout...")

    layout_map = create_layout_map(
        simulator,
        floor_height,
    )

    print("[PASS] Layout created")

    pygame.init()

    screen = pygame.display.set_mode(
        (WINDOW_WIDTH, WINDOW_HEIGHT),
        pygame.RESIZABLE,
    )

    pygame.display.set_caption(
        f"{config['project_name']} | "
        f"HSSD {dataset['scene_id']}"
    )

    title_font = pygame.font.SysFont(
        "Arial",
        22,
        bold=True,
    )

    controls_font = pygame.font.SysFont(
        "Arial",
        17,
    )

    clock = pygame.time.Clock()

    yaw = 0.0
    pitch = 0.0

    # The planned route. Click the layout to plan one, Enter to follow it.
    route = Route(simulator.pathfinder)

    # Habitat's 3D line drawer, for showing the drone and the route inside
    # the house.
    debug_lines = simulator.get_debug_line_render()

    # Counts down after a click on somewhere unreachable, or a refused
    # take-off, so the refusal is visible for a moment instead of flashing
    # past in one frame.
    reject_timer = 0.0
    reject_message = ""

    def refuse(message):
        """Say no, on screen and in the terminal."""
        nonlocal reject_timer, reject_message

        reject_timer = 1.4
        reject_message = message

        print(message)

    running = True

    print()
    print("Controls")
    print("--------")
    print("W/S:       Move the camera forward/backward")
    print("A/D:       Move the camera left/right")
    print("I/K:       Look up/down (camera)")
    print("J/L:       Look left/right (camera)")
    print("Z/X:       Move the camera vertically")
    print("Shift:     Move faster")
    print("Arrows:    Drive the drone by compass -- up is north on the map")
    print("1:         Take off, hold to climb")
    print("0:         Descend, hold to land")
    print("Click map: Plan a rolling route to that spot")
    print("Enter:     Follow the planned route")
    print("C:         Clear the planned route")
    print("P:         Print camera and drone positions")
    print("Escape:    Exit")
    print()

    while running:
        delta_time = clock.tick(60) / 1000.0

        # -------------------------------------------------------------
        # Events
        # -------------------------------------------------------------

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False

                elif event.key == pygame.K_RETURN:
                    if not route.is_planned:
                        pass

                    elif drone.flying:
                        # A rolling route is a route across the floor. Driving
                        # it while airborne would mean tracing a floor path
                        # through the air, which is neither what was planned
                        # nor what the navmesh promised was clear.
                        refuse("Land before following a rolling route.")

                    else:
                        route.start_driving()
                        print("Following the planned route.")

                elif event.key == pygame.K_c:
                    route.clear()
                    print("Route cleared.")

                elif event.key in (pygame.K_1, pygame.K_KP1):
                    if drone.flying:
                        pass

                    elif drone.take_off():
                        print(
                            "Taking off, climbing to "
                            f"{drone.target_altitude:.2f} m."
                        )

                    else:
                        refuse("No room overhead to take off.")

                elif event.key == pygame.K_p:
                    position = camera_node.translation

                    print(
                        "Camera: "
                        f"x={position.x:.3f}, "
                        f"y={position.y:.3f}, "
                        f"z={position.z:.3f}"
                    )

                    ground = drone.floor_position

                    print(
                        "Drone:  "
                        f"x={ground.x:.3f}, "
                        f"y={ground.y:.3f}, "
                        f"z={ground.z:.3f}, "
                        f"altitude={drone.altitude:.3f} m, "
                        f"{'flying' if drone.flying else 'rolling'}"
                    )

            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                current_width, current_height = screen.get_size()

                goal = map_click_to_world(
                    event.pos[0],
                    event.pos[1],
                    layout_map,
                    simulator.pathfinder,
                    current_width // 2,
                    current_height,
                )

                if goal is None:
                    # The click was on the camera panel, not the map.
                    pass

                elif route.plan(drone.floor_position, goal):
                    reject_timer = 0.0

                    print(
                        "Route planned: "
                        f"{route.length:.2f} m, "
                        f"{len(route.points)} corners. "
                        "Press Enter to follow it."
                    )

                else:
                    # Worth saying out loud rather than just refusing: a goal
                    # with no rolling route is exactly the case the drone can
                    # fly to, which is the decision feature 4 automates later.
                    refuse("No rolling route to that spot -- try flying there.")

        keys = pygame.key.get_pressed()

        # -------------------------------------------------------------
        # IJKL look controls
        # -------------------------------------------------------------

        # IJKL always aims the camera, so you can watch the drone from
        # wherever you like while you drive it.
        if keys[pygame.K_j]:
            yaw += LOOK_SPEED * delta_time

        if keys[pygame.K_l]:
            yaw -= LOOK_SPEED * delta_time

        if keys[pygame.K_i]:
            pitch += LOOK_SPEED * delta_time

        if keys[pygame.K_k]:
            pitch -= LOOK_SPEED * delta_time

        pitch = max(
            -89.0,
            min(89.0, pitch),
        )

        yaw_rotation = mn.Quaternion.rotation(
            mn.Deg(yaw),
            mn.Vector3.y_axis(),
        )

        pitch_rotation = mn.Quaternion.rotation(
            mn.Deg(pitch),
            mn.Vector3.x_axis(),
        )

        camera_node.rotation = (
            yaw_rotation * pitch_rotation
        )

        # -------------------------------------------------------------
        # WASD movement controls
        # -------------------------------------------------------------

        fast = keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]

        move_speed = FAST_MOVE_SPEED if fast else NORMAL_MOVE_SPEED

        # -------------------------------------------------------------
        # The camera: WASD to fly, Z/X for height
        # -------------------------------------------------------------

        camera_movement = mn.Vector3(0.0, 0.0, 0.0)

        # The camera flies the way it is looking.
        camera_forward = yaw_rotation.transform_vector(
            mn.Vector3(0.0, 0.0, -1.0)
        )

        camera_right = yaw_rotation.transform_vector(
            mn.Vector3(1.0, 0.0, 0.0)
        )

        if keys[pygame.K_w]:
            camera_movement += camera_forward

        if keys[pygame.K_s]:
            camera_movement -= camera_forward

        if keys[pygame.K_a]:
            camera_movement -= camera_right

        if keys[pygame.K_d]:
            camera_movement += camera_right

        if keys[pygame.K_z]:
            camera_movement += mn.Vector3.y_axis()

        if keys[pygame.K_x]:
            camera_movement -= mn.Vector3.y_axis()

        if camera_movement.length() > 0:
            camera_node.translation += (
                camera_movement.normalized()
                * move_speed
                * delta_time
            )

        # -------------------------------------------------------------
        # The drone: altitude, on 1 and 0
        # -------------------------------------------------------------

        # 1 lifts, 0 lowers. Neither key moves the drone directly -- they
        # only change the altitude it is aiming for, and drone.update() below
        # flies it there smoothly. That is why a tap of 1 rises rather than
        # teleports, and why a landing settles instead of snapping down.
        was_flying = drone.flying

        if keys[pygame.K_1] or keys[pygame.K_KP1]:
            drone.climb(delta_time)

        if keys[pygame.K_0] or keys[pygame.K_KP0]:
            drone.descend(delta_time)

        # The one call that actually changes the drone's height, every frame,
        # whether it is taking off, climbing, descending or landing.
        drone.update(delta_time)

        if was_flying and not drone.flying:
            print("Landed.")

        # -------------------------------------------------------------
        # The drone: arrow keys, by compass
        # -------------------------------------------------------------

        # The drone drives by compass, never by facing. Up is always north on
        # the layout map no matter where the camera points, so what you read
        # off the map is what the keys do. A sphere has no front, which makes
        # compass control the only honest scheme for it -- there is no "its
        # left" to mean anything.
        #
        # The map is drawn straight from world coordinates: its top edge is
        # the smallest z and its right edge the largest x. So north is -Z and
        # east is +X.
        heading = mn.Vector3(0.0, 0.0, 0.0)

        if keys[pygame.K_UP]:
            heading += mn.Vector3(0.0, 0.0, -1.0)     # north, map up

        if keys[pygame.K_DOWN]:
            heading += mn.Vector3(0.0, 0.0, 1.0)      # south, map down

        if keys[pygame.K_LEFT]:
            heading += mn.Vector3(-1.0, 0.0, 0.0)     # west, map left

        if keys[pygame.K_RIGHT]:
            heading += mn.Vector3(1.0, 0.0, 0.0)      # east, map right

        drone_speed = DRONE_FLY_SPEED if drone.flying else DRONE_ROLL_SPEED

        if fast:
            drone_speed *= 2.0

        # Following a route. An arrow key takes control back, because an
        # operator who starts driving has changed their mind. Flying the
        # camera around does not interrupt anything.
        if route.driving:
            if heading.length() > 0:
                route.clear()
                print("Route cancelled -- you took over.")

            else:
                before = drone.floor_position

                after = route.advance(
                    before,
                    drone_speed * delta_time,
                )

                drone.place_at(after)
                drone.roll(after - before)

                if route.is_finished:
                    print("Arrived.")
                    route.clear()

        if heading.length() > 0:
            step = (
                heading.normalized()
                * drone_speed
                * delta_time
            )

            # One call, two rules. On the ground drive() asks the navmesh,
            # which stops at walls, slides along them, and follows ramps. In
            # the air it asks Bullet instead, for the same courtesies against
            # the real geometry. The drone never has to know which it is.
            drone.drive(step, simulator.pathfinder)

        # -------------------------------------------------------------
        # RGB rendering
        # -------------------------------------------------------------

        # Habitat clears debug lines every frame, so everything below has to
        # be re-drawn every frame.
        draw_drone_outline(debug_lines, drone)

        if route.is_planned:
            debug_lines.draw_path_with_endpoint_circles(
                points=route.points,
                radius=0.12,
                color=mn.Color4(0.27, 0.59, 1.0, 1.0),
            )

        observations = (
            simulator.get_sensor_observations()
        )

        rgb_frame = observations[
            "viewer_rgb"
        ][:, :, :3]

        rgb_surface = numpy_to_surface(
            rgb_frame
        )

        # -------------------------------------------------------------
        # Layout rendering
        # -------------------------------------------------------------

        layout_surface = numpy_to_surface(
            layout_map
        )

        current_width, current_height = (
            screen.get_size()
        )

        half_width = current_width // 2

        rgb_surface = pygame.transform.smoothscale(
            rgb_surface,
            (half_width, current_height),
        )

        layout_surface = pygame.transform.smoothscale(
            layout_surface,
            (half_width, current_height),
        )

        screen.fill((15, 15, 18))

        screen.blit(
            rgb_surface,
            (0, 0),
        )

        screen.blit(
            layout_surface,
            (half_width, 0),
        )

        draw_route_on_layout(
            screen,
            route,
            layout_map,
            simulator,
            half_width,
            current_height,
        )

        # The drone first, so the camera dot stays on top when they overlap.
        draw_layout_marker(
            screen,
            drone.floor_position,
            layout_map,
            simulator,
            half_width,
            current_height,
            color=FLYING_COLOR if drone.flying else ROLLING_COLOR,
            hollow=drone.flying,
        )

        draw_layout_marker(
            screen,
            camera_node.translation,
            layout_map,
            simulator,
            half_width,
            current_height,
            color=CAMERA_COLOR,
        )

        # -------------------------------------------------------------
        # Interface labels
        # -------------------------------------------------------------

        pygame.draw.line(
            screen,
            (255, 255, 255),
            (half_width, 0),
            (half_width, current_height),
            3,
        )

        rgb_title = title_font.render(
            "RGB Environment View",
            True,
            (255, 255, 255),
        )

        layout_title = title_font.render(
            "Ground-Truth Layout",
            True,
            (255, 255, 255),
        )

        # Count the refusal flash down.
        if reject_timer > 0.0:
            reject_timer = max(0.0, reject_timer - delta_time)

        if reject_timer > 0.0:
            route_label = reject_message
            route_label_color = REJECT_COLOR

        elif route.driving:
            route_label = "Following the route  (arrows to take over)"
            route_label_color = ROUTE_COLOR

        elif route.is_planned:
            route_label = (
                f"Route ready: {route.length:.1f} m  "
                "(Enter to follow, C to clear)"
            )
            route_label_color = GOAL_COLOR

        else:
            route_label = "Click the layout to plan a rolling route"
            route_label_color = (200, 200, 200)

        route_text = controls_font.render(
            route_label,
            True,
            route_label_color,
        )

        screen.blit(
            route_text,
            (half_width + 18, 44),
        )

        # The mode line. This is the one readout that says what the drone is
        # doing right now, and the project's whole argument is about which of
        # the two it should be in, so it gets its own colour.
        if drone.flying:
            mode_label = f"FLYING   altitude {drone.altitude:.2f} m   (0 to land)"
            mode_color = FLYING_COLOR

        else:
            mode_label = "ROLLING   on the ground   (1 to take off)"
            mode_color = ROLLING_COLOR

        mode_text = controls_font.render(
            mode_label,
            True,
            mode_color,
        )

        screen.blit(
            mode_text,
            (18, 44),
        )

        controls_text = controls_font.render(
            "Camera: WASD move, IJKL look, Z/X height  |  "
            "Drone: arrows drive, 1 up, 0 down  |  Shift speed  |  Esc exit",
            True,
            (255, 255, 255),
        )

        screen.blit(
            rgb_title,
            (18, 15),
        )

        screen.blit(
            layout_title,
            (half_width + 18, 15),
        )

        screen.blit(
            controls_text,
            (18, current_height - 32),
        )

        pygame.display.flip()

    simulator.close()
    pygame.quit()

    print("Environment closed")

    return 0


if __name__ == "__main__":
    sys.exit(main())
