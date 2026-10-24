"""Drive the drone and watch it map the house, live.

    conda activate habitat
    cd ~/COMP-490-Group-Project/simulation
    python demo/run_mapping.py

Five panels. The one that matters is the big one on the right.

    Behind the drone        a camera following it, so you can see where it is
    What the drone believes the map it has built FOR ITSELF, growing as it
                           drives. Nothing in it came from the simulator.
    Left lens              the raw picture the stereo pair is looking at
    Depth being used       stereo's metres, with Depth Anything filling the
                           gaps stereo could not match
    Status                 what each stage costs, and how far the drone's
                           idea of where it is has drifted from the truth

The map is not a floor plan. Every square is a stack of 10 cm layers, and
the colour says what the drone could DO there:

    grey    plain floor, roll across it
    green   something overhead, with room to roll UNDER it
    blue    something in the way, with room to fly OVER it
    slate   the air is clear but the ground was never seen
    red     no way past at any height

Controls:

    arrows          drive the drone, by compass -- up is north on the map
    1 / 0           take off and climb / descend and land
    V               free camera: W A S D to fly it, I J K L to look
    M               show the true layout behind the drone's map
    click the map   plan a route ON THE DRONE'S OWN MAP to that spot
    Enter           follow it
    C               clear it
    Tab             save the map to data/maps/live.npz
    P               print where the drone is, believed and true
    Escape          quit

Pass --no-map to leave Depth Anything out: stereo and localization only,
about three times faster, useful when you are working on the driving.
"""

import argparse
import json
import math
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pygame

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import magnum as mn
from habitat.utils.visualizations import maps

from delta import rig as rig_module
from delta import sim as world
from delta.simple_brain import Brain
from delta.drone import Drone, find_spot_near
from delta.gap_map import AIRBORNE, AROUND, FLOOR, OVER, UNDER, UNSEEN
from delta.imu import FakeIMU
from delta.world_frame import WorldFrame


WINDOW = (1600, 950)

TOP_HEIGHT = 560
PANEL_GAP = 2

DRONE_COLOR = (90, 230, 120)       # where the drone really is
BELIEF_COLOR = (255, 170, 60)      # where the drone thinks it is
ROUTE_COLOR = (90, 160, 255)
GOAL_COLOR = (255, 230, 80)
REJECT_COLOR = (255, 80, 80)

ROLL_SPEED = 0.9                   # metres per second. See the note below.
FLY_SPEED = 1.1
FREE_SPEED = 2.5
FAST = 3.0
LOOK_SPEED = 90.0

# How fast the gimbal may slew the sensor head, in degrees per second.
# Not cosmetic: a head that snaps to a new heading in one frame leaves the
# feature tracker with two views that barely overlap, and the drone loses
# track of where it is entirely. Measured at 52 m of drift when unlimited.
GIMBAL_RATE = 110.0

# Why the drone is this slow: the tracker rejects a visual match implying a
# jump bigger than position.json's max_speed (3 m/s) as impossible, and falls
# back on the IMU. Driving at 3.5 m/s sent 14 frames in 40 to the IMU and the
# drift to 34%. At about 1 m/s it is 1%.


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-map", action="store_true",
                        help="stereo and localization only, no Depth Anything")
    parser.add_argument("--load", default=None,
                        help="start from a saved map instead of an empty one")
    parser.add_argument("--frames", type=int, default=0,
                        help="drive north by itself for this many frames, then "
                             "quit. For checking the whole thing runs without "
                             "anyone at the keyboard.")
    args = parser.parse_args()

    config = world.load_project_config()
    drone_config = config["drone"]
    diameter = float(drone_config["diameter"])

    root, dataset_config, scene_file, navmesh_file = world.resolve_dataset_paths(config)

    world.verify_files(root, dataset_config, scene_file)
    world.verify_physics()

    rig = rig_module.load()

    print()
    print(rig.describe(eye_height=diameter / 2.0))
    print()
    print("Loading the house...")

    simulator = world.create_simulator(dataset_config, scene_file, rig)
    world.load_or_generate_navmesh(simulator, navmesh_file, drone_config)

    start = world.choose_start_position(simulator)

    drone = Drone(simulator, diameter=diameter)
    drone.place_at(find_spot_near(simulator.pathfinder, start))

    print(f"[PASS] Drone at {drone.floor_position}")

    print("Loading Depth Anything..." if not args.no_map else "Map off.")

    brain = Brain(rig, with_map=not args.no_map)

    if args.load:
        from delta.gap_map import GapMap

        brain.map = GapMap.load(args.load)
        print(f"[PASS] Resumed a map of {brain.map.frames} frames")

    imu = FakeIMU(
        json.loads((PROJECT_ROOT / "config" / "position.json").read_text())
        ["fake_imu"]
    )

    frame = WorldFrame.from_drone(drone, drone.heading)

    # The true layout, drawn once, for the M key. ORACLE: it is shown behind
    # the drone's own map so you can see what it has not found yet, and
    # nothing reads it.
    truth_map = maps.get_topdown_map(
        simulator.pathfinder,
        float(drone.floor_position.y),
        map_resolution=720,
        draw_border=True,
    )
    truth_map = maps.colorize_topdown_map(truth_map)

    # -----------------------------------------------------------------

    pygame.init()

    screen = pygame.display.set_mode(WINDOW, pygame.RESIZABLE)
    pygame.display.set_caption(f"DELTA | {config['dataset']['scene_id']}")

    big = pygame.font.SysFont("Arial", 20, bold=True)
    small = pygame.font.SysFont("Arial", 15)
    tiny = pygame.font.SysFont("Consolas", 13)

    clock = pygame.time.Clock()

    state = Session(drone, brain, frame, diameter)
    state.auto = args.frames
    state.bounds = simulator.pathfinder.get_bounds()

    print()
    print(__doc__.split("Controls:")[1].split("Pass --no-map")[0].strip())
    print()

    running = True

    while running:
        dt = clock.tick(30) / 1000.0
        state.now += dt

        running = handle_events(state, simulator, screen, pygame.event.get())

        keys = pygame.key.get_pressed()

        if state.auto:
            auto_drive(state, simulator, dt)

            if state.frames >= state.auto:
                running = False
        else:
            drive(state, simulator, keys, dt)

        world.mount_sensors(simulator, drone)

        rotation, position = world.camera_pose_in_world(simulator, rig)
        reading = imu.read(state.now, rotation, position)

        state.true_position = position

        left, right = world.read_cameras(simulator, rig)

        began = time.perf_counter()
        thought = brain.step(left, right, reading, state.now, dt)
        state.frame_ms = (time.perf_counter() - began) * 1000.0
        state.thought = thought
        state.frames += 1

        if not state.free_camera:
            world.place_spectator(
                simulator, drone, state.cam_yaw, state.cam_pitch,
                state.cam_distance, state.cam_height,
            )

        draw(screen, state, simulator, rig, truth_map, big, small, tiny)

        pygame.display.flip()

    simulator.close()
    pygame.quit()

    if args.frames and state.thought is not None:
        summarise(state)

    print("Closed")
    return 0


def summarise(state):
    drift = state.drift()
    gap_map = state.brain.map

    print()
    print(f"{state.frames} frames driven")

    if drift is not None:
        print(f"  drift from the truth   {drift:.2f} m")

    if gap_map is None:
        print("  no map built")
        return

    kinds = gap_map.classify(state.diameter)
    seen = int((kinds != UNSEEN).sum())

    print(f"  frames into the map    {gap_map.frames}")
    print(f"  squares seen           {seen}"
          f"  ({seen * gap_map.cell ** 2:.1f} m2)")

    for name, value in (
        ("roll across", FLOOR),
        ("roll UNDER", UNDER),
        ("fly OVER", OVER),
        ("air only", AIRBORNE),
        ("no way past", AROUND),
    ):
        number = int((kinds == value).sum())
        print(f"    {name:<14} {number:6d}  {number / max(seen, 1) * 100:5.1f}%")


# ---------------------------------------------------------------------
# What the loop is carrying around
# ---------------------------------------------------------------------


class Session:
    def __init__(self, drone, brain, frame, diameter):
        self.drone = drone
        self.brain = brain
        self.frame = frame
        self.diameter = diameter

        self.now = 0.0
        self.frames = 0
        self.frame_ms = 0.0
        self.thought = None
        self.true_position = None

        self.free_camera = False
        self.show_truth = False

        self.cam_yaw = 0.0
        self.cam_pitch = -22.0
        self.cam_distance = 2.6
        self.cam_height = 1.5

        self.bounds = None       # the pathfinder's extent, for the M key

        # The map only changes on frames that went into it, so the
        # classification and the picture are worked out then and reused
        # in between. Recomputing both every screen frame is what held
        # the window at 2 fps, and a slow window breaks the tracker: it
        # needs frames close enough together to match features across.
        self.kinds = None
        self.map_picture = None
        self.drawn_at_frame = -1

        self.planner = None
        self.plan = None
        self.plan_index = 0
        self.following = False

        self.message = ""
        self.message_until = 0.0

        # Frames to drive unattended before quitting; 0 means a person
        # is at the keyboard.
        self.auto = 0
        self.auto_heading = 0.0
        self.auto_stuck = 0

    def say(self, text, seconds=3.0):
        self.message = text
        self.message_until = self.now + seconds
        print(text)

    @property
    def believed(self):
        """Where the drone thinks it is, in map metres."""
        if self.thought is None:
            return (0.0, 0.0)

        pose = self.thought.track.pose

        return (float(pose[0, 3]), float(pose[2, 3]))

    def drift(self):
        """How far the drone's belief is from the truth, in metres."""
        if self.true_position is None:
            return None

        across, along, _ = self.frame.to_map(*self.true_position)
        believed = self.believed

        return math.dist((across, along), believed)


# ---------------------------------------------------------------------
# Input
# ---------------------------------------------------------------------


def handle_events(state, simulator, screen, events):
    drone = state.drone

    for event in events:
        if event.type == pygame.QUIT:
            return False

        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                return False

            elif event.key == pygame.K_v:
                state.free_camera = not state.free_camera
                state.say("Free camera" if state.free_camera else "Following the drone")

            elif event.key == pygame.K_m:
                state.show_truth = not state.show_truth
                state.say("True layout on" if state.show_truth else "True layout off")

            elif event.key in (pygame.K_1, pygame.K_KP1):
                if not drone.flying:
                    if drone.take_off():
                        state.say(f"Taking off to {drone.target_altitude:.2f} m")
                    else:
                        state.say("No room overhead to take off")

            elif event.key == pygame.K_c:
                state.plan = None
                state.following = False
                state.say("Route cleared")

            elif event.key == pygame.K_RETURN:
                if state.plan is None:
                    state.say("Click the map to plan a route first")
                else:
                    state.following = True
                    state.plan_index = 0
                    state.say(f"Following: {state.plan.describe()}")

            elif event.key == pygame.K_TAB:
                if state.brain.map is None:
                    state.say("Nothing mapped yet")
                else:
                    path = state.brain.map.save(
                        PROJECT_ROOT / "data" / "maps" / "live.npz"
                    )
                    state.say(f"Saved {path.name} ({state.brain.map.frames} frames)")

            elif event.key == pygame.K_p:
                truth = state.frame.to_map(*state.true_position)
                print(f"believed {state.believed[0]:+.2f}, {state.believed[1]:+.2f}"
                      f"   true {truth[0]:+.2f}, {truth[1]:+.2f}"
                      f"   drift {state.drift():.2f} m")

        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            plan_to_click(state, screen, event.pos)

    return True


def plan_to_click(state, screen, position):
    """A click on the map panel: plan a route there, on the drone's own map."""
    if state.brain.map is None:
        state.say("Nothing mapped yet -- drive around first")
        return

    layout = map_panel_rect(screen)

    if not layout.collidepoint(position):
        return

    gap_map = state.brain.map

    # Undo the panel: pixels -> map squares -> metres.
    fraction_x = (position[0] - layout.left) / layout.width
    fraction_y = (position[1] - layout.top) / layout.height

    window = map_window(gap_map)

    col = window[0] + fraction_x * (window[1] - window[0])
    row = window[2] + fraction_y * (window[3] - window[2])

    across, along = gap_map.to_world(row, col)

    from delta.map_navigation import MapPlanner

    if state.planner is None or state.planner.is_stale:
        state.planner = MapPlanner(gap_map, state.diameter)

    plan = state.planner.plan_between(state.believed, (float(across), float(along)))

    if plan is None:
        state.plan = None
        state.say("No route there through what has been mapped", 2.0)
        return

    state.plan = plan
    state.plan_index = 0
    state.following = False
    state.say(f"Planned: {plan.describe()} -- Enter to follow")


def auto_drive(state, simulator, dt):
    """Drive without anyone at the keyboard, for --frames.

    It turns when it stops making progress, and that is not a detail. The
    first version simply drove north until the count ran out, which meant it
    reached a wall and then pressed into it for sixty frames. A camera 15 cm
    from a blank wall has nothing to track, so the drone's idea of where it
    was came apart: 6.4 m of drift, none of it the tracker's fault.

    Nobody drives a robot like that, but an unattended test will, every time,
    unless it is told not to.
    """
    drone = state.drone

    before = drone.floor_position

    heading = mn.Quaternion.rotation(
        mn.Deg(state.auto_heading), mn.Vector3.y_axis()
    ).transform_vector(mn.Vector3(0.0, 0.0, -1.0))

    drone.drive(
        heading * ROLL_SPEED * dt,
        simulator.pathfinder,
        turn_rate=GIMBAL_RATE * dt,
    )
    drone.update(dt)

    moved = (drone.floor_position - before).length()

    # Blocked, or nearly: turn a little and try again. Small turns on
    # purpose -- a big one swings the view so far between frames that the
    # feature tracker has nothing to match, and the drone loses itself.
    if moved < ROLL_SPEED * dt * 0.4:
        state.auto_stuck += 1

        if state.auto_stuck >= 2:
            state.auto_heading = (state.auto_heading + 25.0) % 360.0
            state.auto_stuck = 0
    else:
        state.auto_stuck = 0


def drive(state, simulator, keys, dt):
    drone = state.drone

    fast = keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]

    # Altitude, on 1 and 0. Neither key moves the drone: they move the
    # altitude it is flying toward, and update() closes the gap smoothly.
    if keys[pygame.K_1] or keys[pygame.K_KP1]:
        drone.climb(dt)

    if keys[pygame.K_0] or keys[pygame.K_KP0]:
        drone.descend(dt)

    was_flying = drone.flying
    drone.update(dt)

    if was_flying and not drone.flying:
        state.say("Landed")

    # The free camera takes WASD/IJKL when it is on.
    if state.free_camera:
        speed = FREE_SPEED * (FAST if fast else 1.0)

        if keys[pygame.K_j]:
            state.cam_yaw += LOOK_SPEED * dt
        if keys[pygame.K_l]:
            state.cam_yaw -= LOOK_SPEED * dt
        if keys[pygame.K_i]:
            state.cam_pitch = min(89.0, state.cam_pitch + LOOK_SPEED * dt)
        if keys[pygame.K_k]:
            state.cam_pitch = max(-89.0, state.cam_pitch - LOOK_SPEED * dt)

        node = simulator.agents[world.SPECTATOR].scene_node

        yaw_rotation = mn.Quaternion.rotation(
            mn.Deg(state.cam_yaw), mn.Vector3.y_axis()
        )
        node.rotation = yaw_rotation * mn.Quaternion.rotation(
            mn.Deg(state.cam_pitch), mn.Vector3.x_axis()
        )

        forward = yaw_rotation.transform_vector(mn.Vector3(0.0, 0.0, -1.0))
        right = yaw_rotation.transform_vector(mn.Vector3(1.0, 0.0, 0.0))

        move = mn.Vector3(0.0, 0.0, 0.0)

        if keys[pygame.K_w]:
            move += forward
        if keys[pygame.K_s]:
            move -= forward
        if keys[pygame.K_a]:
            move -= right
        if keys[pygame.K_d]:
            move += right
        if keys[pygame.K_z]:
            move += mn.Vector3.y_axis()
        if keys[pygame.K_x]:
            move -= mn.Vector3.y_axis()

        if move.length() > 0:
            node.translation += move.normalized() * speed * dt

    else:
        if keys[pygame.K_j]:
            state.cam_yaw += LOOK_SPEED * dt
        if keys[pygame.K_l]:
            state.cam_yaw -= LOOK_SPEED * dt

    # The drone: arrow keys, by compass. Up is always north on the map,
    # whatever the camera is doing, so what you read off the map is what the
    # keys do. A sphere has no front, so compass is the only honest scheme.
    heading = mn.Vector3(0.0, 0.0, 0.0)

    if keys[pygame.K_UP]:
        heading += mn.Vector3(0.0, 0.0, -1.0)
    if keys[pygame.K_DOWN]:
        heading += mn.Vector3(0.0, 0.0, 1.0)
    if keys[pygame.K_LEFT]:
        heading += mn.Vector3(-1.0, 0.0, 0.0)
    if keys[pygame.K_RIGHT]:
        heading += mn.Vector3(1.0, 0.0, 0.0)

    speed = (FLY_SPEED if drone.flying else ROLL_SPEED) * (FAST if fast else 1.0)

    if heading.length() > 0:
        if state.following:
            state.following = False
            state.say("Route cancelled -- you took over")

        drone.drive(
            heading.normalized() * speed * dt,
            simulator.pathfinder,
            turn_rate=GIMBAL_RATE * dt,
        )

    elif state.following and state.plan is not None:
        follow_plan(state, simulator, speed * dt)


def follow_plan(state, simulator, distance):
    """Walk the plan the drone made on its own map.

    The plan is in map metres and the drone lives in Habitat's world, so
    every waypoint goes through WorldFrame -- which is fixed at the start and
    never refreshed. If the tracker has drifted, the drone will arrive
    somewhere slightly wrong, and that is the honest outcome rather than a
    bug to paper over.
    """
    plan = state.plan
    drone = state.drone

    while distance > 1e-4 and state.plan_index < len(plan.points):
        across, along, height = plan.points[state.plan_index]

        x, _, z = state.frame.ground_point(across, along)

        here = drone.floor_position
        to_go = mn.Vector3(x - here.x, 0.0, z - here.z)

        if to_go.length() < 0.15:
            state.plan_index += 1
            continue

        # Climb or land as the plan's own height demands.
        wanted_altitude = max(0.0, height - drone.radius)

        if wanted_altitude > 0.05 and not drone.flying:
            drone.take_off()

        if drone.flying:
            drone.target_altitude = wanted_altitude

        travel = min(distance, to_go.length())
        drone.drive(
            to_go.normalized() * travel,
            simulator.pathfinder,
            turn_rate=GIMBAL_RATE * 0.05,
        )
        distance -= travel

    if state.plan_index >= len(plan.points):
        state.following = False
        state.say("Arrived")


# ---------------------------------------------------------------------
# Drawing
# ---------------------------------------------------------------------


def surface_from(image):
    return pygame.surfarray.make_surface(np.transpose(image, (1, 0, 2)))


def map_panel_rect(screen):
    width, height = screen.get_size()
    top = int(height * TOP_HEIGHT / WINDOW[1])

    return pygame.Rect(width // 2 + PANEL_GAP, 0, width // 2 - PANEL_GAP, top)


def map_window(gap_map, margin=12):
    """(left, right, top, bottom) in grid squares, cropped to what is known.

    A 40 m map of a 6 m drive is almost all empty, so the panel shows only
    the part that has been seen, plus a margin to drive into.
    """
    seen = (gap_map.states() != UNSEEN).any(axis=2)
    rows, cols = np.nonzero(seen)

    if len(rows) == 0:
        middle = gap_map.half
        return (middle - 40, middle + 40, middle - 40, middle + 40)

    return (
        max(0, cols.min() - margin),
        min(gap_map.n, cols.max() + margin + 1),
        max(0, rows.min() - margin),
        min(gap_map.n, rows.max() + margin + 1),
    )


def refresh_map_view(state):
    """Re-read the map, but only when it has actually changed."""
    gap_map = state.brain.map

    if gap_map is None or gap_map.frames == state.drawn_at_frame:
        return

    state.kinds = gap_map.classify(state.diameter)
    state.map_picture = gap_map.picture(state.diameter, state.kinds)
    state.drawn_at_frame = gap_map.frames


def draw(screen, state, simulator, rig, truth_map, big, small, tiny):
    refresh_map_view(state)

    width, height = screen.get_size()
    top = int(height * TOP_HEIGHT / WINDOW[1])
    bottom = height - top - 26

    half = width // 2

    screen.fill((16, 16, 20))

    # ---- top left: behind the drone --------------------------------
    view = surface_from(world.spectator_frame(simulator))
    screen.blit(
        pygame.transform.smoothscale(view, (half - PANEL_GAP, top)), (0, 0)
    )

    # ---- top right: the drone's own map ----------------------------
    panel = map_panel_rect(screen)
    draw_map(screen, state, panel, truth_map, small)

    # ---- bottom row -------------------------------------------------
    third = width // 3

    thought = state.thought

    if thought is not None:
        lens = surface_from(np.ascontiguousarray(
            state.brain.stereo.rectify.left_rgb
        ))
        screen.blit(
            pygame.transform.smoothscale(lens, (third - PANEL_GAP, bottom)),
            (0, top + PANEL_GAP),
        )

        shown = thought.dense if thought.dense is not None else thought.depth
        screen.blit(
            pygame.transform.smoothscale(
                surface_from(colour_depth(shown)),
                (third - PANEL_GAP, bottom),
            ),
            (third, top + PANEL_GAP),
        )

    draw_stats(screen, state, tiny, small,
               pygame.Rect(2 * third, top + PANEL_GAP,
                           width - 2 * third, bottom))

    # ---- titles -----------------------------------------------------
    label(screen, big, "Behind the drone -- V for free camera", (14, 10))
    label(screen, big, "What the drone believes", (panel.left + 14, 10))
    label(screen, small,
          "green: really here   orange: believed here   M: true layout",
          (panel.left + 14, 36), (170, 170, 180))

    label(screen, small, "Left lens", (14, top + 10))
    label(screen, small, "Depth the drone is using", (third + 14, top + 10))
    label(screen, small, "Status", (2 * third + 14, top + 10))

    # ---- the message line ------------------------------------------
    if state.now < state.message_until:
        label(screen, small, state.message, (14, height - 22), GOAL_COLOR)
    else:
        label(screen, small,
              "arrows drive | 1 up 0 down | click map to plan | Enter follow"
              " | Tab save | Esc quit",
              (14, height - 22), (150, 150, 160))


def label(screen, font, text, position, colour=(255, 255, 255)):
    screen.blit(font.render(text, True, colour), position)


def colour_depth(depth, far=6.0):
    """Depth as a picture: near is yellow, far is blue, no answer is black."""
    shown = np.nan_to_num(depth, nan=0.0, posinf=0.0)
    shown = np.clip(shown / far, 0.0, 1.0)

    image = cv2.applyColorMap((shown * 255).astype(np.uint8), cv2.COLORMAP_VIRIDIS)
    image[~np.isfinite(depth)] = (0, 0, 0)

    return np.ascontiguousarray(image[:, :, ::-1])


def draw_map(screen, state, panel, truth_map, font):
    gap_map = state.brain.map

    if gap_map is None:
        pygame.draw.rect(screen, (26, 26, 32), panel)
        label(screen, font, "no map yet -- drive the drone around",
              (panel.left + 16, panel.top + 70), (140, 140, 150))
        return

    if state.map_picture is None:
        return

    left, right, top, bottom = map_window(gap_map)

    picture = state.map_picture[top:bottom, left:right]

    if state.show_truth:
        picture = blend_truth(picture, truth_map, gap_map,
                              (left, right, top, bottom), state.frame,
                              state.bounds)

    surface = pygame.transform.scale(
        surface_from(picture), (panel.width, panel.height)
    )
    screen.blit(surface, panel.topleft)

    def to_screen(row, col):
        x = panel.left + (col - left) / max(right - left, 1) * panel.width
        y = panel.top + (row - top) / max(bottom - top, 1) * panel.height
        return int(x), int(y)

    # The planned route, on the map it was planned on.
    if state.plan is not None:
        points = [
            to_screen(*gap_map.to_grid(across, along))
            for across, along, _ in state.plan.points
        ]

        if len(points) >= 2:
            pygame.draw.lines(screen, ROUTE_COLOR, False, points, 3)

        pygame.draw.circle(screen, GOAL_COLOR, points[-1], 7)

    # Where it thinks it is, and where it really is.
    believed = gap_map.to_grid(*state.believed)
    marker(screen, to_screen(*believed), BELIEF_COLOR)

    if state.true_position is not None:
        across, along, _ = state.frame.to_map(*state.true_position)
        marker(screen, to_screen(*gap_map.to_grid(across, along)), DRONE_COLOR)


def marker(screen, position, colour):
    pygame.draw.circle(screen, colour, position, 7)
    pygame.draw.circle(screen, (255, 255, 255), position, 9, 2)


def blend_truth(picture, truth_map, gap_map, window, frame, bounds):
    """Put the true layout faintly behind the drone's map.

    ORACLE, and shown only because the point of a split screen is to see what
    the drone has NOT found. Nothing reads it.
    """
    left, right, top, bottom = window

    # Where each mapped square sits in the true layout's pixels.
    rows = np.arange(top, bottom)
    cols = np.arange(left, right)

    across, along = gap_map.to_world(rows[:, None], cols[None, :])

    # The true map is drawn in Habitat's x/z; convert the map frame back.
    height, width = truth_map.shape[:2]

    # frame.to_world is per-point, so do the rotation here in bulk.
    radians = math.radians(frame.heading)
    forward = (-math.sin(radians), -math.cos(radians))
    rightward = (-forward[1], forward[0])

    x = frame.origin_x + rightward[0] * across + forward[0] * along
    z = frame.origin_z + rightward[1] * across + forward[1] * along

    # get_topdown_map covers exactly the pathfinder's bounds, so those are
    # what turn a world position into one of its pixels.
    lower, upper = bounds

    u = ((x - lower[0]) / max(upper[0] - lower[0], 1e-9) * (width - 1))
    v = ((z - lower[2]) / max(upper[2] - lower[2], 1e-9) * (height - 1))

    u = np.clip(u, 0, width - 1).astype(int)
    v = np.clip(v, 0, height - 1).astype(int)

    behind = truth_map[v, u]

    out = picture.astype(np.float32)
    unseen = (picture.sum(axis=2) == 0)

    out[unseen] = behind[unseen] * 0.45

    return out.astype(np.uint8)


def draw_stats(screen, state, mono, font, panel):
    pygame.draw.rect(screen, (22, 22, 27), panel)

    thought = state.thought

    if thought is None:
        return

    ms = thought.milliseconds
    track = thought.track
    drift = state.drift()

    gap_map = state.brain.map

    lines = [
        ("WHERE IT THINKS IT IS", None),
        (f"  source        {track.source}"
         f"{'  (' + track.why + ')' if track.why else ''}", None),
        (f"  corners       {track.corners}", None),
        (f"  inliers       {track.inliers}", None),
        (f"  believed      {state.believed[0]:+.2f}, {state.believed[1]:+.2f} m", None),
        (f"  drift         {drift:.2f} m" if drift is not None else "", REJECT_COLOR
         if drift is not None and drift > 1.0 else None),
        ("", None),
        ("WHAT IT CAN SEE", None),
        (f"  camera height {thought.camera_height:.2f} m"
         f"{'' if thought.floor_measured else '  (not measured)'}", None),
        (f"  stereo points {len(thought.points)}", None),
        ("", None),
    ]

    if gap_map is not None and state.kinds is not None:
        kinds = state.kinds
        seen = int((kinds != UNSEEN).sum())

        counts = {
            "roll across": int((kinds == FLOOR).sum()),
            "roll UNDER": int((kinds == UNDER).sum()),
            "fly OVER": int((kinds == OVER).sum()),
            "air only": int((kinds == AIRBORNE).sum()),
            "no way past": int((kinds == AROUND).sum()),
        }

        lines += [
            ("WHAT IT HAS MAPPED", None),
            (f"  frames        {gap_map.frames}", None),
            (f"  seen          {seen} squares"
             f"  ({seen * gap_map.cell ** 2:.1f} m2)", None),
        ]

        for name, number in counts.items():
            share = number / seen * 100 if seen else 0.0
            lines.append((f"  {name:<13} {number:5d}  {share:4.1f}%", None))

        lines.append(("", None))

    lines += [
        ("WHAT IT COSTS", None),
        (f"  frame         {state.frame_ms:.0f} ms"
         f"  ({1000.0 / max(state.frame_ms, 1e-9):.1f} fps)", None),
    ]

    for name in ("rectify", "sgbm", "points", "floor", "position", "map"):
        if name in ms:
            lines.append((f"  {name:<13} {ms[name]:.0f} ms", None))

    y = panel.top + 34

    for text, colour in lines:
        if text:
            screen.blit(
                mono.render(text, True, colour or (205, 205, 215)),
                (panel.left + 12, y),
            )

        y += 17

        if y > panel.bottom - 16:
            break


if __name__ == "__main__":
    raise SystemExit(main())
