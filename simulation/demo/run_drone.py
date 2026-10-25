"""Drive the drone around the house and watch it map it.

The simulated half of the system: Habitat, the drone, and its cameras.
Everything else -- depth, RTAB-Map, the map -- is the brain, a separate
process in its own conda env, exactly as it will be with real hardware. This
window sends it the stereo pair ten times a second and draws what it sends
back.

Ported from the Spatial Memory Glasses viewer (spatial-agent-12). The glasses'
other jobs are gone: no YOLOE boxes, no hazard grid, no conversation. What is
left is the part this project is about -- localization and mapping.

    bash run_drone.sh                  # starts the brain and this window
    bash run_drone.sh --pose oracle    # Habitat's true pose instead of RTAB-Map
    bash run_drone.sh --depth oracle   # Habitat's true depth instead of DA3
    bash run_drone.sh --fresh          # forget last session's map

The two oracle switches are how you tell a mapping bug from a localization
bug: swap in the truth for one of them and see which panel stops being wrong.

The window:

    top left       the drone, from behind (V: fly the camera freely instead)
    top right      what the DRONE believes, drawn over Habitat's true layout
                   (M hides the truth). Green: where it really is. Orange:
                   where the brain thinks it is. Blue: the route.
    bottom         the left lens, the depth the brain is using, and the log

Keys:

    Up / Down        forward / back                 Shift: faster
    Left / Right     turn on the spot
    1 / 0            take off and climb / descend and land
    R                roam: drive to a random spot (to build the map quickly)
    G                autopilot: follow the brain's own route
    click the map    plan a route there, on the drone's own map
    C                clear the route
    V                camera: behind the drone / free flight (WASD, IJKL, Z/X)
    M                show / hide the true layout
    Tab              tell the brain to save the map
    F12              screenshot to outputs/screenshots/
    Esc              quit (the brain saves the map)
"""

import argparse
import json
import math
import os
import queue
import sys
import threading
import time
from pathlib import Path

os.environ.setdefault("SDL_FRAMEBUFFER_ACCELERATION", "0")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import cv2  # noqa: E402
import habitat_sim  # noqa: E402
import magnum as mn  # noqa: E402
import numpy as np  # noqa: E402
import pygame  # noqa: E402
from habitat.utils.visualizations import maps  # noqa: E402

from delta import frames, wire  # noqa: E402
from delta.brain import config as brain_config  # noqa: E402
from delta.sim import ClimbRate, Imu, Odometer, Rangefinder, World, first_hit  # noqa: E402

WINDOW = (1600, 960)
TOP = 560
STRIP_TOP = 566
STATUS_TOP = 934

# A 30 cm drone, not a walking adult. Slower, because the tracker matches
# features BETWEEN frames: at 1 m/s and 10 frames a second it moves 10 cm,
# which is about as far as ORB can bridge in this house.
WALK_SPEED = 0.8                    # m/s
TURN_SPEED = math.radians(50.0)     # rad/s
FAST = 2.0                          # Shift multiplies both

# The chase camera sits lower than the glasses' did, because the drone is
# 15 cm off the floor and a camera at head height looks down at a dot. It
# rides up with the drone when it flies, by less, because ceilings here can
# be 2 m.
CHASE_BACK = 1.6
CHASE_UP = 0.95                     # above the drone's centre, on the ground
CHASE_UP_FLYING = 0.5               # ... and in the air
CHASE_MARGIN = 0.2                  # stop this far short of a wall
CHASE_NEAREST = 0.45                # never closer than this to the drone
CHASE_PITCH = -22.0                 # the free camera's starting tilt

KIND_COLOURS = {
    "reply": (110, 210, 255),
    "system": (135, 140, 150),
}

FREE_COLOUR = np.array([78, 92, 110], np.uint8)
OCCUPIED_COLOUR = np.array([235, 232, 220], np.uint8)

# Free floor the drone worked out is OUTSIDE -- nothing overhead. Olive, so
# it reads as ground rather than as another room, and so that the boundary
# between the two is visible at a glance: that line is the building's wall,
# found by the drone rather than given to it.
OUTDOOR_COLOUR = np.array([86, 96, 58], np.uint8)
OUTDOOR_WALL_COLOUR = np.array([150, 160, 110], np.uint8)
TRUTH_FLOOR = np.array([34, 40, 50], np.uint8)
TRUTH_EDGE = np.array([70, 78, 94], np.uint8)

DRONE_TRUE = (80, 220, 110)
DRONE_BELIEVED = (255, 150, 50)
ROUTE_COLOUR = (80, 150, 255)
TARGET_COLOUR = (255, 220, 80)
THING_COLOUR = (215, 120, 255)
PLACE_COLOUR = (255, 220, 80)


def parse_arguments():
    parser = argparse.ArgumentParser(description="Drive the drone and watch it map.")
    parser.add_argument("--pose", choices=["rtabmap", "oracle"], default="rtabmap",
                        help="where the brain's pose comes from (oracle: Habitat's truth)")
    parser.add_argument("--depth", choices=["da3", "oracle"], default="da3",
                        help="where the brain's depth comes from (oracle: Habitat's truth)")
    parser.add_argument("--fresh", action="store_true", help="the brain forgets everything first")
    parser.add_argument("--no-odometer", action="store_true",
                        help="take the cage's axle encoder out, to measure what it adds")
    parser.add_argument("--altitude", choices=["rangefinder", "oracle"], default="rangefinder",
                        help="where the brain's height comes from (oracle: Habitat's truth)")
    parser.add_argument("--trace", type=str, default=None,
                        help="write pose error and RTAB-Map health 4 times a second to this file (JSON lines)")
    parser.add_argument("--no-tilt", action="store_true",
                        help="the brain ignores the IMU's pitch and roll (to see what that does)")
    parser.add_argument("--no-brain", action="store_true", help="just the sim, nothing to talk to")
    parser.add_argument("--script", type=str, help="drive the wearer from a JSON script")
    parser.add_argument("--screenshots", type=float, default=0.0,
                        help="save a screenshot every N seconds")
    parser.add_argument("--headless", action="store_true", help="no window (for scripts)")
    parser.add_argument("--roam", action="store_true",
                        help="start roaming straight away, to build a map without a driver")
    parser.add_argument("--port", type=int, default=None)
    return parser.parse_args()


# ---------------------------------------------------------------------
# Talking to the brain
# ---------------------------------------------------------------------


class BrainLink:
    """The connection, from the sim's side. Receiving runs on its own thread;
    Habitat is only ever touched from the main one."""

    def __init__(self, host, port, hello):
        self.host, self.port, self.hello = host, port, hello
        self.link = None
        self.ready = False
        self.state = None
        self.state_arrays = {}
        self.said = queue.Queue()
        self.error = None
        self.closed = False

        threading.Thread(target=self._run, daemon=True, name="brain-link").start()

    def _run(self):
        try:
            self.link = wire.connect(self.host, self.port, timeout=300.0,
                                     should_stop=lambda: self.closed)
            self.link.send("hello", self.hello)

            while not self.closed:
                message = self.link.receive()

                if message.kind == "say":
                    self.said.put((message.fields["kind"], message.fields["text"]))
                elif message.kind == "status":
                    # The glasses spoke; DELTA writes a line in the log.
                    self.said.put(("system", message.fields["text"]))
                elif message.kind == "state":
                    self.state, self.state_arrays = message.fields, message.arrays
                elif message.kind == "ready":
                    self.ready = True

        except (wire.Closed, OSError) as error:
            if not self.closed:
                self.error = str(error)

    @property
    def connected(self):
        return self.link is not None and self.link.open

    def send(self, kind, fields=None, arrays=None):
        if self.connected:
            try:
                self.link.send(kind, fields, arrays)
            except wire.Closed as error:
                self.error = str(error)

    def close(self):
        if self.connected:
            self.send("bye")

        self.closed = True


# ---------------------------------------------------------------------
# Lining the brain's map up with the truth
# ---------------------------------------------------------------------


class Alignment:
    """Brain world -> Habitat-derived world (x east, y north), on the floor.

    With RTAB-Map the brain's map starts wherever the glasses were switched
    on, facing however they faced, so to draw it over the true layout the
    two have to be lined up once -- at the first frame both know about, and
    again whenever the brain says its frame jumped. Never refitted otherwise:
    drift must show up as drift.
    """

    def __init__(self):
        self.epoch = None
        self.yaw = 0.0
        self.offset = np.zeros(2)

    def update(self, epoch, brain_pose, true_pose):
        if self.epoch == epoch or true_pose is None:
            return

        bx, by, byaw = brain_pose
        tx, ty, tyaw = true_pose

        self.yaw = frames.wrap(tyaw - byaw)
        c, s = math.cos(self.yaw), math.sin(self.yaw)
        self.offset = np.array([tx - (c * bx - s * by), ty - (s * bx + c * by)])
        self.epoch = epoch

    @property
    def ready(self):
        return self.epoch is not None

    def point(self, xy):
        c, s = math.cos(self.yaw), math.sin(self.yaw)
        x, y = xy
        return np.array([c * x - s * y, s * x + c * y]) + self.offset

    def matrix(self):
        """2x3 affine of the same map."""
        c, s = math.cos(self.yaw), math.sin(self.yaw)
        return np.array([[c, -s, self.offset[0]], [s, c, self.offset[1]]])


# ---------------------------------------------------------------------
# The app
# ---------------------------------------------------------------------


class DroneApp:
    def __init__(self, args):
        self.args = args
        self.settings = brain_config.load()

        print("Opening the house...")
        self.world = World(stage="process", true_depth=(args.depth == "oracle"))
        self.drone_config = self.world.config["drone"]
        self.rig = self.world.rig
        self.imu = Imu(seed=7)
        self._tilt_noise = np.random.default_rng(8)
        self._tilt_error = np.zeros(2)
        self.odometer = Odometer(seed=9)
        self._rolled_from = None
        self.rangefinder = Rangefinder(self.rig.rangefinder or {}, self.world.simulator, seed=10)
        self.climb_rate = ClimbRate(seed=11)
        self.frame_rate = float(self.settings.cameras.frame_rate)

        self.floor_height = float(self.world.drone.floor_position.y)
        self._make_truth_layer()

        width, height = self.rig.resolution("process")

        self.brain = None

        if not args.no_brain:
            self.brain = BrainLink(
                self.settings.link.host,
                args.port or int(self.settings.link.port),
                {
                    "scene_id": self.world.scene_id,
                    "width": width,
                    "height": height,
                    "pose": args.pose,
                    "depth": args.depth,
                    "altitude": args.altitude,
                    "fresh": args.fresh,
                    "use_tilt": not args.no_tilt,
                    "map_localize": getattr(args, "map_localize", False),
                    # The lenses sit at the centre of the sphere, so this is
                    # the drone's radius -- and the brain scales every depth
                    # by it, so it is sent rather than assumed.
                    "lens_height": float(self.drone_config["diameter"]) / 2.0,
                },
            )

        self.alignment = Alignment()
        self.truth_by_frame = {}
        self.height_by_frame = {}

        # One trace per run. It is appended to line by line as the run goes,
        # so it has to be emptied at the start -- otherwise a second run with
        # the same file name is silently mixed into the first one's numbers.
        if args.trace:
            Path(args.trace).parent.mkdir(parents=True, exist_ok=True)
            Path(args.trace).write_text("")
        self.walked = 0.0
        self.turned = 0.0
        self._navigable = None
        self.frame_id = 0
        self.clock = 0.0
        self.next_frame = 0.0
        self.last_glasses = None

        self.transcript = []
        self.show_truth = True
        self.free_camera = False
        self.autopilot = False
        self.roam = None
        self.camera_yaw = 0.0
        self.camera_pitch = CHASE_PITCH

        self.script = Script(args.script, self) if args.script else None

        if getattr(args, "roam", False):
            self.start_roam()
        self.next_screenshot = args.screenshots or math.inf
        self.running = True

    # -----------------------------------------------------------------
    # The true layout, drawn once
    # -----------------------------------------------------------------

    def _make_truth_layer(self):
        pathfinder = self.world.pathfinder
        topdown = maps.get_topdown_map(pathfinder, self.floor_height, map_resolution=1024, draw_border=True)

        layer = np.zeros(topdown.shape + (3,), np.uint8)
        layer[topdown == maps.MAP_VALID_POINT] = TRUTH_FLOOR
        layer[topdown == maps.MAP_BORDER_INDICATOR] = TRUTH_EDGE
        self.truth_layer = layer

        lower, upper = pathfinder.get_bounds()
        height, width = topdown.shape

        self._truth_bounds = (lower, upper, width, height)

        # The ground truth's own extent, as the panel's starting view.
        self._fit_view(
            np.array([lower.x - 1.0, -upper.z - 1.0]),
            np.array([upper.x + 1.0, -lower.z + 1.0]),
        )

    def _fit_view(self, view_min, view_max):
        """Set the map panel to show this rectangle of the world.

        Separate from _make_truth_layer, and recomputed as the drone maps,
        because the view must NOT be fixed to the ground truth's extent.

        The truth is a navmesh of one building and its garden. The drone's
        map is whatever its cameras reached, and that can be bigger -- it
        sees through a doorway into a room the navmesh never covered, or its
        pose drifts and it writes cells beyond the edge. Clipping to the
        truth's rectangle hid exactly those cells, which is the opposite of
        useful: the whole reason for drawing the truth underneath is to see
        where the drone's map and the real world disagree, and a disagreement
        off the edge of the truth is still a disagreement.
        """
        self.view_min = np.asarray(view_min, dtype=np.float64)
        self.view_max = np.asarray(view_max, dtype=np.float64)

        span = np.maximum(self.view_max - self.view_min, 1e-6)
        panel = np.array([800.0, float(TOP)])

        self.scale = float(min(panel / span))
        self.panel_offset = (panel - span * self.scale) / 2.0

        lower, upper, width, height = self._truth_bounds

        # Topdown pixel -> world: x along columns, z along rows.
        dx = (upper.x - lower.x) / (width - 1)
        dz = (upper.z - lower.z) / (height - 1)

        self.truth_affine = np.array([
            [dx * self.scale, 0.0,
             (lower.x - self.view_min[0]) * self.scale + self.panel_offset[0]],
            [0.0, dz * self.scale,
             (self.view_max[1] + lower.z) * self.scale + self.panel_offset[1]],
        ])

    def _widen_view_for(self, state, arrays):
        """Grow the panel so everything the drone has mapped is on it.

        Only ever grows, and only by whole half-metres, so the picture does
        not jitter as the map creeps outward cell by cell.
        """
        if "grid" not in state or "grid" not in arrays or not self.alignment.ready:
            return

        grid = arrays["grid"]
        origin = np.asarray(state["grid"]["origin"], dtype=np.float64)
        resolution = float(state["grid"]["resolution"])

        rows, cols = grid.shape[:2]

        corners = np.array([
            origin,
            origin + [cols * resolution, 0.0],
            origin + [0.0, rows * resolution],
            origin + [cols * resolution, rows * resolution],
        ])

        # The grid is in the brain's own frame; the panel is in the truth's.
        in_world = np.array([self.alignment.point(corner) for corner in corners])

        wanted_min = np.minimum(self.view_min, in_world.min(axis=0) - 1.0)
        wanted_max = np.maximum(self.view_max, in_world.max(axis=0) + 1.0)

        wanted_min = np.floor(wanted_min * 2.0) / 2.0
        wanted_max = np.ceil(wanted_max * 2.0) / 2.0

        if np.any(wanted_min < self.view_min) or np.any(wanted_max > self.view_max):
            self._fit_view(wanted_min, wanted_max)

    def to_panel(self, xy):
        """Habitat-derived world (x east, y north) to map-panel pixels."""
        return (
            int((xy[0] - self.view_min[0]) * self.scale + self.panel_offset[0]),
            int((self.view_max[1] - xy[1]) * self.scale + self.panel_offset[1]),
        )

    # -----------------------------------------------------------------
    # Main loop
    # -----------------------------------------------------------------

    def run(self):
        if self.args.headless:
            os.environ["SDL_VIDEODRIVER"] = "dummy"

        pygame.init()
        self.screen = pygame.display.set_mode(WINDOW)
        pygame.display.set_caption(f"DELTA | {self.world.scene_id}")
        pygame.key.set_repeat(400, 40)
        self.font = pygame.font.SysFont("DejaVu Sans", 15)
        self.small = pygame.font.SysFont("DejaVu Sans", 12)
        self.big = pygame.font.SysFont("DejaVu Sans", 18, bold=True)

        timer = pygame.time.Clock()

        while self.running:
            # At most a tenth of a second per step: when the sim stalls (the GPU
            # busy with depth), time slows rather than the wearer teleporting a
            # metre between two camera frames -- which no real camera sees.
            dt = min(timer.tick(30) / 1000.0, 0.1)
            self.clock += dt

            self._events()
            self._move(dt)

            if self.script is not None:
                self.script.step(dt)

            if self.clock >= self.next_frame:
                self._send_frame()
                self.next_frame = self.clock + 1.0 / self.frame_rate

            self._drain_brain()
            self._draw()

            if self.clock >= self.next_screenshot:
                self.screenshot()
                self.next_screenshot = self.clock + self.args.screenshots

        if self.brain is not None:
            self.brain.close()

        self.world.close()
        pygame.quit()

    # -----------------------------------------------------------------
    # Input
    # -----------------------------------------------------------------

    def _events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False

            elif event.type == pygame.KEYDOWN:
                self._key(event)

            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                self._click(event.pos)

    def _key(self, event):
        key = event.key

        if key == pygame.K_ESCAPE:
            self.running = False
        elif key in (pygame.K_1, pygame.K_KP1):
            if not self.world.flying:
                if self.world.climb(0.0):
                    self.note("Taking off.")
                else:
                    self.note("No room overhead to take off.")
        elif key == pygame.K_c:
            self.command("clear", "Route cleared.")
        elif key == pygame.K_TAB:
            self.command("save", "Saving the map.")
        elif key == pygame.K_g:
            self.autopilot = not self.autopilot
            self.note(f"Autopilot {'on' if self.autopilot else 'off'}.")
        elif key == pygame.K_r:
            self.start_roam()
        elif key == pygame.K_v:
            self.free_camera = not self.free_camera
        elif key == pygame.K_m:
            self.show_truth = not self.show_truth
        elif key == pygame.K_F12:
            self.screenshot()

    def command(self, name, said, **fields):
        self.note(said)

        if self.brain is not None:
            self.brain.send("command", dict(name=name, **fields))

    def _click(self, position):
        """A click on the map panel: plan a route there, on the drone's map.

        The click is in panel pixels, which are Habitat-derived world metres.
        The brain plans in ITS world, which is wherever the drone switched on,
        so the point goes back through the alignment that lines the two up --
        and if the brain has drifted, the route will be planned to where the
        brain thinks that spot is. That is the honest behaviour: the drone can
        only go where its own map says.
        """
        if self.brain is None or not self.alignment.ready:
            return

        x, y = position

        if not (800 <= x < WINDOW[0] and y < TOP):
            return

        world_xy = (
            (x - self.panel_offset[0]) / self.scale + self.view_min[0],
            self.view_max[1] - (y - self.panel_offset[1]) / self.scale,
        )

        # World -> the brain's frame: the inverse of Alignment.point.
        c, s_ = math.cos(-self.alignment.yaw), math.sin(-self.alignment.yaw)
        dx, dy = np.array(world_xy) - self.alignment.offset

        self.command(
            "plan",
            f"Plan a route to {world_xy[0]:+.1f}, {world_xy[1]:+.1f}.",
            xy=[float(c * dx - s_ * dy), float(s_ * dx + c * dy)],
        )

    def note(self, text):
        self.transcript.append(("system", text))

    # -----------------------------------------------------------------
    # Moving the wearer
    # -----------------------------------------------------------------

    def _move(self, dt):
        keys = pygame.key.get_pressed()
        fast = FAST if (keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]) else 1.0

        if self.free_camera:
            self._fly(keys, dt, fast)

        # Altitude: 1 up, 0 down. Neither key moves the drone -- they move the
        # altitude it is flying toward, and settle() closes the gap smoothly,
        # so a take-off is a climb rather than a jump.
        if keys[pygame.K_1] or keys[pygame.K_KP1]:
            self.world.climb(dt)

        if keys[pygame.K_0] or keys[pygame.K_KP0]:
            self.world.descend(dt)

        self.world.settle(dt)

        walk = (keys[pygame.K_UP] - keys[pygame.K_DOWN]) * WALK_SPEED * fast * dt
        turn = (keys[pygame.K_LEFT] - keys[pygame.K_RIGHT]) * TURN_SPEED * fast * dt

        if self.script is not None:
            walk += self.script.walk * dt
            turn += self.script.turn * dt

        if walk or turn:
            self.roam = None

        if self.roam is not None:
            self._follow_points(self.roam, dt)
        elif self.autopilot:
            self._follow_brain_route(dt)

        if turn:
            self.world.turn(turn)

        if walk:
            self.world.walk(walk)

    def _follow_points(self, points, dt):
        """Turn toward the next point, then walk to it, like a person would."""
        while points and math.hypot(points[0].x - self.world.drone.floor_position.x,
                                    points[0].z - self.world.drone.floor_position.z) < 0.2:
            points.pop(0)

        if not points:
            self.roam = None
            return

        wanted = self.world.heading_towards(points[0])
        remaining = frames.wrap(wanted - self.world.heading)
        step = TURN_SPEED * dt

        if abs(remaining) > math.radians(4.0):
            self.world.turn(max(-step, min(step, remaining)))
        else:
            self.world.face(wanted)
            self.world.walk(WALK_SPEED * dt)

    def _follow_brain_route(self, dt):
        """Walk the brain's route the way a wearer following its cues would.

        By what the BRAIN believes -- where it thinks you are and which way
        it thinks you face -- not by the truth. Following the truth would
        hide exactly the errors this is for: with 0.8 m of drift, a walker
        steering by the truth went where the route really was and met the
        kitchen island the brain's map had moved aside.

        The brain's pose is a few frames old by the time it arrives, so the
        motion since that frame is added on, as a person's sense of their own
        turning and stepping would.
        """
        state = self.brain.state if self.brain else None

        if not state or "route" not in state or "pose" not in state:
            return

        believed = self._believed_now(state["pose"])

        if believed is None:
            return

        body, yaw = believed
        target = next(
            (np.array(p) for p in state["route"]["points"][1:]
             if math.dist(p, body) > 0.3),
            None,
        )

        if target is None:
            return

        relative = frames.wrap(math.atan2(target[1] - body[1], target[0] - body[0]) - yaw)
        step = TURN_SPEED * dt

        if abs(relative) > math.radians(10.0):
            self.world.turn(max(-step, min(step, relative)))
        else:
            self.world.turn(0.3 * relative)
            self.world.walk(WALK_SPEED * dt)

    def _believed_now(self, pose):
        """The brain's belief about the wearer, carried forward to now."""
        then = self.truth_by_frame.get(pose["frame"])
        now = self.truth_by_frame.get(self.frame_id - 1)

        if then is None or now is None:
            return None

        # The true motion since the brain's frame, in that frame's own axes...
        c, s = math.cos(then[2]), math.sin(then[2])
        dx, dy = now[0] - then[0], now[1] - then[1]
        forward, left = c * dx + s * dy, -s * dx + c * dy
        turned = frames.wrap(now[2] - then[2])

        # ...applied to where the brain thought the lens was.
        c, s = math.cos(pose["yaw"]), math.sin(pose["yaw"])
        x = pose["x"] + c * forward - s * left
        y = pose["y"] + s * forward + c * left
        yaw = pose["yaw"] + turned

        # From the left lens to the middle of the body, as the brain does it.
        back = float(self.settings.wearer.lens_forward)
        right = self.rig.baseline / 2.0
        body = (x - back * math.cos(yaw) + right * math.sin(yaw),
                y - back * math.sin(yaw) - right * math.cos(yaw))

        return body, yaw

    def start_roam(self):
        """Drive to a random reachable spot -- the quick way to build a map.

        Straight off Habitat's navmesh, which is the answer key. That is fine
        here and only here: roaming is a stand-in for a person at the keyboard,
        not for the drone's own navigation. The brain never sees this route --
        it sees the same camera frames it would see if you were driving.
        """
        pathfinder = self.world.pathfinder
        here = self.world.drone.floor_position

        for _ in range(30):
            goal = self.world.random_floor_point()

            request = habitat_sim.ShortestPath()
            request.requested_start = here
            request.requested_end = mn.Vector3(goal)

            if not pathfinder.find_path(request):
                continue

            if request.geodesic_distance <= 3.0:
                continue

            points = [mn.Vector3(point) for point in request.points]

            # find_path starts at our own feet; driving to where we already
            # are is a no-op.
            if points and (points[0] - here).length() < 0.2:
                points = points[1:]

            if not points:
                continue

            self.roam = points
            self.note(f"Roaming {request.geodesic_distance:.0f} m.")
            return

        self.note("Could not find somewhere to roam to.")

    def _fly(self, keys, dt, fast):
        speed = 2.0 * fast * dt
        look = 90.0 * dt

        self.camera_yaw += (keys[pygame.K_j] - keys[pygame.K_l]) * look
        self.camera_pitch = max(-89.0, min(89.0, self.camera_pitch + (keys[pygame.K_i] - keys[pygame.K_k]) * look))

        yaw = mn.Quaternion.rotation(mn.Deg(self.camera_yaw), mn.Vector3.y_axis())
        forward = yaw.transform_vector(mn.Vector3(0, 0, -1))
        right = yaw.transform_vector(mn.Vector3(1, 0, 0))

        move = (forward * (keys[pygame.K_w] - keys[pygame.K_s])
                + right * (keys[pygame.K_d] - keys[pygame.K_a])
                + mn.Vector3.y_axis() * (keys[pygame.K_z] - keys[pygame.K_x]))

        if move.length() > 0:
            self.world.viewer_node.translation += move.normalized() * speed

    # -----------------------------------------------------------------
    # The cameras, to the brain
    # -----------------------------------------------------------------

    def _send_frame(self):
        seen = self.world.observe_rig()
        self.last_glasses = seen

        position, rotation = seen["true_position"], seen["true_rotation"]
        self.truth_by_frame[self.frame_id] = frames.planar(
            frames.world_from_habitat_camera(position, rotation)
        )
        self.truth_by_frame.pop(self.frame_id - 600, None)

        # Height and what is under the drone, per frame, so the brain's
        # estimate is scored against the moment it was made, not a few frames
        # later -- at 0.8 m/s a 0.15 s lag alone is 12 cm.
        self.height_by_frame[self.frame_id] = (round(self._true_altitude(), 3), self._surface_below())
        self.height_by_frame.pop(self.frame_id - 600, None)

        previous = self.truth_by_frame.get(self.frame_id - 1)

        if previous is not None:
            now = self.truth_by_frame[self.frame_id]
            self.walked += math.dist(previous[:2], now[:2])
            self.turned += abs(math.degrees(frames.wrap(now[2] - previous[2])))

        if self.brain is not None and self.brain.connected:
            fields = {
                "id": self.frame_id,
                "stamp": self.clock,
                # The IMU rides the gimbal, so it turns with the sensor
                # head -- which, for a sphere, is the only heading there is.
                "imu_yaw": self.imu.read(self.world.heading, self.clock),
            }

            # The down-facing rangefinder: how far below it the nearest
            # surface is. The brain works out its own height from this
            # (delta/brain/height.py); it is no longer told the altitude.
            reading = self.rangefinder.read(self.world.drone)

            if reading is not None:
                fields["range_down"] = reading

            # The flight controller's vertical speed (accelerometer and
            # barometer): what lets the brain tell its own climbing from the
            # ground rising under it, as on stairs.
            fields["climb_rate"] = self.climb_rate.read(
                float(self.world.drone.object.translation.y), self.clock
            )

            if self.args.altitude == "oracle":
                fields["true_altitude"] = self._true_altitude()

            # Pitch and roll from gravity, as the IMU chip reports them: the
            # head's true tilt plus an error that wanders slowly -- about half
            # a degree, changing over tens of seconds, as a fused orientation
            # does -- not a fresh jump every frame.
            # The gimbal holds the lenses level, so the TRUE tilt is zero
            # and what the chip reports is only its own error: about half a
            # degree, wandering over tens of seconds, as a fused orientation
            # does -- not a fresh jump every frame.
            dt, settle = 1.0 / self.frame_rate, 20.0
            self._tilt_error += (-self._tilt_error * dt / settle
                                 + 0.5 * math.sqrt(2.0 * dt / settle) * self._tilt_noise.normal(0.0, 1.0, 2))
            fields["imu_pitch"] = float(self._tilt_error[0])
            fields["imu_roll"] = float(self._tilt_error[1])

            # The cage's axle encoder: distance rolled along the heading since
            # the last frame, added to a running total. Only on the ground --
            # in the air the cage does not roll, and the reading is withheld
            # rather than frozen, so the brain knows not to lean on it.
            here = self.world.drone.floor_position

            if self._rolled_from is not None and not self.world.flying:
                self.odometer.roll(float(mn.math.dot(here - self._rolled_from, self.world.forward)))

            self._rolled_from = here

            if not self.world.flying and not self.args.no_odometer:
                fields["odometer"] = self.odometer.total

            arrays = {"left": seen["left"], "right": seen["right"]}

            # Oracle fields only when the experiment asked for them.
            if self.args.pose == "oracle":
                fields["true_position"] = position
                fields["true_rotation"] = rotation

            if self.args.depth == "oracle":
                arrays["true_depth"] = seen["true_depth"]

            self.brain.send("frame", fields, arrays)

        self.frame_id += 1

    def _drain_brain(self):
        if self.brain is None:
            return

        while not self.brain.said.empty():
            kind, text = self.brain.said.get()
            self.transcript.append((kind, text))

            if self.script is not None:
                self.script.heard(kind, text)

        del self.transcript[:-200]

        state = self.brain.state

        if state and "pose" in state:
            pose = state["pose"]
            truth = self.truth_by_frame.get(pose["frame"])
            self.alignment.update(state.get("epoch", 0), (pose["x"], pose["y"], pose["yaw"]), truth)

            if self.args.trace and pose["frame"] != getattr(self, "_traced", None):
                self._traced = pose["frame"]
                self._trace(state, truth)

    def _trace(self, state, truth):
        """One line of --trace: where the drone is, where it thinks it is,
        and how RTAB-Map is doing."""
        error = self._pose_error(state)
        stats = state.get("stats", {})
        loc = stats.get("localization", {})
        here = self.world.drone.floor_position
        height_then = self.height_by_frame.get(
            stats.get("altitude_frame"), (round(self._true_altitude(), 3), self._surface_below())
        )
        line = {
            "t": round(self.clock, 2), "frame": state["pose"]["frame"], "walked": round(self.walked, 2),
            "true": [round(float(here.x), 2), round(-float(here.z), 2), round(float(self.world.heading), 3)],
            "error_m": None if error is None else round(error[0], 3),
            "error_deg": None if error is None else round(error[1], 2),
            "inliers": loc.get("inliers"), "lost": loc.get("lost"), "loops": loc.get("loops"),
            "rejected": loc.get("rejected"), "imu_heading": loc.get("imu_heading"),
            "epoch": state.get("epoch"),
            # How high the drone really is, and how high the brain thinks it
            # is. Logged side by side so flying can be judged the same way as
            # driving: by the gap between the two.
            # Above the floor, at the frame the brain's estimate is from.
            "altitude_true": height_then[0],
            "altitude_brain": stats.get("altitude"),
            # What is under the drone: the true height of the surface straight
            # below (0 over the floor, ~0.75 over a table), and what the brain
            # made of its rangefinder.
            "surface_true": height_then[1],
            "surface_brain": stats.get("surface_below"),
            "flying": bool(self.world.flying),
            # The script's current {"label": ...}: which motion this is.
            "label": self.script.label if self.script is not None else None,
        }

        with open(self.args.trace, "a") as out:
            out.write(json.dumps(line) + "\n")

    # -----------------------------------------------------------------
    # Drawing
    # -----------------------------------------------------------------

    def _draw(self):
        screen = self.screen
        screen.fill((14, 16, 20))

        self._draw_viewer()
        self._draw_map()
        self._draw_feeds()
        self._draw_transcript()
        self._draw_status()

        pygame.draw.line(screen, (60, 64, 72), (800, 0), (800, TOP), 2)
        pygame.draw.line(screen, (60, 64, 72), (0, TOP), (WINDOW[0], TOP), 2)

        pygame.display.flip()

    def _place_chase_camera(self):
        """Behind and above the drone, looking at it -- at any height, and
        never through a wall.

        The camera wants to be CHASE_BACK behind the drone and CHASE_UP above
        it (less in the air, where ceilings are close). A ray from the drone
        out to that spot finds the first wall, ceiling or cupboard in the
        way, and the camera stops just short of it, so the drone stays in
        view with a wall at its back. It then turns to look AT the drone,
        so a climb tilts the view up instead of leaving the frame.

        Pulling in is immediate -- a wall cannot be seen through for even
        one frame -- and easing back out is gradual, so the view does not
        jump about as the drone passes doorways.
        """
        drone = self.world.drone
        target = drone.position
        up = CHASE_UP_FLYING if drone.flying else CHASE_UP

        wanted = target + self.world.forward * -CHASE_BACK + mn.Vector3(0.0, up, 0.0)
        offset = wanted - target
        length = offset.length()
        direction = offset / length

        hit = first_hit(self.world.simulator, target, direction, length + CHASE_MARGIN,
                        drone.object.object_id)
        reach = length if hit is None else max(CHASE_NEAREST, hit - CHASE_MARGIN)
        reach = min(reach, length)

        # Ease out, never in.
        previous = getattr(self, "_chase_reach", reach)
        reach = reach if reach < previous else previous + 0.15 * (reach - previous)
        self._chase_reach = reach

        camera = target + direction * reach
        look = target - camera

        node = self.world.viewer_node
        node.translation = camera
        node.rotation = mn.Quaternion.rotation(
            mn.Rad(math.atan2(-look.x, -look.z)), mn.Vector3.y_axis()
        ) * mn.Quaternion.rotation(
            mn.Rad(math.atan2(look.y, math.hypot(look.x, look.z))), mn.Vector3.x_axis()
        )

    def _outline_drone(self):
        """Ring the drone in the 3D view, and its shadow when it is flying.

        The glasses drew the visor's outline here. A 30 cm sphere on a grey
        floor is just as easy to lose, and when it is airborne there is
        nothing in the picture to say HOW high -- so the shadow and the line
        down to it are not decoration, they are the altitude readout.
        """
        lines = self.world.simulator.get_debug_line_render()
        drone = self.world.drone

        green = mn.Color4(0.24, 0.86, 0.35, 1.0)
        blue = mn.Color4(0.31, 0.78, 1.0, 1.0)

        lines.draw_circle(drone.position, drone.radius, blue if drone.flying else green)

        if not drone.flying:
            return

        ground = drone.floor_position
        shadow = mn.Vector3(ground.x, ground.y + 0.01, ground.z)

        lines.draw_circle(shadow, drone.radius, blue)
        lines.draw_transformed_line(shadow, drone.position, blue)

    def _draw_viewer(self):
        if not self.free_camera:
            self._place_chase_camera()
        else:
            self.world.viewer_node.rotation = mn.Quaternion.rotation(
                mn.Deg(self.camera_yaw), mn.Vector3.y_axis()
            ) * mn.Quaternion.rotation(mn.Deg(self.camera_pitch), mn.Vector3.x_axis())

        self._outline_drone()
        image = self.world.observe_viewer()
        surface = pygame.surfarray.make_surface(np.transpose(image, (1, 0, 2)))
        self.screen.blit(pygame.transform.smoothscale(surface, (800, TOP)), (0, 0))

        label = "Free camera (WASD, IJKL, Z/X) - V to follow" if self.free_camera else "Behind you - V for free camera"
        self._text(label, (12, 10), self.big, (240, 240, 240))

    def _draw_map(self):
        canvas = np.zeros((TOP, 800, 3), np.uint8)
        canvas[:] = (20, 23, 28)

        if self.show_truth:
            truth = cv2.warpAffine(self.truth_layer, self.truth_affine, (800, TOP), flags=cv2.INTER_NEAREST)
            canvas = np.maximum(canvas, truth)

        state = self.brain.state if self.brain else None
        arrays = self.brain.state_arrays if self.brain else {}

        if state:
            self._widen_view_for(state, arrays)

            if self.show_truth:
                # The view may have just grown, so the truth has to be
                # redrawn at the new scale rather than the one above.
                canvas[:] = (20, 23, 28)
                truth = cv2.warpAffine(self.truth_layer, self.truth_affine,
                                       (800, TOP), flags=cv2.INTER_NEAREST)
                canvas = np.maximum(canvas, truth)

        if state and "grid" in state and "grid" in arrays and self.alignment.ready:
            grid = arrays["grid"]
            origin = state["grid"]["origin"]
            resolution = state["grid"]["resolution"]

            # Grid cell (col, row) -> brain world -> true world -> panel.
            a = self.alignment.matrix()
            cell = np.array([[resolution, 0.0, origin[0]], [0.0, resolution, origin[1]]])
            world = a[:, :2] @ cell
            world[:, 2] += a[:, 2]
            panel = np.array([[self.scale, 0.0, -self.view_min[0] * self.scale + self.panel_offset[0]],
                              [0.0, -self.scale, self.view_max[1] * self.scale + self.panel_offset[1]]])
            affine = panel[:, :2] @ world
            affine[:, 2] += panel[:, 2]

            colours = np.zeros(grid.shape + (3,), np.uint8)
            colours[grid == 1] = FREE_COLOUR
            colours[grid == 2] = OCCUPIED_COLOUR

            # Cells the drone decided are open to the sky. UNROOFED is 1 in
            # occupancy.py; cells it has no opinion on keep the indoor
            # colours, because "not yet established" is not "outside".
            roof = arrays.get("roof")

            if roof is not None and roof.shape == grid.shape:
                outside = roof == 1
                colours[(grid == 1) & outside] = OUTDOOR_COLOUR
                colours[(grid == 2) & outside] = OUTDOOR_WALL_COLOUR
            mask = cv2.warpAffine((grid > 0).astype(np.uint8), affine, (800, TOP), flags=cv2.INTER_NEAREST)
            warped = cv2.warpAffine(colours, affine, (800, TOP), flags=cv2.INTER_NEAREST)
            canvas[mask > 0] = warped[mask > 0]

        surface = pygame.surfarray.make_surface(np.transpose(canvas, (1, 0, 2)))
        self.screen.blit(surface, (800, 0))

        offset = np.array([800, 0])

        if state and self.alignment.ready:
            for place in state.get("places", []):
                p = np.array(self.to_panel(self.alignment.point(place["xy"]))) + offset
                pygame.draw.polygon(self.screen, PLACE_COLOUR, _star(p, 9))
                self._text(place["name"], p + (10, -8), self.small, PLACE_COLOUR)

            # Only trusted things, and names only for the well seen: a walk
            # through a house glimpses a great deal.
            for thing in state.get("things", []):
                if not thing["confirmed"]:
                    continue

                p = np.array(self.to_panel(self.alignment.point(thing["xy"]))) + offset
                pygame.draw.circle(self.screen, THING_COLOUR, p, 4)

                if thing["seen"] >= 6:
                    self._text(thing["id"], p + (6, -7), self.small, THING_COLOUR)

            if "route" in state:
                points = [np.array(self.to_panel(self.alignment.point(xy))) + offset
                          for xy in state["route"]["points"]]

                if len(points) >= 2:
                    pygame.draw.lines(self.screen, ROUTE_COLOUR, False, points, 3)

                target = np.array(self.to_panel(self.alignment.point(state["route"]["target"]))) + offset
                pygame.draw.circle(self.screen, TARGET_COLOUR, target, 8, 2)

            if "pose" in state:
                pose = state["pose"]
                p = np.array(self.to_panel(self.alignment.point((pose["x"], pose["y"])))) + offset
                heading = pose["yaw"] + self.alignment.yaw
                tip = p + 16 * np.array([math.cos(heading), -math.sin(heading)])
                pygame.draw.line(self.screen, DRONE_BELIEVED, p, tip, 3)
                pygame.draw.circle(self.screen, DRONE_BELIEVED, p, 6)

        # Where the wearer really is.
        lens = self.truth_by_frame.get(self.frame_id - 1)

        if lens is not None:
            p = np.array(self.to_panel(lens[:2])) + offset
            tip = p + 16 * np.array([math.cos(lens[2]), -math.sin(lens[2])])
            pygame.draw.circle(self.screen, DRONE_TRUE, p, 7, 2)
            pygame.draw.line(self.screen, DRONE_TRUE, p, tip, 2)

        self._text("What the drone believes", (812, 10), self.big, (240, 240, 240))
        self._text("green: really here   orange: believed here   "
                   "olive: the drone worked out this is outdoors   "
                   "M: true layout "
                   + ("on" if self.show_truth else "off"),
                   (812, 34), self.small, (170, 175, 185))

        if self.brain is None:
            self._text("Running without a brain (--no-brain)", (812, 56), self.font, (200, 120, 120))
        elif not self.brain.connected:
            message = self.brain.error or "Waiting for the brain to start..."
            self._text(message, (812, 56), self.font, (220, 180, 90))
        elif not self.brain.ready:
            self._text("Brain loading models...", (812, 56), self.font, (220, 180, 90))

    def _draw_feeds(self):
        """The left lens, and the depth the brain made of it.

        The glasses drew YOLOE's boxes and the hazard grid over the lens.
        DELTA has neither, so the lens is shown as the cameras see it -- which
        is also the honest thing, since that picture plus the depth beside it
        is the entire input the map is built from.
        """
        if self.last_glasses is None:
            return

        state = self.brain.state if self.brain else None
        arrays = self.brain.state_arrays if self.brain else {}

        width, height = 512, 320

        self._blit(self.last_glasses["left"], (8, STRIP_TOP + 22), (width, height))
        self._text("Left lens", (8, STRIP_TOP + 2), self.font, (210, 210, 210))

        if "depth" in arrays:
            depth = cv2.applyColorMap(255 - arrays["depth"], cv2.COLORMAP_VIRIDIS)[:, :, ::-1]
            self._blit(depth, (528, STRIP_TOP + 22), (width, height))

            scale = (state or {}).get("stats", {}).get("floor_scale")
            label = "Depth the brain is using (0-6 m)"

            if scale:
                label += f"   floor scale x{scale:.2f}"

            self._text(label, (528, STRIP_TOP + 2), self.font, (210, 210, 210))

    def _draw_transcript(self):
        x, top, bottom = 1050, STRIP_TOP + 2, STATUS_TOP - 30
        width = WINDOW[0] - x - 8

        self._text("Log", (x, top), self.font, (210, 210, 210))

        lines = []

        for kind, text in self.transcript[-40:]:
            for line in _wrap(text, self.small, width):
                lines.append((kind, line))

        y = bottom - 16

        for kind, line in reversed(lines):
            if y < top + 22:
                break

            self._text(line, (x, y), self.small, KIND_COLOURS.get(kind, (200, 200, 200)))
            y -= 16

        box = pygame.Rect(x, STATUS_TOP - 26, width, 22)
        pygame.draw.rect(self.screen, (26, 28, 34), box)
        self._text(
            "arrows drive   1/0 fly, land   click map: plan   C: clear   "
            "G: autopilot   R: roam   Tab: save",
            (x + 4, box.y + 4), self.small, (130, 135, 145),
        )

    def _draw_status(self):
        state = self.brain.state if self.brain else None

        if not state:
            # Say why there is no map, rather than showing an empty panel and
            # leaving it to be guessed: a brain that is still loading looks
            # exactly like one that never started.
            if self.brain is None:
                text = "No brain (--no-brain): the house only, nothing is mapped."
            elif self.brain.error:
                text = f"Brain not reachable: {self.brain.error}. See outputs/logs/brain.log."
            elif not self.brain.connected:
                text = "Waiting for the brain to start... (if this lasts, see outputs/logs/brain.log)"
            else:
                text = "Brain connected, loading models..."

            self._text(text, (8, STATUS_TOP + 5), self.small, (255, 180, 90))
            return

        stats = state.get("stats", {})
        rates = stats.get("rates", {})
        ms = stats.get("ms", {})
        loc = stats.get("localization", {})
        sources = stats.get("sources", {})

        parts = [
            f"pose {sources.get('pose')}  depth {sources.get('depth')}",
            f"frames {rates.get('frames', 0):.0f}/s  perception {rates.get('perception', 0):.1f}/s "
            f"(DA3 {ms.get('depth', 0)} ms, map {ms.get('map', 0)} ms)",
        ]

        self._draw_height(stats)

        roofed = stats.get("roofed")

        if roofed:
            parts.append(
                f"mapped {roofed['indoor']} indoor / {roofed['outdoor']} outdoor cells"
            )

        if sources.get("pose") == "rtabmap":
            parts.append(f"RTAB-Map {loc.get('ms', 0):.0f} ms, nodes {loc.get('nodes', 0)}, "
                         f"loops {loc.get('loops', 0)} kept / {loc.get('rejected', 0)} rejected")

        error = self._pose_error(state)

        if error is not None:
            parts.append(f"pose error {error[0]:.2f} m, {error[1]:.1f} deg")

        if stats.get("relocalizing"):
            parts.append("RECOGNISING LAST SESSION'S MAP...")

        self._text("   |   ".join(parts), (8, STATUS_TOP + 5), self.small, (175, 180, 190))

    def _draw_height(self, stats):
        """Two heights, the brain's next to the truth, on the 3D view:

            above the floor        how high the drone is in the room
            above what is below    how far it would drop to land -- the
                                   same number over the floor, less over
                                   a table

        The truth is the moment the brain's estimate is from, so a climb
        does not show as an error just because the picture is a frame old.
        """
        altitude = stats.get("altitude")

        if altitude is None:
            return

        surface = stats.get("surface_below") or 0.0
        true_altitude, true_surface = self.height_by_frame.get(
            stats.get("altitude_frame"), (self._true_altitude(), self._surface_below())
        )
        true_surface = max(0.0, true_surface or 0.0)

        rows = [
            ("HEIGHT", "brain", "truth"),
            ("above the floor", f"{altitude:.2f} m", f"{true_altitude:.2f} m"),
            ("above what is below", f"{altitude - surface:.2f} m", f"{true_altitude - true_surface:.2f} m"),
            ("what is below (0 = floor)", f"{surface:.2f} m", f"{true_surface:.2f} m"),
        ]

        box = pygame.Surface((380, 92), pygame.SRCALPHA)
        box.fill((10, 12, 16, 200))
        self.screen.blit(box, (8, 34))

        for i, (name, brain, truth) in enumerate(rows):
            y = 40 + i * 21
            colour = (150, 155, 165) if i == 0 else (235, 235, 235)
            self._text(name, (16, y), self.font, colour)
            self._text(brain, (230, y), self.font, (120, 200, 255) if i else colour)
            self._text(truth, (310, y), self.font, (160, 220, 160) if i else colour)

    def _true_altitude(self):
        """How high the drone's underside truly is above the FLOOR.

        Not `world.altitude`: in the air that is measured from the nearest
        navmesh surface in 3D, which beside a counter or a bed is the top of
        the furniture, not the floor below -- it once read 0.18 m for a drone
        a metre up. The floor here is the one the drone took off from.
        """
        centre = self.world.drone.object.translation
        return max(0.0, float(centre.y) - float(self.world.drone.radius) - self.floor_height)

    def _surface_below(self):
        """True height above the floor of whatever is straight under the
        drone -- for scoring the brain's height estimate, never sent to it."""
        centre = self.world.drone.object.translation
        down = first_hit(self.world.simulator, centre, mn.Vector3(0.0, -1.0, 0.0), 10.0,
                         self.world.drone.object.object_id)

        if down is None:
            return None

        return round(float(centre.y) - down - self.floor_height, 3)

    def _pose_error(self, state):
        if "pose" not in state or not self.alignment.ready:
            return None

        pose = state["pose"]
        truth = self.truth_by_frame.get(pose["frame"])

        if truth is None:
            return None

        believed = self.alignment.point((pose["x"], pose["y"]))
        yaw = frames.wrap(pose["yaw"] + self.alignment.yaw - truth[2])

        return math.dist(believed, truth[:2]), abs(math.degrees(yaw))

    # -----------------------------------------------------------------

    def _blit(self, image, position, size):
        surface = pygame.surfarray.make_surface(np.transpose(np.ascontiguousarray(image[:, :, :3]), (1, 0, 2)))
        self.screen.blit(pygame.transform.smoothscale(surface, size), position)

    def _text(self, text, position, font, colour):
        self.screen.blit(font.render(text, True, colour), (int(position[0]), int(position[1])))

    def screenshot(self, name=None):
        folder = PROJECT_ROOT / "outputs" / "screenshots"
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / f"{name or time.strftime('%Y%m%d-%H%M%S')}.png"
        pygame.image.save(self.screen, str(path))
        print(f"Saved {path}")

    def score_map(self, name):
        """Score the brain's live map against the navmesh, for long-walk tests.

        Same rules as tools/check_map.py, but on the map the brain actually
        built -- its own pose, not Habitat's -- lined up the way the map panel
        is: once, never refitted, so drift counts against it. Appends a line
        to outputs/eval/live_map.jsonl and saves a picture:

            grey    true floor nobody mapped       blue    free, correct
            red     free but inside furniture      white   solid, correct
            orange  solid on clear floor (phantom)

        The thin grey frame is the navmesh's rectangle. The picture grows to
        hold the brain's whole map, so anything outside the frame is map the
        brain put where no floor is.
        """
        state = self.brain.state if self.brain else None
        arrays = self.brain.state_arrays if self.brain else {}

        if not (state and "grid" in state and "grid" in arrays and self.alignment.ready):
            print(f"[score] {name}: no map from the brain yet", flush=True)
            return

        if self._navigable is None:
            self._navigable = maps.get_topdown_map(
                self.world.pathfinder, self.floor_height, meters_per_pixel=0.05, draw_border=False
            ) == 1
            lower, _ = self.world.pathfinder.get_bounds()
            self._nav_lower = np.array([lower.x, lower.z])

        resolution, lower = 0.05, self._nav_lower

        # Grid cell -> brain world -> true world -> navmesh pixel (x, z = -north).
        def compose(p, q):
            out = p[:, :2] @ q
            out[:, 2] += p[:, 2]
            return out

        g = state["grid"]
        cell = np.array([[g["resolution"], 0.0, g["origin"][0]], [0.0, g["resolution"], g["origin"][1]]])
        to_pixel = np.array([[1 / resolution, 0.0, -lower[0] / resolution],
                             [0.0, -1 / resolution, -lower[1] / resolution]])
        affine = compose(to_pixel, compose(self.alignment.matrix(), cell))

        # Score on a picture big enough for the brain's whole map, not just
        # the navmesh's rectangle. Warped onto the navmesh picture alone, any
        # cell the brain put beyond the house's edge -- drift, or a lookalike
        # closure that flung a room outside -- fell off and was never counted:
        # the further wrong the map went, the less of it was scored. The
        # padding is empty, so whatever lands there is off the true floor.
        rows, cols = arrays["grid"].shape[:2]
        corners = np.array([[0, 0, 1], [cols, 0, 1], [0, rows, 1], [cols, rows, 1]], float) @ affine.T
        truth_height, truth_width = self._navigable.shape

        left = max(0, int(np.ceil(-corners[:, 0].min())))
        top = max(0, int(np.ceil(-corners[:, 1].min())))
        right = max(0, int(np.ceil(corners[:, 0].max())) - truth_width)
        bottom = max(0, int(np.ceil(corners[:, 1].max())) - truth_height)
        padding = ((top, bottom), (left, right))

        navigable = np.pad(self._navigable.astype(np.uint8), padding)
        inside = np.pad(np.ones((truth_height, truth_width), bool), padding)
        affine[:, 2] += (left, top)
        height, width = navigable.shape

        def disc(metres):
            size = 2 * int(round(metres / resolution)) + 1
            return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))

        clear = cv2.erode(navigable, disc(0.10)).astype(bool)
        floor = cv2.dilate(navigable, disc(0.35)).astype(bool)

        classes = cv2.warpAffine(arrays["grid"].astype(np.uint8), affine, (width, height),
                                 flags=cv2.INTER_NEAREST, borderValue=0)

        free, solid = classes == 1, classes == 2
        seen_clear = clear & (classes != 0)
        error = self._pose_error(state)

        result = {
            "name": name,
            "time_s": round(self.clock, 1),
            "walked_m": round(self.walked, 1),
            "turned_deg": round(self.turned),
            "coverage": round(float((free & clear).sum() / max(clear.sum(), 1)), 4),
            "phantom": round(float((solid & clear).sum() / max(seen_clear.sum(), 1)), 4),
            "unsafe": round(float((free & ~floor).sum() / max(free.sum(), 1)), 4),
            "free_m2": round(float((free & clear).sum()) * resolution ** 2, 1),
            # Mapped (free or solid) outside the navmesh's whole rectangle.
            "beyond_truth_m2": round(float(((classes != 0) & ~inside).sum()) * resolution ** 2, 1),
            "pose_error_m": round(error[0], 3) if error else None,
            "heading_error_deg": round(error[1], 2) if error else None,
            "loops": state.get("stats", {}).get("localization", {}).get("loops"),
            # The lenses sit at the centre of the sphere: its radius when it is
            # on the ground, plus however high it is flying.
            "lens_height_true": round(
                float(self.drone_config["diameter"]) / 2.0 + self._true_altitude(), 3
            ),
            "altitude_true": round(self._true_altitude(), 3),
            "lens_height_brain": state.get("stats", {}).get("lens_height"),
            "map_check": state.get("stats", {}).get("map_localize"),
            # "Stop." warnings said since the last score.
            "stops": sum(1 for kind, text in self.transcript[getattr(self, "_scored_upto", 0):]
                         if text.endswith("Stop.")),
        }
        self._scored_upto = len(self.transcript)
        print(f"[score] {json.dumps(result)}", flush=True)

        folder = PROJECT_ROOT / "outputs" / "eval"
        (folder / "live_map").mkdir(parents=True, exist_ok=True)

        with open(folder / "live_map.jsonl", "a") as log:
            log.write(json.dumps(result) + "\n")

        picture = np.full((height, width, 3), 15, np.uint8)
        picture[navigable > 0] = (60, 60, 60)
        picture[free] = (170, 110, 60)
        picture[solid] = (230, 230, 230)
        picture[free & ~floor] = (40, 40, 230)
        picture[solid & clear] = (0, 150, 255)
        cv2.rectangle(picture, (left, top), (left + truth_width - 1, top + truth_height - 1), (90, 90, 90), 1)
        cv2.imwrite(str(folder / "live_map" / f"{name}.png"), picture)


# ---------------------------------------------------------------------
# Scripts: the same app, driven from a file
# ---------------------------------------------------------------------


class Script:
    """Steps from a JSON list, one after another:

        {"wait": 3}                         do nothing for 3 s
        {"walk": 2.0}                       walk forward for 2 s
        {"turn": 90}                        turn left 90 degrees (negative: right)
        {"turn": 360, "speed": 120}         ...at 120 degrees a second (default 50)
        {"label": "climb"}                  name what follows, in the --trace,
                                            until the next label (tools/segments.py)
        {"say": "take me to the fridge"}    type it to the glasses
        {"key": "identify"}                 Space; also "stop"
        {"roam": 3}                         walk to 3 random spots in turn
        {"autopilot": 60}                   follow the brain's route, up to 60 s,
                                            until the glasses say "arrived"
        {"wait_for": "ready", "timeout": 300}   until the brain is ready
        {"screenshot": "name"}
        {"score": "name"}                   score the live map (outputs/eval/)
        {"fly": 1.2}                        take off if needed and fly to 1.2 m
                                            (underside above the floor); done on
                                            arrival, or when a ceiling stops it
        {"land": true}                      come down and land
        {"quit": true}
    """

    def __init__(self, path, app):
        self.steps = json.loads(Path(path).read_text())
        self.app = app
        self.walk = 0.0
        self.turn = 0.0
        self._index = -1
        self._elapsed = 0.0
        self._target = 0.0
        self._done = True
        self._roams = 0
        self._arrived = False
        self._turn_speed = TURN_SPEED
        self.label = None

    def heard(self, kind, text):
        if "arrived" in text.lower() or "can't find" in text.lower():
            self._arrived = True

    def step(self, dt):
        app = self.app
        self._elapsed += dt

        if not self._done:
            self._continue(dt)
            return

        self._index += 1
        self.walk = self.turn = 0.0

        if self._index >= len(self.steps):
            app.running = False
            return

        step = self.steps[self._index]
        self._elapsed = 0.0
        self._done = False
        print(f"[script] {step}", flush=True)

        if "say" in step:
            app.ask(step["say"])
            self._done = True
        elif "key" in step:
            app.command(step["key"], step["key"])
            self._done = True
        elif "screenshot" in step:
            app.screenshot(step["screenshot"])
            self._done = True
        elif "score" in step:
            app.score_map(step["score"])
            self._done = True
        elif "label" in step:
            self.label = str(step["label"])
            self._done = True
        elif "quit" in step:
            app.running = False
        elif "roam" in step:
            self._roams = int(step["roam"])
            self._roam_mark = None
        elif "autopilot" in step:
            self._arrived = False
            app.autopilot = True
        elif "turn" in step:
            self._target = math.radians(float(step["turn"]))
            self._turn_speed = math.radians(float(step.get("speed", math.degrees(TURN_SPEED))))
        elif "fly" in step:
            drone = app.world.drone

            if not drone.flying and not drone.take_off():
                app.note("Script: no room overhead to take off.")
                self._done = True
            else:
                drone.target_altitude = float(step["fly"])
                self._last_altitude = drone.altitude
                self._still_for = 0.0
        elif "land" in step:
            app.world.drone.land()

    def _continue(self, dt):
        app = self.app
        step = self.steps[self._index]

        if "fly" in step:
            drone = app.world.drone

            # Arrived -- or stopped moving, which is what a ceiling or a beam
            # overhead looks like: the drone holds below it rather than
            # pressing on, and the step must not wait for it for ever.
            moved = abs(drone.altitude - self._last_altitude)
            self._last_altitude = drone.altitude
            self._still_for = self._still_for + dt if moved < 1e-4 else 0.0

            self._done = (
                abs(drone.altitude - drone.target_altitude) < 0.03
                or self._still_for > 1.0
                or self._elapsed >= float(step.get("timeout", 20.0))
            )

        elif "land" in step:
            self._done = (
                not app.world.drone.flying
                or self._elapsed >= float(step.get("timeout", 20.0))
            )

        elif "wait" in step:
            self._done = self._elapsed >= float(step["wait"])

        elif "walk" in step:
            self.walk = WALK_SPEED
            self._done = self._elapsed >= float(step["walk"])

        elif "turn" in step:
            if abs(self._target) < 1e-3:
                self.turn = 0.0
                self._done = True
            else:
                amount = math.copysign(min(self._turn_speed * dt, abs(self._target)), self._target)
                self._target -= amount
                self.turn = amount / dt

        elif "roam" in step:
            # Stuck: in the air, a route planned on the floor can run into a
            # cupboard the drone cannot fly over, and the step used to wait
            # for it for ever. No progress for 8 s (a half-turn takes 3.6)
            # gives up on that spot and moves on to the next.
            here = app.world.drone.floor_position

            if app.roam is not None:
                if self._roam_mark is None or math.hypot(
                    here.x - self._roam_mark[0], here.z - self._roam_mark[1]
                ) > 0.1:
                    self._roam_mark = (here.x, here.z, self._elapsed)
                elif self._elapsed - self._roam_mark[2] > 8.0:
                    app.note("Script: stuck, skipping to the next spot.")
                    app.roam = None
                    self._roam_mark = None

            if app.roam is None:
                if self._roams <= 0:
                    self._done = True
                else:
                    self._roams -= 1
                    app.start_roam()

        elif "autopilot" in step:
            if self._arrived or self._elapsed >= float(step["autopilot"]):
                app.autopilot = False
                self._done = True

        elif "wait_for" in step:
            if app.brain is not None and app.brain.error:
                print(f"[script] the brain went away: {app.brain.error}", flush=True)
                app.running = False
                return

            ready = app.brain is not None and app.brain.ready
            self._done = ready or self._elapsed >= float(step.get("timeout", 300))


def _wrap(text, font, width):
    words, lines, line = text.split(), [], ""

    for word in words:
        candidate = f"{line} {word}".strip()

        if font.size(candidate)[0] <= width:
            line = candidate
        else:
            lines.append(line)
            line = word

    if line:
        lines.append(line)

    return lines


def _star(centre, radius):
    points = []

    for i in range(10):
        angle = -math.pi / 2 + i * math.pi / 5
        r = radius if i % 2 == 0 else radius * 0.45
        points.append((centre[0] + r * math.cos(angle), centre[1] + r * math.sin(angle)))

    return points


def main():
    args = parse_arguments()

    if args.headless:
        os.environ["SDL_VIDEODRIVER"] = "dummy"

    DroneApp(args).run()


if __name__ == "__main__":
    main()
