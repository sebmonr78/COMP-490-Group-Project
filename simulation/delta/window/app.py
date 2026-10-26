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
    top middle     the brain's own 2D map, drawn over the TRUE layout in grey
                   (M hides the truth). Green ring: where the drone really
                   is. Orange: where the brain thinks it is. Blue: the route.
    top right      the brain's 3D replica, ceiling off
    bottom         the left lens, the depth the brain is using, the height
                   box (brain next to truth), and the log

What reads the truth -- the grey layout, the green rings, the truth column,
the scores -- is all in examiner.py; none of it is sent to the brain.

Keys:

    Up / Down        forward / back                 Shift: faster
    Left / Right     turn on the spot
    1 / 0            take off and climb / descend and land
    I / K / O        tilt the cameras up / down / level
    R                roam: drive to a random spot (to build the map quickly)
    G                autopilot: follow the brain's own route
    click the map    plan a route there, on the drone's own map
    C                clear the route
    V                camera: behind the drone / free flight (WASD, IJKL, Z/X)
    M / N            show / hide the true layout / the rooms
    drag, scroll     turn and zoom the 3D replica; F: follow the drone or not
    [ ]              the replica without its ceiling, a flat cut, everything
    B                build the 3D replica now
    Tab              tell the brain to save the map
    F12              screenshot to outputs/screenshots/
    Esc              quit (the brain saves the map)
"""

import argparse
import collections
import contextlib
import math
import os
import time
import warnings
from pathlib import Path

os.environ.setdefault("SDL_FRAMEBUFFER_ACCELERATION", "0")

PROJECT_ROOT = Path(__file__).resolve().parents[2]

import cv2  # noqa: E402
import habitat_sim  # noqa: E402
import magnum as mn  # noqa: E402
import numpy as np  # noqa: E402
import pygame  # noqa: E402

from delta import frames  # noqa: E402
from delta.brain import config as brain_config  # noqa: E402
from delta.drone import FAST, TILT_SPEED, TURN_SPEED, WALK_SPEED  # noqa: E402
from delta.rig_process import RigProcess  # noqa: E402
from delta.sim import ClimbRate, Imu, Odometer, Rangefinder, World, first_hit  # noqa: E402
from delta.window.background import Background  # noqa: E402
from delta.window.colours import (  # noqa: E402
    DOOR_COLOUR, DRONE_BELIEVED, DRONE_TRUE, FREE_COLOUR, KIND_COLOURS, OCCUPIED_COLOUR,
    PLACE_COLOUR, ROLL_UNDER_COLOUR, ROOM_COLOURS, ROUTE_COLOUR, TARGET_COLOUR, THING_COLOUR,
)
from delta.window.examiner import Alignment, Examiner  # noqa: E402
from delta.window.link import BrainLink  # noqa: E402
from delta.window.monitor import Monitor  # noqa: E402
from delta.window.script import Script  # noqa: E402
from delta.window.view3d import OrbitView, Painter  # noqa: E402

# The window is laid out for the screen it opens on (layout(), once, before
# anything is drawn), and drawn at that size: scaling a fixed-size drawing up
# to a 4K screen cost 12-15 ms a frame. Sizes in this file are in points, and
# px() turns them into pixels.
UI = 1.0
WINDOW = (1920, 1080)


def px(points):
    """Points -> pixels. A point is UI pixels: 1.5 on a 4K screen, as at
    Windows' 150%, so the text comes out the size the rest of the screen's
    is."""
    return int(round(points * UI))


def layout(size, ui=1.0):
    """Lay the window out for `size` pixels, `ui` pixels to a point.

    Three panels across the top: the 3D view, the 2D map, the 3D replica.
    Along the bottom, the cameras' pictures at their own 16:10, then the log
    panel with the height box at the top of it; the status line under all.
    """
    global UI, WINDOW, PANEL, TOP, MAP_X, REPLICA_X, STRIP_TOP, STATUS_TOP
    global FEED, LOG_LEFT, HEIGHT_BOX_TOP, HEIGHT_BOX_HEIGHT
    global RESOURCES_LEFT, RESOURCES_TOP, RESOURCES_WIDTH, RESOURCES_HEIGHT, LOG_CLEAR

    UI, WINDOW = float(ui), (int(size[0]), int(size[1]))
    width, height = WINDOW

    PANEL = width // 3
    TOP = int(height * 0.555)
    MAP_X, REPLICA_X = PANEL, 2 * PANEL

    STRIP_TOP = TOP + px(6)
    STATUS_TOP = height - px(26)

    # As big as the strip is tall, leaving the log at least 560 points.
    feed_height = STATUS_TOP - px(8) - (STRIP_TOP + px(22))
    feed_width = min(feed_height * 1.6, (width - px(24) - px(560)) / 2.0)
    FEED = (int(feed_width), int(feed_width / 1.6))
    LOG_LEFT = px(16) + 2 * FEED[0] + px(10)
    HEIGHT_BOX_TOP = STRIP_TOP + px(24)
    HEIGHT_BOX_HEIGHT = px(131)

    # The resources box: beside the height box where the log is wide enough
    # (a 4K screen), under it where it is not.
    RESOURCES_WIDTH, RESOURCES_HEIGHT = px(365), px(10 * 18 + 8)

    if width - LOG_LEFT - px(8) >= px(380 + 10) + RESOURCES_WIDTH:
        RESOURCES_LEFT, RESOURCES_TOP = LOG_LEFT + px(390), HEIGHT_BOX_TOP
    else:
        RESOURCES_LEFT, RESOURCES_TOP = LOG_LEFT, HEIGHT_BOX_TOP + HEIGHT_BOX_HEIGHT + px(6)

    # The log's lines stop above both boxes.
    LOG_CLEAR = max(HEIGHT_BOX_TOP + HEIGHT_BOX_HEIGHT, RESOURCES_TOP + RESOURCES_HEIGHT) + px(6)


layout(WINDOW)


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


# The window runs up to this many loops a second. The view behind the drone
# is drawn on every one; the rest of the window in turn (see _draw).
FPS_CAP = 60
BACKGROUND = (14, 16, 20)

# The replica panel takes the ceiling off so the rooms can be seen into, like
# a dollhouse: every cube within CEILING_MARGIN of the ceiling the BRAIN found
# over it (or above) is not drawn -- the brain keeps them all. [ and ] step
# through, from low to high: a flat cut at any height, the ceiling off (the
# start), and everything drawn.
CEILING_MARGIN = 0.15
REPLICA_CUT_STEP = 0.1
REPLICA_CUT_TOP = 3.0


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
    parser.add_argument("--no-camera-process", action="store_true",
                        help="draw the cameras' pictures in the window instead of a process of their own")
    parser.add_argument("--replica", choices=["live", "later"], default=None,
                        help="build the 3D replica as it goes, or only on B / at the end (default: brain.json)")
    parser.add_argument("--ui", type=float, default=None,
                        help="pixels per point, to make everything bigger or smaller (default: from the screen)")
    parser.add_argument("--profile", action="store_true",
                        help="print where the window's and the brain's time goes, every 10 s")
    return parser.parse_args()


class Stopwatch:
    """Where the window's time goes: milliseconds per loop for each part,
    over the last stretch. One thread does the sim, the drawing and all, so
    whatever is slow here slows the whole simulation."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.parts, self.loops, self.started = {}, 0, time.monotonic()

    @contextlib.contextmanager
    def part(self, name):
        start = time.perf_counter()

        try:
            yield
        finally:
            self.parts[name] = self.parts.get(name, 0.0) + time.perf_counter() - start

    def rate(self):
        return self.loops / max(time.monotonic() - self.started, 1e-6)

    def report(self):
        """One line, the slowest part first; then starts over."""
        loops = max(self.loops, 1)
        parts = sorted(self.parts.items(), key=lambda item: -item[1])
        line = f"{self.rate():.1f} loops/s | " + "  ".join(
            f"{name} {1000.0 * total / loops:.1f}" for name, total in parts)
        self.reset()
        return line + " (ms per loop)"


# ---------------------------------------------------------------------
# The app
# ---------------------------------------------------------------------


class DroneApp(Examiner):
    """The simulated drone and its window: drives the drone (keys, roaming,
    a script), sends the brain what the drone's sensors read, and draws what
    the brain sends back. Everything that reads the truth to CHECK the
    brain is in the Examiner, mixed in from examiner.py."""

    def __init__(self, args):
        self.args = args
        self.settings = brain_config.load()

        # The cameras' pictures are drawn by a process of their own
        # (rig_process.py), started first so it loads the house while this
        # one does. Until it is ready -- or if it fails -- they are drawn here.
        self.cameras = None

        if not args.no_camera_process:
            try:
                self.cameras = RigProcess("process", true_depth=(args.depth == "oracle"))
            except Exception as error:
                print(f"Drawing the cameras in the window: the camera process did not start ({error!r})")

        print("Opening the house...")
        self.world = World(stage="process", true_depth=(args.depth == "oracle"),
                           viewer_size=(int(PANEL / UI), int(TOP / UI)))
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
                    **({} if args.replica is None else {"replica_live": args.replica == "live"}),
                    # The lenses sit at the centre of the sphere, so this is
                    # the drone's radius -- and the brain scales every depth
                    # by it, so it is sent rather than assumed.
                    "lens_height": float(self.drone_config["diameter"]) / 2.0,
                },
            )

        self.alignment = Alignment()
        self.truth_by_frame = {}
        self.height_by_frame = {}
        self.true_views = []

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
        self.show_rooms = True
        self.orbit = OrbitView()
        self.painter = Painter()           # draws the replica on its own thread
        self.watch = Stopwatch()
        self.monitor = Monitor()       # memory, CPU, GPU, every 2 s
        self.replica_cut = "ceiling"   # a height, "ceiling", or None: all
        self._replica_shown = None     # (key, centres, colours) last drawn
        self._replica_picture = None   # (cubes picture, scaled to the panel)
        self._blits = {}               # place -> (image, size, scaled surface)
        self._dragging = None
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
        panel = np.array([float(PANEL), float(TOP)])

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

    def run(self, display):
        self.display = display
        self.screen = pygame.Surface(WINDOW)
        self.screen.fill(BACKGROUND)
        self._rendered = {}            # (text, font, colour) -> rendered text
        self._map_worker = Background("map layers")
        self._map_asked = None         # (state, what is shown, when) of the last layers asked for
        self._poses = collections.deque(maxlen=2)   # (when, pose): the brain's last two
        self._pending = None           # a frame taken, its pictures not all drawn yet
        pygame.display.set_caption(f"DELTA | {self.world.scene_id}")
        pygame.key.set_repeat(400, 40)
        self.font = pygame.font.SysFont("DejaVu Sans", px(15))
        self.small = pygame.font.SysFont("DejaVu Sans", px(12))
        self.big = pygame.font.SysFont("DejaVu Sans", px(18), bold=True)

        timer = self.timer = pygame.time.Clock()

        while self.running:
            # At most a tenth of a second per step: when the sim stalls (the GPU
            # busy with depth), time slows rather than the wearer teleporting a
            # metre between two camera frames -- which no real camera sees.
            with self.watch.part("waiting"):
                dt = min(timer.tick(FPS_CAP) / 1000.0, 0.1)
            self.clock += dt
            self.watch.loops += 1

            with self.watch.part("input"):
                self._events()
            with self.watch.part("physics"):
                self._move(dt)

            if self.script is not None:
                with self.watch.part("script"):
                    self.script.step(dt)

            with self.watch.part("cameras"):
                self._cameras_step()

            with self.watch.part("brain in"):
                self._drain_brain()
            self._draw()

            if self.args.profile and time.monotonic() - self.watch.started >= 10.0:
                self._print_profile()

            if self.clock >= self.next_screenshot:
                self.screenshot()
                self.next_screenshot = self.clock + self.args.screenshots

        if self.brain is not None:
            self.brain.close()

        if self.cameras is not None:
            self.cameras.close()

        self.world.close()
        pygame.quit()

    # -----------------------------------------------------------------
    # Input
    # -----------------------------------------------------------------

    def _events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False

            elif event.type == pygame.VIDEORESIZE:
                self.display = pygame.display.set_mode(event.size, pygame.RESIZABLE)

            elif event.type == pygame.KEYDOWN:
                self._key(event)

            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                position = self._logical(event.pos)

                if REPLICA_X <= position[0] and position[1] < TOP:
                    self._dragging = position      # turning the replica
                else:
                    self._click(position)

            elif event.type == pygame.MOUSEBUTTONUP and event.button == 1:
                self._dragging = None

            elif event.type == pygame.MOUSEMOTION and getattr(self, "_dragging", None):
                position = self._logical(event.pos)
                self.orbit.drag((position[0] - self._dragging[0]) / UI, (position[1] - self._dragging[1]) / UI)
                self._dragging = position

            elif event.type == pygame.MOUSEWHEEL:
                position = self._logical(pygame.mouse.get_pos())

                if REPLICA_X <= position[0] and position[1] < TOP:
                    self.orbit.zoom(event.y)

    def _logical(self, position):
        """A point in the window, in the drawing's own pixels."""
        width, height = self.display.get_size()
        return (position[0] * WINDOW[0] / max(width, 1), position[1] * WINDOW[1] / max(height, 1))

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
        elif key == pygame.K_n:
            self.show_rooms = not self.show_rooms
        elif key == pygame.K_f:
            self.orbit.follow = not self.orbit.follow
            self.note(f"Replica view {'follows the drone' if self.orbit.follow else 'stays put'}.")
        elif key in (pygame.K_LEFTBRACKET, pygame.K_RIGHTBRACKET):
            # In order from low to high: a flat cut, the ceiling off, all.
            # [ steps down that line, ] steps up it.
            down = key == pygame.K_LEFTBRACKET
            ceiling = self._typical_ceiling()
            top = REPLICA_CUT_TOP if ceiling is None else ceiling - CEILING_MARGIN

            if self.replica_cut is None:
                self.replica_cut = "ceiling" if down else None
            elif self.replica_cut == "ceiling":
                self.replica_cut = round(top - REPLICA_CUT_STEP, 2) if down else None
            else:
                cut = self.replica_cut + (-REPLICA_CUT_STEP if down else REPLICA_CUT_STEP)
                self.replica_cut = "ceiling" if cut > top else max(0.2, round(cut, 2))

            if self.replica_cut is None:
                self.note("Replica: everything shown, ceiling too.")
            elif self.replica_cut == "ceiling":
                self.note("Replica: the ceiling taken off.")
            else:
                self.note(f"Replica: cut flat at {self.replica_cut:.1f} m.")
        elif key == pygame.K_b:
            self.command("build_replica", "Build the 3D replica from everything seen so far.")
        elif key == pygame.K_o and not self.free_camera:
            self.world.drone.pitch = 0.0
            self.note("Cameras level.")

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

        if not (MAP_X <= x < MAP_X + PANEL and y < TOP):
            return

        self.plan_to((
            (x - self.panel_offset[0]) / self.scale + self.view_min[0],
            self.view_max[1] - (y - self.panel_offset[1]) / self.scale,
        ))

    def plan_to(self, world_xy):
        """Ask the brain for a route to a spot in true-world metres (x east,
        y north) -- what a click on the map does, and what a script does."""
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
        else:
            # The gimbal's tilt: I looks up, K looks down. (In free-camera
            # mode the same keys look the free camera around instead.)
            self.world.drone.tilt((keys[pygame.K_k] - keys[pygame.K_i]) * TILT_SPEED * fast * dt)

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
        """The brain's belief about the wearer, carried forward to now.

        TO MOVE INTO THE BRAIN: the motion since the brain's frame is the
        TRUE motion here -- a stand-in, from the glasses, for a person's own
        sense of their steps. On the drone it has to come from the IMU and
        the axle encoder, in the brain, when navigation moves there.
        """
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

        # From the left lens to the middle of the body, as the brain does it
        # (loop._body): half a baseline to the right. The glasses also stepped
        # back from the lens to the middle of a head; a sphere's lenses ARE at
        # its middle, so there is no such step -- and the glasses' setting for
        # it, read here after the port, crashed the autopilot.
        right = self.rig.baseline / 2.0
        body = (x + right * math.sin(yaw), y - right * math.cos(yaw))

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

    def _cameras_step(self):
        """Ten frames a second for the brain.

        Each frame: every sensor read at its moment (_send_frame), its
        pictures drawn from where the drone was then, and the frame sent
        when they are done (_finish_frame). The pictures are drawn by the
        camera process when it is running, in parallel with this loop;
        otherwise here, one a loop (~13 ms each) rather than all at once
        (~40 ms, a stutter every third loop) -- and, when behind, the rest
        at once, so the brain still gets its frames.
        """
        due = self.clock >= self.next_frame
        cameras = self.cameras

        if cameras is not None:
            seen = cameras.poll()

            if seen is not None:
                self._finish_frame(seen)

            if cameras.failed:
                self.note(f"Drawing the cameras in the window: {cameras.failed}")
                cameras.close()
                self.cameras = cameras = None

        if self.world.rig_busy:
            self._continue_frame(rest=due)
            return

        if cameras is not None and cameras.ready:
            if due and cameras.busy is None:
                self._send_frame(draw=False)
                cameras.request(self.frame_id - 1, *self.world.rig_pose())
                self.next_frame = self.clock + 1.0 / self.frame_rate
            return

        if due:
            self._send_frame()
            self._continue_frame()
            self.next_frame = self.clock + 1.0 / self.frame_rate

    def _send_frame(self, draw=True):
        """Take a frame: every sensor read now, at the frame's moment. Its
        camera pictures come later, all from where the drone is now --
        drawn here (_continue_frame) or by the camera process -- and the
        frame goes to the brain when the last is done (_finish_frame)."""
        seen = self.world.start_rig(draw=draw)
        self._pending = None

        # The examiner notes where the drone really was for this frame. Its
        # notes stay here, in the window; none of it goes to the brain.
        with self.watch.part("cameras: truth notes"):
            self._note_truth(seen)

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
            # The IMU rides the gimbal, so it also reports the tilt the gimbal
            # has been told to hold -- plus its own error.
            fields["imu_pitch"] = float(self.world.drone.pitch + self._tilt_error[0])
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

            # The pictures to send, once drawn: name in the message -> picture.
            pictures = {"left": "left", "right": "right"}

            # Oracle fields only when the experiment asked for them.
            if self.args.pose == "oracle":
                fields["true_position"] = seen["true_position"]
                fields["true_rotation"] = seen["true_rotation"]

            if self.args.depth == "oracle":
                pictures["true_depth"] = "true_depth"

            self._pending = (fields, pictures)

        self.frame_id += 1

    def _continue_frame(self, rest=False):
        """Draw the next picture of the frame (all the rest if `rest`); send
        the frame when the last is done."""
        while self.world.rig_busy:
            with self.watch.part("cameras: render"):
                seen = self.world.render_rig_part()

            if seen is not None:
                self._finish_frame(seen)

            if not rest:
                break

    def _finish_frame(self, seen):
        self.last_glasses = seen

        if self._pending is not None and self.brain is not None and self.brain.connected:
            fields, pictures = self._pending

            with self.watch.part("cameras: send"):
                self.brain.send("frame", fields, {name: seen[source] for name, source in pictures.items()})

        self._pending = None

    def _shown_pose(self):
        """Where the brain thinks the drone is, as drawn: glided between its
        last two updates instead of jumping ten times a second -- the way
        games draw between physics steps. It is shown one update late (0.1
        s) and smooth; a jump of over a metre (a loop closure) is drawn as a
        jump, not a slide."""
        if not self._poses:
            return None

        if len(self._poses) == 1:
            return self._poses[-1][1]

        (t0, a), (t1, b) = self._poses[-2], self._poses[-1]

        if math.hypot(b["x"] - a["x"], b["y"] - a["y"]) > 1.0:
            return b

        f = min(1.0, (time.monotonic() - t1) / max(t1 - t0, 1e-3))

        def mix(u, v):
            return u + f * (v - u)

        shown = dict(b, x=mix(a["x"], b["x"]), y=mix(a["y"], b["y"]),
                     yaw=a["yaw"] + f * frames.wrap(b["yaw"] - a["yaw"]))

        if "body" in a and "body" in b:
            shown["body"] = [mix(a["body"][0], b["body"][0]), mix(a["body"][1], b["body"][1])]

        return shown

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

        if state and "pose" in state and state is not getattr(self, "_posed_state", None):
            self._posed_state = state
            self._poses.append((time.monotonic(), state["pose"]))

        if state and "pose" in state:
            pose = state["pose"]
            truth = self.truth_by_frame.get(pose["frame"])
            self.alignment.update(state.get("epoch", 0), (pose["x"], pose["y"], pose["yaw"]), truth)

            if self.args.trace and pose["frame"] != getattr(self, "_traced", None):
                self._traced = pose["frame"]
                self._trace(state, truth)

    # -----------------------------------------------------------------
    # Drawing
    # -----------------------------------------------------------------

    def _draw(self):
        """Every panel, every loop -- each from what it keeps, so that is
        cheap: the map's layers, the replica's picture and the cameras'
        pictures are redrawn only when they change, the marks on them every
        loop. The text is drawn again when the brain says something new, or
        twice a second for the numbers. Only what changed goes to the screen.
        """
        screen = self.screen

        with self.watch.part("3D view"):
            self._draw_viewer()
        with self.watch.part("map"):
            self._draw_map()
        with self.watch.part("replica"):
            self._draw_replica()

        changed = [pygame.Rect(0, 0, WINDOW[0], TOP)]

        with self.watch.part("feeds"):
            pictures = (id(self.last_glasses), id((self.brain.state_arrays or {}).get("depth")) if self.brain else None)

            if pictures != getattr(self, "_feeds_shown", None):
                self._feeds_shown = pictures
                strip = pygame.Rect(0, TOP, LOG_LEFT, STATUS_TOP - TOP)
                screen.fill(BACKGROUND, strip)
                self._draw_feeds()
                changed.append(strip)

        with self.watch.part("text"):
            news = (id(self.brain.state) if self.brain else None, len(self.transcript),
                    self.transcript[-1] if self.transcript else None)

            if news != getattr(self, "_text_shown", None) or time.monotonic() - getattr(self, "_text_at", 0.0) > 0.5:
                self._text_shown, self._text_at = news, time.monotonic()
                log = pygame.Rect(LOG_LEFT, TOP, WINDOW[0] - LOG_LEFT, STATUS_TOP - TOP)
                status = pygame.Rect(0, STATUS_TOP, WINDOW[0], WINDOW[1] - STATUS_TOP)
                screen.fill(BACKGROUND, log)
                screen.fill(BACKGROUND, status)
                self._draw_transcript()
                self._draw_status()
                changed += [log, status]

        for x in (MAP_X, REPLICA_X):
            pygame.draw.line(screen, (60, 64, 72), (x, 0), (x, TOP), px(2))

        pygame.draw.line(screen, (60, 64, 72), (0, TOP), (WINDOW[0], TOP), px(2))
        changed.append(pygame.Rect(0, TOP - px(2), WINDOW[0], px(4)))

        # Only what changed goes to the screen -- unless the window was
        # resized and the drawing has to be scaled to it, all of it.
        with self.watch.part("show"):
            if self.display.get_size() == WINDOW:
                for rect in changed:
                    self.display.blit(screen, rect, rect)

                pygame.display.update(changed)
            else:
                self.display.blit(pygame.transform.smoothscale(screen, self.display.get_size()), (0, 0))
                pygame.display.flip()

    def _print_profile(self):
        """The window's parts, and the brain's own, to stdout."""
        print(f"[profile] window {self.watch.report()}", flush=True)
        stats = (self.brain.state or {}).get("stats", {}) if self.brain else {}

        if stats:
            print(f"[profile] brain rates {stats.get('rates')} ms {stats.get('ms')}", flush=True)

    def _draw_replica(self):
        """The brain's 3D replica, in its own frame, orbiting the drone.

        Drag to turn it, scroll to zoom, F to stop following the drone. The
        orange ring is where the brain believes the drone is; the green one
        where it really is, put into the brain's frame for comparison --
        the only truth on this panel, and only on it as a mark.
        """
        replica = self.brain.replica if self.brain else None
        state = self.brain.state if self.brain else None
        markers = []

        shown_pose = self._shown_pose()

        if state and shown_pose is not None:
            body = shown_pose.get("body", [shown_pose["x"], shown_pose["y"]])
            altitude = float(state.get("stats", {}).get("altitude") or 0.0)
            believed = (body[0], body[1], altitude + float(self.drone_config["diameter"]) / 2.0)
            markers.append((believed, (255, 150, 50), 9))

            if self.orbit.follow:
                self.orbit.look_at(believed)

            # Where it really is, in the brain's frame: the alignment, undone.
            here = self.world.drone.position
            if self.alignment.ready:
                c, s = math.cos(-self.alignment.yaw), math.sin(-self.alignment.yaw)
                dx, dy = np.array([here.x, -here.z]) - self.alignment.offset
                markers.append(((c * dx - s * dy, s * dx + c * dy, float(here.y) - self.floor_height),
                                (80, 220, 110), 7))

        # The cubes are drawn at one pixel a point, on the painter's own
        # thread, and only when they or the view changed; the newest picture
        # finished is scaled up to the panel, once per picture.
        size = (int(PANEL / UI), int(TOP / UI))

        if replica is not None:
            centres, colour = self._replica_to_draw(replica)
            self.painter.ask(self.orbit, centres, colour, size, float(replica[2]["voxel"]),
                             version=(id(replica), self.replica_cut))
        else:
            self.painter.ask(self.orbit, np.zeros((0, 3), np.float32), np.zeros((0, 3), np.uint8),
                             size, version=None)

        # (Before the first is done, a blank one.)
        picture = self.painter.latest() or (self.orbit, np.zeros((size[1], size[0], 3), np.uint8))

        if self._replica_picture is None or self._replica_picture[0] is not picture[1]:
            surface = pygame.surfarray.make_surface(np.transpose(picture[1], (1, 0, 2)))
            self._replica_picture = (picture[1], pygame.transform.smoothscale(surface, (PANEL, TOP)))

        self.screen.blit(self._replica_picture[1], (REPLICA_X, 0))

        # The marks with the camera the picture was drawn from, so they sit
        # where the picture has the drone even while it is a moment old.
        for xyz, rgb, radius in markers:
            point = picture[0].project(xyz, size)

            if point is not None:
                pygame.draw.circle(self.screen, rgb, (REPLICA_X + point[0] * UI, point[1] * UI),
                                   px(radius), px(2))

        self._text("3D replica", (REPLICA_X + px(12), px(10)), self.big, (240, 240, 240))
        live = ((state or {}).get("stats", {}).get("replica") or {"live": True})

        if replica:
            count = f"{len(replica[0]):,} cubes from {replica[2].get('views', 0)} views   "
        elif not live["live"]:
            count = f"not built: {live.get('views', 0)} views kept, B builds it   "
        else:
            count = "nothing yet   "
        self._text(count + "drag: turn   scroll: zoom   F: "
                   + ("following the drone" if self.orbit.follow else "fixed"),
                   (REPLICA_X + px(12), px(34)), self.small, (170, 175, 185))

        # What the brain knows of the ceiling, and what the view hides.
        ceiling = self._typical_ceiling()
        known = "ceiling: not seen yet" if ceiling is None else f"ceiling: {ceiling:.2f} m (the brain's)"
        if self.replica_cut is None:
            shown = "showing all"
        elif self.replica_cut == "ceiling":
            shown = "ceiling taken off"
        else:
            shown = f"cut flat at {self.replica_cut:.1f} m"

        self._text(f"{known}   view: {shown}   [ ]: lower / raise",
                   (REPLICA_X + px(12), px(52)), self.small, (190, 170, 120))

    def _replica_to_draw(self, replica):
        """The cubes the panel draws: all of them, less the ceiling or what
        is above a flat cut. Worked out once per replica and cut."""
        key = (id(replica), self.replica_cut)

        if self._replica_shown is not None and self._replica_shown[0] == key:
            return self._replica_shown[1:]

        index, colour, info = replica
        centres = (index.astype(np.float32) + 0.5) * float(info["voxel"])

        if self.replica_cut == "ceiling":
            limit = self._ceiling_over(centres[:, :2], info)

            if limit is not None:
                keep = centres[:, 2] < limit - CEILING_MARGIN
                centres, colour = centres[keep], colour[keep]
        elif self.replica_cut is not None:
            keep = centres[:, 2] < self.replica_cut
            centres, colour = centres[keep], colour[keep]

        self._replica_shown = (key, centres, colour)
        return centres, colour

    @staticmethod
    def _ceiling_over(xy, info):
        """The brain's ceiling height over each point, or None if it has
        seen no ceiling at all. A patch it has not seen one in takes the
        highest of its neighbours', and failing that the usual height."""
        heights = info.get("ceiling")

        if heights is None or np.all(np.isnan(heights)):
            return None

        heights = heights.astype(np.float32)
        rows, cols = heights.shape

        # Each patch and its eight neighbours, the highest seen.
        padded = np.pad(heights, 1, constant_values=np.nan)
        stack = np.stack([padded[r:r + rows, c:c + cols] for r in range(3) for c in range(3)])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            near = np.nanmax(stack, axis=0)
        near = np.where(np.isnan(near), np.nanmedian(heights), near)

        cell, origin = float(info["ceiling_cell"]), np.asarray(info["ceiling_origin"], np.float32)
        col = np.floor((xy[:, 0] - origin[0]) / cell).astype(np.int64)
        row = np.floor((xy[:, 1] - origin[1]) / cell).astype(np.int64)
        inside = (col >= 0) & (col < cols) & (row >= 0) & (row < rows)

        limit = np.full(len(xy), np.nanmedian(heights), np.float32)
        limit[inside] = near[row[inside], col[inside]]
        return limit

    def _typical_ceiling(self):
        """The brain's usual ceiling height, or None until it has seen one."""
        replica = self.brain.replica if self.brain else None
        heights = replica[2].get("ceiling") if replica else None

        if heights is None or np.all(np.isnan(heights)):
            return None

        return float(np.nanmedian(heights.astype(np.float32)))

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
        self._draw_cameras(lines)

        if not drone.flying:
            return

        ground = drone.floor_position
        shadow = mn.Vector3(ground.x, ground.y + 0.01, ground.z)

        lines.draw_circle(shadow, drone.radius, blue)
        lines.draw_transformed_line(shadow, drone.position, blue)

    def _draw_cameras(self, lines):
        """Where the cameras are and where they look, like the glasses' lens
        marks: the stereo board on the front of the sphere with its two
        lenses, a line along each lens's view, and a frame for the edge of
        what they see. Drawn in the gimbal's frame, so it turns with the
        heading and tilts with the pitch -- I and K show up here at once.
        """
        drone = self.world.drone
        orange = mn.Color4(1.0, 0.6, 0.15, 1.0)

        width, height = self.rig.resolution("process")
        left = self.rig.camera(self.rig.stereo_pair[0])
        half_width = math.tan(math.radians(left.hfov) / 2.0)
        half_height = half_width * height / width

        half_base = self.rig.baseline / 2.0
        face = -(drone.radius + 0.006)      # just outside the shell, in front
        reach = 0.22                        # how far out the view frame sits

        lines.push_transform(mn.Matrix4.from_(drone.sensor_rotation.to_matrix(), drone.position))

        # The board, and a lens at each end of it.
        lines.draw_box(mn.Vector3(-half_base - 0.015, -0.012, face - 0.006),
                       mn.Vector3(half_base + 0.015, 0.012, face + 0.006), orange)

        for x in (-half_base, half_base):
            lines.draw_box(mn.Vector3(x - 0.008, -0.008, face - 0.008),
                           mn.Vector3(x + 0.008, 0.008, face), orange)
            lines.draw_transformed_line(mn.Vector3(x, 0.0, face), mn.Vector3(x, 0.0, face - 0.35), orange)

        # The view: a frame `reach` out, joined back to the board.
        centre = mn.Vector3(0.0, 0.0, face)
        corners = [
            mn.Vector3(sx * half_width * reach, sy * half_height * reach, face - reach)
            for sx, sy in ((-1, -1), (1, -1), (1, 1), (-1, 1))
        ]

        for i, corner in enumerate(corners):
            lines.draw_transformed_line(centre, corner, orange)
            lines.draw_transformed_line(corner, corners[(i + 1) % 4], orange)

        lines.pop_transform()

    def _draw_viewer(self):
        """The drone from behind: Habitat renders it at one pixel a point
        (on 4K, 44% of the panel's pixels), scaled up."""
        if not self.free_camera:
            self._place_chase_camera()
        else:
            self.world.viewer_node.rotation = mn.Quaternion.rotation(
                mn.Deg(self.camera_yaw), mn.Vector3.y_axis()
            ) * mn.Quaternion.rotation(mn.Deg(self.camera_pitch), mn.Vector3.x_axis())

        self._outline_drone()
        image = self.world.observe_viewer()
        surface = pygame.surfarray.make_surface(np.transpose(image[:, :, :3], (1, 0, 2)))

        if surface.get_size() != (PANEL, TOP):
            surface = pygame.transform.smoothscale(surface, (PANEL, TOP))

        self.screen.blit(surface, (0, 0))

        label = "Free camera (WASD, IJKL, Z/X) - V to follow" if self.free_camera else "Behind you - V for free camera"
        self._text(label, (px(12), px(10)), self.big, (240, 240, 240))

    def _panel_affine(self, layer):
        """Cell (col, row) of a brain layer -> brain world -> true world ->
        map-panel pixel. `layer` carries its own origin and resolution."""
        origin, resolution = layer["origin"], layer["resolution"]

        a = self.alignment.matrix()
        cell = np.array([[resolution, 0.0, origin[0]], [0.0, resolution, origin[1]]])
        world = a[:, :2] @ cell
        world[:, 2] += a[:, 2]
        panel = np.array([[self.scale, 0.0, -self.view_min[0] * self.scale + self.panel_offset[0]],
                          [0.0, -self.scale, self.view_max[1] * self.scale + self.panel_offset[1]]])
        affine = panel[:, :2] @ world
        affine[:, 2] += panel[:, 2]

        return affine

    def _draw_map(self):
        state = self.brain.state if self.brain else None
        arrays = self.brain.state_arrays if self.brain else {}

        # The layers are drawn again only when the brain has sent a new
        # state (a few times a second) or a key changed what is shown, not
        # on every frame; the marks over them are drawn every frame.
        shown = (self.show_truth, self.show_rooms, self.alignment.ready)
        asked = self._map_asked

        # At most ~7 times a second, and on the worker, not in this loop:
        # at 4K a redraw is ~15 ms -- a spike every time the brain sent news.
        # The marks over the layers (the drone) are drawn here, every loop.
        stale = asked is None or (asked[0] is not state and time.monotonic() - asked[2] >= 0.15)

        if stale or asked[1] != shown:
            if state:
                self._widen_view_for(state, arrays)

            # Where everything goes on the panel, worked out here, so the
            # worker never reads the view while this loop changes it.
            affines = {"truth": self.truth_affine / UI}

            if state and self.alignment.ready:
                for layer in ("grid", "rooms"):
                    if layer in state:
                        affines[layer] = self._panel_affine(state[layer]) / UI

            self._map_worker.ask(self._map_picture, state, arrays, shown, affines)
            self._map_asked = (state, shown, time.monotonic())

        picture = self._map_worker.latest()

        if picture is not None:
            self.screen.blit(picture, (MAP_X, 0))

        self._map_marks(state)

    def _map_picture(self, state, arrays, shown, affines):
        """The map's layers as a panel-sized picture (on the worker)."""
        canvas = self._map_layers(state, arrays, shown, affines)
        surface = pygame.surfarray.make_surface(np.transpose(canvas, (1, 0, 2)))
        return pygame.transform.smoothscale(surface, (PANEL, TOP))

    def _map_layers(self, state, arrays, shown, affines):
        """The map panel's picture: the true layout, the brain's map, its
        rooms."""
        # At one pixel a point: on 4K the panel has 2.25 times the pixels,
        # and each redraw took 35-40 ms. Every warp is the panel's, scaled.
        size = (int(PANEL / UI), int(TOP / UI))
        canvas = np.zeros((size[1], size[0], 3), np.uint8)
        canvas[:] = (20, 23, 28)

        show_truth, show_rooms, ready = shown

        if show_truth:
            # The true layout never changes; only the panel's scale does.
            view = affines["truth"].tobytes()

            if getattr(self, "_truth_warped", (None,))[0] != view:
                self._truth_warped = (view, cv2.warpAffine(self.truth_layer, affines["truth"], size,
                                                           flags=cv2.INTER_NEAREST))

            canvas = np.maximum(canvas, self._truth_warped[1])

        if state and "grid" in state and "grid" in arrays and "grid" in affines:
            grid = arrays["grid"]
            affine = affines["grid"]

            # What is there, from the solid layer -- white where anything
            # stands, at any height, like the glasses' map -- and, from the
            # rolling layer, where the ball can still get underneath it.
            solid = arrays.get("solid")
            solid = solid if solid is not None and solid.shape == grid.shape else grid

            colours = np.zeros(grid.shape + (3,), np.uint8)
            colours[(solid == 1) | (grid == 1)] = FREE_COLOUR
            colours[solid == 2] = OCCUPIED_COLOUR
            colours[(solid == 2) & (grid == 1)] = ROLL_UNDER_COLOUR
            colours[grid == 2] = OCCUPIED_COLOUR

            mask = cv2.warpAffine(((grid > 0) | (solid > 0)).astype(np.uint8), affine, size,
                                  flags=cv2.INTER_NEAREST)
            warped = cv2.warpAffine(colours, affine, size, flags=cv2.INTER_NEAREST)
            canvas[mask > 0] = warped[mask > 0]

        # The brain's rooms, each its own tint over the map (N to hide).
        if show_rooms and state and "rooms" in state and "rooms" in arrays and "rooms" in affines:
            labels = arrays["rooms"]
            warped = cv2.warpAffine(labels.astype(np.uint8), affines["rooms"],
                                    size, flags=cv2.INTER_NEAREST)
            # Only the free floor is tinted: what is occupied stays white, so
            # the rooms never hide the map underneath them.
            tinted = (warped > 0) & np.all(canvas == FREE_COLOUR, axis=2)
            palette = np.array([(0, 0, 0)] + ROOM_COLOURS * 26, np.float32)[:256]
            canvas[tinted] = (0.55 * canvas[tinted] + 0.45 * palette[warped[tinted]]).astype(np.uint8)

        return canvas

    def _map_marks(self, state):
        """What is drawn over the map panel every frame: the drone, the
        route, the names, and the panel's own text."""
        offset = np.array([MAP_X, 0])

        if state and self.alignment.ready:
            for place in state.get("places", []):
                p = np.array(self.to_panel(self.alignment.point(place["xy"]))) + offset
                pygame.draw.polygon(self.screen, PLACE_COLOUR, _star(p, px(9)))
                self._text(place["name"], p + (px(10), -px(8)), self.small, PLACE_COLOUR)

            # Only trusted things, and names only for the well seen: a walk
            # through a house glimpses a great deal.
            for thing in state.get("things", []):
                if not thing["confirmed"]:
                    continue

                p = np.array(self.to_panel(self.alignment.point(thing["xy"]))) + offset
                pygame.draw.circle(self.screen, THING_COLOUR, p, px(4))

                if thing["seen"] >= 6:
                    self._text(thing["id"], p + (px(6), -px(7)), self.small, THING_COLOUR)

            if "route" in state:
                points = [np.array(self.to_panel(self.alignment.point(xy))) + offset
                          for xy in state["route"]["points"]]

                if len(points) >= 2:
                    pygame.draw.lines(self.screen, ROUTE_COLOUR, False, points, px(3))

                target = np.array(self.to_panel(self.alignment.point(state["route"]["target"]))) + offset
                pygame.draw.circle(self.screen, TARGET_COLOUR, target, px(8), px(2))

            if self.show_rooms and "rooms" in state:
                for door in state["rooms"]["doors"]:
                    p = np.array(self.to_panel(self.alignment.point(door["xy"]))) + offset
                    pygame.draw.circle(self.screen, DOOR_COLOUR, p, px(4))

                for room in state["rooms"]["list"]:
                    p = np.array(self.to_panel(self.alignment.point(room["xy"]))) + offset
                    self._text(f"room {room['id']}  {room['area']:.0f} m2",
                               p - (px(30), px(7)), self.small, (250, 250, 250))

            pose = self._shown_pose()

            if pose is not None:
                p = np.array(self.to_panel(self.alignment.point((pose["x"], pose["y"])))) + offset
                heading = pose["yaw"] + self.alignment.yaw
                tip = p + px(16) * np.array([math.cos(heading), -math.sin(heading)])
                pygame.draw.line(self.screen, DRONE_BELIEVED, p, tip, px(3))
                pygame.draw.circle(self.screen, DRONE_BELIEVED, p, px(6))

        # Where the wearer really is.
        lens = self.truth_by_frame.get(self.frame_id - 1)

        if lens is not None:
            p = np.array(self.to_panel(lens[:2])) + offset
            tip = p + px(16) * np.array([math.cos(lens[2]), -math.sin(lens[2])])
            pygame.draw.circle(self.screen, DRONE_TRUE, p, px(7), px(2))
            pygame.draw.line(self.screen, DRONE_TRUE, p, tip, px(2))

        self._text("What the drone believes", (MAP_X + px(12), px(10)), self.big, (240, 240, 240))
        self._text("green: really here   orange: believed here   white: occupied   lilac: can roll under",
                   (MAP_X + px(12), px(34)), self.small, (170, 175, 185))
        self._text("M: true layout " + ("on" if self.show_truth else "off")
                   + "   N: rooms " + ("on" if self.show_rooms else "off") + "   click: plan a route",
                   (MAP_X + px(12), px(50)), self.small, (170, 175, 185))

        silent = time.monotonic() - self.brain.state_at if self.brain is not None and self.brain.state_at else 0.0

        if self.brain is None:
            self._text("Running without a brain (--no-brain)", (MAP_X + px(12), px(70)), self.font, (200, 120, 120))
        elif not self.brain.connected:
            message = (f"The brain stopped ({self.brain.error}). Close this window and see outputs/logs/brain.log."
                       if self.brain.error else "Waiting for the brain to start...")
            self._text(message, (MAP_X + px(12), px(70)), self.font, (240, 120, 90))
        elif not self.brain.ready:
            self._text("Brain loading models...", (MAP_X + px(12), px(70)), self.font, (220, 180, 90))
        elif silent > 3.0:
            # Connected but quiet: the brain is stuck or one of its threads
            # died. Its position and map on screen are frozen, not live.
            self._text(f"The brain has not answered for {silent:.0f} s -- what you see is frozen. "
                       "See outputs/logs/brain.log.", (MAP_X + px(12), px(70)), self.font, (240, 120, 90))

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

        width, height = FEED

        self._blit(self.last_glasses["left"], (px(8), STRIP_TOP + px(22)), (width, height))
        self._text("Left lens", (px(8), STRIP_TOP + px(2)), self.font, (210, 210, 210))

        if "depth" in arrays:
            self._blit(arrays["depth"], (px(16) + width, STRIP_TOP + px(22)), (width, height),
                       colour=lambda d: cv2.applyColorMap(255 - d, cv2.COLORMAP_VIRIDIS)[:, :, ::-1])

            scale = (state or {}).get("stats", {}).get("floor_scale")
            label = "Depth the brain is using (0-6 m)"

            if scale:
                label += f"   floor scale x{scale:.2f}"

            self._text(label, (px(16) + width, STRIP_TOP + px(2)), self.font, (210, 210, 210))

    def _draw_transcript(self):
        x, top, bottom = LOG_LEFT, STRIP_TOP + px(2), STATUS_TOP - px(30)
        width = WINDOW[0] - x - px(8)

        self._text("Log", (x, top), self.font, (210, 210, 210))

        lines = []

        for kind, text in self.transcript[-40:]:
            for line in _wrap(text, self.small, width):
                lines.append((kind, line))

        y = bottom - px(16)

        for kind, line in reversed(lines):
            # Stop under the height box.
            if y < LOG_CLEAR:
                break

            self._text(line, (x, y), self.small, KIND_COLOURS.get(kind, (200, 200, 200)))
            y -= px(16)

        box = pygame.Rect(x, STATUS_TOP - px(26), width, px(22))
        pygame.draw.rect(self.screen, (26, 28, 34), box)
        self._text(
            "arrows drive   1/0 fly, land   I/K tilt   click map: plan   C: clear   "
            "G: autopilot   R: roam   B: build 3D   Tab: save",
            (x + px(4), box.y + px(4)), self.small, (130, 135, 145),
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

            self._text(text, (px(8), STATUS_TOP + px(5)), self.small, (255, 180, 90))
            return

        stats = state.get("stats", {})
        rates = stats.get("rates", {})
        ms = stats.get("ms", {})
        loc = stats.get("localization", {})
        sources = stats.get("sources", {})

        parts = [
            f"screen {self.timer.get_fps():.0f} fps",
            f"pose {sources.get('pose')}  depth {sources.get('depth')}",
            # The cameras run at 10 frames a second, like the real ones: the
            # brain's rate, not the screen's.
            f"brain: camera {rates.get('frames', 0):.0f} frames/s, thought through {rates.get('perception', 0):.1f}/s "
            f"(depth {ms.get('depth', 0)} ms, map {ms.get('map', 0)} ms)",
        ]

        self._draw_height(stats)
        self._draw_resources(stats)

        if sources.get("pose") == "rtabmap":
            parts.append(f"RTAB-Map {loc.get('ms', 0):.0f} ms, nodes {loc.get('nodes', 0)}, "
                         f"loops {loc.get('loops', 0)} kept / {loc.get('rejected', 0)} rejected")

        error = self._pose_error(state)

        if error is not None:
            parts.append(f"pose error {error[0]:.2f} m, {error[1]:.1f} deg")

        if stats.get("relocalizing"):
            parts.append("RECOGNISING LAST SESSION'S MAP...")

        self._text("   |   ".join(parts), (px(8), STATUS_TOP + px(5)), self.small, (175, 180, 190))

    def _draw_resources(self, stats):
        """What it all costs, next to the height box: memory and CPU of the
        three processes, the GPU, and the size of each map. Measured every
        2 s -- the brain's half by the brain itself."""
        brain = stats.get("resources") or {}
        processes = brain.get("processes", {})
        map_2d, replica, rtabmap = brain.get("map_2d", {}), brain.get("replica", {}), brain.get("rtabmap", {})
        gpu, machine = self.monitor.gpu, self.monitor.machine

        def size(mb):
            if mb is None:
                return "-"
            return f"{mb / 1000:.1f} GB" if mb >= 1000 else f"{mb:.0f} MB"

        def usage(process):
            return (size(process.get("mb")), f"{process['cpu']:.0f}%" if "cpu" in process else "-")

        rows = [
            ("RESOURCES", "memory", "CPU"),
            ("window (sim + drawing)", *usage(self.monitor.window)),
            ("brain", *usage(processes.get("brain", {}))),
            ("RTAB-Map", *usage(processes.get("rtabmap", {}))),
        ]
        lines = []

        if gpu:
            watts = f"   {gpu['watts']:.0f} W" if gpu.get("watts") is not None else ""
            lines.append(f"GPU  {gpu['use']:.0f}% busy   {size(gpu['mb'])} of {size(gpu['total_mb'])}{watts}")
        else:
            lines.append("GPU  (nvidia-smi not found)")

        if machine:
            lines.append(f"machine  CPU {machine['cpu']:.0f}% of {machine['threads']} threads   "
                         f"RAM {size(machine['mb'])} / {size(machine['total_mb'])}")

        lines.append(f"2D map  {size(map_2d.get('mb'))}   ({map_2d.get('scans', 0)} views kept)")
        lines.append(f"3D replica  {replica.get('cubes', 0):,} cubes, {size(replica.get('mb'))}   "
                     f"views {size(replica.get('views_mb'))}")
        lines.append(f"RTAB-Map  {rtabmap.get('nodes', 0)} places, database {size(rtabmap.get('db_mb'))}")
        lines.append(f"window {self.timer.get_fps():.0f} fps   brain {stats.get('rates', {}).get('perception', 0):.0f} frames/s")

        x, top = RESOURCES_LEFT, RESOURCES_TOP
        pygame.draw.rect(self.screen, (24, 26, 32), pygame.Rect(x, top, RESOURCES_WIDTH, RESOURCES_HEIGHT))

        for i, (name, memory, cpu) in enumerate(rows):
            y = top + px(4) + i * px(18)
            colour = (150, 155, 165) if i == 0 else (225, 225, 225)
            self._text(name, (x + px(8), y), self.small, colour)
            self._text(memory, (x + px(200), y), self.small, colour)
            self._text(cpu, (x + px(285), y), self.small, colour)

        for i, line in enumerate(lines):
            y = top + px(4) + (len(rows) + i) * px(18)
            self._text(line, (x + px(8), y), self.small, (190, 195, 205))

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
        ceiling, true_ceiling = stats.get("ceiling"), self._true_ceiling()

        rows = [
            ("HEIGHT", "brain", "truth"),
            ("above the floor", f"{altitude:.2f} m", f"{true_altitude:.2f} m"),
            ("above what is below", f"{altitude - surface:.2f} m", f"{true_altitude - true_surface:.2f} m"),
            ("what is below (0 = floor)", f"{surface:.2f} m", f"{true_surface:.2f} m"),
            ("ceiling (above the floor)", "not seen" if ceiling is None else f"{ceiling:.2f} m",
             "none" if true_ceiling is None else f"{true_ceiling:.2f} m"),
        ]

        # At the top of the log panel, clear of the 3D view.
        x, top = LOG_LEFT, HEIGHT_BOX_TOP
        pygame.draw.rect(self.screen, (24, 26, 32), pygame.Rect(x, top, px(380), HEIGHT_BOX_HEIGHT))

        for i, (name, brain, truth) in enumerate(rows):
            y = top + px(4) + i * px(21)
            colour = (150, 155, 165) if i == 0 else (235, 235, 235)
            self._text(name, (x + px(8), y), self.font, colour)
            self._text(brain, (x + px(222), y), self.font, (120, 200, 255) if i else colour)
            self._text(truth, (x + px(302), y), self.font, (160, 220, 160) if i else colour)

        pitch = self.world.drone.pitch
        tilt = "level" if abs(pitch) < 0.5 else f"{abs(pitch):.0f} deg {'down' if pitch > 0 else 'up'}"
        self._text(f"cameras: {tilt}   (I up, K down, O level)", (x + px(8), top + px(4) + len(rows) * px(21)),
                   self.small, (190, 170, 120))

    # -----------------------------------------------------------------

    def _blit(self, image, position, size, colour=None):
        """A picture, scaled to `size` -- once per new picture: the window
        runs 30 times a second, a camera picture is new 10 times, and
        scaling one to these sizes takes milliseconds. `colour` turns
        `image` into RGB first (a depth map into colours)."""
        cached = self._blits.get(position)

        if cached is None or cached[0] is not image or cached[1] != size:
            rgb = image if colour is None else colour(image)
            surface = pygame.surfarray.make_surface(np.transpose(np.ascontiguousarray(rgb[:, :, :3]), (1, 0, 2)))
            cached = self._blits[position] = (image, size, pygame.transform.smoothscale(surface, size))

        self.screen.blit(cached[2], position)

    def _text(self, text, position, font, colour):
        """Text, rendered once and kept: most of the window's text is the
        same from one loop to the next, and rendering it is the slow part."""
        key = (text, id(font), tuple(colour))
        rendered = self._rendered.get(key)

        if rendered is None:
            if len(self._rendered) > 2000:
                self._rendered.clear()

            rendered = self._rendered[key] = font.render(text, True, colour)

        self.screen.blit(rendered, (int(position[0]), int(position[1])))

    def screenshot(self, name=None):
        folder = PROJECT_ROOT / "outputs" / "screenshots"
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / f"{name or time.strftime('%Y%m%d-%H%M%S')}.png"
        pygame.image.save(self.screen, str(path))
        print(f"Saved {path}")


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

    display = open_window(args)
    DroneApp(args).run(display)


def open_window(args):
    """The window, maximised, and the layout for its size.

    Opened before the house is loaded, because the layout decides how big
    Habitat renders the view behind the drone. A point is 1.5 pixels on a
    4K screen (2560 points across, as Windows at 150%), 1 on anything up to
    2560 wide; --ui overrides it.
    """
    pygame.init()

    if args.headless:
        layout(WINDOW)
        return pygame.display.set_mode(WINDOW)

    desktop = pygame.display.get_desktop_sizes()[0]
    ui = args.ui or max(1.0, round(desktop[0] / 2560.0 * 4.0) / 4.0)
    display = pygame.display.set_mode((desktop[0] * 3 // 4, desktop[1] * 3 // 4), pygame.RESIZABLE)

    try:
        from pygame._sdl2 import video
        video.Window.from_display_module().maximize()
    except Exception as error:                     # an old pygame: stay unmaximised
        print(f"Could not maximise the window: {error!r}")

    # The window manager answers with the new size a moment later.
    first = display.get_size()
    deadline = time.monotonic() + 2.0

    while time.monotonic() < deadline and pygame.display.get_window_size() == first:
        pygame.event.pump()
        time.sleep(0.05)

    size = pygame.display.get_window_size()
    layout(size, ui)
    print(f"Window {size[0]} x {size[1]} pixels, {ui:g} pixels a point.")
    return pygame.display.set_mode(size, pygame.RESIZABLE)


if __name__ == "__main__":
    main()
