"""The brain's loops: localize, map, publish. Nothing else.

This is the Spatial Memory Glasses brain with the glasses' other jobs taken
out. That project's brain also detected objects with YOLOE, warned a blind
wearer about obstacles with a fast SGBM loop, spoke directions through Piper,
and answered spoken requests through Gemma. DELTA does none of that, so none
of it is here -- but every part that the mapping depends on is the same file
it was there, unchanged:

    rectify.py      line the stereo pair up
    localize.py     RTAB-Map, through its C++ bridge, with loop closure
    depth.py        Depth Anything 3, scaled by the floor it can see
    occupancy.py    the map, as log-odds per cell per height band
    planner.py      A* across the map the drone built
    navmap.py       the map, inflated for planning
    map_localize.py checking RTAB-Map's pose against the map

Three loops, and each takes the NEWEST frame when it is ready for one, so a
slow loop falls behind by skipping frames rather than by queueing them:

    localization    every frame, CPU     the RTAB-Map bridge, inside localize
    perception      newest frame, GPU    rectify -> DA3 -> the map, at the
                                         frame's pose
    publisher       4 times a second     what the brain knows, to the window

One lock guards what the loops share: the grid and the route.
"""

import json
import math
import shutil
import threading
import time
import traceback
from collections import deque
from dataclasses import dataclass, field

import numpy as np

from delta import frames
from delta.brain import (
    cameras,
    config,
    depth as depth_module,
    localize,
    map_localize,
    navmap,
    occupancy,
    planner,
    rectify,
)
from delta.wire import Closed


@dataclass
class View:
    """The last frame perception finished, for the window."""

    frame: int = -1
    depth: object = None
    pose: object = None
    size: tuple = (0, 0)


@dataclass
class Route:
    points: list = field(default_factory=list)
    target: tuple = (0.0, 0.0)
    label: str = ""
    index: int = 0


class Latest:
    """A one-slot mailbox: whoever reads it gets the newest thing in it."""

    def __init__(self):
        self.item = None
        self.frame_id = -1
        self._new = threading.Condition()

    def put(self, frame_id, item):
        with self._new:
            self.item = item
            self.frame_id = frame_id
            self._new.notify_all()

    def newer_than(self, frame_id, timeout=0.5):
        with self._new:
            if self.frame_id <= frame_id:
                self._new.wait(timeout)

            return self.item if self.frame_id > frame_id else None


class Rate:
    """How often something is happening, smoothed."""

    def __init__(self):
        self.value = 0.0
        self._last = None

    def tick(self):
        now = time.monotonic()

        if self._last is not None:
            gap = now - self._last

            if gap > 0:
                self.value += 0.2 * (1.0 / gap - self.value)

        self._last = now


class Brain:
    def __init__(self, link, hello, settings, rig):
        self.link = link
        self.settings = settings
        self.rig = rig
        self.lock = threading.RLock()
        self.running = True

        self.scene = hello["scene_id"]
        self.size = (int(hello["width"]), int(hello["height"]))
        self.pose_source = hello.get("pose", "rtabmap")
        self.depth_source = hello.get("depth", "da3")

        # How high the lenses sit above the floor. For the glasses this was a
        # standing adult; for DELTA it is the drone's own radius, because the
        # cameras sit at the centre of the sphere. The sim sends it, since the
        # sim is what knows how big the drone is.
        self.lens_height = float(
            hello.get("lens_height") or settings.drone.lens_height
        )

        # While it flies, the lenses rise with it, and every height the map
        # records is measured from the lens. The sim sends the altitude so the
        # map still puts the floor at the floor.
        self.altitude = 0.0

        self.use_tilt = bool(hello.get("use_tilt", True))
        self.map_localize = bool(hello.get("map_localize", False))
        self.corrector = None

        self.memory_dir = (
            config.path(settings.memory.directory) / self.scene / self.pose_source
        )
        self._prepare_memory(fresh=hello.get("fresh", False))

        self.latest = Latest()
        self.view = View()
        self.grid = None
        self.route = None

        self.rates = {name: Rate() for name in ("frames", "perception")}
        self.timing = {}
        self.epoch = 0
        self.last_stamp = 0.0
        self.graph_dirty = False
        self._rebuilt_at = 0.0

        self.rectifier = rectify.Rectifier(rig, self.size)
        self._load_models(hello)
        self._load_memory()

    # -----------------------------------------------------------------
    # Startup
    # -----------------------------------------------------------------

    def _load_models(self, hello):
        s = self.settings

        self.status(f"Starting localization ({self.pose_source})...")

        if self.pose_source == "oracle":
            self.localizer = localize.OracleLocalizer()
        else:
            self.localizer = localize.RtabmapLocalizer(
                s.localization,
                self.rectifier,
                self.size,
                database=self.memory_dir / "rtabmap.db",
                log=self.memory_dir / "rtabmap.log",
                on_graph=self._on_graph,
            )
            self.localizer.use_tilt = self.use_tilt

        if self.map_localize:
            self.corrector = map_localize.MapCorrector(self._body)
            publish = self.localizer.fixes.add

            def corrected(fix):
                if fix.pose is not None:
                    fix.raw_pose = fix.pose
                    fix.pose = self.corrector.apply(fix.pose)

                publish(fix)

            self.localizer.fixes.add = corrected

        self.status(f"Loading depth ({self.depth_source})...")

        if self.depth_source == "oracle":
            self.depth = depth_module.OracleDepth()
        else:
            self.depth = depth_module.DA3Depth(s.depth, self.rectifier.baseline)

        self.floor = depth_module.FloorScale(
            self.lens_height, enabled=s.depth.scale_from_floor
        )

        if self.depth_source != "oracle":
            self.status("Warming up Depth Anything...")

            blank = np.zeros((self.size[1], self.size[0], 3), np.uint8)
            pair = rectify.RectifiedPair(
                blank, blank, self.rectifier.K, self.rectifier.baseline
            )
            self.depth.estimate(pair)

        self.status("Brain ready.")

    def _prepare_memory(self, fresh):
        if fresh and self.memory_dir.exists():
            previous = self.memory_dir.with_name(self.memory_dir.name + ".previous")
            shutil.rmtree(previous, ignore_errors=True)
            self.memory_dir.rename(previous)

        self.memory_dir.mkdir(parents=True, exist_ok=True)

    def _load_memory(self):
        """Pick up last session's map, if there is one.

        This is the proposal's persistent digital representation in its
        simplest form: the drone is meant to come back and compare, and it
        cannot compare against something it threw away at exit.
        """
        meta_path = self.memory_dir / "meta.json"
        self.saved_root_map = None

        if not meta_path.exists():
            return

        meta = json.loads(meta_path.read_text())

        if (self.memory_dir / "grid.npz").exists():
            self.grid = occupancy.OccupancyGrid(self.settings.occupancy, (0.0, 0.0))
            self.grid.load(self.memory_dir / "grid.npz")

        # With RTAB-Map, what was saved is in LAST session's map frame. Until
        # RTAB-Map recognises somewhere from it, this session's poses are in a
        # different frame and must not be written into the old map.
        if self.pose_source == "rtabmap":
            self.saved_root_map = meta.get("root_map", 0)

        self.status(
            f"Remembered a map from {meta.get('saved', 'last time')}; "
            "looking for somewhere I recognise."
        )

    def save(self):
        with self.lock:
            if self.grid is not None:
                self.grid.save(self.memory_dir / "grid.npz")

            latest = self.localizer.fixes.latest

            if self.saved_root_map is not None:
                root_map = self.saved_root_map
            else:
                root_map = max(0, latest.root_map) if latest is not None else 0

            meta = {
                "scene": self.scene,
                "pose": self.pose_source,
                "root_map": root_map,
                "saved": time.strftime("%Y-%m-%d %H:%M:%S"),
            }

            (self.memory_dir / "meta.json").write_text(json.dumps(meta, indent=1))

    # -----------------------------------------------------------------
    # Running
    # -----------------------------------------------------------------

    def run(self):
        """Serve the sim until it disconnects, then save and stop."""
        for loop in (self._perceive, self._publish):
            threading.Thread(
                target=self._guarded(loop), daemon=True, name=loop.__name__
            ).start()

        self.link.send("ready", {})

        try:
            while self.running:
                message = self.link.receive()

                if message.kind == "frame":
                    self._on_frame(message)
                elif message.kind == "command":
                    self._command(message.fields.get("name", ""), message.fields)
                elif message.kind == "bye":
                    break

        except Closed:
            pass

        finally:
            self.running = False
            self.status("Saving the map...")
            self.save()
            self.localizer.close()
            self.link.close()
            print("Brain stopped; map saved to", self.memory_dir, flush=True)

    def _guarded(self, loop):
        def run():
            while self.running:
                try:
                    loop()
                    return
                except Closed:
                    return
                except Exception as error:
                    traceback.print_exc()
                    self.status(f"{loop.__name__} failed: {error!r}; restarting it")
                    time.sleep(1.0)

        return run

    def _on_frame(self, message):
        frame = cameras.from_message(message)
        pair = self.rectifier(frame)

        # While flying, the lenses are the altitude higher above the floor.
        self.altitude = float(message.fields.get("altitude", 0.0) or 0.0)

        self.localizer.submit(frame, pair)
        self.latest.put(frame.id, (frame, pair))
        self.last_stamp = frame.stamp
        self.rates["frames"].tick()

    def _command(self, name, fields=None):
        fields = fields or {}

        if name == "plan":
            self._plan(tuple(fields["xy"]), fields.get("label", "there"))
        elif name == "clear":
            with self.lock:
                self.route = None
        elif name == "save":
            self.save()
            self.status("Map saved.")

    # -----------------------------------------------------------------
    # Where the drone is
    # -----------------------------------------------------------------

    def _body(self, lens_pose):
        """The drone's centre and heading, from the left lens's pose.

        The lens sits half a baseline to the left of the sphere's centre, so
        the centre is half a baseline to its right. The glasses also had to
        step back from the lens to the middle of a head; a sphere's centre is
        the lens's own position, so that term is gone.
        """
        x, y, yaw = frames.planar(lens_pose)
        right = self.rectifier.baseline / 2.0

        return (x + right * math.sin(yaw), y - right * math.cos(yaw)), yaw

    def _footprint(self):
        """The drone as a circle in the base frame of the left lens."""
        return (
            0.0,
            -self.rectifier.baseline / 2.0,
            float(self.settings.drone.radius),
        )

    def _localized(self, fix):
        """False while a saved map has not been recognised yet."""
        if self.saved_root_map is None:
            return True

        if fix.root_map == self.saved_root_map:
            self.saved_root_map = None
            self.epoch += 1
            self.status("I know where we are.")
            return True

        if not hasattr(self, "_waiting_since"):
            self._waiting_since = time.monotonic()

        waited = time.monotonic() - self._waiting_since

        if waited > float(self.settings.localization.relocalize_within):
            self.saved_root_map = None
            self.epoch += 1

            # Kept, not deleted: RTAB-Map may still recognise it later.
            for name in ("grid.npz", "meta.json"):
                saved = self.memory_dir / name

                if saved.exists():
                    saved.rename(saved.with_name(saved.name + ".unrecognised"))

            with self.lock:
                self.grid = None

            self.status("I don't recognise this place; starting a new map.")
            return True

        return False

    def _on_graph(self, poses):
        self.graph_dirty = True

    def _apply_graph(self):
        """Redraw the map after a loop closure.

        This is what RTAB-Map buys that plain odometry cannot: when it
        recognises somewhere it has been, it corrects every pose behind it,
        and the kept scans are redrawn at the corrected poses. Without it, a
        circuit of a house leaves two copies of every wall.

        At most every few seconds -- closures arrive in bursts, and redrawing
        a long drive takes a moment.
        """
        if not self.graph_dirty or time.monotonic() - self._rebuilt_at < 3.0:
            return

        self.graph_dirty = False
        self._rebuilt_at = time.monotonic()
        poses = dict(self.localizer.node_poses)

        with self.lock:
            if self.corrector is not None:
                self.corrector.reset()

            if self.grid is not None:
                self.grid.rebuild(poses)

    # -----------------------------------------------------------------
    # The perception loop
    # -----------------------------------------------------------------

    def _perceive(self):
        last = -1

        while self.running:
            item = self.latest.newer_than(last)

            if item is None:
                continue

            frame, pair = item
            last = frame.id

            fix = self.localizer.fix(frame.id, timeout=2.0)

            if fix is None or not fix.ok or not self._localized(fix):
                continue

            # The lens height the map measures from. On the ground it is the
            # drone's radius; airborne, it is that plus however high it is.
            lens_height = self.lens_height + self.altitude

            started = time.monotonic()
            depth_map = self.depth.estimate(pair, frame)
            measured = time.monotonic()

            tilt = self._tilt(frame)
            tilted = fix.pose.copy()

            if tilt is not None:
                tilted[:3, :3] = fix.pose[:3, :3] @ tilt

            self.floor.lens_height = lens_height
            scale = self.floor.update(depth_map, frames.world_from_optical(tilted))
            floor_seen = scale is not None

            # No floor in view: borrow the recent scale for what is close, and
            # declare nothing free.
            if scale is None:
                scale = 1.0 if self.depth_source == "oracle" else self.floor.fallback()

            if scale is None:
                continue

            depth_map.depth = depth_map.depth * scale
            depth_map.scale = scale

            # Indoors or out? Measured, not assumed: a ceiling is a large
            # flat surface at a constant height above the lens, and open sky
            # is not. See FloorScale.ceiling. -1 means this view could not
            # tell, which is different from "no roof".
            overhead = self.floor.ceiling(
                depth_map, frames.world_from_optical(tilted)
            )
            roofed = -1 if not floor_seen else int(overhead is not None)

            scan = occupancy.make_scan(
                depth_map.depth,
                depth_map.K,
                lens_height,
                self.settings.occupancy,
                self._footprint(),
                fix.anchor,
                fix.anchor_from_base,
                free_space=floor_seen or self.depth_source == "oracle",
                tilt=tilt,
                # On the ground the drone's own footprint is known free. In
                # the air it is not: there may be nothing under it at all.
                footprint_free=self.altitude < 0.05,
                roofed=roofed,
            )

            with self.lock:
                pose = fix.pose

                if self.corrector is not None:
                    pose = getattr(fix, "raw_pose", fix.pose)

                    if floor_seen and self.grid is not None:
                        self.corrector.update(self.grid, scan.hits, pose)

                if self.grid is None:
                    self.grid = occupancy.OccupancyGrid(
                        self.settings.occupancy, pose[:2, 3]
                    )

                self.grid.add(scan, pose)

                self.view = View(frame.id, depth_map, pose, frame.size)

            self.timing.update(
                depth=(measured - started) * 1000.0,
                map=(time.monotonic() - measured) * 1000.0,
            )
            self.rates["perception"].tick()

            self._apply_graph()

    def _tilt(self, frame):
        """This frame's camera tilt as a rotation, or None for level.

        From the IMU's gravity reading, exactly as the glasses did it. For
        DELTA the gimbal holds the lenses level, so the true tilt is zero and
        what arrives is only the chip's own error -- but it is still applied,
        because on real hardware the gimbal will not be perfect, and because
        heights in the map are read straight off the image rows: half a degree
        of unnoticed tilt lifts a point 3 m away by 2.6 cm.
        """
        if not self.use_tilt or (frame.imu_pitch is None and frame.imu_roll is None):
            return None

        return frames.level_from_tilted(frame.imu_pitch or 0.0, frame.imu_roll or 0.0)

    # -----------------------------------------------------------------
    # Planning on the map the drone built
    # -----------------------------------------------------------------

    def _plan(self, target_xy, label):
        with self.lock:
            if self.grid is None:
                self.status("Nothing mapped yet.")
                return

            fix = self.localizer.fixes.latest

            if fix is None or fix.pose is None:
                self.status("I do not know where I am yet.")
                return

            here, _ = self._body(fix.pose)

            nav = navmap.build(
                self.grid,
                self.settings.planner,
                wearer_xy=here,
                wearer_radius=float(self.settings.drone.radius),
            )

            if nav is None:
                self.status("Nothing mapped yet.")
                return

            points = planner.plan(nav, here, target_xy)

            if not points:
                self.status(f"No route to {label} through what I have mapped.")
                self.route = None
                return

            self.route = Route(points=points, target=tuple(target_xy), label=label)

            length = sum(
                math.dist(a, b) for a, b in zip(points, points[1:])
            )
            self.status(f"Route to {label}: {length:.1f} m, {len(points)} corners.")

    # -----------------------------------------------------------------
    # Telling the window what it knows
    # -----------------------------------------------------------------

    def status(self, text):
        print(text, flush=True)

        try:
            self.link.send("status", {"text": text})
        except Closed:
            pass

    def _publish(self):
        saved_at = time.monotonic()
        every = float(self.settings.memory.get("autosave_seconds", 30.0))

        while self.running:
            time.sleep(0.25)

            # Save as it goes, not only on the way out. The map used to be
            # written once, when the sim disconnected -- so a crash, a killed
            # terminal or a closed laptop lost the whole session's work, and
            # the drone that is supposed to come back and compare had nothing
            # to come back to.
            if every > 0 and time.monotonic() - saved_at >= every:
                saved_at = time.monotonic()

                if self.grid is not None:
                    self.save()

            fields, arrays = self._snapshot()

            try:
                self.link.send("state", fields, arrays)
            except Closed:
                return

    def _roof_counts(self):
        """How many mapped cells are indoors and outdoors, for the status bar."""
        if self.grid is None:
            return None

        from delta.brain.occupancy import ROOFED, UNROOFED

        roof = self.grid.roof_classes()

        return {
            "indoor": int((roof == ROOFED).sum()),
            "outdoor": int((roof == UNROOFED).sum()),
        }

    def _snapshot(self):
        fields = {"epoch": self.epoch}
        arrays = {}

        fix = self.localizer.fixes.latest

        if fix is not None and fix.pose is not None:
            x, y, yaw = frames.planar(fix.pose)
            body, _ = self._body(fix.pose)

            fields["pose"] = {
                "frame": fix.frame,
                "x": x,
                "y": y,
                "yaw": yaw,
                "body": list(body),
            }

            if getattr(fix, "raw_pose", None) is not None:
                rx, ry, ryaw = frames.planar(fix.raw_pose)
                fields["pose"]["raw"] = [rx, ry, ryaw]

        with self.lock:
            if self.grid is not None:
                bounds = self.grid.known_bounds(margin=0.5)

                if bounds is not None:
                    r0, r1, c0, c1 = bounds

                    arrays["grid"] = self.grid.classes()[r0:r1, c0:c1]
                    arrays["roof"] = self.grid.roof_classes()[r0:r1, c0:c1]

                    # What the drone could DO in each cell -- roll across,
                    # roll under, fly over -- once the grid keeps height
                    # bands. Until it does, the window falls back to the
                    # plain free/solid classes above.
                    if hasattr(self.grid, "moves"):
                        arrays["moves"] = self.grid.moves(
                            float(self.settings.drone.diameter)
                        )[r0:r1, c0:c1]

                    corner = self.grid.to_world([c0, r0])
                    fields["grid"] = {
                        "origin": list(corner),
                        "resolution": self.grid.resolution,
                    }

            if self.route is not None:
                fields["route"] = {
                    "points": [list(p) for p in self.route.points],
                    "label": self.route.label,
                    "target": list(self.route.target),
                }

            view = self.view

            if view.depth is not None:
                arrays["depth"] = np.clip(
                    view.depth.depth / 6.0 * 255.0, 0, 255
                ).astype(np.uint8)

        fields["stats"] = {
            "rates": {name: round(rate.value, 1) for name, rate in self.rates.items()},
            "ms": {name: round(value) for name, value in self.timing.items()},
            "localization": dict(self.localizer.stats),
            "floor_scale": round(self.floor.scale, 3),
            "lens_height": round(self.lens_height + self.altitude, 3),
            "altitude": round(self.altitude, 3),
            "map_localize": (
                dict(self.corrector.stats) if self.corrector is not None else None
            ),
            "sources": {"pose": self.pose_source, "depth": self.depth_source},
            "relocalizing": self.saved_root_map is not None,
            "roofed": self._roof_counts(),
        }

        return fields, arrays
