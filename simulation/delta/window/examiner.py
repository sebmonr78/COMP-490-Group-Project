"""The examiner: the only code that reads the simulator's truth.

The brain never sees any of this. It is here to CHECK the brain, and to show
the truth beside what the brain believes:

    the true layout    the grey map under the brain's own map (M hides it)
    Alignment          lines the brain's map up with the true one, once, at
                       the start -- only so the two can be drawn on top of
                       each other. Never refitted: drift shows as drift
    true heights       the "truth" column of the height box; true positions
                       for the green ring
    notes              where the drone really was at every frame, so the
                       brain's answer about a frame is checked against it
    scores             score_map, score_rooms, score_replica: the brain's
                       map, rooms and 3D replica against the house, to
                       outputs/eval/
    --trace            position error, 4 times a second, to a file

DroneApp (app.py) mixes this in, so these are the app's own methods and use
its world and its notes.
"""

import json
import math
import os
from pathlib import Path

import cv2
import habitat_sim
import magnum as mn
import numpy as np
from habitat.utils.visualizations import maps

from delta import frames
from delta.sim import first_hit
from delta.window.colours import ROOM_COLOURS, TRUTH_EDGE, TRUTH_FLOOR

PROJECT_ROOT = Path(__file__).resolve().parents[2]


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


class Examiner:
    """The app's examiner methods: the truth, kept, drawn and scored."""

    def _note_truth(self, seen):
        """Where the drone really was at this frame -- kept to check the
        brain's answers about the frame when they come back, a few frames
        later, against the right moment."""
        position, rotation = seen["true_position"], seen["true_rotation"]
        self.truth_by_frame[self.frame_id] = frames.planar(
            frames.world_from_habitat_camera(position, rotation)
        )

        # Where the lens really looked from, every 25 cm or 15 degrees: what
        # the replica is scored against (score_replica).
        last = self.true_views[-1] if self.true_views else None

        if last is None or math.dist(last[0], position) > 0.25 or \
                abs(float(np.dot(last[1], rotation))) < math.cos(math.radians(15.0) / 2.0):
            self.true_views.append((list(position), np.array(rotation)))
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
            # How many frames a second the brain is turning into map.
            "perception_hz": stats.get("rates", {}).get("perception"),
            "rtabmap_ms": loc.get("ms"),
        }

        with open(self.args.trace, "a") as out:
            out.write(json.dumps(line) + "\n")

    def _true_rooms(self):
        """HSSD's own room outlines for this house, as a picture on the
        navmesh grid: (labels, names), 0 where no room is. The answer key --
        for scoring the brain's rooms, never sent to it."""
        if getattr(self, "_rooms_truth", None) is not None:
            return self._rooms_truth

        dataset = self.world.config["dataset"]
        root = Path(os.environ.get(dataset["environment_variable"], dataset["default_root"])).expanduser()
        regions = json.loads((root / dataset["semantic_file"]).read_text()).get("region_annotations", [])

        navigable, lower = self._truth_grid()
        labels = np.zeros(navigable.shape, np.uint8)
        names = [None]

        # Outdoor areas first, so a room drawn over one wins.
        for region in sorted(regions, key=lambda r: r["label"] != "outdoor"):
            points = np.array([[(p[0] - lower[0]) / 0.05, (p[2] - lower[1]) / 0.05] for p in region["poly_loop"]])
            names.append(region["label"])
            cv2.fillPoly(labels, [np.round(points).astype(np.int32)], len(names) - 1)

        self._rooms_truth = (labels, names, self._join_open_rooms(labels, names, lower))
        return self._rooms_truth

    def _join_open_rooms(self, labels, names, lower):
        """HSSD's rooms as WALLS make them: neighbours with no wall between.

        HSSD names rooms by use. An open-plan living room, dining room and
        kitchen are three rooms to it and one space to anything that only
        sees walls -- a robot vacuum too, which shows them as one room until
        someone splits it. So: along every boundary two outlines share, look
        across it with a ray at 1.2 and 1.6 m. If most of it is open, the two
        are one room here. The by-use split needs to know what is IN a room
        (a stove, a sofa), which is a later feature; both scores are kept.
        """
        simulator = self.world.simulator
        me = self.world.drone.object.object_id
        joined = list(range(len(names)))

        def root(i):
            while joined[i] != i:
                i = joined[i]
            return i

        pairs = {}

        for (sa, sb), (da, db) in (
            ((slice(1, None), slice(None)), (1, 0)),
            ((slice(None), slice(1, None)), (0, 1)),
        ):
            here = labels[:labels.shape[0] - da, :labels.shape[1] - db]
            there = labels[sa, sb]
            touching = (here != there) & (here > 0) & (there > 0)

            for row, col in zip(*np.nonzero(touching)):
                a, b = int(here[row, col]), int(there[row, col])

                if "outdoor" in (names[a], names[b]):
                    continue

                pairs.setdefault((min(a, b), max(a, b)), []).append((row, col, da, db))

        for (a, b), cells in pairs.items():
            sample = cells[:: max(1, len(cells) // 40)]
            walls = 0

            for row, col, da, db in sample:
                x = lower[0] + (col + 0.5) * 0.05
                z = lower[1] + (row + 0.5) * 0.05
                across = mn.Vector3(float(db), 0.0, float(da))

                for height in (1.2, 1.6):
                    start = mn.Vector3(x, self.floor_height + height, z) - across * 0.3
                    if first_hit(simulator, start, across, 0.65, me) is not None:
                        walls += 1
                        break

            if walls < 0.5 * len(sample):
                joined[root(b)] = root(a)

        return [root(i) for i in range(len(names))]

    def _truth_grid(self):
        """The navmesh as a 5 cm picture, and its corner (x, z)."""
        if self._navigable is None:
            self._navigable = self._floor_slice(self.world.pathfinder)
            lower, _ = self.world.pathfinder.get_bounds()
            self._nav_lower = np.array([lower.x, lower.z])

        return self._navigable, self._nav_lower

    def _floor_slice(self, pathfinder):
        """A navmesh at FLOOR level, as a 5 cm picture: where its robot can
        be on the floor.

        Not maps.get_topdown_map: that slices with Habitat's default slack
        of 0.5 m, so the tops of beds and sofa seats -- separate navmesh
        islands, 0.4-0.5 m up -- came out as floor, and the scores called
        the brain wrong for marking a bed occupied. 10 cm is the floor, its
        rugs and its thresholds, and nothing standing on it.
        """
        return np.asarray(pathfinder.get_topdown_view(0.05, self.floor_height, 0.10), dtype=bool)

    def score_rooms(self, name):
        """Score the brain's rooms against HSSD's room outlines.

            recall      per true room the brain has mapped (30% or more of
                        it), the share of its mapped floor that lies in ONE
                        brain room -- 1.0 if no room was cut in pieces
            precision   per brain room, the share of it in ONE true room --
                        1.0 if no two rooms were run together

        The measures of Bormann et al.'s survey, on the part of the house
        that was explored. Appends to outputs/eval/rooms.jsonl and saves a
        picture: true outlines over the brain's rooms.
        """
        state = self.brain.state if self.brain else None
        arrays = self.brain.state_arrays if self.brain else {}

        if not (state and "rooms" in state and "rooms" in arrays and self.alignment.ready):
            print(f"[rooms] {name}: no rooms from the brain yet", flush=True)
            return

        truth, names, joined = self._true_rooms()
        _, lower = self._truth_grid()
        height, width = truth.shape

        meta = state["rooms"]
        r = 0.05
        cell = np.array([[meta["resolution"], 0.0, meta["origin"][0]], [0.0, meta["resolution"], meta["origin"][1]]])
        a = self.alignment.matrix()
        world = a[:, :2] @ cell
        world[:, 2] += a[:, 2]
        to_pixel = np.array([[1 / r, 0.0, -lower[0] / r], [0.0, -1 / r, -lower[1] / r]])
        affine = to_pixel[:, :2] @ world
        affine[:, 2] += to_pixel[:, 2]

        brain = cv2.warpAffine(arrays["rooms"].astype(np.uint8), affine, (width, height), flags=cv2.INTER_NEAREST)
        kinds = {room["id"]: room for room in meta["list"]}

        # By walls: HSSD's rooms with the open-plan ones joined.
        group_names = {}

        for g in range(1, len(names)):
            group_names.setdefault(joined[g], set()).add(names[g])

        by_walls = np.array(joined, np.int32)[truth]
        walls_names = {g: "+".join(sorted(n)) for g, n in group_names.items()}
        walls_names[0] = None

        walls_score = _room_scores(brain, by_walls, walls_names, kinds)
        use_score = _room_scores(brain, truth.astype(np.int32), dict(enumerate(names)), kinds)

        result = {
            "name": name,
            "time_s": round(self.clock, 1),
            "rooms_brain": len(kinds),
            "doors": len(meta["doors"]),
            "by_walls": walls_score,
            "by_use": {k: use_score[k] for k in ("rooms_true_explored", "recall", "precision")},
        }
        print(f"[rooms] {json.dumps(result)}", flush=True)

        folder = PROJECT_ROOT / "outputs" / "eval"
        (folder / "rooms").mkdir(parents=True, exist_ok=True)

        with open(folder / "rooms.jsonl", "a") as log:
            log.write(json.dumps(result) + "\n")

        # The brain's rooms in colour; HSSD's outlines by use in grey, and the
        # rooms the walls make -- what the brain is scored against -- in white.
        palette = np.array([(15, 15, 15)] + [c[::-1] for c in ROOM_COLOURS] * 26, np.uint8)[:256]
        picture = palette[brain]
        square = np.ones((3, 3), np.uint8)
        picture[cv2.morphologyEx(truth.astype(np.uint8), cv2.MORPH_GRADIENT, square) > 0] = (110, 110, 110)
        picture[cv2.morphologyEx(by_walls.astype(np.uint8), cv2.MORPH_GRADIENT, square) > 0] = (255, 255, 255)

        for g in range(1, len(names)):
            ys, xs = np.nonzero(truth == g)

            if len(xs):
                cv2.putText(picture, names[g], (int(xs.mean()) - 20, int(ys.mean())),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)

        cv2.imwrite(str(folder / "rooms" / f"{name}.png"), picture)

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

    def _true_ceiling(self, xy=None):
        """True height above the floor of the ceiling straight over the
        drone (or over a point, east/north in the true frame), None if
        nothing is over it -- for scoring the brain's ceiling, never sent."""
        if xy is None:
            origin = self.world.drone.object.translation
            ignore = self.world.drone.object.object_id
        else:
            origin, ignore = mn.Vector3(float(xy[0]), self.floor_height + 1.4, -float(xy[1])), None

        up = first_hit(self.world.simulator, origin, mn.Vector3(0.0, 1.0, 0.0), 10.0, ignore)

        if up is None:
            return None

        return round(float(origin.y) + up - self.floor_height, 3)

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

        self._truth_grid()
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

        # The solid layer -- what is THERE, at any height -- against the truth
        # of the same question (_true_solid). Edges, because that is what a
        # camera sees of a sofa or a wall: its inside stays unknown.
        brain_solid = cv2.warpAffine(arrays.get("solid", arrays["grid"]).astype(np.uint8), affine,
                                     (width, height), flags=cv2.INTER_NEAREST, borderValue=0)
        true_occupied, true_free = (np.pad(a, padding) for a in self._true_solid())
        explored = cv2.dilate((brain_solid > 0).astype(np.uint8), disc(0.10)).astype(bool)
        edges = true_occupied & cv2.dilate(true_free.astype(np.uint8), disc(0.05)).astype(bool) & explored
        near = disc(0.05)
        brain_occupied = brain_solid == 2

        solid_scores = {
            # Of the true edges of things where the drone has looked: marked occupied.
            "solid_found": round(float((edges & cv2.dilate(brain_occupied.astype(np.uint8), near).astype(bool)).sum()
                                       / max(edges.sum(), 1)), 3),
            # Of what it marked occupied: something really there.
            "solid_right": round(float((brain_occupied & cv2.dilate(true_occupied.astype(np.uint8), near).astype(bool)).sum()
                                       / max(brain_occupied.sum(), 1)), 3),
            # Of what it marked free: something really there after all.
            "solid_free_wrong": round(float(((brain_solid == 1) & ~cv2.dilate(true_free.astype(np.uint8), near).astype(bool)).sum()
                                            / max((brain_solid == 1).sum(), 1)), 3),
        }

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
            **solid_scores,
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

        # The solid layer: true things dark grey, the brain's occupied white
        # where right and orange where nothing is there, its free blue, and
        # true edges it has looked at but not marked in red.
        picture = np.full((height, width, 3), 15, np.uint8)
        picture[true_occupied] = (55, 55, 55)
        picture[brain_solid == 1] = (170, 110, 60)
        picture[brain_occupied] = (0, 150, 255)
        picture[brain_occupied & cv2.dilate(true_occupied.astype(np.uint8), near).astype(bool)] = (235, 235, 235)
        picture[edges & ~cv2.dilate(brain_occupied.astype(np.uint8), near).astype(bool)] = (40, 40, 230)
        cv2.imwrite(str(folder / "live_map" / f"{name}_solid.png"), picture)

    def score_replica(self, name):
        """Score the brain's 3D replica against the real house.

        The truth: true depth rendered from every place the lens looked from
        (every 25 cm or 15 degrees of the drive), cut to what the brain
        keeps -- within 4 m, below 3 m -- and thinned to one point per
        2.5 cm. The brain's cubes are put in the true frame by the same
        alignment the map panel uses: drift counts against it.

            accuracy      median distance, brain cube -> nearest true surface
            precision     share of brain cubes within 5 cm of a true surface
            completeness  share of true surface within 5 cm of a brain cube
            F-score       the two together: 2PR / (P + R)

        The measures 3D-reconstruction papers use (5 cm, as on Replica and
        ScanNet). Appends to outputs/eval/replica.jsonl and saves a picture
        from above: true surface grey, brain cubes green within 5 cm and red
        further, true surface it missed blue.
        """
        from scipy.spatial import cKDTree

        replica = self.brain.replica if self.brain else None

        if replica is None or not self.alignment.ready or not self.true_views:
            print(f"[replica] {name}: no replica from the brain yet", flush=True)
            return

        settings = self.settings.replica
        max_range, max_height = float(settings.max_range), float(settings.max_height)
        views = self.true_views[:: max(1, len(self.true_views) // 200)]
        truth = []

        for position, rotation in views:
            depth, K = self.world.true_depth_at(position, rotation)
            height, width = depth.shape
            v, u = np.mgrid[0:height:2, 0:width:2].astype(np.float32)
            d = depth[::2, ::2]

            # Habitat's camera: x right, y up, looking down -z.
            camera = np.stack([(u + 0.5 - K[0, 2]) / K[0, 0] * d, -(v + 0.5 - K[1, 2]) / K[1, 1] * d, -d], axis=-1)
            keep = (d > 0.1) & (np.hypot(camera[..., 0], camera[..., 2]) < max_range)
            # The lens's rotation as a matrix: its columns are where it turns
            # the x, y and z axes to.
            w, x, y, z = rotation
            q = mn.Quaternion(mn.Vector3(x, y, z), w)
            R = np.array([list(q.transform_vector(mn.Vector3(*axis))) for axis in np.eye(3)], np.float32).T
            world = camera[keep] @ R.T + np.asarray(position, np.float32)

            # Habitat (x, y up, z) -> east, north, height above the floor.
            points = np.stack([world[:, 0], -world[:, 2], world[:, 1] - self.floor_height], axis=1)
            truth.append(points[points[:, 2] < max_height])

        truth = np.concatenate(truth)
        truth = np.unique(np.floor(truth / 0.025).astype(np.int64), axis=0).astype(np.float32) * 0.025 + 0.0125

        index, colour, info = replica
        cubes = (index.astype(np.float32) + 0.5) * float(info["voxel"])
        a = self.alignment.matrix()
        cubes = np.stack([a[0, 0] * cubes[:, 0] + a[0, 1] * cubes[:, 1] + a[0, 2],
                          a[1, 0] * cubes[:, 0] + a[1, 1] * cubes[:, 1] + a[1, 2], cubes[:, 2]], axis=1)

        to_truth, _ = cKDTree(truth).query(cubes, distance_upper_bound=1.0)
        to_cubes, _ = cKDTree(cubes).query(truth, distance_upper_bound=1.0)

        precision = float((to_truth < 0.05).mean())
        completeness = float((to_cubes < 0.05).mean())

        result = {
            "name": name,
            "time_s": round(self.clock, 1),
            "cubes": int(len(cubes)),
            "true_points": int(len(truth)),
            "views_scored": len(views),
            "accuracy_m": round(float(np.median(np.minimum(to_truth, 1.0))), 3),
            "precision": round(precision, 3),
            "completeness": round(completeness, 3),
            "f_score": round(2 * precision * completeness / max(precision + completeness, 1e-9), 3),
            "pose_error_m": round(self._pose_error(self.brain.state)[0], 3) if self.brain.state and self._pose_error(self.brain.state) else None,
            "ceiling": self._score_ceiling(info),
        }
        print(f"[replica] {json.dumps(result)}", flush=True)

        folder = PROJECT_ROOT / "outputs" / "eval"
        (folder / "replica").mkdir(parents=True, exist_ok=True)

        with open(folder / "replica.jsonl", "a") as log:
            log.write(json.dumps(result) + "\n")

        # From above, 5 cm a pixel.
        lower = np.minimum(truth.min(axis=0), cubes.min(axis=0))[:2]
        upper = np.maximum(truth.max(axis=0), cubes.max(axis=0))[:2]
        size = np.ceil((upper - lower) / 0.05).astype(int) + 1
        picture = np.full((size[1], size[0], 3), 12, np.uint8)

        def pixels(points):
            cols = ((points[:, 0] - lower[0]) / 0.05).astype(int)
            rows = (size[1] - 1 - (points[:, 1] - lower[1]) / 0.05).astype(int)
            return rows, cols

        picture[pixels(truth)] = (90, 90, 90)
        picture[pixels(truth[to_cubes >= 0.05])] = (220, 120, 40)
        picture[pixels(cubes[to_truth < 0.05])] = (80, 200, 90)
        picture[pixels(cubes[to_truth >= 0.05])] = (40, 40, 230)
        cv2.imwrite(str(folder / "replica" / f"{name}.png"), picture)

    def _score_ceiling(self, info):
        """The brain's ceiling against the real one: for every patch the
        brain has a ceiling for, a ray straight up from 1.4 m at the patch's
        centre, put in the true frame by the alignment. Median error, and
        the share within 10 cm (two cube layers)."""
        heights = info.get("ceiling")

        if heights is None or np.all(np.isnan(heights)):
            return {"patches": 0}

        heights = heights.astype(np.float32)
        cell, origin = float(info["ceiling_cell"]), np.asarray(info["ceiling_origin"], np.float64)
        errors, brain, truth = [], [], []

        for row, col in zip(*np.nonzero(~np.isnan(heights))):
            centre = origin + (np.array([col, row]) + 0.5) * cell
            true = self._true_ceiling(self.alignment.point(centre))

            if true is not None:
                errors.append(float(heights[row, col]) - true)
                brain.append(float(heights[row, col]))
                truth.append(true)

        if not errors:
            return {"patches": int((~np.isnan(heights)).sum()), "scored": 0}

        errors = np.abs(np.array(errors))
        return {
            "patches": int((~np.isnan(heights)).sum()),
            "scored": len(errors),
            "median_error_m": round(float(np.median(errors)), 3),
            "within_10cm": round(float((errors <= 0.10).mean()), 3),
            "brain_typical_m": round(float(np.median(brain)), 2),
            "true_typical_m": round(float(np.median(truth)), 2),
        }

    def _true_solid(self):
        """Where something really stands between just off the floor and
        just under the ceiling, and where nothing does: (occupied, free) on
        the navmesh picture. The answer key for the solid layer.

        Asked of Habitat's own geometry by building a second navmesh for an
        imaginary robot 1.8 m tall and 5 cm wide, which can stand only where
        nothing at all is overhead below 1.8 m. Where it can stand is free;
        what it cannot, near floor, is something there. Built once a run.
        """
        if getattr(self, "_solid_truth", None) is not None:
            return self._solid_truth

        navigable, lower = self._truth_grid()

        settings = habitat_sim.NavMeshSettings()
        settings.set_defaults()
        settings.agent_height = 1.8
        settings.agent_radius = 0.05
        settings.agent_max_climb = 0.08
        settings.agent_max_slope = 45.0
        settings.cell_size = 0.05
        settings.cell_height = 0.02
        settings.include_static_objects = True

        tall = habitat_sim.nav.PathFinder()
        self.world.simulator.recompute_navmesh(tall, settings)

        # Its own picture, then shifted onto the drone navmesh's grid.
        free = self._floor_slice(tall)
        tall_lower, _ = tall.get_bounds()
        dc = int(round((tall_lower.x - lower[0]) / 0.05))
        dr = int(round((tall_lower.z - lower[1]) / 0.05))

        true_free = np.zeros(navigable.shape, bool)
        rows = slice(max(0, dr), min(navigable.shape[0], dr + free.shape[0]))
        cols = slice(max(0, dc), min(navigable.shape[1], dc + free.shape[1]))
        true_free[rows, cols] = free[rows.start - dr:rows.stop - dr, cols.start - dc:cols.stop - dc]

        # Something there: not free, but next to floor someone could reach --
        # walls, furniture, the space under a table -- not the world outside.
        square = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13))
        near_floor = cv2.dilate((navigable | true_free).astype(np.uint8), square).astype(bool)
        true_occupied = near_floor & ~true_free

        self._solid_truth = (true_occupied, true_free)
        return self._solid_truth


def _room_scores(brain, truth, names, kinds):
    """Recall and precision of the brain's rooms against one
    truth picture (see DroneApp.score_rooms). `names` maps a truth id to its
    name; `kinds` the brain's room id to its published description."""
    mapped = brain > 0
    recalls, precisions, rows = [], [], []

    for g in np.unique(truth[truth > 0]):
        inside = truth == g
        seen = inside & mapped

        if seen.sum() < 0.3 * inside.sum() or "outdoor" in (names.get(int(g)) or ""):
            continue

        recalls.append(np.bincount(brain[seen], minlength=256).max() / seen.sum())

    for b, room in kinds.items():
        labelled = (brain == b) & (truth > 0)

        if labelled.sum() == 0:
            continue

        values, counts = np.unique(truth[labelled], return_counts=True)
        best = int(values[counts.argmax()])
        share = counts.max() / labelled.sum()
        precisions.append(share)
        rows.append(f"{b}: {room['area']} m2 -> {names.get(best)} ({share:.0%})")

    return {
        "rooms_true_explored": len(recalls),
        "recall": round(float(np.mean(recalls)), 3) if recalls else None,
        "precision": round(float(np.mean(precisions)), 3) if precisions else None,
        "rooms": rows,
    }
