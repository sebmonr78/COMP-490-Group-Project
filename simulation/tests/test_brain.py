"""Check the brain is honest, and that its parts do what they claim.

No scene, no GPU, no Depth Anything, no RTAB-Map. Everything here is either
read off the source or run on numbers made up in this file, so it takes a
second and can be run after every edit.

    conda activate brain
    cd ~/COMP-490-Group-Project/simulation
    python tests/test_brain.py

Two halves, and the first is the one that matters most.

THE BRAIN MUST NOT CHEAT. Habitat knows exactly where the drone is, exactly
how far away everything is, and the true floor plan. If any of that reaches
the brain, every number this project reports becomes meaningless -- and the
failure is silent, because a brain fed the answers looks like a brain that is
working very well. So the separation is tested rather than trusted:

    nothing under delta/brain/ may import Habitat, Magnum, or the sim
    the sim may only send the truth when an --oracle flag asked for it

The oracle flags are deliberately kept. They are not cheating as long as you
have to ask for them: `--pose oracle` and `--depth oracle` replace one stage
with the truth so you can see which stage is wrong. Using them by accident is
what the second test prevents.

THE PARTS MUST WORK. The rest are unit tests, most of them regressions for
bugs that cost real time: a floor finder that silently found nothing, a
ceiling test that is the whole indoor/outdoor answer, and a map that has to
survive a loop closure.
"""

import ast
import math
import sys
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

from delta.brain import config as brain_config
from delta.brain import depth as depth_module
from delta.brain import occupancy
from delta.brain.occupancy import FREE, OCCUPIED, ROOFED, UNKNOWN, UNROOFED


# ---------------------------------------------------------------------
# The brain must not cheat
# ---------------------------------------------------------------------

FORBIDDEN = ("habitat", "habitat_sim", "magnum", "delta.sim", "delta.drone")


def the_brain_cannot_reach_the_simulator():
    """Not one module under delta/brain/ may import the world.

    Read off the source rather than by trying it, because an import that only
    happens inside one rarely-taken branch would still be a way in.
    """
    offenders = []

    for path in sorted((PROJECT_ROOT / "delta" / "brain").rglob("*.py")):
        tree = ast.parse(path.read_text())

        for node in ast.walk(tree):
            names = []

            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [node.module or ""]

            for name in names:
                root = name.split(".")[0]

                if root in FORBIDDEN or name in FORBIDDEN:
                    offenders.append(f"{path.name}:{node.lineno} imports {name}")

    assert not offenders, "the brain can reach the simulator: " + "; ".join(offenders)

    looked = len(list((PROJECT_ROOT / "delta" / "brain").rglob("*.py")))

    return f"{looked} modules, none can import Habitat or the sim"


def the_truth_is_only_sent_when_asked_for():
    """Every `true_*` field the sim sends must sit inside an oracle test.

    Checked structurally, not by grepping: the assignment has to be nested
    inside an `if` whose text mentions "oracle". Moving one of these lines out
    of its guard by accident is exactly the edit this is here to catch.
    """
    source = (PROJECT_ROOT / "demo" / "run_drone.py").read_text()
    tree = ast.parse(source)
    lines = source.splitlines()

    # Every line that writes a true_* field into a message.
    writes = []

    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue

        for target in node.targets:
            if not isinstance(target, ast.Subscript):
                continue

            key = getattr(target.slice, "value", None)

            if isinstance(key, str) and key.startswith("true_"):
                writes.append((node.lineno, key))

    assert writes, "no true_* fields found at all -- has the sim stopped sending them?"

    # Which lines are inside an oracle-guarded `if`.
    guarded = set()

    for node in ast.walk(tree):
        if not isinstance(node, ast.If):
            continue

        if "oracle" not in ast.unparse(node.test):
            continue

        for inner in ast.walk(node):
            if hasattr(inner, "lineno"):
                guarded.add(inner.lineno)

    loose = [
        f"line {line} sends {key!r}: {lines[line - 1].strip()}"
        for line, key in writes
        if line not in guarded
    ]

    assert not loose, "the truth is sent unguarded: " + "; ".join(loose)

    return f"{len(writes)} true_* fields, all behind an oracle flag"


def the_brain_ignores_truth_it_was_not_given():
    """A frame with no oracle fields must carry none, not silently default."""
    from delta.brain import cameras

    @dataclass
    class FakeMessage:
        fields: dict
        arrays: dict

    picture = np.zeros((8, 8, 3), np.uint8)

    frame = cameras.from_message(
        FakeMessage(
            fields={"id": 1, "stamp": 0.1, "imu_yaw": 0.0},
            arrays={"left": picture, "right": picture},
        )
    )

    assert frame.true_pose is None, "a frame invented a true pose"
    assert frame.true_depth is None, "a frame invented a true depth"
    assert frame.true_altitude is None, "a frame invented a true altitude"

    return "a plain frame carries no truth"


# What the drone's own hardware measures, and so all a frame may carry
# besides oracle fields: the clock, the IMU, the cage's axle encoder, the
# down-facing rangefinder and the flight controller's vertical speed. The
# cameras' pictures travel as arrays.
SENSOR_FIELDS = {"id", "stamp", "imu_yaw", "imu_pitch", "imu_roll", "odometer", "range_down", "climb_rate"}


def the_sim_sends_only_what_sensors_measure():
    """Every field of a frame is a sensor reading, or a true_* oracle field.

    The true_* test above cannot see a truth sent under an innocent name.
    That is not hypothetical: the sim sent Habitat's altitude as plain
    "altitude" for weeks, and the brain built every flying map from it.
    """
    source = (PROJECT_ROOT / "demo" / "run_drone.py").read_text()
    tree = ast.parse(source)

    sender = next(
        node for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "_send_frame"
    )

    sent = set()

    for node in ast.walk(sender):
        # fields = {"id": ..., ...}
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == "fields" for t in node.targets
        ) and isinstance(node.value, ast.Dict):
            sent.update(k.value for k in node.value.keys if isinstance(k, ast.Constant))

        # fields["x"] = ...
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if (isinstance(target, ast.Subscript) and isinstance(target.value, ast.Name)
                        and target.value.id == "fields"):
                    key = getattr(target.slice, "value", None)

                    if isinstance(key, str):
                        sent.add(key)

    assert sent, "found no frame fields at all -- has _send_frame changed shape?"

    unknown = sorted(k for k in sent if k not in SENSOR_FIELDS and not k.startswith("true_"))

    assert not unknown, f"the sim sends fields no sensor measures: {unknown}"

    return f"{len(sent)} fields, all sensor readings or oracle-only"


# ---------------------------------------------------------------------
# The floor finder, at both heights
# ---------------------------------------------------------------------


@dataclass
class FakeDepth:
    depth: np.ndarray
    K: np.ndarray


def _camera(width=960, height=600, hfov=120.0):
    focal = (width / 2.0) / math.tan(math.radians(hfov) / 2.0)

    return np.array(
        [[focal, 0.0, width / 2.0], [0.0, focal, height / 2.0], [0.0, 0.0, 1.0]]
    )


# A level camera: world-up is minus optical-y, so row 2 is (0, -1, 0).
LEVEL = np.eye(4)
LEVEL[:3, :3] = np.array([[1.0, 0, 0], [0, 0, 1.0], [0, -1.0, 0]])


def _room(lens_height, ceiling=None, width=960, height=600):
    """Planar depth of a room seen by a level camera `lens_height` up.

    Planar, not ray length: that is what Habitat renders and what Depth
    Anything is scaled to, and it is what the rays in depth.py assume.
    """
    K = _camera(width, height)

    u, v = np.meshgrid(np.arange(width) + 0.5, np.arange(height) + 0.5)
    rays = np.linalg.inv(K) @ np.stack([u.ravel(), v.ravel(), np.ones(u.size)])

    # Height change per unit of planar depth.
    up = (LEVEL[2, :3] @ rays).reshape(height, width)

    z = np.full((height, width), 12.0)

    down = up < -0.02
    z[down] = np.minimum(z[down], lens_height / -up[down])

    if ceiling is not None:
        above = up > 0.02
        z[above] = np.minimum(z[above], ceiling / up[above])

    return FakeDepth(z.astype(np.float32), K)


def the_floor_is_found_at_a_drone_height():
    """The bug that stopped the map being built at all.

    floor_drop searched a fixed 0.3-4.0 m below the lens. That is right for
    glasses on a head and discards every floor pixel for a robot 15 cm up, so
    the scale never resolved and not one frame was ever mapped. There was no
    error -- the brain just quietly reported no floor, for ever.
    """
    scale = depth_module.FloorScale(0.15)
    measured = scale.update(_room(0.15), LEVEL)

    assert measured is not None, "no floor found with the lens 0.15 m up"
    assert abs(measured - 1.0) < 0.05, (
        f"a perfectly metric depth should score 1.0, got {measured:.3f}"
    )

    return f"lens 0.15 m: scale {measured:.3f}"


def the_floor_is_still_found_at_a_head_height():
    """And the fix must not have broken the height it was tuned at."""
    scale = depth_module.FloorScale(1.741)
    measured = scale.update(_room(1.741), LEVEL)

    assert measured is not None, "no floor found with the lens 1.741 m up"
    assert abs(measured - 1.0) < 0.05, f"expected 1.0, got {measured:.3f}"

    return f"lens 1.741 m: scale {measured:.3f}"


def a_wrong_scale_is_refused():
    """A depth half the true size should be caught, not believed."""
    scale = depth_module.FloorScale(0.15)

    half = _room(0.15)
    half.depth = half.depth * 0.35          # scale would be ~2.9, far outside

    assert scale.update(half, LEVEL) is None, "an implausible scale was accepted"

    return "a scale outside 0.6-1.6 is refused"


# ---------------------------------------------------------------------
# Indoors or out
# ---------------------------------------------------------------------


def a_ceiling_is_measured():
    # 0.7 m: flying 1.1 m up under a 2 m ceiling, which once read as outdoors.
    for height in (0.7, 1.6, 2.35, 3.0):
        scale = depth_module.FloorScale(0.15)
        got = scale.ceiling(_room(0.15, ceiling=height), LEVEL)

        assert got is not None, f"no ceiling found at {height} m"
        assert abs(got - height) < 0.05, f"ceiling {height} m measured as {got:.2f}"

    return "ceilings at 0.7, 1.6, 2.35 and 3.0 m, all within 5 cm"


def open_sky_is_not_a_ceiling():
    """The whole indoor/outdoor answer. Depth Anything returns numbers for
    the sky as happily as for plaster; what it cannot do is make them agree
    on a height."""
    scale = depth_module.FloorScale(0.15)

    assert scale.ceiling(_room(0.15, ceiling=None), LEVEL) is None, (
        "open sky was mistaken for a ceiling"
    )

    return "open sky gives no ceiling"


def the_roof_layer_needs_more_than_one_look():
    """A cell must not flip indoors on a single glimpse of a doorframe."""
    settings = brain_config.load().occupancy
    grid = occupancy.OccupancyGrid(settings, (0.0, 0.0))

    cell = (300, 300)

    grid.roof[cell] = grid.roof_hit                 # one view
    assert grid.roof_classes()[cell] == UNKNOWN, "one view decided it"

    grid.roof[cell] = 3 * grid.roof_hit             # three views
    assert grid.roof_classes()[cell] == ROOFED, "three views did not decide it"

    grid.roof[cell] = 3 * grid.roof_miss            # and back out again
    assert grid.roof_classes()[cell] == UNROOFED, "driving outside did not register"

    return "one view is not enough; three are, either way"


# ---------------------------------------------------------------------
# The map
# ---------------------------------------------------------------------


def a_scan_marks_floor_free_and_walls_solid():
    settings = brain_config.load().occupancy
    room = _room(0.15, ceiling=2.4)

    scan = occupancy.make_scan(
        room.depth, room.K, 0.15, settings,
        footprint=(0.0, 0.0, 0.15), roofed=1,
    )

    assert len(scan.free) >= 3, "a view of a floor produced no free area"
    assert scan.roofed == 1, "the scan forgot what was overhead"

    grid = occupancy.OccupancyGrid(settings, (0.0, 0.0))
    grid.add(scan, np.eye(4))

    classes = grid.classes()

    assert (classes == FREE).sum() > 0, "nothing was marked free"
    assert (grid.roof_classes() == ROOFED).sum() == 0, (
        "one scan should not be enough to call anywhere indoors"
    )

    for _ in range(3):
        grid.add(scan, np.eye(4))

    assert (grid.roof_classes() == ROOFED).sum() > 0, (
        "four views of a ceiling did not mark anywhere indoors"
    )

    return f"{int((classes == FREE).sum())} cells free after one scan"


def a_flying_drone_does_not_clear_the_floor_beneath_it():
    """The glasses always marked the cell underfoot as free floor, correctly:
    a person standing somewhere proves it is standable. A flying drone proves
    nothing of the sort, and there may be nothing under it at all."""
    settings = brain_config.load().occupancy
    room = _room(0.15)

    grounded = occupancy.make_scan(
        room.depth, room.K, 0.15, settings,
        footprint=(0.0, 0.0, 0.15), footprint_free=True,
    )
    airborne = occupancy.make_scan(
        room.depth, room.K, 0.15, settings,
        footprint=(0.0, 0.0, 0.15), footprint_free=False,
    )

    assert grounded.footprint[2] > 0.0, "a grounded drone lost its footprint"
    assert airborne.footprint[2] == 0.0, (
        "a flying drone still cleared the ground beneath it"
    )

    return "the footprint is free only when it is on the ground"


def a_loop_closure_redraws_the_map():
    """What RTAB-Map buys over plain odometry: when it recognises somewhere,
    every pose behind it is corrected and the kept scans are redrawn. Without
    this, going round a house twice leaves two copies of every wall."""
    settings = brain_config.load().occupancy
    room = _room(0.15)

    scan = occupancy.make_scan(
        room.depth, room.K, 0.15, settings, footprint=(0.0, 0.0, 0.15), anchor=7
    )

    grid = occupancy.OccupancyGrid(settings, (0.0, 0.0))
    grid.add(scan, np.eye(4))

    before = grid.classes()
    kept = len(grid.scans)

    assert kept >= 1, "no scan was kept for redrawing"

    # The graph moves node 7 two metres sideways.
    moved = np.eye(4)
    moved[0, 3] = 2.0

    grid.rebuild({7: moved})

    after = grid.classes()

    assert not np.array_equal(before, after), "the map ignored the correction"
    assert (after == FREE).sum() > 0, "the redraw lost the map"

    return f"{kept} scans redrawn at corrected poses"


# ---------------------------------------------------------------------
# Height, from the rangefinder
# ---------------------------------------------------------------------


def the_height_survives_flying_over_a_table():
    """Take off, climb to 1 m, fly over a 0.75 m table, climb a little over
    it, leave it, land -- the readings a rangefinder would give, frame by
    frame at 10 a second. The altitude must follow the climbing and NOT the
    table, and the table must be seen for what it is."""
    from delta.brain import height as height_module

    radius, below, speed = 0.15, 0.10, 0.8
    estimate = height_module.Height(radius, below, speed)

    def reading(altitude, surface=0.0):
        # Sensor to the surface: lens height minus the sensor's offset,
        # minus whatever is under it.
        return radius + altitude - below - surface

    plan = (
        [(0.0, 0.0)] * 10                                         # on the floor
        + [(0.05 * i, 0.0) for i in range(1, 21)]                 # climb at 0.5 m/s
        + [(1.0, 0.0)] * 10                                       # hover
        + [(1.0, 0.75)] * 10                                      # over the table
        + [(1.0 + 0.04 * i, 0.75) for i in range(1, 6)]           # climb over it
        + [(1.2, 0.75)] * 5
        + [(1.2, 0.0)] * 10                                       # past its edge
        + [(1.2 - 0.06 * i, 0.0) for i in range(1, 21)]           # descend, land
        + [(0.0, 0.0)] * 10
    )

    worst = 0.0
    surface_seen = []

    for i, (altitude, surface) in enumerate(plan):
        got = estimate.update(reading(altitude, surface), 0.1 * i)
        worst = max(worst, abs(got - altitude))

        if surface > 0.0:
            surface_seen.append(estimate.surface)

    assert worst < 0.02, f"altitude off by {worst:.3f} m at worst"
    assert abs(min(surface_seen) - 0.75) < 0.02 and abs(max(surface_seen) - 0.75) < 0.02, (
        f"the table was read as {min(surface_seen):.2f}-{max(surface_seen):.2f} m, not 0.75"
    )
    assert estimate.surface == 0.0, "the table stayed behind after the drone left it"

    return f"altitude within {worst * 100:.1f} cm; table seen as 0.75 m, and forgotten after"


def something_low_at_the_edge_of_the_cone_leaves_no_trace():
    """Measured in a real run: at 1 m up, something 9 cm tall slipped into
    the rangefinder's cone in one frame and drifted back out of it over
    several. Taken for furniture on the way in and for climbing on the way
    out, it left the brain 9 cm high all the way down to the floor."""
    from delta.brain import height as height_module

    radius, below = 0.15, 0.10
    estimate = height_module.Height(radius, below, 0.8)

    readings = (
        [radius + 1.0 - below] * 10                                   # 1 m up, over floor
        + [radius + 1.0 - below - 0.09] * 5                            # in: one step
        + [radius + 1.0 - below - 0.09 + 0.018 * i for i in range(1, 6)]   # out: slowly
        + [radius + 1.0 - below] * 10
    )

    for i, reading in enumerate(readings):
        got = estimate.update(reading, 0.1 * i)

    assert abs(got - 1.0) < 0.02, f"it ended {got - 1.0:+.3f} m off"
    assert estimate.surface == 0.0, f"a {estimate.surface:.2f} m surface was left behind"

    return f"ended {abs(got - 1.0) * 100:.1f} cm off, nothing left behind"


def climbing_up_stairs_is_not_taken_for_floor():
    """Seen by hand: flying up a staircase, the brain said it was 0.21 m
    over the floor when it was 0.22 m over a 0.35 m step. Climbing while
    crossing a step, the drone's rise and the step's rise cancel in the
    reading. With the flight controller's climb rate they come apart.

    Five 0.18 m steps 0.25 m deep, crossed at 0.8 m/s while climbing at
    0.6 m/s, then the landing at the top; climb rate with 3 cm/s noise."""
    from delta.brain import height as height_module

    radius, below = 0.15, 0.10
    estimate = height_module.Height(radius, below, 0.8)
    rng = np.random.default_rng(0)

    def step_under(x):
        return 0.0 if x < 0.0 else 0.18 * min(5, int(x / 0.25) + 1)

    worst = 0.0
    altitude, x = 0.0, -0.5

    # Take off from the floor to 0.6 m, then cross the stairs climbing to 1.6.
    for i in range(70):
        taking_off = i < 10
        climbing = i >= 20 and altitude < 1.6
        rate = 0.6 if (taking_off or climbing) else 0.0

        if i >= 20:
            x += 0.08

        altitude += rate * 0.1
        reading = radius + altitude - below - step_under(x)
        got = estimate.update(reading, 0.1 * i, rate + 0.03 * rng.normal())
        worst = max(worst, abs(got - altitude))

    assert worst < 0.05, f"altitude off by {worst:.3f} m at worst on the stairs"
    assert abs(estimate.surface - 0.90) < 0.05, f"the top of the stairs read as {estimate.surface:.2f} m, not 0.90"

    return f"altitude within {worst * 100:.1f} cm; the top step read as {estimate.surface:.2f} m (0.90)"


def a_surface_that_slides_out_of_the_cone_is_forgotten():
    """Seen by hand: hovering 1 m up, something 12 cm tall came into the cone
    in one step and slid back out a few centimetres a frame. The surface
    stayed, and the drone landed believing it sat 14 cm up -- the floor in
    front of it rose into the obstacle band. With the climb rate saying the
    drone did not move, the slow change has to be the ground."""
    from delta.brain import height as height_module

    radius, below = 0.15, 0.10
    estimate = height_module.Height(radius, below, 0.8)
    rng = np.random.default_rng(1)

    altitude, surfaces = 0.0, []
    surfaces += [0.0] * 10                                   # on the floor
    surfaces += [0.0] * 20                                   # take off to 1 m
    surfaces += [0.12] * 5                                   # in, in one step
    surfaces += [0.12 - 0.024 * i for i in range(1, 6)]      # out, slowly
    surfaces += [0.0] * 10                                   # hover
    surfaces += [0.0] * 20                                   # land

    for i, surface in enumerate(surfaces):
        rate = 0.5 if 10 <= i < 30 else (-0.5 if i >= 50 else 0.0)
        altitude = max(0.0, altitude + rate * 0.1)
        got = estimate.update(radius + altitude - below - surface, 0.1 * i, rate + 0.03 * rng.normal())

    assert estimate.surface == 0.0, f"a {estimate.surface:.2f} m surface was left behind"
    assert abs(got - altitude) < 0.02, f"landed believing it was {got - altitude:+.2f} m up"

    return f"landed {abs(got - altitude) * 100:.1f} cm off, nothing left behind"


TESTS = [
    the_brain_cannot_reach_the_simulator,
    the_truth_is_only_sent_when_asked_for,
    the_brain_ignores_truth_it_was_not_given,
    the_sim_sends_only_what_sensors_measure,
    the_floor_is_found_at_a_drone_height,
    the_floor_is_still_found_at_a_head_height,
    a_wrong_scale_is_refused,
    a_ceiling_is_measured,
    open_sky_is_not_a_ceiling,
    the_roof_layer_needs_more_than_one_look,
    a_scan_marks_floor_free_and_walls_solid,
    a_flying_drone_does_not_clear_the_floor_beneath_it,
    a_loop_closure_redraws_the_map,
    the_height_survives_flying_over_a_table,
    something_low_at_the_edge_of_the_cone_leaves_no_trace,
    climbing_up_stairs_is_not_taken_for_floor,
    a_surface_that_slides_out_of_the_cone_is_forgotten,
]


def main():
    print()

    failures = 0

    for test in TESTS:
        name = test.__name__.replace("_", " ")

        try:
            detail = test()
            print(f"PASS  {name:<52} {detail}")

        except AssertionError as error:
            failures += 1
            print(f"FAIL  {name:<52} {error}")

    print()

    if failures:
        print(f"{failures} of {len(TESTS)} checks failed")
        return 1

    print(f"All {len(TESTS)} checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
