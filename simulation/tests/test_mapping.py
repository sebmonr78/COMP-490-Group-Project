"""Check the map understands gaps, and that routes use them.

No scene, no Depth Anything, no GPU. The map is built by hand here, one
obstacle at a time, so that each check has exactly one right answer:

    a sofa    0.1-0.5 m, clear above   -> fly OVER it
    a table   0.6-0.8 m, clear below   -> roll UNDER it
    a wall    floor to ceiling         -> go AROUND, or nowhere

That is the distinction the glasses project's height map could not make, and
it is the whole reason gap_map.py exists -- so it is tested where it can be
tested exactly, rather than hoping a house happens to contain the right
furniture.

Run it:

    conda activate habitat
    cd ~/COMP-490-Group-Project/simulation
    python tests/test_mapping.py
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

from delta.gap_map import FREE, SOLID, UNKNOWN, GapMap
from delta.map_navigation import MapPlanner
from delta.world_frame import WorldFrame


# A small grid, so a test runs in milliseconds: 4 m across, 1.5 m tall.
GRID = {
    "grid": {
        "cell": 0.10,
        "layer": 0.10,
        "size_m": 4.0,
        "height_m": 1.5,
        "min_range": 0.25,
        "max_range": 4.0,
        "stride": 3,
        "carve_stride": 6,
        "carve_samples": 24,
        "carve_margin": 0.12,
        "solid_hits": 2,
        "free_hits": 2,
    }
}

DIAMETER = 0.30          # the drone: three layers tall
HITS = 5                 # comfortably over both thresholds


def corridor():
    """A straight run of floor with clear air above it, and nothing else.

    Columns 18-21 (a 40 cm wide lane), every row. Level 0 is the floor
    itself -- solid, because a floor is a thing you can see -- and levels 1
    upward are carved clear.
    """
    world = GapMap(GRID, camera_height=0.15)

    lane = slice(18, 22)

    world.solid_votes[:, lane, 0] = HITS
    world.free_votes[:, lane, 1:] = HITS

    return world


def put(world, row, low, high, lane=slice(18, 22)):
    """Something solid in these layers, at this row. Clears the free votes
    there, because a layer cannot be both looked through and full."""
    world.solid_votes[row, lane, low : high + 1] = HITS
    world.free_votes[row, lane, low : high + 1] = 0


# ---------------------------------------------------------------------
# The map
# ---------------------------------------------------------------------


def plain_floor_is_rollable(_):
    world = corridor()

    grid = world.states()
    roll = world.rollable(DIAMETER, grid)

    assert grid[20, 20, 0] == SOLID, "the floor should read as solid"
    assert grid[20, 20, 3] == FREE, "the air above it should read as free"
    assert roll[20, 20], "clear floor with clear air over it must be rollable"

    assert not roll[20, 5], "floor that was never seen must not be rollable"
    assert grid[20, 5, 2] == UNKNOWN

    return f"{int(roll.sum())} squares rollable, none of them unseen"


def unknown_is_not_free(_):
    """The distinction the whole design rests on.

    A layer nothing has been said about is not a layer the drone may fly
    into. If UNKNOWN ever counts as passable, the drone will happily plan a
    route through the middle of an unmapped wall.
    """
    world = corridor()

    # Seen floor, but the air above it never carved.
    world.solid_votes[30, 25, 0] = HITS

    grid = world.states()

    assert grid[30, 25, 2] == UNKNOWN
    assert not world.passable(DIAMETER, grid)[30, 25, 1]
    assert not world.rollable(DIAMETER, grid)[30, 25]

    return "unknown space is never passable"


def ground_and_ceiling(_):
    world = corridor()
    put(world, 20, 1, 4)        # a sofa on the floor

    grid = world.states()

    ground = world.ground_level(grid)
    ceiling = world.ceiling_level(grid)

    assert ground[20, 20] == 0, "the floor is still the lowest solid layer"
    assert ground[25, 20] == 0
    assert ground[20, 5] == -1, "nothing seen means no ground"

    # Free space runs to the top of the map where nothing blocks it.
    assert ceiling[25, 20] == world.levels - 1
    assert ceiling[20, 5] == -1

    return f"ground 0, ceiling {int(ceiling[25, 20])} of {world.levels - 1}"


def sofa_can_be_flown_over(_):
    """Solid from the floor up, with room above: over is the only way."""
    world = corridor()
    put(world, 20, 1, 4)        # 0.1 m to 0.5 m: too tall to roll through

    grid = world.states()

    fits = world.passable(DIAMETER, grid)
    roll = world.rollable(DIAMETER, grid)

    assert not roll[20, 20], "a 50 cm sofa is not rollable"
    assert not fits[20, 20, 1], "the drone does not fit at sofa height"

    # Above it: levels 5 and up are free, and the drone needs three of them.
    assert fits[20, 20, 5], "there is room to fly over the sofa"

    clearance = world.clearance(DIAMETER, grid)

    assert clearance[20, 20, 0] == 5, "lowest way past is over the top"

    return f"blocked to {int(clearance[20, 20, 0]) - 1}, passable from layer 5"


def table_can_be_rolled_under(_):
    """Solid higher up, with room below: under is available, and cheaper."""
    world = corridor()
    put(world, 20, 6, 7)        # a tabletop at 0.6-0.8 m

    grid = world.states()

    fits = world.passable(DIAMETER, grid)
    roll = world.rollable(DIAMETER, grid)

    assert roll[20, 20], "the drone fits under a table with 60 cm of clearance"
    assert fits[20, 20, 1], "and it fits in the layers just above the floor"
    assert not fits[20, 20, 6], "but not in the tabletop itself"

    clearance = world.clearance(DIAMETER, grid)

    assert clearance[20, 20, 0] == 1, "lowest way past is underneath"

    return "rollable underneath, blocked at the tabletop"


def wall_blocks_both_ways(_):
    world = corridor()
    put(world, 20, 1, world.levels - 1)

    grid = world.states()

    fits = world.passable(DIAMETER, grid)

    assert not fits[20, 20].any(), "a floor-to-ceiling wall is passable nowhere"
    assert not world.rollable(DIAMETER, grid)[20, 20]

    return "no layer passable, as it should be"


def picture_tells_them_apart(_):
    """The three cases have to look different, or the map cannot be read."""
    world = corridor()
    put(world, 15, 1, 4)                       # sofa  -> over
    put(world, 20, 6, 7)                       # table -> under
    put(world, 25, 1, world.levels - 1)        # wall  -> around

    image = world.picture(DIAMETER)

    sofa = tuple(image[15, 20])
    table = tuple(image[20, 20])
    wall = tuple(image[25, 20])
    floor = tuple(image[30, 20])

    assert len({sofa, table, wall, floor}) == 4, (
        f"the four cases must be four colours, got "
        f"sofa={sofa} table={table} wall={wall} floor={floor}"
    )

    # Blue for over, green for under: the blue channel decides.
    assert sofa[2] > sofa[0], "a fly-over obstacle should read blue"
    assert table[1] > table[0], "a roll-under obstacle should read green"

    return f"over={sofa} under={table} around={wall} floor={floor}"


def saving_and_loading(tmp):
    world = corridor()
    put(world, 20, 1, 4)
    world.frames = 17

    path = world.save(tmp / "map.npz")
    back = GapMap.load(path)

    assert back.frames == 17
    assert back.n == world.n and back.levels == world.levels
    assert np.array_equal(back.solid_votes, world.solid_votes)
    assert np.array_equal(back.free_votes, world.free_votes)
    assert np.array_equal(back.states(), world.states())

    return f"{path.stat().st_size / 1024:.1f} kB round-tripped exactly"


# ---------------------------------------------------------------------
# Navigation on it
# ---------------------------------------------------------------------


def plans_along_clear_floor(_):
    world = corridor()
    planner = MapPlanner(world, DIAMETER)

    start = planner.snap(*world.to_world(12, 20))
    goal = planner.snap(*world.to_world(28, 20))

    assert start is not None and goal is not None, "both ends must be usable"

    plan = planner.plan(start, goal)

    assert plan is not None, "a straight clear corridor must be plannable"
    assert not plan.flies, "there is no reason to fly down a clear corridor"

    straight = abs(28 - 12) * world.cell

    assert plan.length < straight * 1.3, (
        f"route is {plan.length:.2f} m for a {straight:.2f} m corridor"
    )

    return f"{plan.describe()}, {len(plan)} nodes"


def flies_over_what_it_cannot_roll_past(_):
    """The locomotion decision, made by the costs and not by a rule."""
    world = corridor()
    put(world, 20, 1, 4)        # the sofa, right across the lane

    planner = MapPlanner(world, DIAMETER)

    start = planner.snap(*world.to_world(12, 20))
    goal = planner.snap(*world.to_world(28, 20))

    plan = planner.plan(start, goal)

    assert plan is not None, "it should get past by flying over"
    assert plan.flies, "the only way past a sofa in a corridor is over it"

    # And it should be flying over the sofa, not the whole way.
    rolled = sum(1 for mode in plan.modes if mode == "roll")

    assert rolled > 0, "it should still roll the parts it can"

    # The climb is itself flying, and happens below the top of the sofa --
    # that is what climbing is. What must hold is where it CROSSES: every
    # node in the sofa's own row has to be above the sofa, which tops out at
    # layer 4.
    crossing = [level for row, _, level in plan.nodes if row == 20]

    assert crossing, "the route has to cross the sofa's row somewhere"
    assert min(crossing) >= 5, (
        f"it crossed the sofa's row at layer {min(crossing)}, which is inside "
        "the sofa -- it must cross at 5 or above"
    )

    heights = [point[2] for point, mode in zip(plan.points, plan.modes)
               if mode == "fly"]

    return (
        f"{plan.describe()}, crossed at layer {min(crossing)}, "
        f"peak {max(heights):.2f} m"
    )


def prefers_rolling_under_to_flying_over(_):
    """Given both ways past, the cheap one wins. That is feature 4."""
    world = corridor()
    put(world, 20, 6, 7)        # a table: clear underneath AND over the top

    planner = MapPlanner(world, DIAMETER)

    start = planner.snap(*world.to_world(12, 20))
    goal = planner.snap(*world.to_world(28, 20))

    plan = planner.plan(start, goal)

    assert plan is not None
    assert not plan.flies, (
        "rolling under costs less than flying over, so it should roll"
    )

    return f"{plan.describe()} -- stayed on the ground"


def refuses_when_there_is_no_way(_):
    world = corridor()
    put(world, 20, 1, world.levels - 1)     # a wall across the only lane

    planner = MapPlanner(world, DIAMETER)

    start = planner.snap(*world.to_world(12, 20))
    goal = planner.snap(*world.to_world(28, 20))

    plan = planner.plan(start, goal)

    assert plan is None, (
        "there is no way through a floor-to-ceiling wall, and saying there is "
        "would be worse than saying nothing"
    )

    return "no route, correctly"


def snapping_finds_somewhere_usable(_):
    world = corridor()
    put(world, 20, 1, 4)

    planner = MapPlanner(world, DIAMETER)

    # Ask for the middle of the sofa. It should answer with somewhere real.
    across, along = world.to_world(20, 20)
    node = planner.snap(across, along)

    assert node is not None, "snap should find a nearby usable node"
    assert planner.usable(*node), "and it has to actually be usable"

    return f"snapped onto {node}"


def frontier_is_where_the_map_runs_out(_):
    world = corridor()

    planner = MapPlanner(world, DIAMETER)
    frontier = planner.frontier_cells()

    roll = world.rollable(DIAMETER, world.states())

    assert frontier.sum() > 0, "a corridor 40 cm wide is all edge"
    assert not (frontier & ~roll).any(), "frontier cells must be rollable"

    return f"{int(frontier.sum())} of {int(roll.sum())} squares are on the edge"


# ---------------------------------------------------------------------
# Frames
# ---------------------------------------------------------------------


def world_frame_round_trips(_):
    for heading in (0.0, 37.0, 90.0, 180.0, -125.0):
        frame = WorldFrame(1.5, 0.25, -3.0, heading)

        for point in ((0.0, 0.0, 0.0), (2.0, -1.0, 0.4), (-3.5, 7.25, 1.1)):
            x, y, z = frame.to_world(*point)
            back = frame.to_map(x, y, z)

            for before, after in zip(point, back):
                assert abs(before - after) < 1e-9, (
                    f"heading {heading}: {point} -> {back}"
                )

    # A heading of zero means the map's +along is Habitat's -z.
    frame = WorldFrame(0.0, 0.0, 0.0, 0.0)
    x, _, z = frame.to_world(0.0, 1.0)

    assert abs(x) < 1e-9 and abs(z + 1.0) < 1e-9, (
        f"forward should be -z, got ({x}, {z})"
    )

    return "exact both ways at five headings"


TESTS = [
    plain_floor_is_rollable,
    unknown_is_not_free,
    ground_and_ceiling,
    sofa_can_be_flown_over,
    table_can_be_rolled_under,
    wall_blocks_both_ways,
    picture_tells_them_apart,
    saving_and_loading,
    plans_along_clear_floor,
    flies_over_what_it_cannot_roll_past,
    prefers_rolling_under_to_flying_over,
    refuses_when_there_is_no_way,
    snapping_finds_somewhere_usable,
    frontier_is_where_the_map_runs_out,
    world_frame_round_trips,
]


def main():
    import tempfile

    print()

    failures = 0

    with tempfile.TemporaryDirectory() as directory:
        tmp = Path(directory)

        for test in TESTS:
            name = test.__name__.replace("_", " ")

            try:
                detail = test(tmp)
                print(f"PASS  {name:<42} {detail}")

            except AssertionError as error:
                failures += 1
                print(f"FAIL  {name:<42} {error}")

    print()

    if failures:
        print(f"{failures} of {len(TESTS)} checks failed")
        return 1

    print(f"All {len(TESTS)} checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
