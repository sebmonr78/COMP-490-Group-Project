"""Check that the drone really rolls, flies, collides and lands.

Run it:

    conda activate habitat
    cd ~/COMP-490-Group-Project/simulation
    python tests/test_drone.py

No window opens and nothing is drawn. This loads the scene, drives the drone
around by calling the same methods the demo calls, and checks what happened.

Worth having because the interesting failures here are all silent. A drone
whose collision test never fires does not crash -- it flies serenely through
the walls of the house, and the only way to notice is to fly it at a wall on
purpose and check that something stopped it. That is what
`flight_is_blocked_by_the_building` does, and it is the test that caught the
real bug in this code: with the drone as a KINEMATIC object, Bullet filtered
every contact against the static building and the drone crossed 30 m of house
without touching anything.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "demo"))

import magnum as mn

import open_environment as demo
from delta.drone import CEILING_MARGIN, Drone, find_spot_near
from delta.navigation import Route


def build_world():
    """Load the scene and put a drone in it, exactly as the demo does."""

    config = demo.load_project_config()

    _, dataset_config, scene_file, navmesh_file = demo.resolve_dataset_paths(
        config
    )

    simulator = demo.create_simulator(dataset_config, scene_file)

    demo.load_or_generate_navmesh(
        simulator,
        navmesh_file,
        config["drone"],
    )

    start = demo.choose_start_position(simulator)

    drone = Drone(
        simulator,
        diameter=float(config["drone"]["diameter"]),
    )

    drone.place_at(find_spot_near(simulator.pathfinder, start))

    return simulator, drone


# ---------------------------------------------------------------------
# The tests
# ---------------------------------------------------------------------


def starts_on_the_ground(simulator, drone):
    assert not drone.flying
    assert drone.altitude == 0.0

    # The centre of a resting sphere sits exactly one radius up.
    assert abs(drone.position.y - drone.floor_position.y - drone.radius) < 1e-6

    return f"resting at {drone.floor_position}"


def rolls_along_the_floor(simulator, drone):
    before = drone.floor_position
    spin_before = list(drone.object.rotation.vector)

    for _ in range(60):
        drone.drive(mn.Vector3(0.02, 0.0, 0.0), simulator.pathfinder)

    travelled = (drone.floor_position - before).length()
    spin_after = list(drone.object.rotation.vector)

    assert travelled > 0.05, "the drone did not move at all"
    assert spin_after != spin_before, "the sphere slid instead of rolling"
    assert not drone.flying and drone.altitude == 0.0

    return f"rolled {travelled:.2f} m and turned while doing it"


def takes_off(simulator, drone):
    assert drone.take_off(), "could not take off with a clear ceiling"
    assert drone.flying
    assert drone.altitude > 0.0

    return f"airborne at {drone.altitude:.2f} m"


def flight_is_blocked_by_the_building(simulator, drone):
    """Fly at the house until it stops us. Something must.

    This is the test that matters. If collision is not working the drone
    simply keeps going, so the failure is "nothing happened" rather than an
    exception, and only a check like this one will see it.
    """
    start = drone.position
    blocked_after = None

    for attempt in range(600):
        was = drone.position

        drone.drive(mn.Vector3(0.0, 0.0, -0.05), simulator.pathfinder)

        if (drone.position - was).length() < 1e-6:
            blocked_after = attempt
            break

    flown = (drone.position - start).length()

    assert blocked_after is not None, (
        "the drone flew 30 m without hitting anything -- collision "
        "detection is not working"
    )

    return f"stopped by the building after {flown:.2f} m"


def climbing_stops_below_the_ceiling(simulator, drone):
    for _ in range(400):
        drone.change_altitude(0.05)

    clearance = drone.ceiling_above()

    assert drone.flying, "climbing should not have landed the drone"
    assert clearance >= 0.0, "the drone is inside the ceiling"

    # It should have stopped because of the ceiling, not run out of loop.
    assert clearance <= CEILING_MARGIN + 0.05, (
        f"stopped {clearance:.2f} m below the ceiling, which is not the "
        "ceiling stopping it"
    )

    return (
        f"held at {drone.altitude:.2f} m with "
        f"{clearance:.2f} m of clearance left"
    )


def descending_lands(simulator, drone):
    for _ in range(600):
        drone.change_altitude(-0.05)

        if not drone.flying:
            break

    assert not drone.flying, "never landed"
    assert drone.altitude == 0.0
    assert abs(drone.position.y - drone.resting_height) < 1e-6

    return "came all the way down and landed"


def plans_and_follows_a_rolling_route(simulator, drone):
    route = Route(simulator.pathfinder)

    # Not every random point is reachable from here -- this house has more
    # than one floor, and a rolling drone cannot take the stairs. Keep
    # drawing goals until one of them is somewhere we can actually roll.
    goal = None

    for seed in range(1, 80):
        simulator.pathfinder.seed(seed)

        candidate = mn.Vector3(
            simulator.pathfinder.get_random_navigable_point()
        )

        if route.plan(drone.floor_position, candidate):
            goal = candidate
            break

    assert goal is not None, "no reachable goal found in 80 tries"

    route.start_driving()
    assert route.driving

    steps = 0

    while route.driving and steps < 100000:
        was = drone.floor_position
        now = route.advance(was, 0.03)

        drone.place_at(now)
        drone.roll(now - was)

        steps += 1

        if route.is_finished:
            break

    gap = (drone.floor_position - goal).length()

    assert route.is_finished, "the route never finished"
    assert gap < 0.5, f"stopped {gap:.2f} m from the goal"

    return f"planned {route.length:.2f} m and arrived {gap:.2f} m from the goal"


TESTS = [
    starts_on_the_ground,
    rolls_along_the_floor,
    takes_off,
    flight_is_blocked_by_the_building,
    climbing_stops_below_the_ceiling,
    descending_lands,
    plans_and_follows_a_rolling_route,
]


def main():
    print()
    print("Loading the scene once, then running every check against it...")

    simulator, drone = build_world()

    print()

    failures = 0

    # Deliberately in order and sharing one drone: each test leaves the drone
    # somewhere, and the next one starts from there. That is how it is
    # actually flown, and it catches state the tests would miss if every one
    # of them started from a clean drone.
    for test in TESTS:
        name = test.__name__.replace("_", " ")

        try:
            detail = test(simulator, drone)
            print(f"PASS  {name:<42} {detail}")

        except AssertionError as error:
            failures += 1
            print(f"FAIL  {name:<42} {error}")

    simulator.close()

    print()

    if failures:
        print(f"{failures} of {len(TESTS)} checks failed")
        return 1

    print(f"All {len(TESTS)} checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
