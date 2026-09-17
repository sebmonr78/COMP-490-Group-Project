"""Planning a route across the building, and following it.

This is feature 3 of the proposal -- autonomous navigation -- and feature 13,
path planning and obstacle avoidance. When the user eventually picks a room or
an object in the application and says "go there", everything after the click is
this file: work out a route, then follow it. Only where the destination comes
from changes -- a click on the layout today, a remembered object location from
the database later.

Two ideas do all the work, and both come from the navmesh:

    is_navigable(point)     can the drone rest here at all?
    find_path(start, end)   what are the corners of the shortest route?

The navmesh already knows the shape of every floor and doorway, so neither
question needs us to understand the building.

A route here is a ROLLING route, on purpose. The navmesh is a map of ground
the drone can sit on, so a path across it is exactly the energy-cheap way to
get somewhere -- which is the whole argument for the rolling-drone design.
When a goal turns out to have no ground route, plan() says so instead of
inventing one, and "there is no way to roll there" is precisely the signal
that the locomotion-mode chooser (feature 4) will need in order to decide
that this leg has to be flown.
"""

import habitat_sim
import magnum as mn


# How close counts as reaching a corner of the route, in metres. Too small and
# the drone circles a corner forever without ever being exactly on it.
CORNER_RADIUS = 0.15

# If a step moves us less than this, something is blocking the way and
# continuing would spin the loop forever.
STUCK_DISTANCE = 1e-4


class Route:
    """A planned route. Plan it, look at it, then follow it.

    Three states, in order:

        empty     nothing planned
        planned   a route exists and can be drawn, but nothing is moving
        driving   the route is being followed

    Keeping "planned" separate from "driving" is deliberate: it lets you see
    where the drone would be sent before you commit it to the trip. An
    inspection robot that silently drives off the moment you touch the map is
    not one anybody will trust in a hazardous facility.
    """

    def __init__(self, pathfinder):
        self.pathfinder = pathfinder
        self.points = []
        self.index = 0
        self.driving = False
        self.length = 0.0

    # -----------------------------------------------------------------
    # State
    # -----------------------------------------------------------------

    @property
    def is_planned(self):
        return len(self.points) > 0

    @property
    def is_finished(self):
        return self.index >= len(self.points)

    @property
    def goal(self):
        if not self.points:
            return None

        return self.points[-1]

    def clear(self):
        self.points = []
        self.index = 0
        self.driving = False
        self.length = 0.0

    # -----------------------------------------------------------------
    # Planning
    # -----------------------------------------------------------------

    def can_roll_to(self, point):
        """Is this ground the drone could rest on? Walls and furniture say no."""
        return self.pathfinder.is_navigable(mn.Vector3(point))

    def plan(self, start, goal):
        """Work out the route from `start` to `goal`.

        Returns True if a route exists. False means the destination cannot
        be rolled to -- behind a closed door, up a flight of stairs, or simply
        not ground at all. We report that rather than quietly moving the
        destination somewhere easier, because "there is no way to roll there"
        is real information, and it is the question flight exists to answer.
        """
        self.clear()

        if not self.can_roll_to(goal):
            return False

        request = habitat_sim.ShortestPath()
        request.requested_start = mn.Vector3(start)
        request.requested_end = mn.Vector3(goal)

        if not self.pathfinder.find_path(request):
            return False

        self.points = [mn.Vector3(point) for point in request.points]
        self.index = 0
        self.length = float(request.geodesic_distance)

        # find_path starts the list at our own feet. Walking to where we
        # already are is a no-op, so skip it.
        if self.points and (self.points[0] - mn.Vector3(start)).length() < CORNER_RADIUS:
            self.index = 1

        return not self.is_finished

    def start_driving(self):
        """Agree to the planned route."""
        if self.is_planned and not self.is_finished:
            self.driving = True

    # -----------------------------------------------------------------
    # Following
    # -----------------------------------------------------------------

    def advance(self, position, distance):
        """Roll `distance` metres along the route, starting from `position`.

        Returns where you end up. The trip is made in real steps through
        try_step, so the drone slides along walls and follows ramps exactly
        as it does under manual control -- the route says where to aim, the
        navmesh still decides what actually happens.
        """
        position = mn.Vector3(position)

        while distance > STUCK_DISTANCE and not self.is_finished:
            corner = self.points[self.index]

            # Compare on the floor plane only. Being a step below the next
            # corner on a ramp should not count as distance to cover.
            to_corner = mn.Vector3(
                corner.x - position.x,
                0.0,
                corner.z - position.z,
            )

            remaining = to_corner.length()

            if remaining < CORNER_RADIUS:
                self.index += 1
                continue

            travel = min(distance, remaining)

            wanted = position + to_corner.normalized() * travel

            reached = mn.Vector3(
                self.pathfinder.try_step(position, wanted)
            )

            moved = (reached - position).length()

            position = reached
            distance -= travel

            # Something solid is in the way that the route did not expect.
            # Stop rather than grind against it.
            if moved < STUCK_DISTANCE:
                self.driving = False
                break

        if self.is_finished:
            self.driving = False

        return position

    def heading_from(self, position):
        """Which way the drone should be heading, or None if there is nothing
        left to drive toward."""
        if self.is_finished:
            return None

        corner = self.points[self.index]

        return mn.Vector3(
            corner.x - position.x,
            0.0,
            corner.z - position.z,
        )
