"""Converting between the map's frame and Habitat's world.

The drone's map is built in the tracker's start frame: where it switched on
is the origin, with +across to its right and +along straight ahead. Habitat's
world is somewhere else entirely. Something has to join them, and this is it.

Why it is a file of its own, and not three lines in the demo: joining the two
frames is exactly where honest work can quietly stop being honest. The map is
built from the tracker, the tracker drifts, and so the map drifts away from
the world it is a map of. The moment that conversion is casually sprinkled
through a demo, the temptation is to "fix" a drifting map by reading the true
position from the simulator -- and then the map looks perfect and means
nothing.

So the rule this file exists to enforce: the transform is fixed ONCE, when
mapping starts, from where the drone was then. It is never refreshed. Any
difference that opens up between a planned route and the real house is the
tracker's accumulated error, which is a real number worth seeing rather than
a bug worth hiding.

Habitat: +x right, +y up, -z forward.
Map:     +across right, +along forward, +height up, measured from the drone's
         own starting position and heading.
"""

import math


class WorldFrame:
    """Where the map's origin sits in Habitat's world.

    Attributes:
        origin_x, origin_z   the drone's ground position when mapping began
        origin_y             the floor height it started from
        heading              the drone's heading then, in degrees
    """

    def __init__(self, origin_x, origin_y, origin_z, heading_degrees):
        self.origin_x = float(origin_x)
        self.origin_y = float(origin_y)
        self.origin_z = float(origin_z)
        self.heading = float(heading_degrees)

        # The map's +along is the drone's forward at the moment it started.
        # Habitat's forward is -z, and a yaw of t about +y sends (0, 0, -1)
        # to (-sin t, 0, -cos t), so that is the vector +along points along.
        radians = math.radians(self.heading)

        self._forward = (-math.sin(radians), -math.cos(radians))

        # +across is the drone's right: forward turned 90 degrees clockwise
        # when seen from above, which for (fx, fz) is (-fz, fx).
        self._right = (-self._forward[1], self._forward[0])

    # -----------------------------------------------------------------

    @classmethod
    def from_drone(cls, drone, heading_degrees):
        ground = drone.floor_position

        return cls(ground.x, ground.y, ground.z, heading_degrees)

    # -----------------------------------------------------------------

    def to_world(self, across, along, height=0.0):
        """Map metres -> Habitat (x, y, z)."""
        x = (
            self.origin_x
            + self._right[0] * across
            + self._forward[0] * along
        )

        z = (
            self.origin_z
            + self._right[1] * across
            + self._forward[1] * along
        )

        return x, self.origin_y + height, z

    def to_map(self, x, y, z):
        """Habitat (x, y, z) -> map (across, along, height)."""
        dx = x - self.origin_x
        dz = z - self.origin_z

        across = self._right[0] * dx + self._right[1] * dz
        along = self._forward[0] * dx + self._forward[1] * dz

        return across, along, y - self.origin_y

    def ground_point(self, across, along):
        """Map metres -> the Habitat point on the floor, ignoring height."""
        x, _, z = self.to_world(across, along, 0.0)

        return x, self.origin_y, z
