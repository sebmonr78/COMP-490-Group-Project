"""How high the drone is -- worked out from its own rangefinder.

The map measures every height down from the lenses, so the brain has to know
how high the lenses are above the FLOOR. On the ground that is the sphere's
radius. In the air it used to be sent by the simulator, which is the answer
key. Now it comes from the down-facing rangefinder (config/rig.json).

The catch is that the rangefinder measures down to whatever is below, and
the floor is only sometimes what is below:

    r            the reading: sensor to the nearest surface below
    lens         the lenses' height above the floor -- what the map needs
    surface      the height of what is under the drone: 0 over the floor,
                 0.75 over a table

    r + below_lens  =  lens - surface

One equation, two unknowns. What tells them apart is speed. The drone cannot
climb or sink faster than its motors allow -- 0.8 m/s, 8 cm a frame -- but
flying off the edge of a table changes the reading by half a metre in one
frame. So:

    the reading changes slowly  ->  the drone moved: lens follows it, the
                                    surface under it stays what it was
    it jumps                    ->  the ground changed under it: the lens
                                    stays put, and the jump is the new
                                    surface's height

This is the terrain estimator a PX4 flight controller runs. A jump back to
within a few centimetres of the floor is taken as the floor again, so small
errors over furniture do not stay behind once the drone is clear of it.

A jump also has to be big: 10 cm or more. The sensor's cone is 15 degrees
wide and answers with the nearest echo, so something low near the drone --
a rug's edge, a chair's foot -- slips in and out of it by a few centimetres
at a time. Taken for furniture, such a step stayed behind when the thing
drifted back out of the cone gradually, and the brain landed believing it
was still 9 cm up. Read as the drone rising and sinking instead, it cancels
itself out when the thing leaves.
"""


class Height:
    """Lens height above the floor, and the surface under the drone."""

    # A change in the reading this many times faster than the drone can climb
    # is the ground changing, not the drone. 1.5x leaves room for noise and
    # for a frame that arrives late.
    JUMP_FACTOR = 1.5

    # The smallest change that can count as the ground changing, and the
    # nearest to the floor a surface can be and not BE the floor: a rug, a
    # door's threshold, a foot of a chair at the edge of the cone.
    MIN_JUMP = 0.10
    FLOOR_TOLERANCE = 0.10

    # Frames arrive about ten a second, sometimes closer together. Over a very
    # short gap even noise looks fast, so speeds are judged over at least this.
    MIN_INTERVAL = 0.1

    def __init__(self, ground_lens_height, below_lens, max_climb_speed):
        self.ground_lens = float(ground_lens_height)
        self.below_lens = float(below_lens)
        self.jump_speed = self.JUMP_FACTOR * float(max_climb_speed)

        # It starts on the floor: that is where a drone is switched on.
        self.lens = self.ground_lens
        self.surface = 0.0
        self.reading = None
        self.jumps = 0

        self._last = None              # (stamp, lens above the surface)

    @property
    def altitude(self):
        """How high the drone's underside is above the floor."""
        return max(0.0, self.lens - self.ground_lens)

    def update(self, reading, stamp):
        """Take one rangefinder reading (metres, or None) and return the
        altitude it implies."""
        self.reading = reading

        # No echo -- out of range, or a soft surface that swallowed it. The
        # last estimate stands until there is one.
        if reading is None:
            return self.altitude

        above_surface = float(reading) + self.below_lens

        if self._last is not None:
            dt = max(float(stamp) - self._last[0], self.MIN_INTERVAL)
            change = above_surface - self._last[1]

            if abs(change) >= max(self.MIN_JUMP, self.jump_speed * dt):
                # The ground changed under the drone. The lens is where it
                # was; the surface is whatever now puts it that far below.
                surface = self.lens - above_surface
                self.surface = 0.0 if abs(surface) < self.FLOOR_TOLERANCE else surface
                self.jumps += 1

        self.lens = max(self.ground_lens, above_surface + self.surface)
        self._last = (float(stamp), above_surface)

        return self.altitude
