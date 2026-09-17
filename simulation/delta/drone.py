"""The DELTA drone: a sphere that rolls across the floor and can fly.

This is the robot the whole project is about, and its two ways of moving are
the project's central argument. From the proposal:

    Rolling will be preferred for energy-efficient long-duration operation,
    while flight will be used for stairs, obstacles, gaps, high locations,
    blocked paths, or areas that cannot be reached from the ground.

So the class has two modes, and they are not cosmetic variants of each other.
They ask the simulator different questions:

    rolling   the navmesh decides. It is a map of ground the drone can rest
              on, and try_step is the call that stops it at walls, slides it
              along them, and carries it up ramps.

    flying    the navmesh is silent. It describes the floor and has no
              opinion about the air above it, so the question goes to the
              real collision geometry instead (see can_occupy), and the
              ceiling is found by looking up (see ceiling_above).

Getting that split right is most of what this file is. A drone that used the
navmesh while airborne would be snapped back to the floor; one that used
collision tests while rolling would have to solve the floor it is sitting on.

Why a sphere:

    A ball is the one shape with no front, no wrong way up, and no way to
    fall over. Whatever it bumps into, it is still in a legal pose
    afterwards -- which is what we want from a body we move by hand, frame
    by frame, with no physics deciding its orientation for us. It is also
    why the drone is driven by compass rather than by "turn left": a sphere
    has no left.

    And it is honest about size. The sphere is DEFAULT_DIAMETER across,
    smaller than a laptop, so when it does not fit through a gap on screen
    it genuinely would not fit.

Built from Habitat's own primitive shapes: get a primitive template, scale
it, register it under our own name, add it. Same pattern as
habitat-sim/tests/test_gfx.py:87.

Requires a habitat-sim built `withbullet`, and a Simulator created with
enable_physics=True.
"""

import math

import habitat_sim
import magnum as mn


# Smaller than a 13-inch laptop is wide (about 0.30 m), in metres.
DEFAULT_DIAMETER = 0.30

# How far above the floor the drone hovers the moment it takes off. Enough
# to read as "airborne" without launching it at the ceiling.
HOVER_ALTITUDE = 0.40

# A take-off that could not even manage this is refused, in metres. Rising
# two centimetres under a shelf is not a take-off, it is a scrape.
MIN_TAKEOFF_CLEARANCE = 0.10

# How fast the drone climbs and descends, in metres per second.
CLIMB_SPEED = 0.8

# Near its target altitude the drone eases off instead of arriving at full
# speed and stopping dead. The rate is per second: at EASE_RATE = 3.0 it
# aims to close the remaining gap in about a third of a second, so the last
# few centimetres of a landing are a settle rather than an arrival.
EASE_RATE = 3.0

# ...but never slower than this, or the last millimetre takes forever.
SETTLE_SPEED = 0.12

# Altitudes closer together than this count as the same altitude, in metres.
ALTITUDE_EPSILON = 1e-4

# The drone stops this far short of the ceiling. Without it a climb ends
# with the sphere's crown against the plaster.
CEILING_MARGIN = 0.05

# How far up ceiling_above is willing to look before giving up, in metres.
CEILING_SEARCH = 12.0

# Habitat's built-in solid sphere. The stock one has a radius of 1 m, so
# every scale below is the radius we actually want.
PRIMITIVE = "uvSphereSolid"


class Drone:
    """A rolling, flying sphere.

    Attributes:
        radius:    half the diameter, in metres
        flying:    True when off the ground, including while taking off
                   and while coming down to land
        altitude:  height of the sphere's underside above the ground beneath
                   it -- exactly 0.0 when landed
        target_altitude:
                   the altitude it is flying toward. update() closes the gap.
        object:    the habitat_sim rigid object, if you need it directly
    """

    def __init__(self, simulator, diameter=DEFAULT_DIAMETER):
        self.simulator = simulator
        self.radius = float(diameter) / 2.0

        self.flying = False
        self.altitude = 0.0
        self.target_altitude = 0.0

        # The height of the surface the drone is currently over. Updated as
        # it rolls, and re-checked as it flies, so it always knows what it
        # would land on -- which is not always the floor it took off from.
        self.ground_height = 0.0

        template_manager = simulator.get_object_template_manager()
        object_manager = simulator.get_rigid_object_manager()

        handles = template_manager.get_template_handles(PRIMITIVE)

        if not handles:
            raise RuntimeError(
                f"Habitat did not provide the primitive '{PRIMITIVE}'."
            )

        template = template_manager.get_template_by_handle(handles[0])

        # The stock sphere has a radius of 1 m, so the scale factor on each
        # axis is simply the radius we want.
        template.scale = mn.Vector3(
            self.radius,
            self.radius,
            self.radius,
        )

        # Registering under our own name keeps the stock template untouched,
        # so anything else that wants a plain sphere still gets one.
        template_manager.register_template(template, "delta_drone_sphere")

        self.object = object_manager.add_object_by_template_handle(
            "delta_drone_sphere"
        )

        if self.object is None:
            raise RuntimeError("Habitat refused to create the drone.")

        # DYNAMIC, which looks like the wrong answer and is the right one.
        #
        # We move this drone ourselves, every frame, and we never call
        # step_physics. KINEMATIC is the motion type that says exactly that,
        # and it is what the glasses project's avatar used.
        #
        # But Bullet will not report a collision between two bodies that are
        # both non-dynamic. Every wall, floor and piece of furniture in an
        # HSSD scene is STATIC, so a KINEMATIC drone is filtered against the
        # entire building and can_occupy() reports "clear" everywhere -- it
        # flies straight through walls, in silence. Measured, not guessed: a
        # KINEMATIC sphere flew 30 m through this house without one contact,
        # and the same sphere as DYNAMIC stopped at the first wall, 4.25 m
        # out.
        #
        # DYNAMIC restores the collision reports and costs us nothing,
        # because gravity only acts inside step_physics and nothing here
        # calls it. If a later feature ever does step physics, it must move
        # the drone back afterwards, or the drone will fall.
        self.object.motion_type = habitat_sim.physics.MotionType.DYNAMIC

    # -----------------------------------------------------------------
    # Position
    # -----------------------------------------------------------------

    @property
    def position(self):
        """The centre of the sphere. This is also where a camera would sit."""
        return mn.Vector3(self.object.translation)

    @property
    def floor_position(self):
        """The point on the ground directly beneath the drone.

        Route planning and the layout map both work in ground coordinates, so
        this is what they are given -- the drone's shadow, not the drone.
        Climbing does not move it.
        """
        centre = self.object.translation

        return mn.Vector3(
            centre.x,
            self.ground_height,
            centre.z,
        )

    @property
    def resting_height(self):
        """The y the sphere's centre has when it is sitting on the ground."""
        return self.ground_height + self.radius

    def place_at(self, ground_position, altitude=None):
        """Put the drone over this point on the ground.

        A sphere's origin is its centre, so resting on the ground means
        lifting the centre by one radius. Anything beyond that is altitude.
        """
        position = mn.Vector3(ground_position)

        self.ground_height = float(position.y)

        if altitude is not None:
            self.altitude = max(0.0, float(altitude))
            self.target_altitude = self.altitude

        self.object.translation = mn.Vector3(
            position.x,
            self.resting_height + self.altitude,
            position.z,
        )

    # -----------------------------------------------------------------
    # Collision, for when the navmesh cannot help
    # -----------------------------------------------------------------

    def can_occupy(self, centre):
        """Would the drone fit, with its centre here?

        Asked by moving the sphere there, running Bullet's collision pass,
        looking for a contact that involves this drone, and putting the
        sphere back. Unlike the navmesh this is just as true in the air as on
        the ground: it knows the underside of a table and the top of a
        doorframe.

        Only real overlap counts. Bullet reports a contact distance that is
        negative when two shapes actually interpenetrate and positive when
        they are merely near, and we require a negative one -- which is why a
        drone resting exactly on the floor, touching it, is not "blocked" by
        it. Verified in this scene: a sphere sitting on the floor produces no
        contact at all, at any altitude from 0 cm upwards.

        Because a move is rejected when the drone WOULD overlap, it is never
        actually left inside anything.
        """
        original = self.object.translation

        self.object.translation = mn.Vector3(centre)

        self.simulator.perform_discrete_collision_detection()

        blocked = False

        for contact in self.simulator.get_physics_contact_points():
            involves_us = (
                contact.object_id_a == self.object.object_id
                or contact.object_id_b == self.object.object_id
            )

            if involves_us and contact.contact_distance < 0.0:
                blocked = True
                break

        self.object.translation = original

        return not blocked

    def ceiling_above(self):
        """Distance from the drone's crown to whatever is overhead.

        Looks straight up from the centre of the sphere and reports the
        nearest thing it finds, measured from the top of the sphere rather
        than its middle. Returns CEILING_SEARCH when it finds nothing --
        open sky, or a gap in the roof.
        """
        ray = habitat_sim.geo.Ray(
            self.object.translation,
            mn.Vector3(0.0, 1.0, 0.0),
        )

        result = self.simulator.cast_ray(ray, max_distance=CEILING_SEARCH)

        if not result.has_hits():
            return CEILING_SEARCH

        # Hits come back nearest first. The ray starts at the centre, so the
        # first radius of it is inside our own sphere and is not clearance.
        nearest = result.hits[0].ray_distance

        return max(0.0, nearest - self.radius)

    # -----------------------------------------------------------------
    # Rolling
    # -----------------------------------------------------------------

    def roll(self, travel):
        """Turn the sphere as if it had rolled `travel` across the ground.

        A ball that rolls without slipping turns by (distance / radius)
        radians -- one circumference of travel is one full turn. The axis is
        horizontal and square to the direction of travel: for motion along d
        that is up x d, which sends the top of the ball forwards, the way a
        real ball behaves.

        Cosmetic, and deliberately so. Nothing else in this file reads the
        drone's rotation, because a sphere has no front. It exists so that
        the thing on screen looks like it is rolling rather than sliding.
        """
        distance = travel.length()

        if distance < 1e-6 or self.radius <= 0.0:
            return

        axis = mn.math.cross(
            mn.Vector3(0.0, 1.0, 0.0),
            travel.normalized(),
        )

        if axis.length() < 1e-6:
            return

        spin = mn.Quaternion.rotation(
            mn.Rad(distance / self.radius),
            axis.normalized(),
        )

        self.object.rotation = (spin * self.object.rotation).normalized()

    # -----------------------------------------------------------------
    # Flying
    # -----------------------------------------------------------------

    # Height is a thing the drone flies TOWARD, not a thing it is set to.
    #
    # take_off and land used to move the sphere the whole way in the frame
    # you pressed the key, which made a 40 cm hop appear out of nowhere. A
    # real drone spools up and rises. So these methods now only ever state an
    # intention -- `target_altitude` -- and update() closes the gap over the
    # following frames, easing off as it arrives. Everything that wants the
    # drone higher or lower goes through the target, so there is exactly one
    # piece of code that actually moves it vertically.

    def take_off(self):
        """Ask to leave the ground, if there is room overhead.

        Returns False rather than climbing into the ceiling. A drone parked
        under a low shelf genuinely cannot take off, and pretending otherwise
        would put the sphere inside the shelf.

        Returning True means the climb has STARTED, not that it has finished.
        The drone is airborne from this moment, at an altitude of nearly
        zero, and rises over the next half second or so.
        """
        if self.flying:
            return True

        headroom = self.ceiling_above() - CEILING_MARGIN

        wanted = min(HOVER_ALTITUDE, headroom)

        if wanted < MIN_TAKEOFF_CLEARANCE:
            return False

        centre = self.object.translation

        destination = mn.Vector3(
            centre.x,
            self.resting_height + wanted,
            centre.z,
        )

        # The ceiling ray only looked along one line. The sphere is wider
        # than a line, so confirm the whole ball fits before committing to a
        # climb we would have to abandon halfway up.
        if not self.can_occupy(destination):
            return False

        self.target_altitude = wanted
        self.flying = True

        return True

    def land(self):
        """Ask to come down onto whatever is underneath.

        Like take_off, this only sets the intention. The drone descends over
        the following frames and touches down when it arrives, which is what
        clears `flying`.
        """
        self.target_altitude = 0.0

    def climb(self, seconds):
        """Hold the climb key for `seconds`: aim that much higher."""
        self._aim(CLIMB_SPEED * seconds)

    def descend(self, seconds):
        """Hold the descend key for `seconds`: aim that much lower."""
        self._aim(-CLIMB_SPEED * seconds)

    def _aim(self, delta):
        """Move the target altitude, within what the room allows."""
        if not self.flying:
            return

        wanted = self.target_altitude + delta

        if delta > 0.0:
            # Do not aim at a ceiling. Measured from where the drone is now,
            # so the headroom is the real remaining gap.
            ceiling = self.altitude + self.ceiling_above() - CEILING_MARGIN
            wanted = min(wanted, ceiling)

        self.target_altitude = max(0.0, wanted)

    def update(self, seconds):
        """Fly `seconds` worth of the way toward the target altitude.

        Call this once a frame, before anything reads the drone's position.
        This is the only code that changes altitude, so every climb, descent,
        take-off and landing is smooth by construction rather than by each
        caller remembering to be gentle.

        Speed eases with the remaining distance, so the drone slows as it
        arrives instead of stopping dead -- but never below SETTLE_SPEED, or
        the last millimetre would take forever.
        """
        if not self.flying:
            return self.altitude

        gap = self.target_altitude - self.altitude

        if abs(gap) < ALTITUDE_EPSILON:
            # Arrived. Touching down is what ends a flight.
            if self.target_altitude <= 0.0:
                self._settle()

            return self.altitude

        speed = min(CLIMB_SPEED, max(SETTLE_SPEED, abs(gap) * EASE_RATE))

        travel = min(abs(gap), speed * seconds)

        step = travel if gap > 0.0 else -travel

        if step > 0.0:
            # A clear ceiling on the centre-line does not mean the whole
            # sphere fits, so the real geometry gets the final say.
            headroom = self.ceiling_above() - CEILING_MARGIN

            if step > headroom:
                step = max(0.0, headroom)
                self.target_altitude = self.altitude + step

        if abs(step) < ALTITUDE_EPSILON:
            return self.altitude

        centre = self.object.translation

        destination = mn.Vector3(
            centre.x,
            centre.y + step,
            centre.z,
        )

        # Descending is not collision tested: `ground_height` comes from the
        # navmesh, which is a map of surfaces this drone can rest on, so the
        # spot below is a legal resting place by construction.
        if step > 0.0 and not self.can_occupy(destination):
            # Blocked on the way up. Stop wanting to be higher, or the drone
            # would grind against the obstacle for as long as the key is held.
            self.target_altitude = self.altitude

            return self.altitude

        self.object.translation = destination
        self.altitude = max(0.0, self.altitude + step)

        if self.altitude <= ALTITUDE_EPSILON and self.target_altitude <= 0.0:
            self._settle()

        return self.altitude

    def _settle(self):
        """Touch down: sit exactly on the ground and stop being airborne."""
        centre = self.object.translation

        self.object.translation = mn.Vector3(
            centre.x,
            self.resting_height,
            centre.z,
        )

        self.altitude = 0.0
        self.target_altitude = 0.0
        self.flying = False

    # -----------------------------------------------------------------
    # Moving horizontally
    # -----------------------------------------------------------------

    def drive(self, step, pathfinder):
        """Move `step` metres horizontally, by whichever rule applies now.

        On the ground the navmesh decides. try_step stops at walls, slides
        along them when you meet one at an angle, and follows the ground up
        and down ramps -- and it reports the new ground height too, which is
        how the drone knows what it is standing on after rolling up a slope.

        In the air the navmesh is silent, so collision stands in for it: try
        the whole step, and if the sphere would overlap something, try each
        axis on its own. That one fallback is what makes the drone slide
        along a wall it meets at an angle instead of stopping dead against
        it, which is the same courtesy try_step extends on the ground.
        """
        if step.length() < 1e-9:
            return

        if not self.flying:
            before = self.floor_position

            reached = mn.Vector3(
                pathfinder.try_step(before, before + step)
            )

            self.place_at(reached)
            self.roll(reached - before)

            return

        centre = self.object.translation

        for candidate in self._slide_candidates(centre, step):
            if self.can_occupy(candidate):
                self.object.translation = candidate

                self._follow_ground_below(pathfinder)

                return

    def _slide_candidates(self, centre, step):
        """The moves to try, best first: all of it, then one axis at a time."""
        return [
            centre + step,
            centre + mn.Vector3(step.x, 0.0, 0.0),
            centre + mn.Vector3(0.0, 0.0, step.z),
        ]

    def _follow_ground_below(self, pathfinder):
        """Update what the drone would land on, now that it has moved.

        Flying across a building means flying over different surfaces, and a
        landing spot halfway up the stairs is not the one it took off from.
        snap_point returns NaN where there is no resting surface at all, in
        which case the drone keeps the last real one it knew about -- a drone
        over a stairwell should not forget how to land.
        """
        centre = self.object.translation

        snapped = pathfinder.snap_point(centre)

        if not all(value == value for value in snapped):
            return

        before = self.altitude

        self.ground_height = float(snapped[1])
        self.altitude = max(0.0, centre.y - self.resting_height)

        # Flying over a table changes what "one metre up" means. Shift the
        # target by the same amount, so crossing a surface does not silently
        # command a climb or a dive the operator never asked for.
        self.target_altitude = max(
            0.0,
            self.target_altitude + (self.altitude - before),
        )

    def remove(self):
        """Take the drone out of the scene."""
        manager = self.simulator.get_rigid_object_manager()
        manager.remove_object_by_id(self.object.object_id)
        self.object = None


# ---------------------------------------------------------------------
# Placement helper
# ---------------------------------------------------------------------


def find_spot_near(pathfinder, origin, distance=2.0):
    """Find a navigable point about `distance` metres from `origin`.

    Used to put the drone somewhere the camera can see it, rather than inside
    the camera. Tries eight directions and takes the first that lands on
    ground; falls back to `origin` if the spot is too enclosed for any of
    them to work.
    """
    origin = mn.Vector3(origin)

    for step in range(8):
        angle = step * (math.pi / 4.0)

        candidate = [
            origin.x + math.cos(angle) * distance,
            origin.y,
            origin.z + math.sin(angle) * distance,
        ]

        snapped = pathfinder.snap_point(candidate)

        # snap_point returns NaN when there is no navigable ground nearby.
        if not all(value == value for value in snapped):
            continue

        moved = mn.Vector3(snapped) - origin

        if moved.length() > distance * 0.5:
            return mn.Vector3(snapped)

    return origin
