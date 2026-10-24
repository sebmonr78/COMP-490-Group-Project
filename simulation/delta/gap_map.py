"""The map: what is solid, what is empty, and crucially WHERE THE GAPS ARE.

This replaces the glasses project's 2.5D height map, and the reason it had to
be replaced is the whole point of a drone that flies.

The glasses map stored one number per floor square: the tallest thing in it.
That is exactly right for a walking person, because a person's answer to every
obstacle is "go around". Height only had to answer one question -- would I hit
it -- so "the tallest thing here is 90 cm" was the end of the story.

A drone has three answers: around, OVER, and UNDER. None of those can be read
off a single height:

    a sofa         90 cm tall, nothing above it -> fly OVER
    a dining table 75 cm tall, clear underneath -> roll UNDER
    a bookcase     200 cm tall, to the ceiling  -> go AROUND

To the glasses map the sofa and the table are nearly the same square. The
difference is not in how tall they are, it is in WHERE THE EMPTY SPACE IS
above and below them. So this map stores the whole vertical column.

    Every floor square is split into `levels` layers of `layer` metres.
    Each layer is one of:

        SOLID     something was seen in it, in `solid_hits` separate frames
        FREE      the drone looked THROUGH it, in `free_hits` separate frames
        UNKNOWN   never established either way

FREE is the part that does not come for free, and it is why this file carves
rays. A depth pixel at 3 m says two things: there is something at 3 m, and
there is nothing in the 3 m in between. Without the second half the space
above the sofa is merely UNKNOWN -- indistinguishable from the inside of the
bookcase -- and the drone can never justify flying over anything. So every
sampled pixel marks its own ray clear along its length, up to just short of
where it stopped. That is the only reason this map can say "go over" rather
than "nothing has contradicted going over".

One vote per layer per frame, never per pixel. Fifty pixels of one wall
landing in one layer is one observation of that wall from one viewpoint, not
fifty, and counting pixels let a single frame's speckle clear `solid_hits` on
its own.

Heights are measured from the floor the camera could see at the time, not
from any world datum. That keeps "60 cm up" meaning 60 cm above the ground
right there, which is the question over/under actually asks. It also means
this map does not stack storeys: a second floor would be written on top of
the first. V1 maps one storey, which is what the drone can reach by rolling.

Frame: the tracker's start frame -- where the drone switched on is (0, 0),
with +across to its right and +along straight ahead of it.
"""

import json
from pathlib import Path

import numpy as np


UNKNOWN, FREE, SOLID = 0, 1, 2

# What `classify` answers with: what the drone can do in a square.
UNSEEN, FLOOR, UNDER, OVER, AIRBORNE, AROUND = 0, 1, 2, 3, 4, 5

CONFIG_PATH = Path(__file__).resolve().parents[1] / "config" / "map.json"


def load_config(path=None):
    return json.loads(Path(path or CONFIG_PATH).read_text())


class GapMap:
    """A grid of vertical columns, each one a stack of layers.

    Attributes:
        cell     metres across one floor square
        layer    metres tall for one layer of a column
        n        squares along each side of the map
        levels   layers in every column
        solid_votes / free_votes   (n, n, levels) counts, in frames
    """

    def __init__(self, config, camera_height):
        g = config["grid"]

        self.g = g
        self.cell = float(g["cell"])
        self.layer = float(g["layer"])
        self.n = int(round(g["size_m"] / self.cell))
        self.levels = int(round(g["height_m"] / self.layer))
        self.half = self.n // 2

        # The camera height when the map was started, kept only so a saved
        # map can say what it was built from.
        self.start_camera_height = float(camera_height)

        self.solid_votes = np.zeros((self.n, self.n, self.levels), np.uint16)
        self.free_votes = np.zeros((self.n, self.n, self.levels), np.uint16)

        self.frames = 0

        # Scratch for the one-vote-per-frame scatter, allocated once. A fresh
        # 4.8-million-entry array per frame is most of the cost of mapping.
        self._touched = np.zeros(self.n * self.n * self.levels, bool)

    # -----------------------------------------------------------------
    # Where things are
    # -----------------------------------------------------------------

    @property
    def shape(self):
        return (self.n, self.n, self.levels)

    def _place(self, x, y, z, pose, camera_height):
        """Camera-frame points -> (row, column, level, inside).

        Only the pose's YAW is used, and the height comes from this frame's
        own measured floor rather than from the pose. That is deliberate, and
        inherited from the glasses project where it was measured: a tracker
        that believes the camera has tipped three degrees forward lifts a
        point 3 m away by 16 cm, which is enough to invent an obstacle, and
        that tilt error only grows. Yaw drifts too, but it does not turn
        floor into furniture.
        """
        height = camera_height - y

        forward = pose[:3, 2]
        yaw = np.arctan2(forward[0], forward[2])

        across = pose[0, 3] + np.cos(yaw) * x + np.sin(yaw) * z
        along = pose[2, 3] - np.sin(yaw) * x + np.cos(yaw) * z

        column = np.floor(across / self.cell).astype(np.int64) + self.half
        row = np.floor(along / self.cell).astype(np.int64) + self.half
        level = np.floor(height / self.layer).astype(np.int64)

        inside = (
            (column >= 0)
            & (column < self.n)
            & (row >= 0)
            & (row < self.n)
            & (level >= 0)
            & (level < self.levels)
        )

        return row, column, level, inside

    def _vote(self, votes, row, column, level):
        """Add one vote to every named layer, counting each layer once.

        A scatter into a boolean and then one add, rather than np.add.at,
        which is both slower and would count the same layer once per pixel.
        """
        if len(row) == 0:
            return

        flat = (row * self.n + column) * self.levels + level

        self._touched[:] = False
        self._touched[flat] = True

        votes += self._touched.reshape(self.shape)

    # -----------------------------------------------------------------
    # Building it
    # -----------------------------------------------------------------

    def add(self, depth, fx, fy, cx, cy, pose, camera_height):
        """One depth image in metres, and the pose it was taken from.

        Two passes over the same picture, and the second one is the
        interesting half:

            solid   where each pixel's surface is
            free    the space each pixel looked through to get there
        """
        g = self.g

        self._add_solid(depth, fx, fy, cx, cy, pose, camera_height)
        self._carve_free(depth, fx, fy, cx, cy, pose, camera_height)

        self.frames += 1

    def _add_solid(self, depth, fx, fy, cx, cy, pose, camera_height):
        g = self.g
        stride = g["stride"]

        sample = depth[::stride, ::stride]

        rows, cols = np.nonzero(
            np.isfinite(sample)
            & (sample > g["min_range"])
            & (sample <= g["max_range"])
        )

        if len(rows) == 0:
            return

        z = sample[rows, cols].astype(np.float64)
        u = cols * stride
        v = rows * stride

        x = (u - cx) * z / fx
        y = (v - cy) * z / fy

        row, column, level, inside = self._place(x, y, z, pose, camera_height)

        self._vote(
            self.solid_votes,
            row[inside],
            column[inside],
            level[inside],
        )

    def _carve_free(self, depth, fx, fy, cx, cy, pose, camera_height):
        """Mark the space between the camera and each surface as empty.

        Samples every ray at `carve_samples` points spread along it, stopping
        `carve_margin` short of the surface so the surface itself is not
        carved away. Sampling in proportion to distance rather than at a
        fixed spacing keeps the cost per ray constant, and puts the samples
        where they matter: close in, where the grid squares are small
        compared with the distance between rays.

        A coarser stride than the solid pass on purpose. Free space is a
        large, smooth thing and does not need every pixel to establish it,
        and this pass does `carve_samples` times the work per pixel.
        """
        g = self.g
        stride = g["carve_stride"]
        count = int(g["carve_samples"])

        sample = depth[::stride, ::stride]

        rows, cols = np.nonzero(
            np.isfinite(sample)
            & (sample > g["min_range"])
            & (sample <= g["max_range"])
        )

        if len(rows) == 0 or count <= 0:
            return

        z = sample[rows, cols].astype(np.float64)
        u = cols * stride
        v = rows * stride

        # The ray through this pixel, as a direction with z = 1.
        dx = (u - cx) / fx
        dy = (v - cy) / fy

        # How far along each ray to stop: just short of the surface.
        reach = np.maximum(z - g["carve_margin"], 0.0)

        # Fractions of the way along, skipping 0 (the camera's own square).
        steps = (np.arange(1, count + 1) / count)[None, :]

        along = reach[:, None] * steps

        x = (dx[:, None] * along).ravel()
        y = (dy[:, None] * along).ravel()
        depth_along = along.ravel()

        row, column, level, inside = self._place(
            x, y, depth_along, pose, camera_height
        )

        self._vote(
            self.free_votes,
            row[inside],
            column[inside],
            level[inside],
        )

    # -----------------------------------------------------------------
    # Reading it
    # -----------------------------------------------------------------

    def states(self):
        """(n, n, levels) of UNKNOWN / FREE / SOLID.

        SOLID wins ties. A layer that has been both looked through and seen
        to contain something is a layer with something in it -- the carve is
        an inference, the surface is a measurement.
        """
        solid = self.solid_votes >= self.g["solid_hits"]
        free = (self.free_votes >= self.g["free_hits"]) & ~solid

        grid = np.full(self.shape, UNKNOWN, np.int8)
        grid[free] = FREE
        grid[solid] = SOLID

        return grid

    def ground_level(self, grid=None):
        """(n, n) lowest SOLID layer per column; -1 where nothing was found.

        The floor, in other words -- or the top of whatever is resting on it.
        """
        grid = self.states() if grid is None else grid

        solid = grid == SOLID
        any_solid = solid.any(axis=2)

        lowest = np.argmax(solid, axis=2).astype(np.int16)
        lowest[~any_solid] = -1

        return lowest

    def ceiling_level(self, grid=None):
        """(n, n) highest layer that is FREE; -1 where none is.

        How high the drone could get here before running out of known-empty
        space. Not the architectural ceiling -- the top of what has been
        SEEN to be empty, which is the only ceiling the drone can act on.
        """
        grid = self.states() if grid is None else grid

        free = grid == FREE
        any_free = free.any(axis=2)

        highest = (
            self.levels - 1 - np.argmax(free[:, :, ::-1], axis=2)
        ).astype(np.int16)
        highest[~any_free] = -1

        return highest

    def layers_needed(self, diameter):
        """How many layers a drone of this diameter occupies."""
        return max(1, int(np.ceil(diameter / self.layer)))

    def passable(self, diameter, grid=None):
        """(n, n, levels) where the drone's whole body fits in free space.

        A layer is passable when it and the layers above it, enough of them
        to hold the drone, are ALL free. Not merely "not solid" -- free.
        Flying into unknown space is how a drone finds a wall with its shell.

        This single array is what makes over and under the same question.
        Passable high up over a sofa means fly over it; passable low down
        under a table means roll under it; a bookcase is passable at no
        level at all, so it has to be gone around.
        """
        grid = self.states() if grid is None else grid

        free = grid == FREE
        need = self.layers_needed(diameter)

        fits = free.copy()

        # fits[.., L] = all of free[.., L : L+need]. Shifting and ANDing is
        # the cheap way to ask that at every level at once.
        for offset in range(1, need):
            shifted = np.zeros_like(free)
            shifted[:, :, : self.levels - offset] = free[:, :, offset:]
            fits &= shifted

        # A body that would stick out of the top of the map does not fit.
        fits[:, :, self.levels - need + 1 :] = False

        return fits

    def rollable(self, diameter, grid=None):
        """(n, n) squares the drone could sit on and roll across.

        Needs a floor that has actually been seen, and enough free space
        directly above it. A square whose floor is unknown is not rollable
        however empty the air above it looks -- that is a hole until proven
        otherwise.
        """
        grid = self.states() if grid is None else grid

        ground = self.ground_level(grid)
        fits = self.passable(diameter, grid)

        rows, cols = np.nonzero(ground >= 0)
        above = ground[rows, cols] + 1

        ok = np.zeros((self.n, self.n), bool)

        valid = above < self.levels
        rows, cols, above = rows[valid], cols[valid], above[valid]

        ok[rows, cols] = fits[rows, cols, above]

        return ok

    def clearance(self, diameter, grid=None):
        """(n, n, 2): the lowest and highest passable layer in each column.

        A compact way to read a column: `low` is the gap nearest the floor
        -- rolling, or squeezing under something -- and `high` is the
        highest the drone could fly there. Where they differ by more than
        the drone's own height, there is more than one way through.
        """
        fits = self.passable(diameter, grid)
        any_fit = fits.any(axis=2)

        low = np.argmax(fits, axis=2).astype(np.int16)
        high = (self.levels - 1 - np.argmax(fits[:, :, ::-1], axis=2)).astype(np.int16)

        low[~any_fit] = -1
        high[~any_fit] = -1

        return np.stack([low, high], axis=2)

    # -----------------------------------------------------------------
    # Grid and world
    # -----------------------------------------------------------------

    def to_world(self, row, column, level=None):
        """Grid indices -> (across, along[, height]) in metres."""
        across = (np.asarray(column) - self.half + 0.5) * self.cell
        along = (np.asarray(row) - self.half + 0.5) * self.cell

        if level is None:
            return across, along

        height = (np.asarray(level) + 0.5) * self.layer

        return across, along, height

    def to_grid(self, across, along, height=None):
        """Metres -> grid indices. No bounds checking; see `inside`."""
        column = int(np.floor(across / self.cell)) + self.half
        row = int(np.floor(along / self.cell)) + self.half

        if height is None:
            return row, column

        return row, column, int(np.floor(height / self.layer))

    def inside(self, row, column, level=None):
        ok = 0 <= row < self.n and 0 <= column < self.n

        if level is None:
            return ok

        return ok and 0 <= level < self.levels

    # -----------------------------------------------------------------
    # Remembering it
    # -----------------------------------------------------------------

    def save(self, path):
        """Write the map to one .npz file.

        This is the beginning of the proposal's persistent digital
        representation: the drone is meant to come back later and compare,
        and it cannot compare against something it threw away when the
        process exited.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        np.savez_compressed(
            path,
            solid_votes=self.solid_votes,
            free_votes=self.free_votes,
            frames=self.frames,
            start_camera_height=self.start_camera_height,
            config=json.dumps({"grid": self.g}),
        )

        return path

    @classmethod
    def load(cls, path):
        """Read a map back. The grid settings come from the file, not the
        current config -- a saved map means nothing read on a different grid."""
        data = np.load(Path(path), allow_pickle=False)

        config = json.loads(str(data["config"]))

        this = cls(config, float(data["start_camera_height"]))

        if this.shape != data["solid_votes"].shape:
            raise ValueError(
                "Saved map does not match the grid it says it was built on: "
                f"{data['solid_votes'].shape} vs {this.shape}"
            )

        this.solid_votes = data["solid_votes"]
        this.free_votes = data["free_votes"]
        this.frames = int(data["frames"])

        return this

    # -----------------------------------------------------------------
    # Looking at it
    # -----------------------------------------------------------------

    def coverage(self):
        """A few numbers worth printing: how much has actually been mapped."""
        grid = self.states()

        known = int((grid != UNKNOWN).sum())
        solid = int((grid == SOLID).sum())
        columns = int((grid != UNKNOWN).any(axis=2).sum())

        return {
            "frames": self.frames,
            "layers_known": known,
            "layers_solid": solid,
            "columns_seen": columns,
            "floor_area_m2": columns * self.cell * self.cell,
        }

    def classify(self, diameter, grid=None):
        """(n, n) of what the drone can do in each square. The map, summarised.

        Six answers:

            0  UNSEEN    nothing known
            1  FLOOR     roll straight across
            2  UNDER     something overhead, and room to roll beneath it
            3  OVER      something in the way, and room to fly above it
            4  AIRBORNE  the air is clear, but no ground was ever seen here
            5  AROUND    something in the way with no gap either side

        AIRBORNE is the one worth explaining, because collapsing it into OVER
        is the easy mistake. Both are squares the drone can fly through and
        cannot roll across, but they are not the same situation at all:

            OVER      there is a measured obstacle under the clear air. The
                      drone knows what it is flying over.
            AIRBORNE  there is clear air and nothing underneath it has ever
                      been established. The drone could fly here but has no
                      idea whether it could land.

        Calling the second one "fly over" would report a confident obstacle
        where the truth is an unanswered question -- and since these squares
        are most of the edge of any partly-built map, it would make the map
        look far more certain than it is. They are also exactly where
        exploring next would pay off.

        The awkward case, and the one that decides how this is written, is the
        CEILING. "Something solid overhead with room underneath" describes a
        dining table, and it also describes every square of floor in a
        building. If the ceiling counts, UNDER means nothing -- the first
        version of this coloured almost an entire house as roll-under.

        What separates a table from a ceiling is not height; a mezzanine is
        high and a crawlspace is low. It is that a table has FREE SPACE ABOVE
        IT and a ceiling does not. So that is the test, and it needs no
        threshold to tune.
        """
        grid = self.states() if grid is None else grid

        fits = self.passable(diameter, grid)
        roll = self.rollable(diameter, grid)
        ground = self.ground_level(grid)

        solid = grid == SOLID
        free = grid == FREE

        out = np.zeros((self.n, self.n), np.uint8)
        out[(grid != UNKNOWN).any(axis=2)] = AROUND   # pessimistic until shown

        # Flyable but not rollable. Whether that is an obstacle to cross or
        # simply unmapped ground depends on what is UNDER the lowest layer
        # the drone fits in.
        flyable = fits.any(axis=2) & ~roll

        # Every question below is "is any layer above/below this one X?", and
        # all of them are asked of every square at once. Doing it a square at
        # a time in Python cost 300 ms a frame on a half-built map, which was
        # most of why the live window ran at 2 fps -- and a slow window is not
        # just uncomfortable, it wrecks the tracker, which needs frames close
        # enough together to match features between.
        level_of = np.arange(self.levels)[None, None, :]

        lowest_fit = np.argmax(fits, axis=2)

        below_is_solid = (solid & (level_of < lowest_fit[:, :, None])).any(axis=2)

        out[flyable] = AIRBORNE
        out[flyable & below_is_solid] = OVER

        out[roll] = FLOOR

        # Something solid standing above the drone's own slot: the only
        # squares where UNDER is even a question.
        floor_of = np.maximum(ground, 0)

        solid_above = solid & (level_of > floor_of[:, :, None])
        standing = solid_above.any(axis=2) & roll

        # The top of that solid stack.
        tops = (
            self.levels - 1 - np.argmax(solid_above[:, :, ::-1], axis=2)
        ).astype(np.int16)

        # Free space above it means it is a thing in the room. No free space
        # above it means it IS the room -- see the note about ceilings above.
        free_above = (free & (level_of > tops[:, :, None])).any(axis=2)

        out[standing & free_above] = UNDER

        return out

    def picture(self, diameter, kinds=None):
        """An RGB image of the map, one pixel per floor square.

        Pass `kinds` from `classify` when you already have it. The live
        window draws the map and prints the same counts beside it, and
        classifying twice a frame is the whole cost of both.

        Colour says what the drone can do there, because that -- not height
        -- is what this map exists to answer. See `classify` for the five
        cases and why the ceiling is the hard one.

            dark      never seen
            grey      plain floor: roll across it
            green     something overhead, and room to roll UNDER it
            blue      something in the way, and room to fly OVER it
            slate     air is clear but the ground was never seen: AIRBORNE
            orange    in the way with no gap either side, shading to dark red
                      with height
        """
        grid = self.states()

        if kinds is None:
            kinds = self.classify(diameter, grid)

        image = np.zeros((self.n, self.n, 3), np.uint8)

        image[kinds == AROUND] = (40, 40, 46)
        image[kinds == FLOOR] = (205, 205, 205)
        image[kinds == UNDER] = (70, 200, 110)
        image[kinds == OVER] = (80, 170, 255)
        image[kinds == AIRBORNE] = (70, 85, 110)

        # The squares with no way past shade by how tall the blockage is, so
        # a wall reads differently from a crate.
        solid = grid == SOLID
        top = np.where(
            solid.any(axis=2),
            self.levels - 1 - np.argmax(solid[:, :, ::-1], axis=2),
            0,
        )
        tall = np.clip((top * self.layer) / 1.8, 0.0, 1.0)

        blocked = kinds == AROUND

        image[blocked, 0] = (255 - 120 * tall[blocked]).astype(np.uint8)
        image[blocked, 1] = (150 - 130 * tall[blocked]).astype(np.uint8)
        image[blocked, 2] = (40 - 20 * tall[blocked]).astype(np.uint8)

        # Unseen squares stay black.
        image[kinds == UNSEEN] = (0, 0, 0)

        return image
