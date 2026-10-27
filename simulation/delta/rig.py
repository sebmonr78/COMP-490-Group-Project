"""The camera rig: where the lenses are, and what they can see.

The drone is a place to stand. This file is what gets bolted to it.

Ported from the Spatial Memory Glasses project, which built it backwards from
a real OAK-FFC-4P baseboard. The geometry is identical -- two parallel cameras
a measured baseline apart -- so this is reused rather than rewritten. What
changes for DELTA is where the rig is MOUNTED:

    glasses   on a head, which turns when the wearer turns
    drone     on a gimbal inside the sphere, which keeps it level

A sphere that rolls tumbles end over end. Cameras bolted to the shell would
tumble with it, and a stereo pair upside down is not a stereo pair. Real
rolling robots solve this with a gimbal, so the rig takes its heading from
the drone and nothing else -- see Drone.heading.

Everything here is driven by `config/rig.json`, and that is the point. The
project is being built backwards from real hardware -- an OAK-FFC-4P baseboard
with two AR0234 global-shutter modules on it -- and none of the numbers that
matter are known yet. The baseline will be whatever the printed frame actually
measures. The field of view will be whatever the lens actually gives, not what
the listing claims. So no camera is written down in code: adding the downward
pair, or correcting the baseline after measuring it with callipers, is an edit
to a JSON file.

Two things this file exists to stop:

    The aspect-ratio trap.
        The AR0234 is 1920x1200, which is 16:10. The rig this replaces was
        640x480, which is 4:3. For the SAME horizontal field of view, a 16:10
        sensor has a markedly smaller VERTICAL field -- and vertical field is
        what sees the floor. Fitting the stock 82 degree lens would quietly
        move the ground horizon several metres out and make the thing useless
        as a navigation aid, while every horizontal number looked fine.
        `vfov()` and `ground_horizon()` are here so that trap is one call away.

    Pinhole creep.
        Habitat renders a perfect pinhole. Real M12 lenses at 120 degrees do
        not. If the undistort step is added later, it has to be threaded back
        through code that assumed it away. So `undistort()` exists now, with
        identity coefficients, and does nothing until real calibration numbers
        land in the config.

Conventions, all of them Habitat's and all of them paid for once already:

    -Z is forward, +Y is up, +X is right.
    Positive yaw turns LEFT.
    Positive pitch, in this file, tilts DOWN.
    Sensor `orientation` composes as Rz * Ry * Rx -- yaw outside pitch.
    `SensorSpec.position` must hold floats; an int fails the Vector3 binding.
    `hfov` is horizontal and pixels are square, so fx == fy at any aspect.
"""

import json
import math
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np


CONFIG_PATH = (
    Path(__file__).resolve().parents[1] / "config" / "rig.json"
)


# ---------------------------------------------------------------------
# What a camera is
# ---------------------------------------------------------------------


@dataclass(frozen=True)
class Camera:
    """One lens, positioned in the sensor head's own frame.

    `x`, `y`, `z` are metres from the centre of the drone: +x right,
    +y up, -z forward.

    `yaw` and `pitch` are degrees. Positive yaw points left, positive pitch
    points down. Roll is not offered -- a camera rolled on a frame is a
    mistake, not a design, and the gimbal exists to keep it that way however
    the sphere tumbles.
    """

    name: str
    x: float
    y: float
    z: float
    yaw: float
    pitch: float
    hfov: float

    @property
    def position(self):
        """As Habitat wants it: three floats, in the mount's frame."""
        return [float(self.x), float(self.y), float(self.z)]

    @property
    def orientation(self):
        """As Habitat wants it: three floats of radians, Rz * Ry * Rx.

        The pitch sign flips here. Habitat rotates about +X by the first
        element, and a positive rotation about +X tilts a -Z-forward camera
        UP. We want positive pitch to mean down, because that is how a person
        describes glasses, so it is negated exactly once -- here.
        """
        return [
            float(math.radians(-self.pitch)),
            float(math.radians(self.yaw)),
            0.0,
        ]


@dataclass(frozen=True)
class Rig:
    """Every camera on the drone, plus the two resolutions.

    Attributes:
        cameras:        the lenses, in config order
        capture:        (width, height) the sensor actually reads out
        process:        (width, height) stereo actually runs on
        stereo_pair:    (left_name, right_name)
        distortion:     camera name -> OpenCV plumb-bob coefficients
        baseline:       metres between the stereo pair, centre to centre
        source:         where this came from, for printing
        rangefinder:    the down-facing distance sensor (see rig.json)
    """

    cameras: tuple
    capture: tuple
    process: tuple
    stereo_pair: tuple
    distortion: dict
    baseline: float
    source: str = "config/rig.json"
    rangefinder: dict = None

    # -----------------------------------------------------------------
    # Looking things up
    # -----------------------------------------------------------------

    @property
    def names(self):
        return [camera.name for camera in self.cameras]

    def camera(self, name):
        for camera in self.cameras:
            if camera.name == name:
                return camera

        raise KeyError(
            f"No camera called '{name}'. The rig has: {self.names}"
        )

    def rgb_uuid(self, name):
        """The sensor key this camera's colour frame arrives under."""
        return f"drone_{name}_rgb"

    def depth_uuid(self, name):
        """ORACLE. The key ground-truth depth arrives under.

        There is no depth sensor on the real drone. Anything that reads
        this is scoring perception, never feeding it.
        """
        return f"drone_{name}_depth"

    # -----------------------------------------------------------------
    # Geometry
    # -----------------------------------------------------------------

    def resolution(self, stage="capture"):
        if stage == "capture":
            return self.capture

        if stage == "process":
            return self.process

        raise ValueError(
            f"stage must be 'capture' or 'process', not {stage!r}"
        )

    def intrinsics(self, name, stage="capture"):
        """The 3x3 pinhole matrix for one camera at one stage.

        Habitat builds its projection from the HORIZONTAL fov with square
        pixels, so fx == fy whatever the aspect ratio. Verified against the
        simulator's own projection matrix, not assumed.
        """
        width, height = self.resolution(stage)

        return intrinsics_for(
            width,
            height,
            self.camera(name).hfov,
        )

    def vfov(self, name, stage="capture"):
        """Vertical field of view in degrees.

        The number the aspect-ratio trap hides. At 120 degrees horizontal,
        4:3 gives about 100 degrees vertical and 16:10 gives 94.5 -- and at
        82 degrees horizontal, 16:10 gives 57.
        """
        width, height = self.resolution(stage)

        return vfov_for(
            width,
            height,
            self.camera(name).hfov,
        )

    def ground_horizon(self, name, eye_height, stage="capture"):
        """Nearest point of flat floor this camera can see, in metres.

        Everything closer is a blind spot directly beneath the drone -- which is
        where the floor it is about to roll onto lives, so it is the one
        blind spot a mapping robot cannot afford.

        `eye_height` is the height of the LENS above the RENDERED FLOOR, and
        those last two words are the trap: the wearer's origin sits on the
        navmesh, which floats above the floor. Measure it, do not assume it.

        `below` is how far under the horizon the bottom edge of the frame
        points, and the two limits go in opposite directions:

            below <= 0     the frame stops at or above the horizon, so no
                           flat floor is ever in it, at any distance: inf.
            below >= 90    the frame reaches past straight down, so floor
                           is visible right under the lens: zero.

        The version of this inherited from the old three-camera rig returned
        inf for the second case, which is the first case's answer. It never
        fired there -- 110 degrees at 4:3 pitched 20 down comes to 67 -- and
        it fires immediately for a downward-facing camera, where it claims a
        camera pointed at the floor cannot see the floor. Caught by
        `tools/measure_rig.py`, which measured zero against it.
        """
        camera = self.camera(name)

        below = math.radians(
            self.vfov(name, stage) / 2.0 + camera.pitch
        )

        if below <= 0.0:
            return math.inf

        if below >= math.pi / 2.0:
            return 0.0

        return eye_height / math.tan(below)

    def rectifiable(self, margin=5.0):
        """Whether planar stereo rectification is geometrically possible.

        A ray 90 degrees off the rectified axis projects to infinity, so
        `|yaw| + hfov/2` has to stay under 90 for both cameras of the pair --
        and it is already badly conditioned before it gets there.

        Parallel mounting makes this trivially true, which is the whole
        reason the splayed three-camera layout is gone. At 0 degrees of yaw
        and 120 degrees of fov the worst ray is 60, against a limit of 90.

        Returns (ok, worst_ray_degrees).
        """
        worst = max(
            abs(self.camera(name).yaw) + self.camera(name).hfov / 2.0
            for name in self.stereo_pair
        )

        return worst < (90.0 - margin), worst

    def stereo_overlap(self):
        """Angular width both cameras of the pair can see, in degrees.

        Depth exists only here. Parallel cameras overlap completely, so this
        equals the fov; it stops being trivial the moment anyone splays them
        again.
        """
        spans = [
            (
                self.camera(name).yaw - self.camera(name).hfov / 2.0,
                self.camera(name).yaw + self.camera(name).hfov / 2.0,
            )
            for name in self.stereo_pair
        ]

        low = max(span[0] for span in spans)
        high = min(span[1] for span in spans)

        return max(0.0, high - low)

    def coverage(self):
        """Angular width every camera sees between them, in degrees."""

        low = min(
            camera.yaw - camera.hfov / 2.0
            for camera in self.cameras
        )

        high = max(
            camera.yaw + camera.hfov / 2.0
            for camera in self.cameras
        )

        return high - low

    # -----------------------------------------------------------------
    # Trying a different rig without editing the file
    # -----------------------------------------------------------------

    def with_hfov(self, hfov):
        """The same rig with every lens changed. For sweeping candidates."""

        return replace(
            self,
            cameras=tuple(
                replace(camera, hfov=float(hfov))
                for camera in self.cameras
            ),
            source=f"{self.source} (hfov overridden to {hfov})",
        )

    def with_pitch(self, pitch):
        """The same rig tilted. For sweeping the ground/overlap trade-off."""

        return replace(
            self,
            cameras=tuple(
                replace(camera, pitch=float(pitch))
                for camera in self.cameras
            ),
            source=f"{self.source} (pitch overridden to {pitch})",
        )

    def with_baseline(self, baseline):
        """The same rig re-spaced, once the printed frame is measured."""

        scale = float(baseline) / self.baseline if self.baseline else 0.0

        return replace(
            self,
            baseline=float(baseline),
            cameras=tuple(
                replace(camera, x=camera.x * scale)
                for camera in self.cameras
            ),
            source=f"{self.source} (baseline overridden to {baseline})",
        )

    # -----------------------------------------------------------------
    # Saying what it is
    # -----------------------------------------------------------------

    def describe(self, eye_height=None):
        """A few lines fit to print at startup."""

        width, height = self.capture
        process_width, process_height = self.process

        ok, worst = self.rectifiable()

        lines = [
            f"Rig: {len(self.cameras)} camera(s) from {self.source}",
            f"  capture   {width}x{height}"
            f"   process {process_width}x{process_height}",
            f"  baseline  {self.baseline * 100:.1f} cm",
            f"  rectifiable  {'yes' if ok else 'NO'}"
            f"  (worst ray {worst:.1f} deg, limit 90)",
        ]

        for camera in self.cameras:
            line = (
                f"  {camera.name:<10}"
                f" hfov {camera.hfov:5.1f}"
                f"  vfov {self.vfov(camera.name):5.1f}"
                f"  yaw {camera.yaw:+5.1f}"
                f"  pitch {camera.pitch:+5.1f}"
            )

            if eye_height is not None:
                horizon = self.ground_horizon(camera.name, eye_height)
                line += f"  ground from {horizon:5.2f} m"

            lines.append(line)

        return "\n".join(lines)


# ---------------------------------------------------------------------
# The maths, free of any rig
# ---------------------------------------------------------------------


def intrinsics_for(width, height, hfov_degrees):
    """Pinhole intrinsics for a Habitat camera.

    fx == fy == (W/2) / tan(hfov/2), and the principal point is the centre.
    Verified against the simulator's projection matrix at 1920x1200: both
    came back 554.256 px at 120 degrees.
    """
    focal = (width / 2.0) / math.tan(math.radians(hfov_degrees) / 2.0)

    return np.array(
        [
            [focal, 0.0, width / 2.0],
            [0.0, focal, height / 2.0],
            [0.0, 0.0, 1.0],
        ]
    )


def vfov_for(width, height, hfov_degrees):
    """Vertical field of view in degrees, from the horizontal one.

        vfov = 2 * atan( tan(hfov/2) * height / width )

    Because pixels are square, this is pure aspect ratio. It is also the
    single line that separates a usable rig from a blind one.
    """
    half = math.tan(math.radians(hfov_degrees) / 2.0) * (height / width)

    return 2.0 * math.degrees(math.atan(half))


def hfov_for_vfov(width, height, vfov_degrees):
    """The horizontal fov needed to buy a given vertical fov.

    The inverse of `vfov_for`, and the question actually being asked when
    someone shops for a lens: the old rig had 94 degrees vertical, so what
    does a 16:10 sensor need horizontally to keep it?
    """
    half = math.tan(math.radians(vfov_degrees) / 2.0) * (width / height)

    return 2.0 * math.degrees(math.atan(half))


# ---------------------------------------------------------------------
# The undistort stage
# ---------------------------------------------------------------------


def undistort(image, intrinsics, coefficients):
    """Remove lens distortion. Identity while the coefficients are zero.

    This is a placeholder doing real work: in simulation the coefficients
    are all zero and Habitat's pinhole render is already undistorted, so
    the function returns the image untouched and costs one array comparison.

    It exists anyway, and is called anyway, so that the day a checkerboard
    calibration produces real coefficients, they go into `config/rig.json`
    and not one line of the pipeline changes. Adding this stage afterwards
    would mean threading it through code that had assumed a pinhole model.
    """
    coefficients = np.asarray(coefficients, dtype=np.float64)

    if not np.any(coefficients):
        return image

    import cv2

    return cv2.undistort(
        image,
        np.asarray(intrinsics, dtype=np.float64),
        coefficients,
    )


# ---------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------


def load(path=None):
    """Read `config/rig.json` into a Rig.

    Two conveniences in the file, both to keep the pending hardware numbers
    in one place each:

        hfov: null   means "use measured.hfov", so one edit re-lenses the rig
        x            is in BASELINES, so one edit re-spaces it
    """
    path = Path(path) if path else CONFIG_PATH

    if not path.is_file():
        raise FileNotFoundError(f"Rig configuration not found: {path}")

    data = json.loads(path.read_text())

    measured = data["measured"]

    baseline = float(measured["baseline"])
    default_hfov = float(measured["hfov"])
    default_pitch = float(measured.get("pitch", 0.0))

    cameras = []

    for entry in data["cameras"]:
        hfov = entry.get("hfov")
        pitch = entry.get("pitch")

        cameras.append(
            Camera(
                name=entry["name"],
                # x is written in baselines, so -0.5 and +0.5 are one
                # baseline apart and the measured number lives in one place.
                x=float(entry["x"]) * baseline,
                y=float(entry.get("y", 0.0)),
                z=float(entry.get("z", 0.0)),
                yaw=float(entry.get("yaw", 0.0)),
                pitch=default_pitch if pitch is None else float(pitch),
                hfov=default_hfov if hfov is None else float(hfov),
            )
        )

    distortion = {
        name: values
        for name, values in data.get("distortion", {}).items()
        if isinstance(values, list)
    }

    return Rig(
        cameras=tuple(cameras),
        capture=(
            int(data["capture"]["width"]),
            int(data["capture"]["height"]),
        ),
        process=(
            int(data["process"]["width"]),
            int(data["process"]["height"]),
        ),
        stereo_pair=tuple(data["stereo_pair"]),
        distortion=distortion,
        baseline=baseline,
        source=str(path.name),
        rangefinder={
            key: value
            for key, value in data.get("rangefinder", {}).items()
            if not key.startswith("_")
        },
    )


# ---------------------------------------------------------------------
# Turning it into Habitat sensors
# ---------------------------------------------------------------------


def sensor_specs(rig, stage="capture", ground_truth_depth=False):
    """Habitat sensor specifications for every camera on the rig.

    These hang off whichever agent mounts them, so their `position` is
    relative to that agent's node -- which, for DELTA, is the centre of the
    drone sphere.

    `ground_truth_depth` adds a depth sensor beside the left lens. The
    real drone has none. It is an evaluation oracle: score perception
    against it, never feed perception from it.
    """
    import habitat_sim

    width, height = rig.resolution(stage)

    specs = []

    for camera in rig.cameras:
        kinds = [
            (rig.rgb_uuid(camera.name), habitat_sim.SensorType.COLOR),
        ]

        # Only beside the left lens: that is the depth the brain's map is
        # in. Rendering the right one too cost a quarter of every frame.
        if ground_truth_depth and camera.name == rig.stereo_pair[0]:
            kinds.append(
                (rig.depth_uuid(camera.name), habitat_sim.SensorType.DEPTH)
            )

        for uuid, kind in kinds:
            spec = habitat_sim.CameraSensorSpec()

            spec.uuid = uuid
            spec.sensor_type = kind
            spec.sensor_subtype = habitat_sim.SensorSubType.PINHOLE
            spec.resolution = [height, width]

            # Floats, not ints. An int in this list fails the Vector3
            # binding with a thoroughly unhelpful error.
            spec.position = camera.position
            spec.orientation = camera.orientation

            spec.hfov = float(camera.hfov)

            specs.append(spec)

    return specs
