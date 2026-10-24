"""Position V1: where the glasses are, from their own two cameras and the IMU.

Each frame:
    1. find corners in the left image (ORB) -- spots that can be found again
    2. give each corner a distance, read from the stereo depth hazard already
       computed (reused: no second stereo matching)
    3. match last frame's corners to this frame's by their fingerprints
    4. solve "what camera move explains how they shifted" (PnP + RANSAC,
       which throws out the bad matches)
    5. if that fails -- blank wall, too few corners, an impossible jump --
       fall back on the IMU: gyro for the turn, last known speed for the step

The answer is the pose of the camera now, in the frame of the camera at the
first frame: (0, 0, 0) is where you started, OpenCV axes (x right, y down,
z forward). No loop closure yet, so error only grows; that is what the
split screen is for.
"""

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class Track:
    pose: np.ndarray        # 4x4, camera now -> start frame
    source: str             # "start", "camera" or "imu"
    corners: int
    inliers: int
    why: str = ""           # when the camera failed: why


class StereoTracker:
    def __init__(self, fx, cx, cy, config):
        self.c = config
        self.fx, self.cx, self.cy = fx, cx, cy
        self.K = np.array([[fx, 0.0, cx], [0.0, fx, cy], [0.0, 0.0, 1.0]])

        self.orb = cv2.ORB_create(nfeatures=config["features"])
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING)

        self.pose = np.eye(4)
        self.velocity = np.zeros(3)     # metres/second, in the start frame
        self._previous = None
        self._keyframe_pose = np.eye(4)
        self._keyframe_age = 0.0        # seconds since the keyframe

        # Which way is down, in the start frame, learned from the IMU. Over
        # time a walker's own acceleration averages out and only gravity is
        # left, so the average of what the chip feels points straight up.
        self._felt_sum = np.zeros(3)
        self._felt_count = 0

    def step(self, gray, depth, imu, dt):
        corners, fingerprints = self.orb.detectAndCompute(gray, None)
        pixels = np.array([k.pt for k in corners], dtype=np.float64).reshape(-1, 2)

        points, keep = self._lift(pixels, depth)
        fingerprints = fingerprints if fingerprints is not None else np.zeros((0, 32), np.uint8)

        if self._previous is None:
            self._keyframe(fingerprints[keep], points[keep])
            return Track(self.pose.copy(), "start", len(pixels), 0)

        self._keyframe_age += dt

        # Measured against the KEYFRAME, not against last frame. See
        # _keyframe for why this robot needs that and the glasses did not.
        motion, inliers, why = self._from_camera(
            pixels, fingerprints, self._keyframe_age
        )
        source = "camera"

        before = self.pose[:3, 3].copy()

        if motion is None:
            # The camera could not answer, so dead-reckon this one step from
            # wherever we had got to.
            self.pose = self.pose @ self._from_imu(imu, dt)
            source = "imu"
        else:
            self.pose = self._keyframe_pose @ motion

        if source == "camera":
            self.velocity = (self.pose[:3, 3] - before) / dt

        if imu is not None and imu.accel is not None:
            self._felt_sum += self.pose[:3, :3] @ imu.accel
            self._felt_count += 1

        if self._needs_new_keyframe(motion, inliers):
            self._keyframe(fingerprints[keep], points[keep])

        return Track(self.pose.copy(), source, len(pixels), inliers, why)

    # -----------------------------------------------------------------
    # Keyframes
    # -----------------------------------------------------------------

    def _keyframe(self, fingerprints, points):
        """Make this frame the one future frames are measured against.

        The glasses compared every frame with the one before it, and for a
        walking person that is fine: a head moves about 10 cm between frames,
        which is plenty of parallax to solve a camera move from.

        This drone is 30 cm across and rolls at about 1 m/s, so at 30 frames a
        second it moves 3 cm -- and 3 cm of parallax against a wall 2 m away
        is close enough to nothing that the answer is mostly noise. Measured:
        frame-to-frame tracking kept reporting "camera", never failed, and
        simply STOPPED ACCUMULATING -- the drone's believed position stalled
        at 0.9 m and wandered backwards while it actually travelled 1.8 m.
        Noise does not add up into distance.

        So the reference is held still until the drone has actually moved far
        enough to have something to measure, and every pose is composed from
        the keyframe's pose rather than from the last frame's. That also stops
        per-frame error compounding thirty times a second.
        """
        self._previous = (fingerprints, points)
        self._keyframe_pose = self.pose.copy()
        self._keyframe_age = 0.0

    def _needs_new_keyframe(self, motion, inliers):
        """Time for a new reference frame?

        Three reasons, and the first is the point of the whole scheme:

            moved far enough    there is parallax to measure against now
            turned far enough   a turn changes what is even visible
            running out of
            matched corners     the view has moved on from the keyframe
        """
        if motion is None:
            return False

        shifted = np.linalg.norm(motion[:3, 3])
        turned = np.degrees(np.linalg.norm(cv2.Rodrigues(motion[:3, :3])[0]))

        return (
            shifted >= self.c.get("keyframe_metres", 0.10)
            or turned >= self.c.get("keyframe_degrees", 8.0)
            or inliers < self.c.get("keyframe_min_inliers", 40)
        )

    def _lift(self, pixels, depth):
        """Each corner's 3D position in this camera's frame, from depth."""
        h, w = depth.shape
        u = np.clip(np.round(pixels[:, 0]).astype(int), 0, w - 1)
        v = np.clip(np.round(pixels[:, 1]).astype(int), 0, h - 1)

        z = depth[v, u]
        keep = np.isfinite(z) & (z >= self.c["min_depth"]) & (z <= self.c["max_depth"])

        z = np.where(keep, z, 0.0)
        points = np.stack(
            [(pixels[:, 0] - self.cx) * z / self.fx, (pixels[:, 1] - self.cy) * z / self.fx, z],
            axis=1,
        )

        return points, keep

    def _from_camera(self, pixels, fingerprints, elapsed):
        """Camera move since the KEYFRAME (4x4, now in the keyframe), or None."""
        old_prints, old_points = self._previous

        if len(old_prints) < self.c["min_inliers"] or len(fingerprints) < self.c["min_inliers"]:
            return None, 0, "too few corners with depth"

        pairs = self.matcher.knnMatch(old_prints, fingerprints, k=2)
        good = [p[0] for p in pairs if len(p) == 2 and p[0].distance < self.c["ratio"] * p[1].distance]

        if len(good) < self.c["min_inliers"]:
            return None, 0, "too few matches"

        where_then = old_points[[m.queryIdx for m in good]]
        where_now = pixels[[m.trainIdx for m in good]]

        ok, rvec, tvec, inliers = cv2.solvePnPRansac(
            where_then,
            where_now,
            self.K,
            None,
            iterationsCount=200,
            reprojectionError=self.c["reprojection_px"],
            confidence=0.999,
        )

        count = 0 if inliers is None else len(inliers)

        if not ok or count < self.c["min_inliers"]:
            return None, count, "matches disagree"

        # PnP gives "the keyframe's points, seen from now": X_now = R X_then + t.
        # The camera's own move is the inverse of that.
        rotation, _ = cv2.Rodrigues(rvec)
        seen = np.eye(4)
        seen[:3, :3] = rotation
        seen[:3, 3] = tvec.ravel()
        motion = np.linalg.inv(seen)

        # `elapsed` is the age of the keyframe, not the frame time: the
        # move being checked covers every frame since the keyframe.
        too_fast = np.linalg.norm(motion[:3, 3]) > self.c["max_speed"] * elapsed
        turned = np.degrees(np.linalg.norm(cv2.Rodrigues(motion[:3, :3])[0]))
        too_quick = turned > self.c["max_turn_deg_per_s"] * elapsed

        if too_fast:
            return None, count, "impossible jump"

        if too_quick:
            return None, count, "impossible turn"

        return motion, count, ""

    def _from_imu(self, imu, dt):
        """No usable picture: turn by the gyro; change speed by what the
        accelerometer feels, minus gravity; step at that speed."""
        motion = np.eye(4)

        if imu is None:
            motion[:3, 3] = self.pose[:3, :3].T @ self.velocity * dt
            return motion

        motion[:3, :3], _ = cv2.Rodrigues(imu.gyro * dt)

        if imu.accel is not None and self._felt_count > 0:
            gravity = -self._felt_sum / self._felt_count
            orientation = self.pose[:3, :3] @ motion[:3, :3]

            self.velocity = self.velocity + (orientation @ imu.accel + gravity) * dt

        # Integrating a chip's small bias makes speed creep up without end.
        # Fade the speed out instead: a blind stretch can coast for about a
        # second, not run away.
        self.velocity = self.velocity * np.exp(-dt / self.c.get("blind_coast_s", 1.0))

        motion[:3, 3] = self.pose[:3, :3].T @ self.velocity * dt

        return motion
