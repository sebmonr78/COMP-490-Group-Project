"""Estimate wearer position -- RTAB-Map on the slide.

Where are the glasses, and which way do they face? Two answers behind one
interface, chosen when the sim says hello:

    RtabmapLocalizer   the real thing. Runs the C++ bridge in rtabmap_bridge/
                       as a subprocess: stereo pairs in, poses out. Odometry
                       on every frame, plus a map of keyframes that corrects
                       the drift when the wearer comes back somewhere seen.

    OracleLocalizer    Habitat's own pose. For measuring the other boxes with
                       localisation error taken out, never for the product.

Frames go in through `submit`; `fix(frame_id)` waits for that frame's answer.

Anchors. When RTAB-Map closes a loop it corrects its past: nodes move. A wall
or a chair placed in the map from some pose must move with the node it was
seen from, or the map tears in two. So every Fix names the node it hangs off
and where it sits relative to that node, and `node_poses` always holds where
each node is now. The oracle never corrects anything: its single node, 0, is
the world origin and never moves.
"""

import json
import math
import os
import struct
import subprocess
import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

from delta import frames
from delta.brain import config


@dataclass
class Fix:
    """Where the glasses were when one frame was taken."""

    frame: int
    ok: bool
    pose: Optional[np.ndarray] = None          # world_from_base, 4x4
    anchor: int = 0                            # node the pose hangs off
    anchor_from_base: Optional[np.ndarray] = None
    inliers: int = 0
    loop: int = 0
    map: int = -1                              # RTAB-Map session of the anchor
    root_map: int = -1                         # oldest session joined to it
    ms: float = 0.0


def _matrix(values):
    if values is None:
        return None

    matrix = np.eye(4)
    matrix[:3, :] = np.asarray(values, dtype=float).reshape(3, 4)

    return matrix


class _Fixes:
    """Recent fixes by frame id, with a way to wait for one."""

    KEEP = 300

    def __init__(self):
        self._fixes = {}
        self._order = deque()
        self._changed = threading.Condition()
        self.latest = None
        self.dead = False

    def add(self, fix):
        with self._changed:
            self._fixes[fix.frame] = fix
            self._order.append(fix.frame)

            while len(self._order) > self.KEEP:
                self._fixes.pop(self._order.popleft(), None)

            if fix.ok:
                self.latest = fix

            self._changed.notify_all()

    def wait(self, frame, timeout):
        with self._changed:
            self._changed.wait_for(
                lambda: frame in self._fixes or self.dead, timeout
            )

            return self._fixes.get(frame)

    def kill(self):
        with self._changed:
            self.dead = True
            self._changed.notify_all()


class OracleLocalizer:
    """Habitat's pose of the left lens, converted into the brain's world."""

    name = "oracle"

    def __init__(self):
        self.node_poses = {0: np.eye(4)}
        self.fixes = _Fixes()
        self.stats = {"source": "oracle"}

    def submit(self, frame, pair):
        if frame.true_pose is None:
            raise RuntimeError(
                "The oracle localizer needs the sim to send true poses "
                "(run the sim with --pose oracle)."
            )

        position, rotation = frame.true_pose
        # Heading and position only: the head's tilt reaches the brain through
        # the IMU, as it would on the real glasses, and is applied once.
        pose = frames.level(frames.world_from_habitat_camera(position, rotation))

        self.fixes.add(
            Fix(frame=frame.id, ok=True, pose=pose, anchor_from_base=pose,
                map=0, root_map=0)
        )

    def fix(self, frame_id, timeout=1.0):
        return self.fixes.wait(frame_id, timeout)

    def close(self):
        self.fixes.kill()


class RtabmapLocalizer:
    """RTAB-Map stereo SLAM, through the bridge program."""

    name = "rtabmap"

    def __init__(self, settings, rectifier, size, database=None, log=None,
                 on_graph=None):
        bridge = config.path(settings.bridge)
        self.use_tilt = True     # pass the IMU's pitch and roll on to RTAB-Map

        if not bridge.is_file():
            raise FileNotFoundError(
                f"No RTAB-Map bridge at {bridge}. Run: bash setup/setup_rtabmap.sh"
            )

        width, height = size
        K = rectifier.K

        command = [
            str(bridge),
            "--fx", repr(float(K[0, 0])), "--fy", repr(float(K[1, 1])),
            "--cx", repr(float(K[0, 2])), "--cy", repr(float(K[1, 2])),
            "--baseline", repr(rectifier.baseline),
            "--width", str(width), "--height", str(height),
        ]

        if database:
            command += ["--database", str(database)]

        if log:
            command += ["--log", str(log)]

        for key, value in settings.parameters.items():
            command += ["--param", f"{key}={value}"]

        # How far a loop closure may move the robot before it is taken for a
        # lookalike. Sized to the robot: see brain.json localization.lookalike.
        lookalike = settings.get("lookalike", {})

        if "jump_per_metre" in lookalike:
            command += ["--allowed-jump-per-metre", repr(float(lookalike["jump_per_metre"]))]

        if "jump_per_lost_frame" in lookalike:
            command += ["--allowed-jump-per-lost", repr(float(lookalike["jump_per_lost_frame"]))]

        self.size = size
        self.keyframe_every = float(settings.keyframe_every)
        self.on_graph = on_graph
        self.node_poses = {}
        self.fixes = _Fixes()
        self.stats = {
            "source": "rtabmap", "nodes": 0, "loops": 0, "rejected": 0, "lost": 0,
            "inliers": 0, "ms": 0.0, "nodes_loaded": 0,
        }

        self._last_node = 0
        self._last_keyframe = -math.inf
        self._errors = []
        self._pending = deque(maxlen=3)
        self._wake = threading.Condition()
        self._closing = False
        self._ready = threading.Event()

        # RTAB-Map's libraries -- OpenCV, and OpenMP under it -- start one
        # worker per hardware thread unless told otherwise, and idle workers
        # spin instead of sleeping. Measured: 664% CPU, one thread working and
        # nineteen at ~32% each waiting for work, on cores the window needed.
        # See brain.json localization.threads.
        threads = str(int(settings.get("threads", 2)))
        environment = dict(
            os.environ,
            OMP_NUM_THREADS=threads,
            OMP_WAIT_POLICY="PASSIVE",
            OPENCV_FOR_THREADS_NUM=threads,
        )

        self.process = subprocess.Popen(
            command,
            env=environment,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        threading.Thread(target=self._read, daemon=True, name="rtabmap-read").start()
        threading.Thread(target=self._write, daemon=True, name="rtabmap-write").start()
        threading.Thread(target=self._drain_errors, daemon=True).start()

        if not self._ready.wait(60.0):
            self.process.kill()
            raise RuntimeError("The RTAB-Map bridge did not start.")

        if self.fixes.dead:
            raise RuntimeError(
                "The RTAB-Map bridge exited at startup: "
                + " ".join(self._errors[-5:])
            )

    # -----------------------------------------------------------------
    # In
    # -----------------------------------------------------------------

    def submit(self, frame, pair):
        """Queue one rectified pair. Never blocks: if the bridge is behind,
        the oldest waiting frame is dropped, which odometry tolerates far
        better than falling further and further behind."""
        keyframe = frame.stamp - self._last_keyframe >= self.keyframe_every

        if keyframe:
            self._last_keyframe = frame.stamp

        with self._wake:
            if len(self._pending) == self._pending.maxlen:
                dropped = self._pending.popleft()

                # Never drop a keyframe silently: let the next one be one.
                if dropped[4]:
                    self._last_keyframe = -math.inf

            self._pending.append(
                (frame.id, frame.stamp, pair.left, pair.right, keyframe, frame.imu_yaw,
                 frame.imu_pitch if self.use_tilt else None, frame.imu_roll if self.use_tilt else None,
                 frame.odometer)
            )
            self._wake.notify()

    def _write(self):
        width, height = self.size

        while True:
            with self._wake:
                self._wake.wait_for(lambda: self._pending or self._closing)

                if self._closing and not self._pending:
                    break

                frame_id, stamp, left, right, keyframe, imu_yaw, pitch, roll, odometer = self._pending.popleft()

            grey_left = cv2.cvtColor(left, cv2.COLOR_RGB2GRAY)
            grey_right = cv2.cvtColor(right, cv2.COLOR_RGB2GRAY)

            tilted = pitch is not None or roll is not None
            flags = ((1 if keyframe else 0) | (2 if imu_yaw is not None else 0) | (4 if tilted else 0)
                     | (8 if odometer is not None else 0))

            # Tilt in radians, as the bridge reads it (bridge.cpp, wire format).
            header = b"FRAM" + struct.pack(
                "<idiiidddd", frame_id, stamp, width, height, flags,
                0.0 if imu_yaw is None else float(imu_yaw),
                math.radians(pitch or 0.0), math.radians(roll or 0.0),
                0.0 if odometer is None else float(odometer),
            )

            try:
                self.process.stdin.write(header + grey_left.tobytes() + grey_right.tobytes())
                self.process.stdin.flush()

            except (BrokenPipeError, ValueError):
                break

        try:
            self.process.stdin.write(b"QUIT")
            self.process.stdin.close()

        except (BrokenPipeError, ValueError):
            pass

    # -----------------------------------------------------------------
    # Out
    # -----------------------------------------------------------------

    def _read(self):
        for raw in self.process.stdout:
            if not raw.startswith(b"@@ "):
                continue

            message = json.loads(raw[3:])
            kind = message["type"]

            if kind == "ready":
                self.stats["nodes_loaded"] = message["nodes_loaded"]
                self._ready.set()

            elif kind == "pose":
                self._on_pose(message)

            elif kind == "graph":
                poses = {
                    int(node): _matrix(values)
                    for node, values in message["poses"].items()
                }

                self.node_poses.update(poses)

                if self.on_graph:
                    self.on_graph(poses)

        self.fixes.kill()
        self._ready.set()

    def _on_pose(self, message):
        pose = _matrix(message["pose"]) if message["ok"] else None

        if message["node"] > 0 and message["node_pose"] is not None:
            self.node_poses[message["node"]] = _matrix(message["node_pose"])
            self._last_node = message["node"]

        anchor = self._last_node
        anchor_from_base = None

        if pose is not None:
            if anchor in self.node_poses:
                anchor_from_base = frames.invert(self.node_poses[anchor]) @ pose

            else:
                # Before the first node there is nothing to hang off but the
                # origin of the map.
                anchor = 0
                anchor_from_base = pose

        stats = self.stats
        stats["ms"] = message["ms"]
        stats["inliers"] = message["inliers"]
        stats["nodes"] = max(stats["nodes"], message["node"])

        if message["loop"] > 0:
            stats["loops"] += 1

        if message["rejected"] > 0:
            stats["rejected"] += 1

        if not message["ok"]:
            stats["lost"] += 1

        stats["imu_heading"] = stats.get("imu_heading", 0) + int(bool(message.get("imu_heading")))

        self.fixes.add(
            Fix(
                frame=message["id"],
                ok=bool(message["ok"]),
                pose=pose,
                anchor=anchor,
                anchor_from_base=anchor_from_base,
                inliers=message["inliers"],
                loop=message["loop"],
                map=message["map"],
                root_map=message["root_map"],
                ms=message["ms"],
            )
        )

    def _drain_errors(self):
        for raw in self.process.stderr:
            self._errors.append(raw.decode("utf-8", "replace").strip())
            del self._errors[:-50]

    # -----------------------------------------------------------------

    def fix(self, frame_id, timeout=1.0):
        return self.fixes.wait(frame_id, timeout)

    def close(self, timeout=30.0):
        """Stop, letting the bridge save its database first."""
        with self._wake:
            self._closing = True
            self._wake.notify()

        try:
            self.process.wait(timeout)

        except subprocess.TimeoutExpired:
            self.process.kill()

        self.fixes.kill()
