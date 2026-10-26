"""Scripted drives: the same window, driven from a JSON file
(tests/scripts/*.json) instead of the keys -- how every test drive is run."""

import json
import math
from pathlib import Path

from delta.drone import TILT_SPEED, TURN_SPEED, WALK_SPEED


class Script:
    """Steps from a JSON list, one after another:

        {"wait": 3}                         do nothing for 3 s
        {"walk": 2.0}                       walk forward for 2 s
        {"turn": 90}                        turn left 90 degrees (negative: right)
        {"turn": 360, "speed": 120}         ...at 120 degrees a second (default 50)
        {"label": "climb"}                  name what follows, in the --trace,
                                            until the next label (tools/segments.py)
        {"say": "take me to the fridge"}    type it to the glasses
        {"key": "identify"}                 Space; also "stop"
        {"roam": 3}                         walk to 3 random spots in turn
        {"autopilot": 60}                   follow the brain's route, up to 60 s,
                                            until the glasses say "arrived"
        {"wait_for": "ready", "timeout": 300}   until the brain is ready
        {"screenshot": "name"}
        {"score": "name"}                   score the live map (outputs/eval/)
        {"score_rooms": "name"}             score the brain's rooms against
                                            HSSD's room outlines
        {"build_replica": true}             build the 3D replica now (when it
                                            is not built as the drone goes)
        {"score_replica": "name"}           score the brain's 3D replica against
                                            the real house (true depth)
        {"plan_ahead": 3}                   plan a route to the spot 3 m ahead,
                                            as a click on the map would
        {"fly": 1.2}                        take off if needed and fly to 1.2 m
                                            (underside above the floor); done on
                                            arrival, or when a ceiling stops it
        {"land": true}                      come down and land
        {"tilt": 30}                        tilt the cameras to 30 degrees down
                                            (negative: up; 0: level)
        {"quit": true}
    """

    def __init__(self, path, app):
        self.steps = json.loads(Path(path).read_text())
        self.app = app
        self.walk = 0.0
        self.turn = 0.0
        self._index = -1
        self._elapsed = 0.0
        self._target = 0.0
        self._done = True
        self._roams = 0
        self._arrived = False
        self._turn_speed = TURN_SPEED
        self.label = None

    def heard(self, kind, text):
        if "arrived" in text.lower() or "can't find" in text.lower():
            self._arrived = True

    def step(self, dt):
        app = self.app
        self._elapsed += dt

        if not self._done:
            self._continue(dt)
            return

        self._index += 1
        self.walk = self.turn = 0.0

        if self._index >= len(self.steps):
            app.running = False
            return

        step = self.steps[self._index]
        self._elapsed = 0.0
        self._done = False
        print(f"[script] {step}", flush=True)

        if "say" in step:
            app.ask(step["say"])
            self._done = True
        elif "key" in step:
            app.command(step["key"], step["key"])
            self._done = True
        elif "screenshot" in step:
            app.screenshot(step["screenshot"])
            self._done = True
        elif "score" in step:
            app.score_map(step["score"])
            self._done = True
        elif "score_rooms" in step:
            app.score_rooms(step["score_rooms"])
            self._done = True
        elif "build_replica" in step:
            app.command("build_replica", "Build the 3D replica.")
            self._done = True
        elif "score_replica" in step:
            app.score_replica(step["score_replica"])
            self._done = True
        elif "plan_ahead" in step:
            # A spot this far straight ahead, planned to as a click would.
            here, ahead = app.world.drone.floor_position, app.world.forward * float(step["plan_ahead"])
            self.route_target = (float(here.x + ahead.x), -float(here.z + ahead.z))
            app.plan_to(self.route_target)
            self._done = True
        elif "label" in step:
            self.label = str(step["label"])
            self._done = True
        elif "quit" in step:
            app.running = False
        elif "roam" in step:
            self._roams = int(step["roam"])
            self._roam_mark = None
        elif "autopilot" in step:
            self._arrived = False
            app.autopilot = True
        elif "turn" in step:
            self._target = math.radians(float(step["turn"]))
            self._turn_speed = math.radians(float(step.get("speed", math.degrees(TURN_SPEED))))
        elif "fly" in step:
            drone = app.world.drone

            if not drone.flying and not drone.take_off():
                app.note("Script: no room overhead to take off.")
                self._done = True
            else:
                drone.target_altitude = float(step["fly"])
                self._last_altitude = drone.altitude
                self._still_for = 0.0
        elif "land" in step:
            app.world.drone.land()
        elif "tilt" in step:
            self._tilt_target = float(step["tilt"])

    def _continue(self, dt):
        app = self.app
        step = self.steps[self._index]

        if "tilt" in step:
            # At the same speed as holding I or K.
            drone = app.world.drone
            gap = self._tilt_target - drone.pitch
            drone.tilt(math.copysign(min(abs(gap), TILT_SPEED * dt), gap))
            self._done = abs(self._tilt_target - drone.pitch) < 0.5 or self._elapsed >= 10.0

        elif "fly" in step:
            drone = app.world.drone

            # Arrived -- or stopped moving, which is what a ceiling or a beam
            # overhead looks like: the drone holds below it rather than
            # pressing on, and the step must not wait for it for ever.
            moved = abs(drone.altitude - self._last_altitude)
            self._last_altitude = drone.altitude
            self._still_for = self._still_for + dt if moved < 1e-4 else 0.0

            self._done = (
                abs(drone.altitude - drone.target_altitude) < 0.03
                or self._still_for > 1.0
                or self._elapsed >= float(step.get("timeout", 20.0))
            )

        elif "land" in step:
            self._done = (
                not app.world.drone.flying
                or self._elapsed >= float(step.get("timeout", 20.0))
            )

        elif "wait" in step:
            self._done = self._elapsed >= float(step["wait"])

        elif "walk" in step:
            self.walk = WALK_SPEED
            self._done = self._elapsed >= float(step["walk"])

        elif "turn" in step:
            if abs(self._target) < 1e-3:
                self.turn = 0.0
                self._done = True
            else:
                amount = math.copysign(min(self._turn_speed * dt, abs(self._target)), self._target)
                self._target -= amount
                self.turn = amount / dt

        elif "roam" in step:
            # Stuck: in the air, a route planned on the floor can run into a
            # cupboard the drone cannot fly over, and the step used to wait
            # for it for ever. No progress for 8 s (a half-turn takes 3.6)
            # gives up on that spot and moves on to the next.
            here = app.world.drone.floor_position

            if app.roam is not None:
                if self._roam_mark is None or math.hypot(
                    here.x - self._roam_mark[0], here.z - self._roam_mark[1]
                ) > 0.1:
                    self._roam_mark = (here.x, here.z, self._elapsed)
                elif self._elapsed - self._roam_mark[2] > 8.0:
                    app.note("Script: stuck, skipping to the next spot.")
                    app.roam = None
                    self._roam_mark = None

            if app.roam is None:
                if self._roams <= 0:
                    self._done = True
                else:
                    self._roams -= 1
                    app.start_roam()

        elif "autopilot" in step:
            if self._arrived or self._elapsed >= float(step["autopilot"]):
                app.autopilot = False
                self._done = True

                # How close it got to a spot planned with plan_ahead.
                if getattr(self, "route_target", None) is not None:
                    here = app.world.drone.floor_position
                    off = math.dist((here.x, -here.z), self.route_target)
                    # Routes stop short of the spot on purpose: by
                    # navigation.approach in brain.json, 0.5 m at the least.
                    print(f"[route] ended {off:.2f} m from the spot it was sent to "
                          f"({'reached' if off < 0.8 else 'NOT reached'}; routes stop ~0.5 m short)",
                          flush=True)

        elif "wait_for" in step:
            if app.brain is not None and app.brain.error:
                print(f"[script] the brain went away: {app.brain.error}", flush=True)
                app.running = False
                return

            ready = app.brain is not None and app.brain.ready
            self._done = ready or self._elapsed >= float(step.get("timeout", 300))
