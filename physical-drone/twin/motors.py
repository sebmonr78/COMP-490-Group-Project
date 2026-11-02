"""The motors and props, from bench data.

A bench test spins one motor and prop on a thrust stand and writes down,
step by step up the throttle, the thrust, the current and the speed. Two
curves come out of that, and they are all a motor is to the physics:

    thrust = k * speed^2        a prop pushes air in proportion to speed
                                squared (a fit through the rows)
    power  = c * thrust^n       what it costs, fitted on a log-log line;
                                momentum theory says n = 1.5, the bench
                                says less (props get better as they speed up)

The motor's speed can't jump: it follows the ESC's command with a time
constant (spin_up). Its twist on the frame comes in two parts: the props'
air drag (shaft power / speed), and the kick of speeding a rotor up or
slowing it down (rotor inertia x acceleration).

Below the lowest row (180 g, about 5% of full thrust) the power curve is an
extrapolation -- which is where rolling lives. Step 3 checks it.
"""

import math

import numpy as np

G = 9.81


class MotorModel:
    """The curves of one motor + prop, fitted from config/body.json's bench."""

    def __init__(self, spec):
        bench = spec["bench"]
        rows = np.array(bench["rows"], dtype=float)        # grams, amps, rpm
        grams, amps, rpm = rows.T
        watts = amps * bench["volts"]

        # thrust = k * w^2, least squares through zero.
        w = rpm * 2.0 * math.pi / 60.0
        newtons = grams * G / 1000.0
        self.k = float(np.sum(newtons * w ** 2) / np.sum(w ** 4))

        # power = c * thrust_grams^n, a straight line in log-log.
        self.n, log_c = np.polyfit(np.log(grams), np.log(watts), 1)
        self.c = float(math.exp(log_c))
        self.n = float(self.n)

        self.max_thrust = bench["max_grams"] * G / 1000.0
        self.max_speed = math.sqrt(self.max_thrust / self.k)
        self.listed_max_watts = bench["max_watts"]
        self.idle_speed = spec["idle"] * self.max_speed
        self.spin_up = spec["spin_up"]
        self.spin_down = spec["spin_down"]
        self.rotor_inertia = spec["rotor_inertia"]
        self.shaft_efficiency = spec["shaft_efficiency"]

    def watts(self, thrust):
        """Electrical power in, at `thrust` newtons (the bench's voltage)."""
        grams = max(thrust, 0.0) * 1000.0 / G
        return self.c * grams ** self.n

    def twist_ratio(self, thrust):
        """The props' drag torque per newton of thrust (metres), at `thrust`:
        shaft power / speed / thrust."""
        speed = math.sqrt(thrust / self.k)
        return self.shaft_efficiency * self.watts(thrust) / speed / thrust

    def fit_report(self):
        rpm = 60.0 / (2.0 * math.pi)
        return {
            "k_newton_per_rad2": self.k,
            "full_thrust_n": self.max_thrust,
            "full_rpm": self.max_speed * rpm,
            "power_exponent": self.n,
            "watts_at_full_fit": self.watts(self.max_thrust),
            "watts_at_full_listed": self.listed_max_watts,
        }


class Motor:
    """One motor turning: commanded a thrust, it gets there with a lag."""

    def __init__(self, model):
        self.model = model
        self.speed = 0.0          # rad/s
        self.target = 0.0
        self.acceleration = 0.0
        self.armed = False

    def command(self, thrust):
        """The ESC's job: the speed that would give `thrust` (PX4 sends
        thrust, linearised; here exactly), never below idle while armed."""
        m = self.model

        if not self.armed:
            self.target = 0.0
            return

        thrust = min(max(thrust, 0.0), m.max_thrust)
        self.target = max(math.sqrt(thrust / m.k), m.idle_speed)

    def step(self, dt):
        m = self.model
        tau = m.spin_up if self.target > self.speed else m.spin_down
        before = self.speed
        self.speed += (self.target - self.speed) * (1.0 - math.exp(-dt / tau))
        self.acceleration = (self.speed - before) / dt

    @property
    def thrust(self):
        return self.model.k * self.speed ** 2

    @property
    def watts(self):
        return self.model.watts(self.thrust)
