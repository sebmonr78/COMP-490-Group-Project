"""The flight brain: PX4's multicopter controller, rebuilt small.

PX4 flies a quadcopter with loops inside loops, each faster than the one
around it:

    where to be     position error -> velocity wanted         (P)        50 Hz
    how fast        velocity error -> acceleration wanted     (PID)      50 Hz
                    acceleration + gravity = the thrust vector: the frame
                    must point its +z along it (tilt capped) and push this hard
    which way up    attitude error -> turn rates wanted       (P)       500 Hz
    how fast it turns  rate error -> twisting torques         (PID)     500 Hz
    mixer           thrust + three torques -> four motors

The parameter names are PX4's (MPC_*, MC_*), and so are the default values
where PX4's are in physical units. PX4's rate gains are normalised to its
motors; here they are in rad/s^2 per rad/s and multiplied by the frame's
inertia, so they mean the same on any frame.

It needs a state: where it is, how fast, which way up, how fast it turns.
In step 1 that comes straight from the simulation (an oracle, like the
brain's --pose oracle). Step 2 replaces it with an estimate from the
sensors: the gyro, the accelerometer, the rangefinder and the flow sensor.
"""

import math
from collections import deque
from dataclasses import dataclass, field

import numpy as np

G = 9.81


def quat_mul(a, b):
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return np.array([aw * bw - ax * bx - ay * by - az * bz,
                     aw * bx + ax * bw + ay * bz - az * by,
                     aw * by - ax * bz + ay * bw + az * bx,
                     aw * bz + ax * by - ay * bx + az * bw])


def quat_conj(q):
    return np.array([q[0], -q[1], -q[2], -q[3]])


def quat_to_matrix(q):
    w, x, y, z = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def matrix_to_quat(R):
    t = np.trace(R)

    if t > 0:
        s = 2.0 * math.sqrt(t + 1.0)
        q = [0.25 * s, (R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s]
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        q = [(R[2, 1] - R[1, 2]) / s, 0.25 * s, (R[0, 1] + R[1, 0]) / s, (R[0, 2] + R[2, 0]) / s]
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        q = [(R[0, 2] - R[2, 0]) / s, (R[0, 1] + R[1, 0]) / s, 0.25 * s, (R[1, 2] + R[2, 1]) / s]
    else:
        s = 2.0 * math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        q = [(R[1, 0] - R[0, 1]) / s, (R[0, 2] + R[2, 0]) / s, (R[1, 2] + R[2, 1]) / s, 0.25 * s]

    q = np.array(q)
    return q / np.linalg.norm(q)


def heading(q):
    """Which way the frame's nose points, radians from +x, turning left."""
    R = quat_to_matrix(q)
    return math.atan2(R[1, 0], R[0, 0])


def tilt(q):
    """How far the frame's +z leans from straight up, radians."""
    return math.acos(max(-1.0, min(1.0, quat_to_matrix(q)[2, 2])))


@dataclass
class State:
    """What the flight brain knows. World frame: +z up."""
    position: np.ndarray
    velocity: np.ndarray
    attitude: np.ndarray         # quaternion w, x, y, z: world from frame
    rates: np.ndarray            # rad/s, in the frame's own axes


@dataclass
class Setpoint:
    position: np.ndarray
    yaw: float = 0.0
    velocity: np.ndarray = field(default_factory=lambda: np.zeros(3))       # feed-forward
    acceleration: np.ndarray = field(default_factory=lambda: np.zeros(3))   # feed-forward


class Trajectory:
    """A straight line from a to b, the way PX4's auto modes fly one: speed
    up, cruise, slow down, so the target never jumps and nothing overshoots.
    Speeds and accelerations are capped per direction (PX4's MPC_* limits).

    The acceleration can't jump either: the frame needs ~0.2 s to lean into
    it, so a target that starts accelerating at once leaves it behind, and
    one that stops at once lets it run past (12 cm on a 2 m line). PX4 caps
    the jerk; here the profile goes through a critically damped filter
    (time constant `smooth`), whose own velocity and acceleration are the
    feed-forward. It still never overshoots: the filter only ever lags.

    And the lean itself trails what is asked by the attitude loop's time
    constant (1 / MC_PITCH_P, 0.15 s), which alone let it run 12 cm past
    the end of a 2 m line: so the acceleration is handed over `lead`
    seconds early, the position and velocity on time."""

    def __init__(self, a, b, t0, gains=None, accel=1.5, smooth=0.12, lead=None):
        g = gains or Gains()
        self.a, self.b, self.t0 = np.asarray(a, float), np.asarray(b, float), t0
        line = self.b - self.a
        self.length = float(np.linalg.norm(line))
        self.dir = line / self.length if self.length > 1e-9 else np.zeros(3)

        flat, up = math.hypot(self.dir[0], self.dir[1]), self.dir[2]
        caps = [g.MPC_XY_VEL_MAX / flat if flat > 1e-9 else math.inf,
                (g.MPC_Z_VEL_MAX_UP if up > 0 else g.MPC_Z_VEL_MAX_DN) / abs(up) if abs(up) > 1e-9 else math.inf]
        self.v = min(caps)
        self.acc = accel
        # Short lines never reach cruise: a triangle instead of a trapezoid.
        self.v = min(self.v, math.sqrt(self.acc * self.length))
        self.ramp = self.v / self.acc if self.acc > 0 else 0.0
        self.cruise = (self.length - self.v * self.ramp) / self.v if self.v > 0 else 0.0
        self.duration = 2 * self.ramp + self.cruise + 6 * smooth      # + the filter settling
        self.w = 1.0 / smooth
        self.lead = 1.0 / g.MC_PITCH_P if lead is None else lead
        self.s, self.sv, self.sa, self.last = 0.0, 0.0, 0.0, t0
        self.history = deque([(t0, 0.0, 0.0)])          # (time, s, sv), up to `lead` ago

    def sample(self, t):
        """(position, velocity, acceleration) wanted at time t. Call with t
        rising: the filter runs `lead` ahead of it."""
        head = t + self.lead
        while self.last < head - 1e-12:
            dt = min(0.001, head - self.last)
            raw, _, _ = self._along(min(max(self.last + dt - self.t0, 0.0), self.duration))
            self.sa = self.w ** 2 * (raw - self.s) - 2.0 * self.w * self.sv
            self.sv += self.sa * dt
            self.s += self.sv * dt
            self.last += dt
            self.history.append((self.last, self.s, self.sv))

        while len(self.history) > 1 and self.history[1][0] <= t:
            self.history.popleft()
        _, s, sv = self.history[0]
        return self.a + s * self.dir, sv * self.dir, self.sa * self.dir

    def _along(self, t):
        r, c, v, acc = self.ramp, self.cruise, self.v, self.acc
        if t < r:
            return 0.5 * acc * t * t, acc * t, acc
        if t < r + c:
            return 0.5 * acc * r * r + v * (t - r), v, 0.0
        if t < 2 * r + c:
            u = t - r - c
            return 0.5 * acc * r * r + v * c + v * u - 0.5 * acc * u * u, v - acc * u, -acc
        return self.length, 0.0, 0.0


@dataclass
class Gains:
    # Position and velocity: PX4 v1.14's defaults.
    MPC_XY_P: float = 0.95
    MPC_Z_P: float = 1.0
    MPC_XY_VEL_P_ACC: float = 1.8
    MPC_XY_VEL_I_ACC: float = 0.4
    MPC_XY_VEL_D_ACC: float = 0.2
    MPC_Z_VEL_P_ACC: float = 4.0
    MPC_Z_VEL_I_ACC: float = 2.0
    MPC_XY_VEL_MAX: float = 1.5          # indoors: PX4's 12 m/s is for fields
    MPC_Z_VEL_MAX_UP: float = 1.0
    MPC_Z_VEL_MAX_DN: float = 0.7
    MPC_ACC_HOR_MAX: float = 3.0
    MPC_TILTMAX_AIR: float = 35.0        # degrees
    # Attitude: PX4's (rad/s per rad).
    MC_ROLL_P: float = 6.5
    MC_PITCH_P: float = 6.5
    MC_YAW_P: float = 2.8
    MC_ROLLRATE_MAX: float = 220.0       # degrees/s
    MC_YAWRATE_MAX: float = 200.0
    # Rates, physical: rad/s^2 per rad/s of error.
    RATE_P: tuple = (25.0, 25.0, 12.0)
    RATE_I: tuple = (10.0, 10.0, 5.0)
    RATE_D: tuple = (0.25, 0.25, 0.0)


class Mixer:
    """Thrust + torques -> four motor thrusts, and what to give up when the
    motors can't do it all: yaw first, then the total, never roll and
    pitch (PX4's order -- staying upright matters most)."""

    def __init__(self, rotors, spin, twist, max_thrust, min_thrust=0.0):
        # [thrust, torque x, torque y, torque z] = A @ motor thrusts
        x, y = rotors[:, 0], rotors[:, 1]
        self.A = np.vstack([np.ones(4), y, -x, -spin * twist])
        self.A_inv = np.linalg.inv(self.A)
        self.max = max_thrust
        self.min = min_thrust

    def mix(self, thrust, torque):
        collective = self.A_inv[:, 0] * thrust
        tilt = self.A_inv[:, 1:3] @ torque[:2]
        yaw = self.A_inv[:, 3] * torque[2]
        room = self.max - self.min

        spread = tilt.max() - tilt.min()
        if spread > room:                                  # can't even do roll and pitch
            tilt *= room / spread

        f = collective + tilt
        f += max(self.min - f.min(), 0.0) - max(f.max() - self.max, 0.0)   # slide the total

        # As much yaw as still fits.
        k = 1.0
        for fi, yi in zip(f, yaw):
            if yi > 0:
                k = min(k, (self.max - fi) / yi)
            elif yi < 0:
                k = min(k, (self.min - fi) / yi)

        return f + max(k, 0.0) * yaw


class FlightController:
    def __init__(self, mass, inertia, rotors, spin, twist, max_thrust, gains=None):
        self.g = gains or Gains()
        self.mass = mass
        self.J = np.asarray(inertia, float)
        self.mixer = Mixer(np.asarray(rotors), np.asarray(spin, float), twist, max_thrust)
        self.max_total = 4 * max_thrust
        self.airborne = False

        self.vel_integral = np.zeros(3)
        self.rate_integral = np.zeros(3)
        self.last_rates = None
        self.rate_d = np.zeros(3)
        self.last_vel_error = None
        self.vel_d = np.zeros(3)
        self.slow_due = 0.0

        self.thrust = mass * G                  # what the slow loop last asked for
        self.attitude_sp = np.array([1.0, 0, 0, 0])
        self.rates_sp = np.zeros(3)
        self.out = {}

    # -- 50 Hz: position and velocity -> thrust vector -> attitude ----------

    def _position(self, state, sp, dt):
        g = self.g
        error = sp.position - state.position
        v_sp = sp.velocity + np.array([g.MPC_XY_P * error[0], g.MPC_XY_P * error[1], g.MPC_Z_P * error[2]])

        speed = math.hypot(v_sp[0], v_sp[1])
        if speed > g.MPC_XY_VEL_MAX:
            v_sp[:2] *= g.MPC_XY_VEL_MAX / speed
        v_sp[2] = min(max(v_sp[2], -g.MPC_Z_VEL_MAX_DN), g.MPC_Z_VEL_MAX_UP)

        v_error = v_sp - state.velocity
        if self.last_vel_error is not None:
            raw = (v_error - self.last_vel_error) / dt
            self.vel_d += (raw - self.vel_d) * min(1.0, dt * 2 * math.pi * 5.0)    # 5 Hz filter
        self.last_vel_error = v_error

        P = np.array([g.MPC_XY_VEL_P_ACC, g.MPC_XY_VEL_P_ACC, g.MPC_Z_VEL_P_ACC])
        I = np.array([g.MPC_XY_VEL_I_ACC, g.MPC_XY_VEL_I_ACC, g.MPC_Z_VEL_I_ACC])
        D = np.array([g.MPC_XY_VEL_D_ACC, g.MPC_XY_VEL_D_ACC, 0.0])
        accel = sp.acceleration + P * v_error + I * self.vel_integral + D * self.vel_d

        horizontal = math.hypot(accel[0], accel[1])
        if horizontal > g.MPC_ACC_HOR_MAX:
            accel[:2] *= g.MPC_ACC_HOR_MAX / horizontal

        # The thrust vector, its lean capped.
        F = self.mass * (accel + np.array([0.0, 0.0, G]))
        F[2] = max(F[2], 0.1 * self.mass * G)
        lean = math.tan(math.radians(g.MPC_TILTMAX_AIR))
        side = math.hypot(F[0], F[1])
        saturated = side > F[2] * lean
        if saturated:
            F[:2] *= F[2] * lean / side

        if self.airborne and not saturated:
            self.vel_integral += v_error * dt

        # Frame +z along F; nose toward the heading wanted.
        z = F / np.linalg.norm(F)
        nose = np.array([math.cos(sp.yaw), math.sin(sp.yaw), 0.0])
        y = np.cross(z, nose)
        y /= np.linalg.norm(y)
        x = np.cross(y, z)
        self.attitude_sp = matrix_to_quat(np.column_stack([x, y, z]))

        # Push along where +z points now (PX4 does the same).
        body_z = quat_to_matrix(state.attitude)[:, 2]
        self.thrust = min(max(float(F @ body_z), 0.0), 0.9 * self.max_total)
        self.out.update(v_sp=v_sp, accel_sp=accel)

    # -- 500 Hz: attitude -> rates -> torques -> motors ------------------------

    def update(self, state, sp, t, dt):
        """Four motor thrusts (N) for now. Call at 500 Hz."""
        g = self.g

        if t >= self.slow_due:
            self._position(state, sp, 0.02)
            self.slow_due = t + 0.02

        # Attitude error in the frame's own axes (Brescianini, as PX4).
        qe = quat_mul(quat_conj(state.attitude), self.attitude_sp)
        if qe[0] < 0:
            qe = -qe
        K = np.array([g.MC_ROLL_P, g.MC_PITCH_P, g.MC_YAW_P])
        rates_sp = 2.0 * K * qe[1:]
        tilt_max = math.radians(g.MC_ROLLRATE_MAX)
        rates_sp[:2] = np.clip(rates_sp[:2], -tilt_max, tilt_max)
        rates_sp[2] = np.clip(rates_sp[2], -math.radians(g.MC_YAWRATE_MAX), math.radians(g.MC_YAWRATE_MAX))
        self.rates_sp = rates_sp

        w = state.rates
        if self.last_rates is not None:
            raw = -(w - self.last_rates) / dt                # D on the measurement
            self.rate_d += (raw - self.rate_d) * min(1.0, dt * 2 * math.pi * 30.0)    # 30 Hz filter
        self.last_rates = w.copy()

        error = rates_sp - w
        if self.airborne:
            self.rate_integral = np.clip(self.rate_integral + error * dt, -0.5, 0.5)

        alpha = np.array(g.RATE_P) * error + np.array(g.RATE_I) * self.rate_integral \
            + np.array(g.RATE_D) * self.rate_d
        torque = self.J @ alpha + np.cross(w, self.J @ w)

        thrusts = self.mixer.mix(self.thrust, torque)
        self.out.update(torque=torque, rates_sp=rates_sp, thrust=self.thrust)
        return thrusts

    def reset(self):
        """On the ground: the integrators would only wind up against it."""
        self.vel_integral[:] = 0.0
        self.rate_integral[:] = 0.0
