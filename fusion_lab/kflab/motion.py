"""Motion models: how the target is expected to move between two measurements.

Every model exposes the same small interface:

    f(x, dt)          predicted state after dt seconds
    F(x, dt)          jacobian of f with respect to x (equal to the matrix for linear models)
    Q(x, dt)          process noise covariance added during the prediction
    pv(x)             position and velocity [px, py, vx, vy] seen from outside
    pv_jacobian(x)    d pv / d x, used by the RADAR measurement model

Position is always the first two entries of the state, so the position covariance
used to draw the ellipse is always P[:2, :2], whatever the model.
"""

import numpy as np

from .angles import wrap


class ConstantVelocity:
    """x = [px, py, vx, vy]. The target keeps its velocity, accelerations are noise."""

    name = "CV"
    labels = ["px", "py", "vx", "vy"]
    units = ["m", "m", "m/s", "m/s"]
    angle_idx = []
    linear = True

    def __init__(self, sigma_a=1.0):
        self.sigma_a = sigma_a  # std of the random acceleration, m/s^2

    @property
    def dim(self):
        return len(self.labels)

    def F(self, x, dt):
        F = np.eye(4)
        F[0, 2] = dt
        F[1, 3] = dt
        return F

    def f(self, x, dt):
        return self.F(x, dt) @ x

    def Q(self, x, dt):
        # A random acceleration a, constant during dt, moves the state by G * a.
        G = np.array([[0.5 * dt**2, 0.0],
                      [0.0, 0.5 * dt**2],
                      [dt, 0.0],
                      [0.0, dt]])
        return G @ G.T * self.sigma_a**2

    def pv(self, x):
        return x[:4].copy()

    def pv_jacobian(self, x):
        return np.eye(4)

    def initial_state(self, pos, vel, pos_var, vel_var):
        x = np.array([pos[0], pos[1], vel[0], vel[1]])
        P = np.diag([pos_var[0], pos_var[1], vel_var, vel_var])
        return x, P


class ConstantAcceleration:
    """x = [px, py, vx, vy, ax, ay]. The target keeps its acceleration, jerk is noise."""

    name = "CA"
    labels = ["px", "py", "vx", "vy", "ax", "ay"]
    units = ["m", "m", "m/s", "m/s", "m/s2", "m/s2"]
    angle_idx = []
    linear = True

    def __init__(self, sigma_j=1.0, init_acc_std=2.0):
        self.sigma_j = sigma_j  # std of the random jerk, m/s^3
        self.init_acc_std = init_acc_std

    @property
    def dim(self):
        return len(self.labels)

    def F(self, x, dt):
        F = np.eye(6)
        F[0, 2] = F[1, 3] = dt
        F[2, 4] = F[3, 5] = dt
        F[0, 4] = F[1, 5] = 0.5 * dt**2
        return F

    def f(self, x, dt):
        return self.F(x, dt) @ x

    def Q(self, x, dt):
        # A random jerk j, constant during dt, moves the state by G * j.
        g = np.array([dt**3 / 6.0, dt**2 / 2.0, dt])
        G = np.zeros((6, 2))
        G[[0, 2, 4], 0] = g
        G[[1, 3, 5], 1] = g
        return G @ G.T * self.sigma_j**2

    def pv(self, x):
        return x[:4].copy()

    def pv_jacobian(self, x):
        J = np.zeros((4, 6))
        J[:, :4] = np.eye(4)
        return J

    def initial_state(self, pos, vel, pos_var, vel_var):
        x = np.array([pos[0], pos[1], vel[0], vel[1], 0.0, 0.0])
        a_var = self.init_acc_std**2
        P = np.diag([pos_var[0], pos_var[1], vel_var, vel_var, a_var, a_var])
        return x, P


class CTRV:
    """Constant Turn Rate and Velocity. x = [px, py, v, psi, omega].

    v is the speed along the heading psi, omega the yaw rate. The model is
    nonlinear, so the prediction needs an EKF (jacobian) or a UKF (sigma points).
    """

    name = "CTRV"
    labels = ["px", "py", "v", "psi", "omega"]
    units = ["m", "m", "m/s", "rad", "rad/s"]
    angle_idx = [3]
    linear = False

    # Below this yaw rate the turn is treated as a straight line (avoids 0 / 0).
    OMEGA_EPS = 1e-4

    def __init__(self, sigma_a=1.0, sigma_yawacc=0.5, init_omega_std=0.3):
        self.sigma_a = sigma_a              # std of the longitudinal acceleration, m/s^2
        self.sigma_yawacc = sigma_yawacc    # std of the yaw acceleration, rad/s^2
        self.init_omega_std = init_omega_std

    @property
    def dim(self):
        return len(self.labels)

    def f(self, x, dt):
        px, py, v, psi, omega = x
        if abs(omega) > self.OMEGA_EPS:
            px += v / omega * (np.sin(psi + omega * dt) - np.sin(psi))
            py += v / omega * (np.cos(psi) - np.cos(psi + omega * dt))
        else:
            px += v * np.cos(psi) * dt
            py += v * np.sin(psi) * dt
        psi = wrap(psi + omega * dt)
        return np.array([px, py, v, psi, omega])

    def F(self, x, dt):
        _, _, v, psi, omega = x
        F = np.eye(5)
        F[3, 4] = dt
        if abs(omega) > self.OMEGA_EPS:
            s0, c0 = np.sin(psi), np.cos(psi)
            s1, c1 = np.sin(psi + omega * dt), np.cos(psi + omega * dt)
            F[0, 2] = (s1 - s0) / omega
            F[0, 3] = v / omega * (c1 - c0)
            F[0, 4] = v * dt * c1 / omega - v / omega**2 * (s1 - s0)
            F[1, 2] = (c0 - c1) / omega
            F[1, 3] = v / omega * (s1 - s0)
            F[1, 4] = v * dt * s1 / omega - v / omega**2 * (c0 - c1)
        else:
            s0, c0 = np.sin(psi), np.cos(psi)
            F[0, 2] = c0 * dt
            F[0, 3] = -v * s0 * dt
            F[0, 4] = -0.5 * v * s0 * dt**2
            F[1, 2] = s0 * dt
            F[1, 3] = v * c0 * dt
            F[1, 4] = 0.5 * v * c0 * dt**2
        return F

    def Q(self, x, dt):
        # Noise inputs: longitudinal acceleration and yaw acceleration.
        psi = x[3]
        G = np.array([[0.5 * dt**2 * np.cos(psi), 0.0],
                      [0.5 * dt**2 * np.sin(psi), 0.0],
                      [dt, 0.0],
                      [0.0, 0.5 * dt**2],
                      [0.0, dt]])
        noise = np.diag([self.sigma_a**2, self.sigma_yawacc**2])
        return G @ noise @ G.T

    def pv(self, x):
        px, py, v, psi, _ = x
        return np.array([px, py, v * np.cos(psi), v * np.sin(psi)])

    def pv_jacobian(self, x):
        _, _, v, psi, _ = x
        J = np.zeros((4, 5))
        J[0, 0] = J[1, 1] = 1.0
        J[2, 2], J[2, 3] = np.cos(psi), -v * np.sin(psi)
        J[3, 2], J[3, 3] = np.sin(psi), v * np.cos(psi)
        return J

    def initial_state(self, pos, vel, pos_var, vel_var):
        speed = float(np.hypot(vel[0], vel[1]))
        psi = float(np.arctan2(vel[1], vel[0])) if speed > 0.5 else 0.0
        # Heading is unknown until the target has moved: start with a wide spread.
        psi_var = (np.pi / 2) ** 2 if speed > 0.5 else np.pi**2
        x = np.array([pos[0], pos[1], speed, psi, 0.0])
        P = np.diag([pos_var[0], pos_var[1], vel_var, psi_var, self.init_omega_std**2])
        return x, P


MODELS = {"CV": ConstantVelocity, "CA": ConstantAcceleration, "CTRV": CTRV}


def heading_of(model, x):
    """Heading in rad for display. CV and CA derive it from the velocity."""
    if model.name == "CTRV":
        return float(x[3])
    return float(np.arctan2(x[3], x[2]))


def speed_of(model, x):
    _, _, vx, vy = model.pv(x)
    return float(np.hypot(vx, vy))
