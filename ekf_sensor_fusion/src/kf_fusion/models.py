"""Motion and measurement models for the lidar + radar EKF.

The state we estimate everywhere in this project is the 2D constant-velocity
state of a single tracked vehicle::

    x = [px, py, vx, vy]^T

That single state is corrected by *both* sensors:

* a **lidar** measurement is linear in the state (it reports px, py directly),
* a **radar** measurement is *non-linear* in the state (range, bearing and
  range-rate in polar coordinates), which is exactly why we need the
  *Extended* Kalman Filter and its Jacobian.

Keeping both models next to each other is deliberate: the whole point of the
course module is that one filter, one state, many sensors -- the "big Kalman
filter" idea.
"""

from __future__ import annotations

import numpy as np

# --- Sensor noise, tuned to the REAL nuScenes measurements ---
# These are not textbook guesses: they come from measuring the extracted
# lidar-centroid and radar returns against ground truth (see the notebook's
# noise-analysis cell). The lidar variance is deliberately inflated because the
# in-box centroid is biased toward the vehicle's visible face by ~1 m -- a real
# effect we fold into R rather than pretend away.
LIDAR_R = np.diag([1.0, 1.0])          # (m^2) on px, py  (centroid, biased+noisy)
RADAR_R = np.diag([0.5, 0.01, 4.0])    # (rho[m^2], phi[rad^2], rho_dot[m^2/s^2])

# Process-noise acceleration (m/s^2)^2 -- how much we let velocity drift between
# updates. Tuned later in the consistency notebook via NIS; these are sane
# defaults for an urban vehicle.
DEFAULT_NOISE_AX = 9.0
DEFAULT_NOISE_AY = 9.0


def transition_matrix(dt: float) -> np.ndarray:
    """Constant-velocity state transition F for a time step ``dt``."""
    return np.array(
        [
            [1.0, 0.0, dt, 0.0],
            [0.0, 1.0, 0.0, dt],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )


def process_noise(dt: float, noise_ax: float = DEFAULT_NOISE_AX,
                  noise_ay: float = DEFAULT_NOISE_AY) -> np.ndarray:
    """Process-noise covariance Q from a white-acceleration model.

    Acceleration is modelled as zero-mean white noise; integrating it into the
    position/velocity state gives the familiar dt^4/dt^3/dt^2 structure.
    """
    dt2 = dt * dt
    dt3 = dt2 * dt
    dt4 = dt3 * dt
    return np.array(
        [
            [dt4 / 4 * noise_ax, 0.0, dt3 / 2 * noise_ax, 0.0],
            [0.0, dt4 / 4 * noise_ay, 0.0, dt3 / 2 * noise_ay],
            [dt3 / 2 * noise_ax, 0.0, dt2 * noise_ax, 0.0],
            [0.0, dt3 / 2 * noise_ay, 0.0, dt2 * noise_ay],
        ]
    )


# --- Lidar: linear measurement model ---------------------------------------
LIDAR_H = np.array(
    [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
    ]
)


def lidar_measurement(x: np.ndarray) -> np.ndarray:
    """Predicted lidar measurement h(x) = [px, py]."""
    return LIDAR_H @ x


# --- Radar: non-linear measurement model -----------------------------------
# The radar sits on the ego vehicle, which MOVES every frame, so its origin in
# the global tracking frame is a per-measurement input ``sensor`` = (sx, sy).
# The measurement is polar about that origin: range, bearing, range-rate.
def radar_measurement(x: np.ndarray, sensor: np.ndarray = None) -> np.ndarray:
    """Predicted radar measurement h(x) = [rho, phi, rho_dot] about ``sensor``."""
    sx, sy = (0.0, 0.0) if sensor is None else (sensor[0], sensor[1])
    px, py, vx, vy = x
    dx, dy = px - sx, py - sy
    rho = max(np.hypot(dx, dy), 1e-6)
    phi = np.arctan2(dy, dx)
    rho_dot = (dx * vx + dy * vy) / rho
    return np.array([rho, phi, rho_dot])


def radar_jacobian(x: np.ndarray, sensor: np.ndarray = None) -> np.ndarray:
    """Jacobian Hj of the radar model about ``sensor``, evaluated at ``x``.

    This linearisation of the polar measurement about the current estimate is
    the single idea that turns a plain KF into an EKF.
    """
    sx, sy = (0.0, 0.0) if sensor is None else (sensor[0], sensor[1])
    px, py, vx, vy = x
    dx, dy = px - sx, py - sy
    c1 = max(dx * dx + dy * dy, 1e-6)
    c2 = np.sqrt(c1)
    c3 = c1 * c2
    return np.array(
        [
            [dx / c2, dy / c2, 0.0, 0.0],
            [-dy / c1, dx / c1, 0.0, 0.0],
            [dy * (vx * dy - vy * dx) / c3, dx * (vy * dx - vx * dy) / c3, dx / c2, dy / c2],
        ]
    )


def normalize_angle(angle: float) -> float:
    """Wrap an angle to (-pi, pi]; essential for the radar bearing residual."""
    return (angle + np.pi) % (2 * np.pi) - np.pi
