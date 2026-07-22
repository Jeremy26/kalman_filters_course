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

# --- Sensor noise (standard deviations reflect typical automotive sensors) ---
# Lidar is accurate in Cartesian position; radar is noisier in range/bearing
# but adds a direct velocity observation through range-rate.
LIDAR_R = np.diag([0.0225, 0.0225])  # (m^2) on px, py
RADAR_R = np.diag([0.09, 0.0009, 0.09])  # (rho[m^2], phi[rad^2], rho_dot[m^2/s^2])

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
def radar_measurement(x: np.ndarray) -> np.ndarray:
    """Predicted radar measurement h(x) = [rho, phi, rho_dot] (polar)."""
    px, py, vx, vy = x
    rho = np.hypot(px, py)
    phi = np.arctan2(py, px)
    # Guard the range-rate against a division by zero at the sensor origin.
    rho = max(rho, 1e-6)
    rho_dot = (px * vx + py * vy) / rho
    return np.array([rho, phi, rho_dot])


def radar_jacobian(x: np.ndarray) -> np.ndarray:
    """Jacobian Hj of the radar measurement model, evaluated at ``x``.

    This linearisation of the polar measurement about the current estimate is
    the single line that turns a plain KF into an EKF.
    """
    px, py, vx, vy = x
    c1 = px * px + py * py
    c1 = max(c1, 1e-6)
    c2 = np.sqrt(c1)
    c3 = c1 * c2
    return np.array(
        [
            [px / c2, py / c2, 0.0, 0.0],
            [-py / c1, px / c1, 0.0, 0.0],
            [py * (vx * py - vy * px) / c3, px * (vy * px - vx * py) / c3, px / c2, py / c2],
        ]
    )


def normalize_angle(angle: float) -> float:
    """Wrap an angle to (-pi, pi]; essential for the radar bearing residual."""
    return (angle + np.pi) % (2 * np.pi) - np.pi
