"""A single Extended Kalman Filter fused by lidar *and* radar.

This is intentionally written from scratch (no filterpy) so students can read
every line of the predict/update cycle. One filter instance holds one state and
one covariance; whichever sensor produces the next measurement calls the
matching ``update_*`` method. That is the "big filter" architecture: the sensors
do not each own a filter -- they share one.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import models


@dataclass
class EKF:
    """Extended Kalman Filter for a 2D constant-velocity target.

    Attributes
    ----------
    x : state estimate [px, py, vx, vy]
    P : state covariance (4x4)
    noise_ax, noise_ay : white-acceleration process noise (tunable)
    """

    x: np.ndarray = field(default_factory=lambda: np.zeros(4))
    P: np.ndarray = field(default_factory=lambda: np.eye(4))
    noise_ax: float = models.DEFAULT_NOISE_AX
    noise_ay: float = models.DEFAULT_NOISE_AY
    _initialized: bool = False

    # -- lifecycle ----------------------------------------------------------
    def initialize(self, x0: np.ndarray, P0: np.ndarray | None = None) -> None:
        self.x = np.asarray(x0, dtype=float).reshape(4)
        if P0 is not None:
            self.P = np.asarray(P0, dtype=float)
        self._initialized = True

    @property
    def initialized(self) -> bool:
        return self._initialized

    # -- predict ------------------------------------------------------------
    def predict(self, dt: float) -> None:
        """Advance the state to now using the constant-velocity model."""
        if dt <= 0:
            return
        F = models.transition_matrix(dt)
        Q = models.process_noise(dt, self.noise_ax, self.noise_ay)
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + Q

    # -- generic linear update ---------------------------------------------
    def _update(self, y: np.ndarray, H: np.ndarray, R: np.ndarray) -> float:
        """Shared Kalman correction. Returns the NIS (for consistency checks)."""
        S = H @ self.P @ H.T + R
        S_inv = np.linalg.inv(S)
        K = self.P @ H.T @ S_inv
        self.x = self.x + K @ y
        ImKH = np.eye(self.x.shape[0]) - K @ H
        # Joseph form keeps P symmetric positive-definite even with rounding.
        self.P = ImKH @ self.P @ ImKH.T + K @ R @ K.T
        return float(y.T @ S_inv @ y)

    # -- lidar (linear) update ---------------------------------------------
    def update_lidar(self, z: np.ndarray, R: np.ndarray | None = None) -> float:
        R = models.LIDAR_R if R is None else R
        H = models.LIDAR_H
        y = z - models.lidar_measurement(self.x)
        return self._update(y, H, R)

    # -- radar (non-linear) update -----------------------------------------
    def update_radar(self, z: np.ndarray, R: np.ndarray | None = None) -> float:
        R = models.RADAR_R if R is None else R
        H = models.radar_jacobian(self.x)
        y = z - models.radar_measurement(self.x)
        y[1] = models.normalize_angle(y[1])  # wrap the bearing residual
        return self._update(y, H, R)
