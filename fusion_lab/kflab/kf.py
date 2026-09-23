"""Linear Kalman filter. Used for CV and CA predictions and for every LiDAR update."""

import numpy as np


def predict(x, P, F, Q):
    x = F @ x
    P = F @ P @ F.T + Q
    return x, P


def update(x, P, z, H, R, residual=None):
    """Returns the new (x, P) and the innovation y with its covariance S.

    residual(z, zp) replaces z - zp when the measurement has angles.
    """
    zp = H @ x
    y = residual(z, zp) if residual else z - zp
    S = H @ P @ H.T + R
    K = P @ H.T @ np.linalg.inv(S)
    x = x + K @ y
    # Joseph form: same result as (I - K H) P, but P stays symmetric and positive.
    I_KH = np.eye(len(x)) - K @ H
    P = I_KH @ P @ I_KH.T + K @ R @ K.T
    return x, P, y, S
