"""Extended Kalman filter: linearize f and h around the current estimate.

Same equations as the linear KF, with F and H replaced by jacobians and the
nonlinear f and h used for the mean.
"""

import numpy as np

from .angles import wrap


def predict(x, P, model, dt):
    F = model.F(x, dt)
    x = model.f(x, dt)
    P = F @ P @ F.T + model.Q(x, dt)
    return x, P


def update(x, P, z, meas):
    H = meas.H(x)
    y = meas.residual(z, meas.h(x))
    S = H @ P @ H.T + meas.R
    K = P @ H.T @ np.linalg.inv(S)
    x = x + K @ y
    for i in meas.model.angle_idx:
        x[i] = wrap(x[i])
    I_KH = np.eye(len(x)) - K @ H
    P = I_KH @ P @ I_KH.T + K @ meas.R @ K.T
    return x, P, y, S
