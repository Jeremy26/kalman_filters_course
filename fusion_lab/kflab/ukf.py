"""Unscented Kalman filter: push a few chosen points (sigma points) through the
nonlinear f or h, then fit a Gaussian to where they land. No jacobian needed.

Sigma points follow van der Merwe's scaled form. With ALPHA = 1 and KAPPA = 0 the
central point has zero weight for the mean and every other point is placed at
sqrt(n) standard deviations; all covariance weights stay positive.
"""

import numpy as np

from .angles import circular_mean, wrap

ALPHA = 1.0
BETA = 2.0
KAPPA = 0.0


def weights(n):
    lam = ALPHA**2 * (n + KAPPA) - n
    Wm = np.full(2 * n + 1, 1.0 / (2.0 * (n + lam)))
    Wc = Wm.copy()
    Wm[0] = lam / (n + lam)
    Wc[0] = lam / (n + lam) + (1.0 - ALPHA**2 + BETA)
    return lam, Wm, Wc


def sigma_points(x, P):
    n = len(x)
    lam, _, _ = weights(n)
    L = np.linalg.cholesky((n + lam) * P)
    pts = np.empty((2 * n + 1, n))
    pts[0] = x
    for i in range(n):
        pts[1 + i] = x + L[:, i]
        pts[1 + n + i] = x - L[:, i]
    return pts


def _mean(points, Wm, angle_idx):
    m = Wm @ points
    for i in angle_idx:
        m[i] = circular_mean(points[:, i], Wm)
    return m


def _diff(points, mean, angle_idx):
    d = points - mean
    for i in angle_idx:
        d[:, i] = wrap(d[:, i])
    return d


def predict(x, P, model, dt):
    n = len(x)
    _, Wm, Wc = weights(n)
    pts = np.array([model.f(p, dt) for p in sigma_points(x, P)])
    x_pred = _mean(pts, Wm, model.angle_idx)
    d = _diff(pts, x_pred, model.angle_idx)
    P_pred = d.T @ np.diag(Wc) @ d + model.Q(x_pred, dt)
    return x_pred, P_pred


def update(x, P, z, meas):
    n = len(x)
    _, Wm, Wc = weights(n)
    X = sigma_points(x, P)
    Z = np.array([meas.h(p) for p in X])
    zp = _mean(Z, Wm, meas.angle_idx)
    dz = _diff(Z, zp, meas.angle_idx)
    dx = _diff(X, x, meas.model.angle_idx)
    S = dz.T @ np.diag(Wc) @ dz + meas.R
    Pxz = dx.T @ np.diag(Wc) @ dz
    K = Pxz @ np.linalg.inv(S)
    y = meas.residual(z, zp)
    x = x + K @ y
    for i in meas.model.angle_idx:
        x[i] = wrap(x[i])
    P = P - K @ S @ K.T
    P = 0.5 * (P + P.T)
    return x, P, y, S
