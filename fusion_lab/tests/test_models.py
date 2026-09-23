import numpy as np
import pytest

from kflab.angles import wrap
from kflab.measurement import CameraBearing, RadarPolar
from kflab.motion import CTRV, ConstantAcceleration, ConstantVelocity


def numeric_jacobian(fn, x, eps=1e-6, angle_rows=()):
    y0 = fn(x)
    J = np.zeros((len(y0), len(x)))
    for i in range(len(x)):
        dx = np.zeros(len(x))
        dx[i] = eps
        d = fn(x + dx) - fn(x - dx)
        for r in angle_rows:
            d[r] = wrap(d[r])
        J[:, i] = d / (2 * eps)
    return J


@pytest.mark.parametrize("omega", [0.3, -0.5, 1e-3])
def test_ctrv_jacobian(omega):
    m = CTRV()
    x = np.array([3.0, -2.0, 7.0, 0.4, omega])
    np.testing.assert_allclose(m.F(x, 0.3), numeric_jacobian(lambda s: m.f(s, 0.3), x, angle_rows=[3]),
                               atol=1e-5)


def test_ctrv_jacobian_is_continuous_at_zero_turn_rate():
    m = CTRV()
    x0 = np.array([3.0, -2.0, 7.0, 0.4, 0.0])
    x1 = x0 + np.array([0, 0, 0, 0, 2 * m.OMEGA_EPS])
    np.testing.assert_allclose(m.F(x0, 0.3), m.F(x1, 0.3), atol=1e-3)


def test_ctrv_straight_line_matches_cv():
    ctrv, cv = CTRV(), ConstantVelocity()
    x = ctrv.f(np.array([0.0, 0.0, 10.0, np.pi / 4, 0.0]), 2.0)
    y = cv.f(np.array([0.0, 0.0, 10 * np.cos(np.pi / 4), 10 * np.sin(np.pi / 4)]), 2.0)
    np.testing.assert_allclose(x[:2], y[:2])


def test_ctrv_full_circle_comes_back():
    m = CTRV()
    x = np.array([1.0, 2.0, 5.0, 0.3, 2 * np.pi / 10])
    np.testing.assert_allclose(m.f(x, 10.0)[:2], x[:2], atol=1e-9)


@pytest.mark.parametrize("model", [ConstantVelocity(), ConstantAcceleration(),
                                   CTRV()])
def test_radar_jacobian(model):
    x = {4: np.array([20.0, 5.0, -3.0, 2.0]),
         6: np.array([20.0, 5.0, -3.0, 2.0, 0.5, 0.1]),
         5: np.array([20.0, 5.0, 4.0, 2.5, 0.2])}[model.dim]
    meas = RadarPolar(model, (1.0, -0.5), 0.3, 1, 1, 1)
    np.testing.assert_allclose(meas.H(x), numeric_jacobian(meas.h, x, angle_rows=[1]), atol=1e-6)


def test_camera_jacobian():
    m = ConstantVelocity()
    meas = CameraBearing(m, (1.0, 2.0), -2.5, 0.01)
    x = np.array([-10.0, -3.0, 1.0, 0.0])
    np.testing.assert_allclose(meas.H(x), numeric_jacobian(meas.h, x, angle_rows=[0]), atol=1e-6)


def test_radar_bearing_residual_wraps():
    meas = RadarPolar(ConstantVelocity(), (0, 0), 0.0, 1, 1, 1)
    y = meas.residual(np.array([10, np.pi - 0.01, 0]), np.array([10, -np.pi + 0.01, 0]))
    assert abs(y[1] + 0.02) < 1e-9


def test_radar_inverse():
    m = ConstantVelocity()
    meas = RadarPolar(m, (2.0, 1.0), 0.7, 1, 1, 1)
    x = np.array([30.0, 12.0, -4.0, 1.0])
    pos, vel = meas.to_position_velocity(meas.h(x))
    np.testing.assert_allclose(pos, x[:2])
    # Only the radial part of the velocity is observed.
    u = (x[:2] - [2.0, 1.0]) / np.linalg.norm(x[:2] - [2.0, 1.0])
    np.testing.assert_allclose(vel, (x[2:] @ u) * u)


def test_process_noise_is_symmetric_positive():
    for m, x in [(ConstantVelocity(), np.zeros(4)), (ConstantAcceleration(), np.zeros(6)),
                 (CTRV(), np.array([0, 0, 5, 1.0, 0.1]))]:
        Q = m.Q(x, 0.1)
        np.testing.assert_allclose(Q, Q.T)
        assert np.all(np.linalg.eigvalsh(Q) > -1e-12)
