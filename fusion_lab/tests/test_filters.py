import numpy as np

from kflab import ekf, kf, ukf
from kflab.measurement import LidarPosition, RadarPolar
from kflab.motion import CTRV, ConstantVelocity


def test_kf_scalar_by_hand():
    # One position, one measurement: K = P / (P + R).
    x, P = np.array([0.0]), np.array([[4.0]])
    x, P, y, S = kf.update(x, P, np.array([2.0]), np.array([[1.0]]), np.array([[1.0]]))
    assert np.isclose(x[0], 2.0 * 4 / 5)
    assert np.isclose(P[0, 0], 4 * 1 / 5)
    assert np.isclose(S[0, 0], 5.0)


def test_kf_predict_cv():
    m = ConstantVelocity(sigma_a=0.0)
    x, P = kf.predict(np.array([0, 0, 1.0, 2.0]), np.eye(4), m.F(None, 2.0), m.Q(None, 2.0))
    np.testing.assert_allclose(x, [2, 4, 1, 2])
    assert np.isclose(P[0, 0], 1 + 4)  # var(px) + dt^2 var(vx)


def test_ekf_equals_kf_on_linear_problem():
    m = ConstantVelocity(sigma_a=1.0)
    x, P = np.array([1.0, 2, 3, 4]), np.diag([1.0, 2, 3, 4])
    a = kf.predict(x, P, m.F(x, 0.5), m.Q(x, 0.5))
    b = ekf.predict(x, P, m, 0.5)
    np.testing.assert_allclose(a[0], b[0])
    np.testing.assert_allclose(a[1], b[1])
    meas = LidarPosition(m, 0.3)
    z = np.array([2.0, 2.5])
    a = kf.update(*b, z, meas.H(None), meas.R)
    c = ekf.update(*b, z, meas)
    np.testing.assert_allclose(a[0], c[0])
    np.testing.assert_allclose(a[1], c[1])


def test_ukf_equals_kf_on_linear_problem():
    m = ConstantVelocity(sigma_a=1.0)
    x, P = np.array([1.0, 2, 3, 4]), np.diag([1.0, 2, 3, 4])
    a = kf.predict(x, P, m.F(x, 0.5), m.Q(x, 0.5))
    b = ukf.predict(x, P, m, 0.5)
    np.testing.assert_allclose(a[0], b[0], atol=1e-9)
    np.testing.assert_allclose(a[1], b[1], atol=1e-9)
    meas = LidarPosition(m, 0.3)
    z = np.array([2.0, 2.5])
    a = kf.update(*a, z, meas.H(None), meas.R)
    c = ukf.update(*b, z, meas)
    np.testing.assert_allclose(a[0], c[0], atol=1e-9)
    np.testing.assert_allclose(a[1], c[1], atol=1e-9)


def test_ekf_and_ukf_agree_on_far_radar_target():
    # Far target, small uncertainty: the radar model is almost linear here.
    m = CTRV()
    x, P = np.array([60.0, 10, 8, 0.2, 0.05]), np.diag([0.1, 0.1, 0.2, 0.01, 0.001])
    meas = RadarPolar(m, (0.0, 0.0), 0.0, 0.5, 0.01, 0.3)
    z = meas.h(x) + np.array([0.3, 0.003, 0.2])
    a = ekf.update(x, P, z, meas)
    b = ukf.update(x, P, z, meas)
    # Not identical: range rate is v * cos(psi - bearing), nonlinear in v and psi.
    np.testing.assert_allclose(a[0], b[0], atol=0.03)
    np.testing.assert_allclose(a[1], b[1], atol=0.01)


def test_ukf_heading_across_pi():
    m = CTRV(sigma_a=0.1, sigma_yawacc=0.1)
    x, P = np.array([0.0, 0, 5, np.pi - 0.01, 0.0]), np.diag([0.1, 0.1, 0.1, 0.05, 0.01])
    xp, _ = ukf.predict(x, P, m, 0.1)
    assert abs(abs(xp[3]) - (np.pi - 0.01)) < 0.02
