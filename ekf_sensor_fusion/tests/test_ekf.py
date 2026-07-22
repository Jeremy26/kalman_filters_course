"""Tests for the fusion EKF.

These double as the course's "definition of done": the filter must track a
known trajectory accurately *and* stay statistically consistent. Students run
``pytest`` and get an objective pass/fail instead of eyeballing a plot.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

# Make ``kf_fusion`` importable when running from the repo without an install.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "data"))

from kf_fusion import EKF, run_fusion, read_log  # noqa: E402
from kf_fusion import models  # noqa: E402
import generate_dataset  # noqa: E402


@pytest.fixture(scope="module")
def dataset(tmp_path_factory):
    path = tmp_path_factory.mktemp("data") / "fusion_log.txt"
    lines = generate_dataset.simulate(n=500, dt=0.05, seed=26)
    path.write_text("\n".join(lines) + "\n")
    return path


def test_radar_jacobian_matches_numerical():
    """The hand-derived Jacobian must match a finite-difference approximation."""
    x = np.array([4.0, 3.0, 2.0, -1.0])
    analytic = models.radar_jacobian(x)
    numeric = np.zeros((3, 4))
    eps = 1e-6
    for j in range(4):
        dx = np.zeros(4)
        dx[j] = eps
        h_plus = models.radar_measurement(x + dx)
        h_minus = models.radar_measurement(x - dx)
        numeric[:, j] = (h_plus - h_minus) / (2 * eps)
    np.testing.assert_allclose(analytic, numeric, rtol=1e-4, atol=1e-5)


def test_covariance_stays_symmetric_and_psd(dataset):
    ekf = EKF()
    for m in read_log(dataset):
        if not ekf.initialized:
            ekf.initialize(m.initial_state())
            last = m.timestamp
            continue
        ekf.predict(m.timestamp - last)
        last = m.timestamp
        (ekf.update_lidar if m.sensor == "lidar" else ekf.update_radar)(m.z)
        np.testing.assert_allclose(ekf.P, ekf.P.T, atol=1e-8)
        assert np.all(np.linalg.eigvalsh(ekf.P) > -1e-8)


def test_fusion_is_accurate(dataset):
    result = run_fusion(dataset)
    s = result.summary()
    # Positions should be tracked to well under a metre on this trajectory.
    assert s["rmse_pos"] < 0.5
    assert s["rmse_vx"] < 1.5 and s["rmse_vy"] < 1.5


def test_fusion_beats_single_sensor(dataset):
    ms = read_log(dataset)
    fused = run_fusion(ms).summary()["rmse_pos"]
    lidar_only = run_fusion(ms, use_lidar=True, use_radar=False).summary()["rmse_pos"]
    radar_only = run_fusion(ms, use_lidar=False, use_radar=True).summary()["rmse_pos"]
    # The point of fusion, evaluated on a common timeline with a lidar occlusion:
    # it beats BOTH single-sensor configs -- lidar-only drifts while blocked,
    # radar-only is always noisy.
    assert fused < lidar_only
    assert fused < radar_only


def test_filter_is_consistent(dataset):
    result = run_fusion(dataset)
    s = result.summary()
    # A well-tuned filter keeps ~95% of innovations under the chi-square bound.
    assert 0.80 <= s["nis_lidar_below_95"] <= 1.0
    assert 0.80 <= s["nis_radar_below_95"] <= 1.0
