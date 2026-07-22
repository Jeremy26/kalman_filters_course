"""Tests for the fusion EKF, run on REAL committed nuScenes tracks.

These need no dataset download — they use the small extracted ``.npz`` tracks in
``data/``. They double as the course's "definition of done": the filter must
track real vehicles, fusion must beat a single sensor on the sparse-lidar case,
and the filter must stay statistically consistent.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from kf_fusion import EKF, load_track, run_fusion  # noqa: E402
from kf_fusion import models  # noqa: E402

DENSE = ROOT / "data" / "track_ed634e83.npz"      # 41 lidar + 41 radar
SPARSE = ROOT / "data" / "track_437fe13d.npz"     # 5 lidar + 22 radar (radar carries it)


def test_tracks_are_present():
    assert DENSE.exists() and SPARSE.exists(), "extracted real tracks must be committed"


def test_radar_jacobian_matches_numerical():
    """Hand-derived Jacobian must match finite differences, about a moving sensor."""
    x = np.array([1310.0, 1040.0, 6.0, -3.0])
    sensor = np.array([1300.0, 1035.0])
    analytic = models.radar_jacobian(x, sensor)
    numeric = np.zeros((3, 4))
    eps = 1e-4
    for j in range(4):
        dx = np.zeros(4); dx[j] = eps
        numeric[:, j] = (models.radar_measurement(x + dx, sensor)
                         - models.radar_measurement(x - dx, sensor)) / (2 * eps)
    np.testing.assert_allclose(analytic, numeric, rtol=1e-4, atol=1e-5)


def test_covariance_stays_symmetric_and_psd():
    track = load_track(DENSE)
    ekf = EKF()
    last = None
    for m in track.measurements:
        if not ekf.initialized:
            ekf.initialize(m.initial_state()); last = m.timestamp; continue
        ekf.predict(m.timestamp - last); last = m.timestamp
        if m.sensor == "lidar":
            ekf.update_lidar(m.z)
        else:
            ekf.update_radar(m.z, sensor=m.sensor_pos)
        np.testing.assert_allclose(ekf.P, ekf.P.T, atol=1e-6)
        assert np.all(np.linalg.eigvalsh(ekf.P) > -1e-6)


def test_fusion_tracks_real_vehicle():
    s = run_fusion(DENSE).summary()
    # Raw-return tracking of a real car: position within a couple of metres,
    # velocity within a few m/s. (Position is bounded by the ~1.5 m lidar bias.)
    assert s["rmse_pos"] < 2.5
    assert s["rmse_vel"] < 2.5


def test_fusion_beats_lidar_only_when_lidar_is_sparse():
    """The real 'why fuse' case: 5 lidar hits in 11 s -> lidar-only drifts."""
    fused = run_fusion(SPARSE).summary()["rmse_pos"]
    lidar_only = run_fusion(SPARSE, use_lidar=True, use_radar=False).summary()["rmse_pos"]
    assert lidar_only > 5.0          # lidar alone is lost
    assert fused < 3.0               # fusion stays locked on
    assert fused < lidar_only


def test_filter_is_consistent():
    s = run_fusion(DENSE).summary()
    # A reasonable (here slightly conservative) filter keeps most innovations
    # under the 95% chi-square bound.
    assert s["nis_lidar_below_95"] >= 0.85
    assert s["nis_radar_below_95"] >= 0.85
