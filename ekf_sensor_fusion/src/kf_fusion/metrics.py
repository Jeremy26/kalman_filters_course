"""Evaluation metrics -- how a *great* filter course proves it works.

Two things separate an engineered filter from a toy one:

1. **Accuracy** -- RMSE of the estimate against ground truth.
2. **Consistency** -- is the filter's own uncertainty honest? A filter can be
   accurate on average yet wildly over- or under-confident. The Normalized
   Innovation Squared (NIS) follows a chi-square distribution when the filter
   is consistent, so we check what fraction of updates fall under the 95%
   chi-square bound (should be ~95%).
"""

from __future__ import annotations

import numpy as np

# 95th-percentile chi-square thresholds by degrees of freedom (dof = meas. dim).
CHI2_95 = {1: 3.841, 2: 5.991, 3: 7.815, 4: 9.488}


def rmse(estimates: np.ndarray, truth: np.ndarray) -> np.ndarray:
    """Per-component RMSE for [px, py, vx, vy]."""
    estimates = np.asarray(estimates)
    truth = np.asarray(truth)
    return np.sqrt(np.mean((estimates - truth) ** 2, axis=0))


def nis_consistency(nis_values: np.ndarray, dof: int) -> float:
    """Fraction of NIS values below the 95% chi-square bound (target ~0.95)."""
    nis_values = np.asarray(nis_values)
    if nis_values.size == 0:
        return float("nan")
    bound = CHI2_95[dof]
    return float(np.mean(nis_values <= bound))


def summarize(estimates, truth, nis_lidar, nis_radar) -> dict:
    e = rmse(estimates, truth)
    return {
        "rmse_px": e[0],
        "rmse_py": e[1],
        "rmse_vx": e[2],
        "rmse_vy": e[3],
        "rmse_pos": float(np.hypot(e[0], e[1])),
        "nis_lidar_below_95": nis_consistency(nis_lidar, dof=2),
        "nis_radar_below_95": nis_consistency(nis_radar, dof=3),
    }
