"""Compare the track with ground truth at the annotated keyframes (2 Hz).

NEES (normalized estimation error squared) on position: e^T P^-1 e with e the
position error and P the 2x2 position covariance. If the filter is consistent,
NEES follows a chi-square law with 2 degrees of freedom: about 95 % of the values
stay below 5.99. Mostly above: the filter is overconfident (Q or R too small).
Mostly near zero: it is too cautious (Q or R too large).
"""

from dataclasses import dataclass

import numpy as np

CHI2_95 = {1: 3.841, 2: 5.991, 3: 7.815}


@dataclass
class Evaluation:
    t: np.ndarray
    error: np.ndarray       # position error, m
    nees: np.ndarray
    rmse: float
    mean_nees: float
    inside_95: float        # fraction of NEES values below the 95 % bound


def evaluate(scene, track):
    ts, errors, nees = [], [], []
    for t, gt_xy in zip(scene.gt["t"], scene.gt["xy"]):
        x, P, _ = track.state_at(t)
        if x is None:
            continue
        e = x[:2] - gt_xy
        ts.append(t)
        errors.append(float(np.hypot(*e)))
        nees.append(float(e @ np.linalg.solve(P[:2, :2], e)))
    ts, errors, nees = np.array(ts), np.array(errors), np.array(nees)
    if len(ts) == 0:
        return Evaluation(ts, errors, nees, float("nan"), float("nan"), float("nan"))
    return Evaluation(ts, errors, nees, float(np.sqrt(np.mean(errors**2))), float(np.mean(nees)),
                      float(np.mean(nees < CHI2_95[2])))
