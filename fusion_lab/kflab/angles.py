"""Angle helpers. Angles are in radians, always kept in [-pi, pi)."""

import numpy as np


def wrap(a):
    """Wrap an angle (or an array of angles) to [-pi, pi)."""
    return (np.asarray(a) + np.pi) % (2.0 * np.pi) - np.pi


def circular_mean(angles, weights):
    """Weighted mean of angles. A plain average of 179 deg and -179 deg gives 0 deg."""
    return float(np.arctan2(np.sum(weights * np.sin(angles)), np.sum(weights * np.cos(angles))))
