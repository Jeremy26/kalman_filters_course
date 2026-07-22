"""Load a real extracted nuScenes track into a stream of measurements.

The ``.npz`` files under ``data/`` are produced by ``nuscenes_extract`` from the
real dataset. Each holds one tracked vehicle's genuine, asynchronous lidar and
radar measurements (plus ground truth, for scoring only). This loader turns that
into the ``Measurement`` stream the filter consumes -- no dataset required at
run time.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


@dataclass
class Measurement:
    sensor: str            # "lidar" or "radar"
    z: np.ndarray          # lidar: [px, py] ; radar: [rho, phi, rho_dot]
    timestamp: float       # seconds (relative to track start)
    sensor_pos: np.ndarray = field(default_factory=lambda: np.zeros(2))  # radar origin (global)
    gt: np.ndarray | None = None  # ground-truth [px, py, vx, vy] at this time

    def initial_state(self) -> np.ndarray:
        """Best first-guess state from this measurement (to seed the filter)."""
        if self.sensor == "lidar":
            return np.array([self.z[0], self.z[1], 0.0, 0.0])
        rho, phi, rho_dot = self.z
        sx, sy = self.sensor_pos
        px, py = sx + rho * np.cos(phi), sy + rho * np.sin(phi)
        # Radar range-rate is along the line of sight -> a rough velocity seed.
        vx, vy = rho_dot * np.cos(phi), rho_dot * np.sin(phi)
        return np.array([px, py, vx, vy])


@dataclass
class Track:
    """A loaded real track: the measurement stream plus ground truth for eval."""
    measurements: list[Measurement]
    gt_time: np.ndarray
    gt_state: np.ndarray            # (N, 4) [px, py, vx, vy]
    instance: str
    category: str
    box_size: np.ndarray            # w, l, h
    lidar_pts_time: np.ndarray
    lidar_pts: np.ndarray           # object array of (2, k) point sets (viz)

    def gt_at(self, t: float) -> np.ndarray:
        """Ground-truth state interpolated to time ``t`` (for scoring)."""
        return np.array([np.interp(t, self.gt_time, self.gt_state[:, i]) for i in range(4)])


def load_track(path: str | Path) -> Track:
    d = np.load(path, allow_pickle=True)
    gt_time = d["gt_time"]
    gt_state = d["gt_state"]
    measurements: list[Measurement] = []
    for t, s, z, extra in zip(d["ev_time"], d["ev_sensor"], d["ev_z"], d["ev_extra"]):
        sensor_pos = np.asarray(extra, dtype=float)[:2] if s == "radar" else np.zeros(2)
        gt = np.array([np.interp(t, gt_time, gt_state[:, i]) for i in range(4)])
        measurements.append(Measurement(str(s), np.asarray(z, dtype=float),
                                        float(t), sensor_pos, gt))
    return Track(
        measurements=measurements,
        gt_time=gt_time,
        gt_state=gt_state,
        instance=str(d["instance"]),
        category=str(d["category"]),
        box_size=d["box_size"],
        lidar_pts_time=d["lidar_pts_time"],
        lidar_pts=d["lidar_pts"],
    )
