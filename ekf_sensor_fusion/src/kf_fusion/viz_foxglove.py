"""Log the running EKF on a real nuScenes track to MCAP for Foxglove.

This view is built to answer "what does the filter actually give me?". On every
frame it draws, in the global map frame:

* the **real lidar points** on the object (white)  -- proof this is real data;
* the incoming **measurement** (yellow = lidar, red = radar);
* the filter's **estimate** (blue) and its **1-sigma covariance** (breathing sphere);
* the estimated **velocity arrow** -- the thing no single measurement gave us;
* a **1-second-ahead prediction** (orange) -- where the filter says it's going;
* **ground truth** (green) for reference.

Open the ``.mcap`` in Foxglove (desktop or app.foxglove.dev) and load
``layouts/ekf_fusion.json``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

import foxglove
from foxglove.messages import (
    Color,
    CubePrimitive,
    LinePrimitive,
    LinePrimitiveLineType,
    Point3,
    Pose,
    Quaternion,
    SceneEntity,
    SceneUpdate,
    SpherePrimitive,
    Vector3,
)

from . import metrics
from .dataset import Measurement, Track

_GT = Color(r=0.2, g=0.85, b=0.2, a=1.0)
_EST = Color(r=0.2, g=0.55, b=1.0, a=1.0)
_COV = Color(r=0.2, g=0.55, b=1.0, a=0.20)
_BOX = Color(r=0.2, g=0.55, b=1.0, a=0.15)
_VEL = Color(r=1.0, g=0.6, b=0.0, a=1.0)
_PRED = Color(r=1.0, g=0.5, b=0.0, a=0.9)
_LIDAR = Color(r=1.0, g=1.0, b=0.3, a=1.0)
_RADAR = Color(r=1.0, g=0.3, b=0.3, a=1.0)
_PTS = Color(r=0.9, g=0.9, b=0.9, a=0.9)


def _p3(x, y, z=0.0):
    return Point3(x=float(x), y=float(y), z=float(z))


class FoxgloveLogger:
    def __init__(self, path: str | Path, track: Track, frame_id: str = "map"):
        self.path = str(path)
        self.frame_id = frame_id
        self.track = track
        self._writer = foxglove.open_mcap(self.path, allow_overwrite=True)
        self._gt_path: list[Point3] = []
        self._est_path: list[Point3] = []
        w, l, h = track.box_size
        self._box_size = Vector3(x=float(l), y=float(w), z=float(h))  # nuScenes: w,l,h
        # index the object lidar points by time for quick lookup
        self._pts_time = list(track.lidar_pts_time)
        self._pts = list(track.lidar_pts)

    def _meas_xy(self, m: Measurement) -> np.ndarray:
        if m.sensor == "lidar":
            return m.z[:2]
        rho, phi, _ = m.z
        sx, sy = m.sensor_pos
        return np.array([sx + rho * np.cos(phi), sy + rho * np.sin(phi)])

    def _nearest_pts(self, t: float, cx: float, cy: float):
        if not self._pts_time:
            return []
        i = int(np.argmin(np.abs(np.array(self._pts_time) - t)))
        if abs(self._pts_time[i] - t) > 0.15:
            return []
        pts = self._pts[i]  # (2, k) object-relative
        return [(cx + pts[0, j], cy + pts[1, j]) for j in range(pts.shape[1])]

    def log_step(self, m: Measurement, x: np.ndarray, P: np.ndarray, nis) -> None:
        log_time = int(m.timestamp * 1e9)
        gt = m.gt
        self._gt_path.append(_p3(gt[0], gt[1]))
        self._est_path.append(_p3(x[0], x[1]))

        sigma = float(np.sqrt(max(P[0, 0] + P[1, 1], 1e-9)))
        mx = self._meas_xy(m)
        speed = float(np.hypot(x[2], x[3]))
        pred = x[:2] + x[2:4] * 1.0  # 1 s ahead

        spheres = [
            SpherePrimitive(pose=Pose(position=Vector3(x=float(x[0]), y=float(x[1]), z=0.0)),
                            size=Vector3(x=2 * sigma, y=2 * sigma, z=0.1), color=_COV),
            SpherePrimitive(pose=Pose(position=Vector3(x=float(mx[0]), y=float(mx[1]), z=0.0)),
                            size=Vector3(x=0.6, y=0.6, z=0.6),
                            color=_LIDAR if m.sensor == "lidar" else _RADAR),
            SpherePrimitive(pose=Pose(position=Vector3(x=float(pred[0]), y=float(pred[1]), z=0.0)),
                            size=Vector3(x=0.7, y=0.7, z=0.7), color=_PRED),
        ]
        # real lidar points on the object
        for (px, py) in self._nearest_pts(m.timestamp, gt[0], gt[1]):
            spheres.append(SpherePrimitive(
                pose=Pose(position=Vector3(x=float(px), y=float(py), z=0.0)),
                size=Vector3(x=0.18, y=0.18, z=0.18), color=_PTS))

        lines = [
            LinePrimitive(type=LinePrimitiveLineType.LineStrip, thickness=2.0,
                          scale_invariant=True, points=list(self._gt_path), color=_GT),
            LinePrimitive(type=LinePrimitiveLineType.LineStrip, thickness=2.0,
                          scale_invariant=True, points=list(self._est_path), color=_EST),
            # velocity arrow (estimate -> estimate + v)
            LinePrimitive(type=LinePrimitiveLineType.LineStrip, thickness=4.0,
                          scale_invariant=True,
                          points=[_p3(x[0], x[1]), _p3(x[0] + x[2], x[1] + x[3])], color=_VEL),
            # line to the 1 s prediction
            LinePrimitive(type=LinePrimitiveLineType.LineStrip, thickness=1.0,
                          scale_invariant=True,
                          points=[_p3(x[0], x[1]), _p3(pred[0], pred[1])], color=_PRED),
        ]
        box = CubePrimitive(
            pose=Pose(position=Vector3(x=float(x[0]), y=float(x[1]), z=0.0),
                      orientation=_yaw_quat(np.arctan2(x[3], x[2]))),
            size=self._box_size, color=_BOX)

        entity = SceneEntity(frame_id=self.frame_id, id="ekf",
                             spheres=spheres, lines=lines, cubes=[box])
        foxglove.log("/scene", SceneUpdate(entities=[entity]), log_time=log_time)

        foxglove.log("/state",
                     {"px": float(x[0]), "py": float(x[1]),
                      "vx": float(x[2]), "vy": float(x[3]), "speed": speed},
                     log_time=log_time)
        pos_err = float(np.hypot(x[0] - gt[0], x[1] - gt[1]))
        rec = {"pos_err": pos_err}
        if not np.isnan(gt[2]):
            rec["vel_err"] = float(np.hypot(x[2] - gt[2], x[3] - gt[3]))
        foxglove.log("/error", rec, log_time=log_time)
        if nis is not None:
            dof = 2 if m.sensor == "lidar" else 3
            foxglove.log("/nis",
                         {"sensor": m.sensor, "nis": float(nis), "chi2_95": metrics.CHI2_95[dof]},
                         log_time=log_time)

    def close(self) -> None:
        self._writer.close()

    def __enter__(self) -> "FoxgloveLogger":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


def _yaw_quat(yaw: float) -> "Quaternion":
    return Quaternion(x=0.0, y=0.0, z=float(np.sin(yaw / 2)), w=float(np.cos(yaw / 2)))
