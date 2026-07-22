"""Log the running EKF to an MCAP file for Foxglove.

Why this matters for the course: a baked-in ``.mp4`` shows *where* the estimate
went, but not *why*. Foxglove lets students scrub the timeline and watch the
covariance ellipse breathe -- growing while the filter dead-reckons, snapping
smaller the instant a measurement lands. That is the intuition the whole course
is trying to build, made tangible.

Open the resulting ``.mcap`` in the Foxglove app (desktop or app.foxglove.dev,
which also runs straight from a Colab download) and load ``layouts/ekf_fusion.json``.

Topics logged
-------------
``/scene``            3D: ground-truth path, estimate path, covariance sphere,
                      and the latest lidar/radar measurement.
``/state``            estimated [px, py, vx, vy] (Plot panel).
``/error``            per-axis position/velocity error vs. ground truth.
``/nis``              Normalized Innovation Squared + its 95% chi-square bound.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

import foxglove
from foxglove.messages import (
    Color,
    LinePrimitive,
    LinePrimitiveLineType,
    Point3,
    Pose,
    SceneEntity,
    SceneUpdate,
    SpherePrimitive,
    Vector3,
)

from . import metrics

_GT_COLOR = Color(r=0.2, g=0.8, b=0.2, a=1.0)      # green  = truth
_EST_COLOR = Color(r=0.2, g=0.5, b=1.0, a=1.0)     # blue   = estimate
_COV_COLOR = Color(r=0.2, g=0.5, b=1.0, a=0.25)    # translucent uncertainty
_LIDAR_COLOR = Color(r=1.0, g=1.0, b=0.2, a=1.0)   # yellow = lidar
_RADAR_COLOR = Color(r=1.0, g=0.3, b=0.3, a=1.0)   # red    = radar


class FoxgloveLogger:
    """Accumulates paths and writes 3D + plot topics to an MCAP file."""

    def __init__(self, path: str | Path, frame_id: str = "map"):
        self.path = str(path)
        self.frame_id = frame_id
        self._writer = foxglove.open_mcap(self.path, allow_overwrite=True)
        self._gt_path: list[Point3] = []
        self._est_path: list[Point3] = []

    # -- one filter step ----------------------------------------------------
    def log_step(
        self,
        t: float,
        x: np.ndarray,
        P: np.ndarray,
        gt: np.ndarray,
        sensor: str,
        z_xy: np.ndarray,
        nis: float | None,
    ) -> None:
        log_time = int(t * 1e9)  # foxglove wants nanoseconds

        self._gt_path.append(Point3(x=float(gt[0]), y=float(gt[1]), z=0.0))
        self._est_path.append(Point3(x=float(x[0]), y=float(x[1]), z=0.0))

        # 1-sigma position uncertainty -> a translucent sphere the eye can read.
        sigma = float(np.sqrt(max(P[0, 0] + P[1, 1], 1e-9)))
        cov_sphere = SpherePrimitive(
            pose=Pose(position=Vector3(x=float(x[0]), y=float(x[1]), z=0.0)),
            size=Vector3(x=2 * sigma, y=2 * sigma, z=0.05),
            color=_COV_COLOR,
        )
        meas_color = _LIDAR_COLOR if sensor == "lidar" else _RADAR_COLOR
        meas_sphere = SpherePrimitive(
            pose=Pose(position=Vector3(x=float(z_xy[0]), y=float(z_xy[1]), z=0.0)),
            size=Vector3(x=0.4, y=0.4, z=0.4),
            color=meas_color,
        )

        entity = SceneEntity(
            frame_id=self.frame_id,
            id="ekf",
            lines=[
                LinePrimitive(
                    type=LinePrimitiveLineType.LineStrip,
                    thickness=2.0,
                    scale_invariant=True,
                    points=list(self._gt_path),
                    color=_GT_COLOR,
                ),
                LinePrimitive(
                    type=LinePrimitiveLineType.LineStrip,
                    thickness=2.0,
                    scale_invariant=True,
                    points=list(self._est_path),
                    color=_EST_COLOR,
                ),
            ],
            spheres=[cov_sphere, meas_sphere],
        )
        foxglove.log("/scene", SceneUpdate(entities=[entity]), log_time=log_time)

        # Plot topics (schemaless JSON -> Foxglove Plot/State panels).
        foxglove.log(
            "/state",
            {"px": float(x[0]), "py": float(x[1]), "vx": float(x[2]), "vy": float(x[3])},
            log_time=log_time,
        )
        foxglove.log(
            "/error",
            {
                "pos_err": float(np.hypot(x[0] - gt[0], x[1] - gt[1])),
                "vel_err": float(np.hypot(x[2] - gt[2], x[3] - gt[3])),
            },
            log_time=log_time,
        )
        if nis is not None:
            dof = 2 if sensor == "lidar" else 3
            foxglove.log(
                "/nis",
                {"sensor": sensor, "nis": float(nis), "chi2_95": metrics.CHI2_95[dof]},
                log_time=log_time,
            )

    def close(self) -> None:
        self._writer.close()

    def __enter__(self) -> "FoxgloveLogger":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
