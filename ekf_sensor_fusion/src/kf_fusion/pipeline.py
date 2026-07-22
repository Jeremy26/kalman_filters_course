"""End-to-end fusion pipeline: log in, filtered track + metrics (+ MCAP) out.

This is the function the notebook calls. It walks the measurement stream, and
for each measurement it:

    1. predicts the shared state forward by the elapsed dt, then
    2. corrects it with the matching sensor's update.

The first measurement seeds the state instead of updating it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .dataset import Measurement, read_log
from .ekf import EKF
from . import metrics


@dataclass
class FusionResult:
    estimates: np.ndarray                 # (N, 4) filtered states
    ground_truth: np.ndarray              # (N, 4)
    timestamps: np.ndarray                # (N,)
    nis_lidar: list[float] = field(default_factory=list)
    nis_radar: list[float] = field(default_factory=list)

    def summary(self) -> dict:
        return metrics.summarize(
            self.estimates, self.ground_truth, self.nis_lidar, self.nis_radar
        )


def run_fusion(
    measurements: list[Measurement] | str | Path,
    noise_ax: float = 9.0,
    noise_ay: float = 9.0,
    mcap_path: str | Path | None = None,
    use_lidar: bool = True,
    use_radar: bool = True,
) -> FusionResult:
    """Run the lidar+radar EKF over a measurement stream.

    Parameters
    ----------
    measurements : a list of Measurement, or a path to a fusion log.
    noise_ax, noise_ay : process-noise acceleration (tune these with NIS).
    mcap_path : if given, also write a Foxglove MCAP recording.
    use_lidar, use_radar : which sensors may *correct* the filter. A disabled
        sensor still advances time (predict-only) so every configuration is
        evaluated on the *same* timeline -- this is what makes the "lidar-only
        vs fused" ablation fair: lidar-only must coast through a lidar occlusion
        instead of silently skipping those frames.
    """
    if isinstance(measurements, (str, Path)):
        measurements = read_log(measurements)
    if not (use_lidar or use_radar):
        raise ValueError("At least one sensor must be enabled.")

    ekf = EKF(noise_ax=noise_ax, noise_ay=noise_ay)
    estimates, truth, times = [], [], []
    nis_lidar: list[float] = []
    nis_radar: list[float] = []

    logger = None
    if mcap_path is not None:
        from .viz_foxglove import FoxgloveLogger  # optional dependency

        logger = FoxgloveLogger(mcap_path)

    def allowed(sensor: str) -> bool:
        return use_lidar if sensor == "lidar" else use_radar

    last_t = None
    try:
        for m in measurements:
            if not ekf.initialized:
                if not allowed(m.sensor):
                    continue  # wait for a usable sensor to seed the state
                ekf.initialize(m.initial_state(), P0=np.diag([1.0, 1.0, 100.0, 100.0]))
                last_t = m.timestamp
                nis = None
            else:
                ekf.predict(m.timestamp - last_t)
                last_t = m.timestamp
                if not allowed(m.sensor):
                    nis = None  # sensor ablated: predict only, no correction
                elif m.sensor == "lidar":
                    nis = ekf.update_lidar(m.z)
                    nis_lidar.append(nis)
                else:
                    nis = ekf.update_radar(m.z)
                    nis_radar.append(nis)

            estimates.append(ekf.x.copy())
            truth.append(m.gt.copy())
            times.append(m.timestamp)

            if logger is not None:
                # Measurement position in Cartesian for the 3D view.
                z_xy = (
                    m.z[:2]
                    if m.sensor == "lidar"
                    else np.array([m.z[0] * np.cos(m.z[1]), m.z[0] * np.sin(m.z[1])])
                )
                logger.log_step(m.timestamp, ekf.x, ekf.P, m.gt, m.sensor, z_xy, nis)
    finally:
        if logger is not None:
            logger.close()

    return FusionResult(
        estimates=np.asarray(estimates),
        ground_truth=np.asarray(truth),
        timestamps=np.asarray(times),
        nis_lidar=nis_lidar,
        nis_radar=nis_radar,
    )
