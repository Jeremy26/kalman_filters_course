"""End-to-end fusion pipeline on a real nuScenes track.

For each real measurement we (1) predict the shared state forward by the elapsed
dt, then (2) correct it with the matching sensor -- lidar (linear) or radar
(non-linear, about the moving ego/radar origin). The first usable measurement
seeds the state.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .dataset import Measurement, Track, load_track
from .ekf import EKF
from . import metrics


@dataclass
class FusionResult:
    estimates: np.ndarray                 # (N, 4) filtered states
    ground_truth: np.ndarray              # (N, 4) GT at each event time
    timestamps: np.ndarray                # (N,)
    sensors: list[str] = field(default_factory=list)
    nis_lidar: list[float] = field(default_factory=list)
    nis_radar: list[float] = field(default_factory=list)

    def summary(self) -> dict:
        return metrics.summarize(
            self.estimates, self.ground_truth, self.nis_lidar, self.nis_radar
        )


def run_fusion(
    track: Track | str | Path,
    noise_ax: float = 4.0,
    noise_ay: float = 4.0,
    mcap_path: str | Path | None = None,
    use_lidar: bool = True,
    use_radar: bool = True,
) -> FusionResult:
    """Run the lidar+radar EKF over a real track.

    Parameters
    ----------
    track : a loaded :class:`Track`, or a path to an extracted ``.npz``.
    noise_ax, noise_ay : process-noise acceleration (tune with NIS).
    mcap_path : if given, also write a Foxglove MCAP recording.
    use_lidar, use_radar : which sensors may *correct* the filter. A disabled
        sensor still advances time (predict-only), so every configuration is
        scored on the same timeline -- the fair way to ask "what does radar add?"
    """
    if isinstance(track, (str, Path)):
        track = load_track(track)
    if not (use_lidar or use_radar):
        raise ValueError("At least one sensor must be enabled.")

    ekf = EKF(noise_ax=noise_ax, noise_ay=noise_ay)
    estimates, truth, times, sensors = [], [], [], []
    nis_lidar: list[float] = []
    nis_radar: list[float] = []

    logger = None
    if mcap_path is not None:
        from .viz_foxglove import FoxgloveLogger

        logger = FoxgloveLogger(mcap_path, track)

    def allowed(sensor: str) -> bool:
        return use_lidar if sensor == "lidar" else use_radar

    last_t = None
    try:
        for m in track.measurements:
            if not ekf.initialized:
                if not allowed(m.sensor):
                    continue
                ekf.initialize(m.initial_state(), P0=np.diag([2.0, 2.0, 100.0, 100.0]))
                last_t = m.timestamp
                nis = None
            else:
                ekf.predict(m.timestamp - last_t)
                last_t = m.timestamp
                if not allowed(m.sensor):
                    nis = None
                elif m.sensor == "lidar":
                    nis = ekf.update_lidar(m.z)
                    nis_lidar.append(nis)
                else:
                    nis = ekf.update_radar(m.z, sensor=m.sensor_pos)
                    nis_radar.append(nis)

            estimates.append(ekf.x.copy())
            truth.append(m.gt.copy())
            times.append(m.timestamp)
            sensors.append(m.sensor)

            if logger is not None:
                logger.log_step(m, ekf.x, ekf.P, nis)
    finally:
        if logger is not None:
            logger.close()

    return FusionResult(
        estimates=np.asarray(estimates),
        ground_truth=np.asarray(truth),
        timestamps=np.asarray(times),
        sensors=sensors,
        nis_lidar=nis_lidar,
        nis_radar=nis_radar,
    )
