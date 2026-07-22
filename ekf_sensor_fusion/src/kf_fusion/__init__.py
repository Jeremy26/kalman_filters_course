"""kf_fusion -- a single Extended Kalman Filter fused by lidar and radar.

The public surface is small on purpose so the teaching notebooks read cleanly::

    from kf_fusion import EKF, read_log, run_fusion

See ``run_ekf.py`` for the end-to-end pipeline.
"""

from .ekf import EKF
from .dataset import Measurement, Track, load_track
from .pipeline import run_fusion, FusionResult

__all__ = [
    "EKF",
    "Measurement",
    "Track",
    "load_track",
    "run_fusion",
    "FusionResult",
]
