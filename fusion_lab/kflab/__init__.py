"""kflab: Kalman filters for single-target multi-sensor tracking.

Reading order for students:
    motion.py       how the target moves (CV, CA, CTRV)
    measurement.py  what each sensor sees (LiDAR, RADAR, camera)
    kf.py           the linear Kalman filter
    ekf.py          the extended Kalman filter
    ukf.py          the unscented Kalman filter
    fusion.py       the event loop that fuses the sensors in time order
    metrics.py      error and NEES against ground truth
"""

from .fusion import Config, Track, run
from .metrics import evaluate
from .scene import available_scenes, load_scene

__all__ = ["Config", "Track", "run", "evaluate", "available_scenes", "load_scene"]
