"""Fuse asynchronous sensors into one track.

The loop is the whole idea of multi-sensor fusion with a Kalman filter:

    for each event, in timestamp order:
        predict the state from the last event time to this event's time
        update with the measurement of that sensor, using that sensor's model

The filter does not care which sensor comes next or at what rate. It only needs
the time gap to predict, and the right measurement model to update.
"""

from dataclasses import dataclass, field

import numpy as np

from . import ekf, kf, ukf
from .measurement import CameraBearing, LidarPosition, RadarPolar
from .motion import CTRV, ConstantAcceleration, ConstantVelocity


@dataclass
class Config:
    model: str = "CV"                   # "CV", "CA" or "CTRV"
    nonlinear: str = "EKF"              # "EKF" or "UKF": used for every nonlinear step
    use_lidar: bool = True
    use_radar: bool = True
    use_camera: bool = True
    camera_bearing_update: bool = False  # off: camera gives the class only

    # Process noise. sigma_a is the acceleration for CV and CTRV, the jerk for CA.
    sigma_a: float = 2.0
    sigma_yawacc: float = 0.5

    # Measurement noise (standard deviations). The RADAR values are large on purpose:
    # a radar cluster sits on the near face of the vehicle, not at its center, which on
    # a 12 m bus means metres of range error and several degrees of bearing error.
    lidar_std: float = 0.3                          # m
    radar_range_std: float = 3.5                    # m
    radar_bearing_std: float = np.deg2rad(8.0)      # rad
    radar_range_rate_std: float = 0.5               # m/s
    camera_bearing_std: float = np.deg2rad(1.0)     # rad

    init_vel_std: float = 5.0                       # m/s, before any velocity is observed

    def make_model(self):
        if self.model == "CV":
            return ConstantVelocity(sigma_a=self.sigma_a)
        if self.model == "CA":
            return ConstantAcceleration(sigma_j=self.sigma_a)
        if self.model == "CTRV":
            return CTRV(sigma_a=self.sigma_a, sigma_yawacc=self.sigma_yawacc)
        raise ValueError(self.model)


@dataclass
class Step:
    """The filter at one event time: just before (prior) and just after the update."""

    t: float
    sensor: str
    channel: str
    event_index: int
    x_prior: np.ndarray
    P_prior: np.ndarray
    x: np.ndarray
    P: np.ndarray
    z: np.ndarray
    y: np.ndarray | None = None  # innovation, None for the step that starts the track
    nis: float | None = None     # normalized innovation squared, y^T S^-1 y


@dataclass
class Track:
    config: Config
    model: object
    steps: list = field(default_factory=list)

    def last_step_before(self, t):
        """Index of the last step with step.t <= t, or None before the track starts."""
        i = None
        lo, hi = 0, len(self.steps)
        while lo < hi:
            mid = (lo + hi) // 2
            if self.steps[mid].t <= t:
                i, lo = mid, mid + 1
            else:
                hi = mid
        return i

    def state_at(self, t):
        """State predicted from the last update to time t. Nothing is changed in the track."""
        i = self.last_step_before(t)
        if i is None:
            return None, None, None
        s = self.steps[i]
        x, P = predict(s.x, s.P, self.model, t - s.t, self.config)
        return x, P, i


def predict(x, P, model, dt, config):
    if dt <= 0:
        return x.copy(), P.copy()
    if model.linear:
        return kf.predict(x, P, model.F(x, dt), model.Q(x, dt))
    return (ukf if config.nonlinear == "UKF" else ekf).predict(x, P, model, dt)


def measurement_for(event, model, config):
    """Build the measurement model and the measurement z for the gated detection.

    Returns (meas, z) or (None, None) when this event does not update the state.
    """
    det = event.target_detection
    if det is None:
        return None, None
    if event.sensor == "lidar" and config.use_lidar:
        return LidarPosition(model, config.lidar_std), np.array(det["xy"])
    if event.sensor == "radar" and config.use_radar:
        meas = RadarPolar(model, event.sensor_xy, event.sensor_yaw, config.radar_range_std,
                          config.radar_bearing_std, config.radar_range_rate_std)
        return meas, np.array(det["z"])
    if event.sensor == "camera" and config.use_camera and config.camera_bearing_update:
        meas = CameraBearing(model, event.sensor_xy, event.sensor_yaw, config.camera_bearing_std)
        return meas, np.array([det["bearing"]])
    return None, None


def update(x, P, z, meas, config):
    if meas.linear:
        return kf.update(x, P, z, meas.H(x), meas.R, meas.residual)
    return (ukf if config.nonlinear == "UKF" else ekf).update(x, P, z, meas)


def initialize(event, meas, z, model, config):
    """Start the track from the first LiDAR or RADAR detection of the target."""
    if isinstance(meas, LidarPosition):
        pos, vel = z, np.zeros(2)
        pos_var = (config.lidar_std**2, config.lidar_std**2)
    elif isinstance(meas, RadarPolar):
        pos, vel = meas.to_position_velocity(z)
        # Cross-range spread grows with range: r * sigma_bearing.
        var = max(config.radar_range_std**2, (z[0] * config.radar_bearing_std) ** 2)
        pos_var = (var, var)
    else:
        return None, None  # a bearing alone cannot place the target
    return model.initial_state(pos, vel, pos_var, config.init_vel_std**2)


def sensor_enabled(sensor, config):
    return {"lidar": config.use_lidar, "radar": config.use_radar, "camera": config.use_camera}[sensor]


def run(scene, config):
    """Run the filter over the whole scene. Returns a Track with one Step per update."""
    model = config.make_model()
    track = Track(config, model)
    x = P = None
    t_last = None

    for index, event in enumerate(scene.events):
        if not sensor_enabled(event.sensor, config):
            continue
        meas, z = measurement_for(event, model, config)
        if meas is None:
            # No gated detection, or a camera in class-only mode: nothing to fuse.
            continue

        if x is None:
            x, P = initialize(event, meas, z, model, config)
            if x is None:
                continue
            t_last = event.t
            track.steps.append(Step(event.t, event.sensor, event.channel, index,
                                    x.copy(), P.copy(), x.copy(), P.copy(), z))
            continue

        x_prior, P_prior = predict(x, P, model, event.t - t_last, config)
        x, P, y, S = update(x_prior, P_prior, z, meas, config)
        t_last = event.t
        nis = float(y @ np.linalg.solve(S, y))
        track.steps.append(Step(event.t, event.sensor, event.channel, index,
                                x_prior, P_prior, x, P, z, y, nis))
    return track
