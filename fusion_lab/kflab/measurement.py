"""Measurement models: what each sensor measures, as a function of the state.

    h(x)               predicted measurement for state x
    H(x)               jacobian of h (equal to the matrix for linear models)
    R                  measurement noise covariance
    residual(z, zp)    z - zp, with angles wrapped

Each measurement model is built for one sensor event, because RADAR and camera
measurements depend on where the sensor was at that instant (sensor_xy, sensor_yaw).
"""

import numpy as np

from .angles import wrap


class LidarPosition:
    """LiDAR box center: z = [px, py] in the scene frame. Linear."""

    name = "lidar"
    labels = ["px", "py"]
    angle_idx = []
    linear = True

    def __init__(self, model, std_pos):
        self.model = model
        self.R = np.eye(2) * std_pos**2
        self._H = np.zeros((2, model.dim))
        self._H[0, 0] = self._H[1, 1] = 1.0

    def h(self, x):
        return x[:2].copy()

    def H(self, x):
        return self._H

    def residual(self, z, zp):
        return z - zp


class RadarPolar:
    """RADAR cluster: z = [range, bearing, range_rate] in the radar's own frame.

    The radar sits at sensor_xy and looks along sensor_yaw. nuScenes gives
    velocities compensated for the ego motion, so the range rate is the target's
    ground velocity projected on the line of sight.
    """

    name = "radar"
    labels = ["range", "bearing", "range_rate"]
    angle_idx = [1]
    linear = False

    def __init__(self, model, sensor_xy, sensor_yaw, std_range, std_bearing, std_range_rate):
        self.model = model
        self.sx, self.sy = sensor_xy
        self.syaw = sensor_yaw
        self.R = np.diag([std_range**2, std_bearing**2, std_range_rate**2])

    def h(self, x):
        px, py, vx, vy = self.model.pv(x)
        dx, dy = px - self.sx, py - self.sy
        r = max(np.hypot(dx, dy), 1e-3)
        bearing = wrap(np.arctan2(dy, dx) - self.syaw)
        range_rate = (dx * vx + dy * vy) / r
        return np.array([r, bearing, range_rate])

    def H(self, x):
        # Chain rule: d h / d x = d h / d [px, py, vx, vy] @ d [px, py, vx, vy] / d x
        px, py, vx, vy = self.model.pv(x)
        dx, dy = px - self.sx, py - self.sy
        r2 = max(dx * dx + dy * dy, 1e-6)
        r = np.sqrt(r2)
        cross = vx * dy - vy * dx
        H_pv = np.array([
            [dx / r, dy / r, 0.0, 0.0],
            [-dy / r2, dx / r2, 0.0, 0.0],
            [dy * cross / r**3, -dx * cross / r**3, dx / r, dy / r],
        ])
        return H_pv @ self.model.pv_jacobian(x)

    def residual(self, z, zp):
        y = z - zp
        y[1] = wrap(y[1])
        return y

    def to_position_velocity(self, z):
        """Invert a measurement, used only to start the track from a RADAR detection."""
        r, bearing, range_rate = z
        a = bearing + self.syaw
        u = np.array([np.cos(a), np.sin(a)])
        return np.array([self.sx, self.sy]) + r * u, range_rate * u


class CameraBearing:
    """Camera box center seen as a direction: z = [bearing] in the camera frame.

    Optional (off by default). A camera alone gives no range, so this update only
    narrows the ellipse across the line of sight, never along it.
    """

    name = "camera"
    labels = ["bearing"]
    angle_idx = [0]
    linear = False

    def __init__(self, model, sensor_xy, sensor_yaw, std_bearing):
        self.model = model
        self.sx, self.sy = sensor_xy
        self.syaw = sensor_yaw
        self.R = np.array([[std_bearing**2]])

    def h(self, x):
        dx, dy = x[0] - self.sx, x[1] - self.sy
        return np.array([wrap(np.arctan2(dy, dx) - self.syaw)])

    def H(self, x):
        dx, dy = x[0] - self.sx, x[1] - self.sy
        r2 = max(dx * dx + dy * dy, 1e-6)
        H = np.zeros((1, self.model.dim))
        H[0, 0], H[0, 1] = -dy / r2, dx / r2
        return H

    def residual(self, z, zp):
        return wrap(z - zp)
