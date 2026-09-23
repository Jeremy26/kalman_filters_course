"""RADAR: cluster the raw points of each sweep and express each cluster as a
polar measurement (range, bearing, range rate) in the radar's own frame."""

from pathlib import Path

import numpy as np
from nuscenes.utils.data_classes import RadarPointCloud
from sklearn.cluster import DBSCAN

from .geometry import axis_yaw, transform, wrap

CLUSTER_EPS = 2.0   # m, DBSCAN neighborhood
GATE_MARGIN = 1.0   # m added around the ground truth box


def sweep(nusc, frames, sd):
    """Raw points of one radar sweep in the scene frame.

    Returns sensor pose (xy, yaw), points (N, 2) and ego-motion compensated
    velocities (N, 2). nuScenes stores these velocities in the sensor frame.
    """
    pc = RadarPointCloud.from_file(str(Path(nusc.dataroot) / sd["filename"]))
    T = frames.sensor_to_scene(sd)
    p = pc.points
    xy = transform(T, np.vstack([p[0], p[1], np.zeros(p.shape[1])]).T)[:, :2]
    vel = (T[:3, :3] @ np.vstack([p[8], p[9], np.zeros(p.shape[1])]))[:2].T
    return T[:2, 3].copy(), axis_yaw(T, 0), xy, vel


def clusters(sensor_xy, sensor_yaw, xy, vel):
    """One detection per DBSCAN cluster, plus the cluster label of every point."""
    if len(xy) == 0:
        return [], np.zeros(0, int)
    labels = DBSCAN(eps=CLUSTER_EPS, min_samples=1).fit_predict(xy)
    dets = []
    for k in range(labels.max() + 1):
        m = labels == k
        c = xy[m].mean(axis=0)
        d = c - sensor_xy
        los = xy[m] - sensor_xy
        los /= np.linalg.norm(los, axis=1, keepdims=True)
        range_rate = float(np.mean(np.sum(vel[m] * los, axis=1)))
        dets.append({
            "z": [round(float(np.hypot(*d)), 3),
                  round(float(wrap(np.arctan2(d[1], d[0]) - sensor_yaw)), 5),
                  round(range_rate, 3)],
            "xy": [round(float(c[0]), 3), round(float(c[1]), 3)],
            "n_points": int(m.sum()),
        })
    return dets, labels


def gate(truth, t_us, dets, labels, xy):
    """The cluster with the most points inside the (grown) target box, or None."""
    inside = truth.contains(t_us, xy, GATE_MARGIN)
    if not inside.any():
        return None
    counts = np.bincount(labels[inside], minlength=len(dets))
    return int(np.argmax(counts))
