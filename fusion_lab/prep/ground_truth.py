"""Ground truth of the target instance, and its interpolation between keyframes.

Interpolated boxes are used only to gate RADAR and camera detections, which arrive
between the 2 Hz annotations. They are never shown to students as ground truth.
"""

import numpy as np

from .geometry import wrap, yaw_of


class TargetTruth:
    def __init__(self, nusc, frames, instance_token):
        inst = nusc.get("instance", instance_token)
        self.category = nusc.get("category", inst["category_token"])["name"]
        self.rows = []
        token = inst["first_annotation_token"]
        while token:
            a = nusc.get("sample_annotation", token)
            t_us = nusc.get("sample", a["sample_token"])["timestamp"]
            vel = nusc.box_velocity(token)[:2]
            xy = frames.to_scene_xy(a["translation"])
            self.rows.append({
                "t_us": t_us,
                "sample_token": a["sample_token"],
                "xy": [float(xy[0]), float(xy[1])],
                "z": float(a["translation"][2]),
                "yaw": yaw_of(a["rotation"]),
                "speed": float(np.hypot(*vel)) if np.all(np.isfinite(vel)) else 0.0,
                "size_wlh": [float(v) for v in a["size"]],
                "visibility": nusc.get("visibility", a["visibility_token"])["level"],
                "num_lidar_pts": int(a["num_lidar_pts"]),
                "num_radar_pts": int(a["num_radar_pts"]),
            })
            token = a["next"]
        self.t = np.array([r["t_us"] for r in self.rows], dtype=np.int64)
        self.xy = np.array([r["xy"] for r in self.rows])
        self.zs = np.array([r["z"] for r in self.rows])
        self.yaws = np.unwrap([r["yaw"] for r in self.rows])
        self.size_wlh = np.array(self.rows[0]["size_wlh"])
        self.by_sample = {r["sample_token"]: r for r in self.rows}

    def box_at(self, t_us):
        """(center_xyz, yaw) interpolated at t_us, or None outside the annotated span."""
        if t_us < self.t[0] or t_us > self.t[-1]:
            return None
        c = np.array([np.interp(t_us, self.t, self.xy[:, 0]), np.interp(t_us, self.t, self.xy[:, 1]),
                      np.interp(t_us, self.t, self.zs)])
        return c, float(wrap(np.interp(t_us, self.t, self.yaws)))

    def contains(self, t_us, xy, margin):
        """Boolean mask: which (N, 2) points fall inside the box grown by margin metres."""
        box = self.box_at(t_us)
        if box is None:
            return np.zeros(len(xy), bool)
        c, yaw = box
        d = np.asarray(xy) - c[:2]
        cy, sy = np.cos(yaw), np.sin(yaw)
        along = d[:, 0] * cy + d[:, 1] * sy
        across = -d[:, 0] * sy + d[:, 1] * cy
        w, l, _ = self.size_wlh
        return (np.abs(along) <= l / 2 + margin) & (np.abs(across) <= w / 2 + margin)

    def corners_at(self, t_us):
        """(8, 3) box corners in the scene frame, or None."""
        box = self.box_at(t_us)
        if box is None:
            return None
        c, yaw = box
        w, l, h = self.size_wlh
        x = np.array([1, 1, 1, 1, -1, -1, -1, -1]) * l / 2
        y = np.array([1, -1, -1, 1, 1, -1, -1, 1]) * w / 2
        z = np.array([1, 1, -1, -1, 1, 1, -1, -1]) * h / 2
        R = np.array([[np.cos(yaw), -np.sin(yaw), 0], [np.sin(yaw), np.cos(yaw), 0], [0, 0, 1]])
        return (R @ np.vstack([x, y, z])).T + c
