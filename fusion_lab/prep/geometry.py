"""Poses and frames. The scene frame is the nuScenes global frame shifted so that
the ego car starts at (0, 0) on the first keyframe. Axes are not rotated."""

import numpy as np
from pyquaternion import Quaternion


def pose_matrix(translation, rotation):
    T = np.eye(4)
    T[:3, :3] = Quaternion(rotation).rotation_matrix
    T[:3, 3] = translation
    return T


class Frames:
    def __init__(self, nusc, origin_xy):
        self.nusc = nusc
        self.origin = np.array([origin_xy[0], origin_xy[1], 0.0])

    def sensor_to_scene(self, sd):
        """4x4 transform from the sensor frame of sample_data sd to the scene frame."""
        cs = self.nusc.get("calibrated_sensor", sd["calibrated_sensor_token"])
        ep = self.nusc.get("ego_pose", sd["ego_pose_token"])
        T = pose_matrix(ep["translation"], ep["rotation"]) @ pose_matrix(cs["translation"],
                                                                         cs["rotation"])
        T[:3, 3] -= self.origin
        return T

    def ego_xy_yaw(self, sd):
        ep = self.nusc.get("ego_pose", sd["ego_pose_token"])
        xy = np.array(ep["translation"][:2]) - self.origin[:2]
        return [float(xy[0]), float(xy[1]), yaw_of(ep["rotation"])]

    def ego_z(self, sd):
        return float(self.nusc.get("ego_pose", sd["ego_pose_token"])["translation"][2])

    def to_scene_xy(self, global_xyz):
        return np.asarray(global_xyz)[..., :2] - self.origin[:2]


def yaw_of(rotation):
    R = Quaternion(rotation).rotation_matrix
    return float(np.arctan2(R[1, 0], R[0, 0]))


def axis_yaw(T, axis):
    """Heading in the scene frame of one sensor axis (0 = x forward for radar, 2 = z for camera)."""
    return float(np.arctan2(T[1, axis], T[0, axis]))


def wrap(a):
    return (np.asarray(a) + np.pi) % (2 * np.pi) - np.pi


def transform(T, pts):
    """Apply a 4x4 transform to (N, 3) points."""
    return pts @ T[:3, :3].T + T[:3, 3]
