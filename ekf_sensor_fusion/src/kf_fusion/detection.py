"""Real detectors and cross-sensor geometry for single-object tracking.

Nothing here uses the human annotation. Detections come from:

* **camera** -- a real 2D object detector (YOLO) run on the raw image. Used to
  let the user *pick* the target ("camera pick"), and as clutter the filter
  must gate against.
* **lidar** -- ground removal + Euclidean clustering (DBSCAN) on the real point
  cloud. Each cluster centroid is a 3D detection (with real clutter).
* **radar** -- the raw radar returns are already detections (position + Doppler).

The only cross-sensor glue is projecting lidar into the camera so a 2D pick can
be turned into a 3D track seed, and gating detections to the filter's own
prediction (real single-target data association).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from pyquaternion import Quaternion

VEHICLE_CLASSES = [2, 5, 7]  # COCO: car, bus, truck


# --------------------------------------------------------------------------
# coordinate helpers (nuScenes sensor <-> ego <-> global)
# --------------------------------------------------------------------------
def sensor_to_global(nusc, xyz: np.ndarray, sd_token: str) -> np.ndarray:
    sd = nusc.get("sample_data", sd_token)
    cs = nusc.get("calibrated_sensor", sd["calibrated_sensor_token"])
    ego = nusc.get("ego_pose", sd["ego_pose_token"])
    p = Quaternion(cs["rotation"]).rotation_matrix @ xyz + np.array(cs["translation"])[:, None]
    p = Quaternion(ego["rotation"]).rotation_matrix @ p + np.array(ego["translation"])[:, None]
    return p


def global_to_sensor(nusc, xyz: np.ndarray, sd_token: str) -> np.ndarray:
    sd = nusc.get("sample_data", sd_token)
    cs = nusc.get("calibrated_sensor", sd["calibrated_sensor_token"])
    ego = nusc.get("ego_pose", sd["ego_pose_token"])
    p = xyz - np.array(ego["translation"])[:, None]
    p = Quaternion(ego["rotation"]).rotation_matrix.T @ p
    p = p - np.array(cs["translation"])[:, None]
    p = Quaternion(cs["rotation"]).rotation_matrix.T @ p
    return p


# --------------------------------------------------------------------------
# camera: real 2D detector
# --------------------------------------------------------------------------
@dataclass
class Detection2D:
    box: np.ndarray      # [x1, y1, x2, y2] pixels
    score: float
    label: str

    @property
    def center(self) -> np.ndarray:
        return np.array([(self.box[0] + self.box[2]) / 2, (self.box[1] + self.box[3]) / 2])


# Ultralytics fetches weights from GitHub releases, which some networks block.
# Mirror on Hugging Face (reachable where GitHub releases are not).
_YOLO_HF_URL = "https://huggingface.co/Ultralytics/YOLO11/resolve/main/yolo11n.pt"


def ensure_yolo_weights(path: str = "models/yolo_det.pt") -> str:
    """Download the YOLO weights from Hugging Face if not already present."""
    import urllib.request
    p = Path(path)
    if not p.exists():
        p.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(_YOLO_HF_URL, p)
    return str(p)


class Yolo2DDetector:
    """Thin wrapper around Ultralytics YOLO restricted to vehicles."""

    def __init__(self, weights: str = "models/yolo_det.pt", conf: float = 0.25):
        from ultralytics import YOLO
        self.model = YOLO(ensure_yolo_weights(weights))
        self.conf = conf

    def detect(self, image_path: str) -> list[Detection2D]:
        r = self.model(image_path, verbose=False, classes=VEHICLE_CLASSES, conf=self.conf)[0]
        out = []
        for b in r.boxes:
            out.append(Detection2D(box=b.xyxy[0].cpu().numpy(),
                                   score=float(b.conf),
                                   label=self.model.names[int(b.cls)]))
        return out


# --------------------------------------------------------------------------
# lidar: real clustering detector
# --------------------------------------------------------------------------
def lidar_detections(nusc, lidar_sd: str, max_range: float = 45.0,
                     eps: float = 0.9, min_samples: int = 6) -> np.ndarray:
    """Return (M, 2) global-frame centroids of clustered lidar objects.

    Ground removal + Euclidean clustering. No labels used.
    """
    from nuscenes.utils.data_classes import LidarPointCloud
    from sklearn.cluster import DBSCAN

    lpc = LidarPointCloud.from_file(nusc.get_sample_data_path(lidar_sd))
    g = sensor_to_global(nusc, lpc.points[:3], lidar_sd)
    ego = np.array(nusc.get("ego_pose", nusc.get("sample_data", lidar_sd)["ego_pose_token"])["translation"])
    keep = ((g[2] > ego[2] - 1.4) & (g[2] < ego[2] + 3.0)
            & (np.hypot(g[0] - ego[0], g[1] - ego[1]) < max_range))
    pts = g[:2, keep].T
    if len(pts) < min_samples:
        return np.empty((0, 2))
    labels = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(pts)
    cents = [pts[labels == c].mean(0) for c in set(labels) if c != -1]
    return np.array(cents) if cents else np.empty((0, 2))


# --------------------------------------------------------------------------
# 2D pick -> 3D seed: project lidar into the camera, take points in the box
# --------------------------------------------------------------------------
def lidar_points_in_box_2d(nusc, lidar_sd: str, cam_sd: str,
                           box2d: np.ndarray) -> np.ndarray:
    """Global (x, y) of lidar points that project inside a camera 2D box."""
    from nuscenes.utils.data_classes import LidarPointCloud
    from nuscenes.utils.geometry_utils import view_points

    lpc = LidarPointCloud.from_file(nusc.get_sample_data_path(lidar_sd))
    g = sensor_to_global(nusc, lpc.points[:3], lidar_sd)
    cam = global_to_sensor(nusc, g, cam_sd)             # points in camera frame
    cs = nusc.get("calibrated_sensor", nusc.get("sample_data", cam_sd)["calibrated_sensor_token"])
    K = np.array(cs["camera_intrinsic"])
    depth = cam[2]
    front = depth > 1.0
    uv = view_points(cam, K, normalize=True)[:2]
    x1, y1, x2, y2 = box2d
    inside = front & (uv[0] > x1) & (uv[0] < x2) & (uv[1] > y1) & (uv[1] < y2)
    return g[:2, inside]


def seed_from_pick(nusc, lidar_sd: str, cam_sd: str, box2d: np.ndarray) -> np.ndarray | None:
    """Turn a picked 2D box into a 3D (x, y) seed via the nearest lidar cluster."""
    from sklearn.cluster import DBSCAN

    pts = lidar_points_in_box_2d(nusc, lidar_sd, cam_sd, box2d).T
    if len(pts) < 5:
        return None
    labels = DBSCAN(eps=0.9, min_samples=5).fit_predict(pts)
    labels_valid = labels[labels != -1]
    if len(labels_valid) == 0:
        return pts.mean(0)
    # the closest cluster to the camera is the picked object (foreground)
    ego = np.array(nusc.get("ego_pose", nusc.get("sample_data", cam_sd)["ego_pose_token"])["translation"])[:2]
    best, bestd = None, np.inf
    for c in set(labels_valid):
        cen = pts[labels == c].mean(0)
        d = np.hypot(cen[0] - ego[0], cen[1] - ego[1])
        if d < bestd:
            best, bestd = cen, d
    return best


def gate_nearest(detections: np.ndarray, predicted_xy: np.ndarray,
                 max_dist: float = 4.0) -> np.ndarray | None:
    """Real single-target association: nearest detection within a gate."""
    if len(detections) == 0:
        return None
    d = np.hypot(detections[:, 0] - predicted_xy[0], detections[:, 1] - predicted_xy[1])
    j = int(np.argmin(d))
    return detections[j] if d[j] <= max_dist else None
