"""LiDAR: keyframe point clouds for display, and 3D boxes from public detections."""

import json
from pathlib import Path

import numpy as np
from nuscenes.utils.data_classes import LidarPointCloud

from .geometry import transform, yaw_of

CROP_RADIUS = 80.0      # m around the ego car
VOXEL = 0.25            # m, one point kept per voxel
GROUND_CLEARANCE = 0.3  # m above the ego wheels; lower points are dropped as ground
MIN_SCORE = 0.2         # detections below this score are not kept, even for display
LIDAR_GATE = 2.0        # m, max distance between detection and ground truth centers


def keyframe_points(nusc, frames, sd):
    pc = LidarPointCloud.from_file(str(Path(nusc.dataroot) / sd["filename"]))
    pts = transform(frames.sensor_to_scene(sd), pc.points[:3].T)
    ego = np.array(frames.ego_xy_yaw(sd)[:2])
    z0 = frames.ego_z(sd)
    keep = (np.hypot(*(pts[:, :2] - ego).T) < CROP_RADIUS) & (pts[:, 2] - z0 > GROUND_CLEARANCE)
    pts = pts[keep]
    _, first = np.unique(np.floor(pts / VOXEL).astype(np.int64), axis=0, return_index=True)
    pts = pts[np.sort(first)]
    pts[:, 2] -= z0  # height above the ego wheels, used for the color
    return pts.astype(np.float32)


def load_detections(path, sample_tokens):
    """Keep only the boxes of the chosen keyframes from a nuScenes detection file."""
    print(f"  reading {Path(path).name} ({Path(path).stat().st_size / 1e6:.0f} MB)")
    results = json.loads(Path(path).read_text())["results"]
    return {t: results.get(t, []) for t in sample_tokens}


def boxes_for_keyframe(raw_boxes, frames, ego_xy, gt_row):
    """Convert boxes to the scene frame and pick the one gated to the target."""
    dets = []
    for b in raw_boxes:
        if b["detection_score"] < MIN_SCORE:
            continue
        xy = frames.to_scene_xy(b["translation"])
        if np.hypot(*(xy - ego_xy)) > CROP_RADIUS + 20:
            continue
        dets.append({
            "xy": [round(float(xy[0]), 3), round(float(xy[1]), 3)],
            "yaw": round(yaw_of(b["rotation"]), 4),
            "size_wlh": [round(float(v), 2) for v in b["size"]],
            "name": b["detection_name"],
            "score": round(float(b["detection_score"]), 3),
        })
    target = None
    if gt_row is not None and dets:
        d = [np.hypot(det["xy"][0] - gt_row["xy"][0], det["xy"][1] - gt_row["xy"][1]) for det in dets]
        i = int(np.argmin(d))
        if d[i] < LIDAR_GATE:
            target = i
    return dets, target
