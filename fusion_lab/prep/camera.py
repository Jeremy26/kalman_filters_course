"""Camera: 2D boxes from YOLOX-s (ONNX, run with OpenCV DNN on CPU).

The detector runs only on images where the target's ground truth box projects
into the image, which is enough for a single-target demo and keeps prep short.
"""

from pathlib import Path

import cv2
import numpy as np

from .geometry import axis_yaw, transform, wrap

INPUT_SIZE = 640
SCORE_MIN = 0.3
NMS_IOU = 0.45
IOU_GATE = 0.3
MIN_VISIBLE_FRACTION = 0.3  # of the projected box that must be inside the image

COCO = {0: "person", 1: "bicycle", 2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}


class YoloX:
    def __init__(self, onnx_path):
        self.net = cv2.dnn.readNetFromONNX(str(onnx_path))
        grids, strides = [], []
        for s in (8, 16, 32):
            n = INPUT_SIZE // s
            gx, gy = np.meshgrid(np.arange(n), np.arange(n))
            grids.append(np.stack([gx, gy], -1).reshape(-1, 2))
            strides.append(np.full((n * n, 1), s))
        self.grids = np.concatenate(grids)
        self.strides = np.concatenate(strides)

    def __call__(self, bgr):
        h, w = bgr.shape[:2]
        r = min(INPUT_SIZE / h, INPUT_SIZE / w)
        canvas = np.full((INPUT_SIZE, INPUT_SIZE, 3), 114, np.uint8)
        canvas[:int(h * r), :int(w * r)] = cv2.resize(bgr, (int(w * r), int(h * r)))
        blob = canvas.transpose(2, 0, 1)[None].astype(np.float32)
        self.net.setInput(blob)
        out = self.net.forward()[0]
        xy = (out[:, :2] + self.grids) * self.strides
        wh = np.exp(out[:, 2:4]) * self.strides
        cls = out[:, 5:].argmax(1)
        score = out[:, 4] * out[np.arange(len(out)), 5 + cls]
        keep = (score > SCORE_MIN) & np.isin(cls, list(COCO))
        xy, wh, cls, score = xy[keep], wh[keep], cls[keep], score[keep]
        boxes = np.hstack([xy - wh / 2, wh]) / r
        idx = cv2.dnn.NMSBoxes(boxes.tolist(), score.tolist(), SCORE_MIN, NMS_IOU)
        idx = np.array(idx).reshape(-1)
        return [(boxes[i, 0], boxes[i, 1], boxes[i, 0] + boxes[i, 2], boxes[i, 1] + boxes[i, 3],
                 COCO[int(cls[i])], float(score[i])) for i in idx]


def camera_model(nusc, frames, sd):
    cs = nusc.get("calibrated_sensor", sd["calibrated_sensor_token"])
    T = frames.sensor_to_scene(sd)
    return np.array(cs["camera_intrinsic"]), T


def project_target(K, T, corners, width, height):
    """2D box [x1, y1, x2, y2] of the target in the image, or None when not visible."""
    cam = transform(np.linalg.inv(T), corners)
    if np.any(cam[:, 2] < 0.5):
        return None
    uv = (K @ cam.T).T
    uv = uv[:, :2] / uv[:, 2:3]
    x1, y1 = uv.min(0)
    x2, y2 = uv.max(0)
    cx1, cy1, cx2, cy2 = max(x1, 0), max(y1, 0), min(x2, width), min(y2, height)
    if cx2 <= cx1 or cy2 <= cy1:
        return None
    if (cx2 - cx1) * (cy2 - cy1) < MIN_VISIBLE_FRACTION * (x2 - x1) * (y2 - y1):
        return None
    return [cx1, cy1, cx2, cy2]


def bearing(K, T, u, v):
    """Bearing of pixel (u, v) in the scene frame, relative to the camera heading."""
    ray = T[:3, :3] @ (np.linalg.inv(K) @ np.array([u, v, 1.0]))
    return float(wrap(np.arctan2(ray[1], ray[0]) - axis_yaw(T, 2)))


def iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def detect(yolo, nusc, frames, sd, gt_box_2d):
    """Run YOLOX on one image. Returns (sensor_xy, sensor_yaw, detections, target index)."""
    K, T = camera_model(nusc, frames, sd)
    img = cv2.imread(str(Path(nusc.dataroot) / sd["filename"]))
    dets = []
    for x1, y1, x2, y2, cls, score in yolo(img):
        vc = 0.5 * (y1 + y2)
        dets.append({
            "box_xyxy": [round(float(v), 1) for v in (x1, y1, x2, y2)],
            "cls": cls,
            "score": round(score, 3),
            "bearing": round(bearing(K, T, 0.5 * (x1 + x2), vc), 5),
            # Left edge of the image box has the larger (counter-clockwise) bearing.
            "bearing_span": [round(bearing(K, T, x2, vc), 5), round(bearing(K, T, x1, vc), 5)],
        })
    target = None
    if dets:
        scores = [iou(d["box_xyxy"], gt_box_2d) for d in dets]
        i = int(np.argmax(scores))
        if scores[i] > IOU_GATE:
            target = i
    return T[:2, 3].copy(), axis_yaw(T, 2), dets, target
