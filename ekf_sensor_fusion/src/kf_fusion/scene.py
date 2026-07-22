"""Single-object tracking on a full real nuScenes scene, logged for Foxglove.

This is the "camera pick" pipeline and the rich recording in one place:

1. On the first keyframe, a real 2D detector (YOLO) runs on the camera. The user
   *picks* the target box (``select``). That pick is turned into a 3D seed by
   projecting lidar into the box (``detection.seed_from_pick``).
2. Every keyframe after that, real detections are produced with **no labels**:
   lidar clusters (DBSCAN) and raw radar returns. The filter gates the nearest
   detection to its own prediction (real single-target association) and fuses.
3. Everything is logged to MCAP: the camera image + 2D detections, the full
   lidar cloud, the radar returns, the tf tree, and the tracking output
   (estimate, covariance, velocity, 1 s prediction, ground truth) + plots.

The nuScenes annotation is used only to (a) auto-pick the demo target for a
reproducible run and (b) score against ground truth.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from pyquaternion import Quaternion

import foxglove
from foxglove.messages import (
    CameraCalibration, Color, CompressedImage, CubePrimitive, FrameTransform,
    ImageAnnotations, LinePrimitive, LinePrimitiveLineType, PackedElementField,
    PackedElementFieldNumericType, Point2, Point3, PointCloud, PointsAnnotation,
    PointsAnnotationType, Pose, Quaternion as FQuat, SceneEntity, SceneUpdate,
    SpherePrimitive, Vector3,
)

from . import detection
from .ekf import EKF

RADARS = ["RADAR_FRONT", "RADAR_FRONT_LEFT", "RADAR_FRONT_RIGHT",
          "RADAR_BACK_LEFT", "RADAR_BACK_RIGHT"]
CAM = "CAM_FRONT"


# --------------------------------------------------------------------------
# small geometry / packing helpers
# --------------------------------------------------------------------------
def _map_to_sensor(nusc, sd_token):
    """(translation, quaternion) placing a sensor's frame in the global map."""
    sd = nusc.get("sample_data", sd_token)
    cs = nusc.get("calibrated_sensor", sd["calibrated_sensor_token"])
    ego = nusc.get("ego_pose", sd["ego_pose_token"])
    q_ego, q_cs = Quaternion(ego["rotation"]), Quaternion(cs["rotation"])
    t = q_ego.rotate(np.array(cs["translation"])) + np.array(ego["translation"])
    q = q_ego * q_cs
    return t, q


def _pack_xyz(pts_3xn: np.ndarray) -> bytes:
    return np.ascontiguousarray(pts_3xn.T.astype(np.float32)).tobytes()


_XYZ_FIELDS = [PackedElementField(name=n, offset=i * 4,
                                  type=PackedElementFieldNumericType.Float32)
               for i, n in enumerate(["x", "y", "z"])]


def _radar_returns(nusc, sample):
    """All radar returns for a keyframe, in the global frame, with Doppler."""
    from nuscenes.utils.data_classes import RadarPointCloud
    out = []
    for rch in RADARS:
        rsd = sample["data"][rch]
        rpc = RadarPointCloud.from_file(nusc.get_sample_data_path(rsd))
        if rpc.points.shape[1] == 0:
            continue
        g = detection.sensor_to_global(nusc, rpc.points[:3], rsd)
        cs = nusc.get("calibrated_sensor", nusc.get("sample_data", rsd)["calibrated_sensor_token"])
        ego = nusc.get("ego_pose", nusc.get("sample_data", rsd)["ego_pose_token"])
        R = Quaternion(ego["rotation"]).rotation_matrix[:2, :2] @ Quaternion(cs["rotation"]).rotation_matrix[:2, :2]
        vg = R @ rpc.points[8:10]
        s = detection.sensor_to_global(nusc, np.zeros((3, 1)), rsd)[:2, 0]
        for i in range(g.shape[1]):
            out.append((g[0, i], g[1, i], s, vg[0, i], vg[1, i]))
    return out


def _yaw_quat(yaw: float) -> FQuat:
    return FQuat(x=0.0, y=0.0, z=float(np.sin(yaw / 2)), w=float(np.cos(yaw / 2)))


def _p3(x, y, z=0.0):
    return Point3(x=float(x), y=float(y), z=float(z))


# --------------------------------------------------------------------------
# the pipeline
# --------------------------------------------------------------------------
def run_scene_sot(nusc, instance_token: str, mcap_path: str | Path,
                  detector: "detection.Yolo2DDetector | None" = None,
                  select=None, verbose: bool = True) -> dict:
    """Track one picked object through a real scene and write a rich MCAP.

    ``select`` chooses the target on the first frame. ``None`` auto-picks the
    2D box overlapping the demo instance (reproducible). Otherwise pass a pixel
    ``(u, v)`` and the box containing it is picked -- that's the "camera click".
    """
    from nuscenes.utils.data_classes import LidarPointCloud

    inst = nusc.get("instance", instance_token)
    if detector is None:
        detector = detection.Yolo2DDetector()

    ekf = EKF(noise_ax=4.0, noise_ay=4.0)
    writer = foxglove.open_mcap(str(mcap_path), allow_overwrite=True)
    gt_path, est_path = [], []
    rmse_pos, n = 0.0, 0

    ann_tok = inst["first_annotation_token"]
    last_t = None
    frame = 0
    try:
        while ann_tok:
            ann = nusc.get("sample_annotation", ann_tok)
            sample = nusc.get("sample", ann["sample_token"])
            lidar_sd = sample["data"]["LIDAR_TOP"]
            cam_sd = sample["data"][CAM]
            t = nusc.get("sample_data", lidar_sd)["timestamp"] / 1e6
            log_time = int(t * 1e9)
            gt = np.array(ann["translation"][:2])

            # --- transforms: map -> ego, map -> camera ---
            for child, sd in [("ego", lidar_sd), (CAM, cam_sd)]:
                if child == "ego":
                    ep = nusc.get("ego_pose", nusc.get("sample_data", sd)["ego_pose_token"])
                    tr, q = np.array(ep["translation"]), Quaternion(ep["rotation"])
                else:
                    tr, q = _map_to_sensor(nusc, sd)
                foxglove.log("/tf", FrameTransform(
                    parent_frame_id="map", child_frame_id=child,
                    translation=Vector3(x=float(tr[0]), y=float(tr[1]), z=float(tr[2])),
                    rotation=FQuat(x=q.elements[1], y=q.elements[2], z=q.elements[3], w=q.elements[0])),
                    log_time=log_time)

            # --- camera image + calibration + real 2D detections ---
            cam_path = nusc.get_sample_data_path(cam_sd)
            with open(cam_path, "rb") as fh:
                foxglove.log("/camera/image",
                             CompressedImage(frame_id=CAM, format="jpeg", data=fh.read()),
                             log_time=log_time)
            cs_cam = nusc.get("calibrated_sensor", nusc.get("sample_data", cam_sd)["calibrated_sensor_token"])
            K = np.array(cs_cam["camera_intrinsic"])
            sdc = nusc.get("sample_data", cam_sd)
            foxglove.log("/camera/calibration", CameraCalibration(
                frame_id=CAM, width=sdc["width"], height=sdc["height"],
                distortion_model="plumb_bob", D=[0.0] * 5,
                K=K.flatten().tolist(), R=[1, 0, 0, 0, 1, 0, 0, 0, 1],
                P=[K[0, 0], 0, K[0, 2], 0, 0, K[1, 1], K[1, 2], 0, 0, 0, 1, 0]),
                log_time=log_time)

            dets2d = detector.detect(cam_path)

            # --- pick the target on the first frame ---
            if not ekf.initialized:
                picked = _pick_box(nusc, dets2d, ann_tok, cam_sd, select)
                if picked is None:
                    ann_tok = ann["next"]; frame += 1; continue
                seed = detection.seed_from_pick(nusc, lidar_sd, cam_sd, picked.box)
                if seed is None:
                    ann_tok = ann["next"]; frame += 1; continue
                # Seed VELOCITY from the nearest radar Doppler return -- without
                # it the filter starts at rest and a fast car escapes the gate.
                v0 = np.zeros(2)
                rret = _radar_returns(nusc, sample)
                if rret:
                    arr = np.array([[r[0], r[1]] for r in rret])
                    d = np.hypot(arr[:, 0] - seed[0], arr[:, 1] - seed[1])
                    if d.min() < 4.0:
                        v0 = np.array(rret[int(np.argmin(d))][3:5])
                ekf.initialize(np.array([seed[0], seed[1], v0[0], v0[1]]),
                               P0=np.diag([2.0, 2.0, 25.0, 25.0]))
                last_t = t
            else:
                ekf.predict(t - last_t); last_t = t
                pred_xy = ekf.x[:2]
                # lidar detections (clusters) -> gate -> update
                ldets = detection.lidar_detections(nusc, lidar_sd)
                lz = detection.gate_nearest(ldets, pred_xy, max_dist=5.0)
                if lz is not None:
                    ekf.update_lidar(lz)
                # radar detections -> gate -> update (with Doppler)
                rret = _radar_returns(nusc, sample)
                if rret:
                    arr = np.array([[r[0], r[1]] for r in rret])
                    rz = detection.gate_nearest(arr, ekf.x[:2], max_dist=5.0)
                    if rz is not None:
                        j = int(np.argmin(np.hypot(arr[:, 0] - rz[0], arr[:, 1] - rz[1])))
                        _gx, _gy, s, vgx, vgy = rret[j]
                        rho = float(np.hypot(_gx - s[0], _gy - s[1]))
                        phi = float(np.arctan2(_gy - s[1], _gx - s[0]))
                        rdot = float(((_gx - s[0]) * vgx + (_gy - s[1]) * vgy) / max(rho, 1e-6))
                        ekf.update_radar(np.array([rho, phi, rdot]), sensor=s)

            # --- log full lidar cloud (global) + radar returns (global) ---
            lpc = LidarPointCloud.from_file(nusc.get_sample_data_path(lidar_sd))
            gl = detection.sensor_to_global(nusc, lpc.points[:3], lidar_sd)
            foxglove.log("/lidar", PointCloud(
                frame_id="map", point_stride=12, fields=_XYZ_FIELDS, data=_pack_xyz(gl)),
                log_time=log_time)
            rret = _radar_returns(nusc, sample)
            if rret:
                rr = np.array([[r[0], r[1], 0.0] for r in rret]).T
                foxglove.log("/radar", PointCloud(
                    frame_id="map", point_stride=12, fields=_XYZ_FIELDS, data=_pack_xyz(rr)),
                    log_time=log_time)

            # --- 2D detection overlay (annotations) ---
            _log_annotations(dets2d, log_time, ekf.initialized, select, nusc, ann_tok, cam_sd)

            # --- tracking output ---
            x = ekf.x
            gt_path.append(_p3(gt[0], gt[1])); est_path.append(_p3(x[0], x[1]))
            _log_track(x, ekf.P, gt, gt_path, est_path, log_time)
            pos_err = float(np.hypot(x[0] - gt[0], x[1] - gt[1]))
            foxglove.log("/error", {"pos_err": pos_err}, log_time=log_time)
            foxglove.log("/state", {"px": float(x[0]), "py": float(x[1]),
                                    "speed": float(np.hypot(x[2], x[3]))}, log_time=log_time)
            rmse_pos += pos_err ** 2; n += 1
            if verbose:
                print(f"  frame {frame}: pos_err={pos_err:.2f} m", flush=True)
            ann_tok = ann["next"]; frame += 1
    finally:
        writer.close()

    return {"frames": n, "rmse_pos": float(np.sqrt(rmse_pos / max(n, 1))), "mcap": str(mcap_path)}


def _pick_box(nusc, dets2d, ann_tok, cam_sd, select):
    """Resolve the user's target selection to one Detection2D."""
    if not dets2d:
        return None
    if select is None:
        # reproducible: pick the detection overlapping the annotated target
        from nuscenes.utils.geometry_utils import view_points
        _, boxes, K = nusc.get_sample_data(cam_sd, selected_anntokens=[ann_tok])
        if not boxes:
            return max(dets2d, key=lambda d: d.score)
        c = view_points(boxes[0].corners(), K, normalize=True)[:2]
        tgt = np.array([c[0].mean(), c[1].mean()])
        return min(dets2d, key=lambda d: np.hypot(*(d.center - tgt)))
    u, v = select  # a pixel "click"
    inside = [d for d in dets2d if d.box[0] <= u <= d.box[2] and d.box[1] <= v <= d.box[3]]
    return min(inside, key=lambda d: np.hypot(*(d.center - np.array([u, v])))) if inside else None


def _log_annotations(dets2d, log_time, initialized, select, nusc, ann_tok, cam_sd):
    tgt_center = None
    if select is None:
        from nuscenes.utils.geometry_utils import view_points
        _, boxes, K = nusc.get_sample_data(cam_sd, selected_anntokens=[ann_tok])
        if boxes:
            c = view_points(boxes[0].corners(), K, normalize=True)[:2]
            tgt_center = np.array([c[0].mean(), c[1].mean()])
    points = []
    for d in dets2d:
        x1, y1, x2, y2 = d.box
        is_target = tgt_center is not None and (x1 <= tgt_center[0] <= x2 and y1 <= tgt_center[1] <= y2)
        col = Color(r=0.1, g=0.9, b=0.3, a=1.0) if is_target else Color(r=1.0, g=0.8, b=0.1, a=0.8)
        points.append(PointsAnnotation(
            type=PointsAnnotationType.LineLoop,
            points=[Point2(x=float(x1), y=float(y1)), Point2(x=float(x2), y=float(y1)),
                    Point2(x=float(x2), y=float(y2)), Point2(x=float(x1), y=float(y2))],
            outline_color=col, thickness=3.0))
    foxglove.log("/camera/annotations", ImageAnnotations(points=points), log_time=log_time)


def _log_track(x, P, gt, gt_path, est_path, log_time):
    sigma = float(np.sqrt(max(P[0, 0] + P[1, 1], 1e-9)))
    pred = x[:2] + x[2:4] * 1.0
    spheres = [
        SpherePrimitive(pose=Pose(position=Vector3(x=float(x[0]), y=float(x[1]), z=0.5)),
                        size=Vector3(x=2 * sigma, y=2 * sigma, z=0.1),
                        color=Color(r=0.2, g=0.55, b=1.0, a=0.25)),
        SpherePrimitive(pose=Pose(position=Vector3(x=float(pred[0]), y=float(pred[1]), z=0.5)),
                        size=Vector3(x=0.9, y=0.9, z=0.9), color=Color(r=1.0, g=0.5, b=0.0, a=0.9)),
    ]
    lines = [
        LinePrimitive(type=LinePrimitiveLineType.LineStrip, thickness=2.0, scale_invariant=True,
                      points=list(gt_path), color=Color(r=0.2, g=0.85, b=0.2, a=1.0)),
        LinePrimitive(type=LinePrimitiveLineType.LineStrip, thickness=2.0, scale_invariant=True,
                      points=list(est_path), color=Color(r=0.2, g=0.55, b=1.0, a=1.0)),
        LinePrimitive(type=LinePrimitiveLineType.LineStrip, thickness=4.0, scale_invariant=True,
                      points=[_p3(x[0], x[1], 0.5), _p3(x[0] + x[2], x[1] + x[3], 0.5)],
                      color=Color(r=1.0, g=0.6, b=0.0, a=1.0)),
    ]
    box = CubePrimitive(pose=Pose(position=Vector3(x=float(x[0]), y=float(x[1]), z=0.75),
                                  orientation=_yaw_quat(float(np.arctan2(x[3], x[2])))),
                        size=Vector3(x=4.5, y=2.0, z=1.5), color=Color(r=0.2, g=0.55, b=1.0, a=0.25))
    foxglove.log("/track", SceneUpdate(entities=[SceneEntity(
        frame_id="map", id="target", spheres=spheres, lines=lines, cubes=[box])]),
        log_time=log_time)
