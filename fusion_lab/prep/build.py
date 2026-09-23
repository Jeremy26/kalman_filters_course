"""Build the cache for one scene. See DESIGN.md, section "Cache format"."""

import json
import time
from pathlib import Path

import numpy as np

from . import camera, lidar, radar
from .geometry import Frames
from .ground_truth import TargetTruth

SCHEMA_VERSION = 1
RADAR_CHANNELS = ["RADAR_FRONT", "RADAR_FRONT_LEFT", "RADAR_FRONT_RIGHT", "RADAR_BACK_LEFT",
                  "RADAR_BACK_RIGHT"]
CAMERA_CHANNELS = ["CAM_FRONT", "CAM_FRONT_LEFT", "CAM_FRONT_RIGHT", "CAM_BACK", "CAM_BACK_LEFT",
                   "CAM_BACK_RIGHT"]
CHANNEL_ID = {c: i for i, c in enumerate(["LIDAR_TOP"] + RADAR_CHANNELS + CAMERA_CHANNELS)}


def scene_samples(nusc, scene):
    out, t = [], scene["first_sample_token"]
    while t:
        s = nusc.get("sample", t)
        out.append(s)
        t = s["next"]
    return out


def scene_sample_data(nusc, samples, channel):
    """Every sample_data of one channel in the scene (keyframes and sweeps), in time order."""
    tokens = {s["token"] for s in samples}
    sensor = {s["token"]: s["channel"] for s in nusc.sensor}
    cs_channel = {c["token"]: sensor[c["sensor_token"]] for c in nusc.calibrated_sensor}
    sds = [sd for sd in nusc.sample_data
           if sd["sample_token"] in tokens and cs_channel[sd["calibrated_sensor_token"]] == channel]
    return sorted(sds, key=lambda sd: sd["timestamp"])


def _r(v, n=3):
    return round(float(v), n)


def build_scene(nusc, spec, detections, yolo, out_dir, camera_stride=1):
    t_start = time.time()
    scene = next(s for s in nusc.scene if s["name"] == spec["name"])
    samples = scene_samples(nusc, scene)
    lidar_kf = [nusc.get("sample_data", s["data"]["LIDAR_TOP"]) for s in samples]
    first_ego = nusc.get("ego_pose", lidar_kf[0]["ego_pose_token"])["translation"]
    frames = Frames(nusc, first_ego[:2])
    truth = TargetTruth(nusc, frames, spec["target"])
    kf_times = np.array([s["timestamp"] for s in samples], dtype=np.int64)

    events = []          # (t_us, dict) before splitting into frames
    radar_points = []    # (t_us, channel, xy, vel) kept for the npz files

    # LiDAR: one event per keyframe with the public 3D boxes.
    lidar_pts = {}
    for s, sd in zip(samples, lidar_kf):
        ego = frames.ego_xy_yaw(sd)
        dets, target = lidar.boxes_for_keyframe(detections.get(s["token"], []), frames,
                                                np.array(ego[:2]), truth.by_sample.get(s["token"]))
        T = frames.sensor_to_scene(sd)
        events.append((sd["timestamp"], {
            "sensor": "lidar", "channel": "LIDAR_TOP", "sd_token": sd["token"],
            "sensor_xy_yaw": [_r(T[0, 3]), _r(T[1, 3]), _r(ego[2], 5)],
            "detections": dets, "target": target}))
        lidar_pts[s["token"]] = lidar.keyframe_points(nusc, frames, sd)
    n_lidar = sum(e["target"] is not None for _, e in events)

    # RADAR: every sweep of the five radars, clustered.
    n_radar = 0
    for ch in RADAR_CHANNELS:
        for sd in scene_sample_data(nusc, samples, ch):
            sxy, syaw, xy, vel = radar.sweep(nusc, frames, sd)
            dets, labels = radar.clusters(sxy, syaw, xy, vel)
            target = radar.gate(truth, sd["timestamp"], dets, labels, xy) if dets else None
            n_radar += target is not None
            events.append((sd["timestamp"], {
                "sensor": "radar", "channel": ch, "sd_token": sd["token"],
                "sensor_xy_yaw": [_r(sxy[0]), _r(sxy[1]), _r(syaw, 5)],
                "detections": dets, "target": target}))
            radar_points.append((sd["timestamp"], ch, sd["token"], xy, vel))

    # Camera: YOLOX on the images where the target projects into the frame.
    n_images = n_camera = 0
    for ch in CAMERA_CHANNELS:
        sds = scene_sample_data(nusc, samples, ch)[::camera_stride]
        for sd in sds:
            corners = truth.corners_at(sd["timestamp"])
            if corners is None:
                continue
            K, T = camera.camera_model(nusc, frames, sd)
            gt2d = camera.project_target(K, T, corners, sd["width"], sd["height"])
            if gt2d is None:
                continue
            sxy, syaw, dets, target = camera.detect(yolo, nusc, frames, sd, gt2d)
            n_images += 1
            n_camera += target is not None
            events.append((sd["timestamp"], {
                "sensor": "camera", "channel": ch, "sd_token": sd["token"],
                "sensor_xy_yaw": [_r(sxy[0]), _r(sxy[1]), _r(syaw, 5)],
                "detections": dets, "target": target}))

    write_scene(out_dir, spec, scene, truth, samples, lidar_kf, frames, kf_times, events,
                lidar_pts, radar_points)
    print(f"  {spec['name']}: {len(events)} events, target gated in {n_lidar}/{len(samples)} "
          f"LiDAR keyframes, {n_radar} RADAR sweeps, {n_camera}/{n_images} images "
          f"({time.time() - t_start:.0f} s)")


def write_scene(out_dir, spec, scene, truth, samples, lidar_kf, frames, kf_times, events,
                lidar_pts, radar_points):
    d = Path(out_dir) / spec["name"]
    (d / "frames").mkdir(parents=True, exist_ok=True)

    meta = {
        "name": spec["name"],
        "description": scene["description"],
        "lesson": spec.get("lesson", ""),
        "origin_global_xy": [float(v) for v in frames.origin[:2]],
        "target": {"instance_token": spec["target"], "category": truth.category,
                   "size_wlh": [float(v) for v in truth.size_wlh]},
        "keyframes": [{"sample_token": s["token"], "t_us": int(s["timestamp"]),
                       "ego_xy_yaw": [_r(v, 5) for v in frames.ego_xy_yaw(sd)]}
                      for s, sd in zip(samples, lidar_kf)],
        "ground_truth": [{k: r[k] for k in ("t_us", "xy", "yaw", "speed", "visibility",
                                             "num_lidar_pts", "num_radar_pts")}
                         for r in truth.rows],
    }
    (d / "scene.json").write_text(json.dumps(meta, indent=1))

    frame_of = lambda t: max(0, int(np.searchsorted(kf_times, t, side="right")) - 1)
    per_frame = [[] for _ in samples]
    for t, e in sorted(events, key=lambda te: te[0]):
        per_frame[frame_of(t)].append({"t_us": int(t), **e})
    points_by_frame = [[] for _ in samples]
    for t, ch, sd_token, xy, vel in radar_points:
        points_by_frame[frame_of(t)].append((sd_token, ch, xy, vel))

    for i, s in enumerate(samples):
        t_end = int(kf_times[i + 1]) if i + 1 < len(samples) else int(max(
            [kf_times[i]] + [e["t_us"] for e in per_frame[i]]))
        (d / "frames" / f"{s['token']}.json").write_text(json.dumps({
            "sample_token": s["token"], "t_start_us": int(kf_times[i]), "t_end_us": t_end,
            "events": per_frame[i]}))
        index = {e["sd_token"]: j for j, e in enumerate(per_frame[i])}
        rxy, rvel, rmeta = [np.zeros((0, 2), np.float32)], [np.zeros((0, 2), np.float32)], \
            [np.zeros((0, 2), np.int32)]
        for sd_token, ch, xy, vel in points_by_frame[i]:
            rxy.append(xy.astype(np.float32))
            rvel.append(vel.astype(np.float32))
            rmeta.append(np.tile([CHANNEL_ID[ch], index[sd_token]], (len(xy), 1)).astype(np.int32))
        np.savez_compressed(d / "frames" / f"{s['token']}.npz",
                            lidar_xyz=lidar_pts[s["token"]],
                            radar_xy=np.concatenate(rxy), radar_vel=np.concatenate(rvel),
                            radar_meta=np.concatenate(rmeta))


def write_manifest(out_dir, names, lidar_source):
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "created": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "nuscenes_version": "v1.0-mini",
        "lidar_detection_source": lidar_source,
        "camera_detector": "yolox_s (ONNX, OpenCV DNN)",
        "scenes": names,
    }
    (Path(out_dir) / "manifest.json").write_text(json.dumps(manifest, indent=1))
