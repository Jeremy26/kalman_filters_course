# Fusion Lab: design

This document records the decisions behind the tool and the cache format. The README
covers how to run it and what to try with students.

## 1. Facts measured on the data

Checked against nuScenes v1.0-mini, the devkit `splits.py` and the nuscenes.org server.

| Fact | Consequence |
|---|---|
| `v1.0-mini.tgz` is 4.17 GB; its JSON tables sit at the start of the archive | Prep reads the tables first, then extracts only the files of the chosen scenes (717 MB, 24 s) |
| Rates in scene-0061: LIDAR_TOP 19.9 Hz, each RADAR 12.7 to 13.7 Hz, each camera 11.3 to 11.7 Hz, keyframes 2 Hz | Real asynchrony exists, but only in the sweeps |
| Annotations and public LiDAR detections exist only at keyframes (2 Hz) | LiDAR updates arrive at 2 Hz; RADAR and camera use every sweep |
| Mini scenes 0103, 0553, 0796, 0916 are in the official val split; the six others are in train | Scenes are taken from val only, so detector outputs are genuine predictions |
| `detection-megvii.zip` (459 MB) on nuscenes.org holds train, val and test results in nuScenes format | Scriptable source for the LiDAR boxes |
| CenterPoint predictions are on an MIT SharePoint folder | Not scriptable. Accepted as a local file with `--lidar-detections` |
| RADAR range rate matches the ground truth radial velocity (bias 0.1 to 0.2 m/s) | Frames and ego-motion compensated velocities are handled correctly |
| RADAR clusters sit on the near face of the target: range 1.6 to 2.1 m short, bearing up to 7 deg off on a bus | RADAR noise defaults are 3.5 m and 8 deg. The point-target model is itself a lesson |

## 2. Decisions

| Topic | Decision | Why |
|---|---|---|
| LiDAR boxes | MEGVII val by default, any nuScenes-format file with `--lidar-detections` | Downloadable without login; same format as CenterPoint |
| Camera | Class only by default; optional bearing-only update | As specified, the camera never changes the estimate. The option shows what a camera can bring, and what it cannot (range) |
| Camera detector | YOLOX-s, ONNX, OpenCV DNN, run once in prep | Apache-2.0, official ONNX export, no torch. Ultralytics YOLOv8 and later are AGPL and ship `.pt` weights |
| Display | Gradio for controls, Rerun viewer embedded with `gradio_rerun` | Native timeline, zoom, layer toggles and synced time series. Gradio keeps the Q and R sliders, which Rerun does not have |
| Working frame | Global nuScenes frame shifted to the first ego position, axes not rotated | A CV model written in the moving ego frame is wrong |
| Association | Oracle gating against ground truth, done in prep | Out of scope; shown as such in the UI |
| Style | Colors and fonts taken from thinkautonomous.ai (navy to steel blue gradient, Fira Sans, Inter, blue gradient buttons) | Consistent with the course |

## 3. Filter

| Model | State | Heading |
|---|---|---|
| CV | `[px, py, vx, vy]` | derived, `atan2(vy, vx)` |
| CA | `[px, py, vx, vy, ax, ay]` | derived |
| CTRV | `[px, py, v, psi, omega]` | in the state |

Position is always first, so the ellipse is always `P[:2, :2]`.

| Step | CV / CA | CTRV |
|---|---|---|
| Predict | linear KF | EKF or UKF |
| LiDAR update | linear KF | linear KF |
| RADAR update | EKF or UKF | EKF or UKF |
| Camera bearing update (optional) | EKF or UKF | EKF or UKF |

The RADAR model `h(x)` starts from the position and heading of the radar that produced
the sweep (five radars, five origins). Angles in residuals and in the UKF means are
wrapped. `fusion.run()` merges every event of the scene, sorts it by timestamp, and for
each one predicts to its time and updates with that sensor's model. Events without a
gated detection, and camera events in class-only mode, do not touch the filter.

UKF sigma points use van der Merwe's form with alpha = 1, beta = 2, kappa = 0: the
central point has no weight in the mean and every covariance weight stays positive.

## 4. Cache format (schema 1)

```
cache/
  manifest.json
  <scene>/
    scene.json
    frames/<sample_token>.json    events whose timestamp falls in [keyframe k, keyframe k+1)
    frames/<sample_token>.npz     point clouds of that interval
  _downloads/                     archives, kept for reruns unless --delete-archives
```

`manifest.json`: `schema_version`, `created`, `nuscenes_version`,
`lidar_detection_source`, `camera_detector`, `scenes`.

`scene.json`: `name`, `description`, `lesson`, `origin_global_xy`, `target`
(`instance_token`, `category`, `size_wlh`), `keyframes` (`sample_token`, `t_us`,
`ego_xy_yaw`), `ground_truth` (`t_us`, `xy`, `yaw`, `speed`, `visibility`,
`num_lidar_pts`, `num_radar_pts`), one row per annotation.

Each event in a frame file:

```json
{"t_us": 1533151603547590, "sensor": "radar", "channel": "RADAR_FRONT",
 "sd_token": "...", "sensor_xy_yaw": [x, y, yaw],
 "detections": [{"z": [range, bearing, range_rate], "xy": [x, y], "n_points": 5}],
 "target": 0}
```

LiDAR detections carry `xy`, `yaw`, `size_wlh`, `name`, `score`. Camera detections
carry `box_xyxy`, `cls`, `score`, `bearing` and `bearing_span` (directions of the box
edges, relative to the camera heading). `target` is the index of the gated detection,
or `null`.

The npz holds `lidar_xyz` (float32, keyframe sweep, ground removed, cropped to 80 m,
0.25 m voxels, z above the ego wheels), `radar_xy`, `radar_vel` (float32, scene frame)
and `radar_meta` (int32: channel id, index of the event in the frame file).

Measured size: 22.7 MB for the three scenes.

## 5. Viewer recording

`app/rerun_log.py` builds one Rerun recording per filter configuration:

- sensor data and ground truth, precomputed once per scene as column batches;
- one or two filter runs: path, heading arrow, the 95 % ellipse at every update and
  predicted at 20 Hz in between (so it visibly grows), the ellipse just before each
  update, error, NEES with its bound, update timeline, and a state text;
- a blueprint: top-down 3D view, state, three time series, playback starting at t = 0.

Each configuration gets a new recording id. With a shared id the viewer merges
recordings, and old runs stayed on screen. A full recording takes 0.2 to 0.4 s to build
and weighs 5 to 7 MB.
