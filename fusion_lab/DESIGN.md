# Fusion Lab: design proposal

Status: proposal, nothing implemented yet. This document fixes the project layout,
the cache format and the open decisions before any app code is written.

Fusion Lab is a local teaching tool. The student picks a nuScenes scene, toggles
LiDAR / RADAR / camera, picks a motion model and a filter (EKF or UKF), tunes Q and R
with sliders, and replays the scene in a bird's eye view with the covariance ellipse.

## 1. Facts checked before designing

These were verified against the real data (nuScenes v1.0-mini metadata, devkit
`splits.py`, nuscenes.org download server) on 2026-09-23.

| Fact | Consequence for the design |
|---|---|
| `https://www.nuscenes.org/data/v1.0-mini.tgz` is 4.17 GB, JSON tables are at the start of the archive | The first run downloads 4 GB even for 3 scenes. Extraction can be selective. |
| Sensor rates in scene-0061: LIDAR_TOP 19.9 Hz, each RADAR 12.7 to 13.7 Hz, each camera 11.3 to 11.7 Hz, keyframes 2 Hz | Real asynchrony exists, but only in the sweeps. |
| Annotations (ground truth) and detector outputs (CenterPoint, MEGVII) exist only at keyframes (2 Hz, one per `sample_token`) | LiDAR detections arrive at 2 Hz. RADAR and camera can run at ~13 and ~12 Hz from sweeps. |
| Mini split: scene-0103, 0553, 0796, 0916 belong to the official **val** split. The other six belong to **train**. | Detector outputs on train scenes are predictions on training data and look unrealistically good. We pick scenes from val only. |
| `https://www.nuscenes.org/data/detection-megvii.zip` (459 MB) contains `megvii_train.json`, `megvii_val.json`, `megvii_test.json` in nuScenes detection format | Official, scriptable fallback source for LiDAR boxes. |
| CenterPoint model zoo links point to an MIT SharePoint folder; the README refers to GitHub issue #249 for train/val/test predictions, and I could not confirm the files from here | CenterPoint cannot be downloaded reliably by a script. See decision D1. |

## 2. Scene and target selection (proposal)

Scored from the mini annotations: net heading change, LiDAR points in box, visibility
level, distance to ego, RADAR points in box. All three are val scenes.

| Scene | Target instance | Class | What it teaches |
|---|---|---|---|
| scene-0553 | `2e19253f...` | vehicle.bus.rigid | **Occlusion**: keyframes 5 to 9 have visibility < 40 % and 0 to 13 LiDAR points, then 61 to 151 points. Then a **94 deg turn** (51 to 145 deg). Then drives away to 133 m where LiDAR has 0 points while RADAR still has 3 to 7 points in the box. One scene, three lessons. |
| scene-0916 | `8a26f7d3...` | vehicle.bus.rigid | **Clean sharp turn**: heading -41 to -135 deg over keyframes 5 to 22, 17 to 20 m from ego, dense LiDAR (200 to 500 points). RADAR returns stop after keyframe 20, so the second half is LiDAR only. |
| scene-0103 | `200c440c...` | vehicle.car | **Distant object**: drives away from 19 to 86 m. LiDAR points fall from ~50 to 0 to 4, RADAR returns stop after keyframe 16. Good for watching the ellipse grow. |

Caveats, to verify once detections are filtered: these counts are points inside the
ground truth box. Whether the detector actually fires on the target at each keyframe is
unknown until `prep` runs. The final instance tokens live in `scenes.yaml` so they can
be swapped without code changes.

## 3. Project structure

```
fusion_lab/
  README.md                 how to run: prep once, then app
  DESIGN.md                 this file
  scenes.yaml               scene names, target instance tokens, optional time window
  docker-compose.yml        two services sharing ./cache: prep (run once), app
  docker/
    app.Dockerfile          python:3.11-slim + requirements-app.txt (no torch)
    prep.Dockerfile         app deps + nuscenes-devkit, scikit-learn, YOLO runtime
  requirements-app.txt      gradio, numpy, matplotlib, pyyaml
  requirements-prep.txt     nuscenes-devkit, scikit-learn, YOLO (see D3)

  kflab/                    filter library. No Gradio, no nuScenes imports. Students read this.
    motion.py               CV, CA, CTRV: f(x, dt), jacobian F(x, dt), Q(dt, params)
    measurement.py          LidarPosition (linear H), RadarPolar (h, jacobian H), CameraClass
    kf.py                   linear predict and update, a dozen lines each
    ekf.py                  EKF predict and update using the jacobians
    ukf.py                  sigma points (Merwe), unscented transform, angle wrapping
    fusion.py               event loop: sort by timestamp, predict to t, update with the event's sensor
    scene.py                load one scene from the cache into plain dataclasses
    metrics.py              position RMSE vs ground truth, NEES

  prep/                     data preparation. Separate container, runs once.
    __main__.py             CLI: python -m prep --accept-nuscenes-license
    download.py             resumable download, checksum, selective extraction of the chosen scenes
    lidar.py                crop and voxel-downsample LiDAR keyframes, filter detection JSON to scenes
    radar.py                load radar sweeps, DBSCAN per sweep and per sensor, polar measurement
    camera.py               YOLO on the camera sweeps that see the target, projected GT for gating
    gating.py               oracle gating of each detection to the target instance
    writer.py               writes the cache format below, with a schema version

  app/
    main.py                 Gradio layout and callbacks
    render.py               BEV drawing with matplotlib (Agg)

  tests/
    test_kf.py              KF against a hand-computed step
    test_ekf_ukf.py         EKF and UKF agree on a nearly linear case, angle wrap on bearing
    test_fusion.py          out-of-order events are applied in timestamp order
```

Filter code and UI are split so that `kflab` runs in a notebook with no Gradio. The
course notebooks use `filterpy`; `kflab` is written from scratch in numpy so every
equation is visible, with filterpy-like names (`x`, `P`, `F`, `H`, `Q`, `R`, `predict`,
`update`) so students recognize them.

## 4. Filter design

**Working frame.** A fixed scene frame: global nuScenes coordinates minus the ego
position at the first keyframe, axes not rotated. The ego frame moves and rotates, and a
constant velocity model written in the ego frame is simply wrong. Rendering can still be
ego-centered.

**States.** One state per model, and a common view for display.

| Model | State | Heading |
|---|---|---|
| CV | `[px, py, vx, vy]` | derived `atan2(vy, vx)`, undefined near zero speed |
| CA | `[px, py, vx, vy, ax, ay]` | derived |
| CTRV | `[px, py, v, psi, omega]` | in the state |

Position is always the first two entries, so the ellipse is always `P[:2, :2]`. Note:
the course notebooks use the interleaved order `[x, vx, y, vy]`; grouping by position
here keeps the ellipse code identical across models.

**Which filter does what.**

| Step | CV / CA | CTRV |
|---|---|---|
| Predict | linear KF | EKF (jacobian) or UKF |
| LiDAR update (position) | linear KF | linear KF (H picks px, py) |
| RADAR update (range, bearing, range rate) | EKF or UKF | EKF or UKF |
| Camera update | no state change | no state change |

The EKF / UKF toggle only changes the nonlinear steps. With CV and LiDAR only, both
settings give the same result, which is itself a lesson.

**RADAR measurement.** `h(x)` is computed from the radar sensor's own position in the
scene frame (5 radars, 5 origins), not from the ego center. nuScenes provides ego-motion
compensated velocities (`vx_comp`, `vy_comp`), so the range rate is the target's world
velocity projected on the line of sight. Bearing residuals are wrapped to [-pi, pi].

**Time ordering.** All events of a scene are merged into one list sorted by timestamp
(integer microseconds). For each event: predict from the last update time to the event
time, then update. No event is ever applied out of order, and no event is dropped for
being late since the whole scene is known offline.

**Run model.** Running the filter on a full scene is about 1000 events, a few
milliseconds in numpy. The app reruns the whole scene on every slider change and stores
the history. Playback is only rendering from that history. This keeps the filter a pure
function `run(scene, config) -> history` that students can call from a notebook.

## 5. Cache format

Everything lives in `./cache`, mounted as a volume, never in the image.

```
cache/
  manifest.json
  scene-0553/
    scene.json
    frames/
      <sample_token>.json      events and detections in [t_k, t_k+1)
      <sample_token>.npz       point clouds for that keyframe interval
  scene-0916/ ...
  scene-0103/ ...
```

Frames are keyed by keyframe `sample_token` as requested. Each frame file holds every
sensor event whose timestamp falls between that keyframe and the next one, so a frame
contains 1 LiDAR detection event, ~6 RADAR sweeps per radar and ~6 camera images. The
asynchrony is preserved inside the frame.

### manifest.json

```json
{
  "schema_version": 1,
  "created": "2026-09-23T12:00:00Z",
  "nuscenes_version": "v1.0-mini",
  "lidar_detection_source": "megvii_val.json | centerpoint_val.json",
  "camera_detector": "yolov8n | yolov3-opencv",
  "scenes": ["scene-0553", "scene-0916", "scene-0103"]
}
```

### scene.json

```json
{
  "name": "scene-0553",
  "description": "Wait at intersection, bicycle, large truck, ...",
  "origin_global_xy": [x0, y0],
  "target": {"instance_token": "...", "category": "vehicle.bus.rigid", "size_wlh": [w, l, h]},
  "sensors": {
    "RADAR_FRONT": {"id": 0, "xy_in_ego": [x, y], "yaw_in_ego": 0.0},
    "CAM_FRONT":   {"id": 5, "xy_in_ego": [x, y], "yaw_in_ego": 0.0, "hfov_rad": 1.2, "K": [[...]]}
  },
  "keyframes": [
    {"sample_token": "...", "t_us": 1533151603547590, "ego_xy_yaw": [x, y, yaw]}
  ],
  "ground_truth": [
    {"t_us": 1533151603547590, "xy": [x, y], "yaw": 0.89, "speed": 4.1,
     "visibility": "v0-40", "num_lidar_pts": 0, "num_radar_pts": 1}
  ]
}
```

Ground truth exists only at keyframes. The app draws it as dots at keyframes plus a
line between them, and computes RMSE and NEES only at keyframes, so no interpolated
value is ever presented as ground truth.

### frames/&lt;sample_token&gt;.json

```json
{
  "sample_token": "...",
  "t_start_us": 1533151603547590,
  "t_end_us": 1533151604047590,
  "events": [
    {
      "t_us": 1533151603547590, "sensor": "lidar", "channel": "LIDAR_TOP",
      "sd_token": "...", "sensor_xy_yaw": [x, y, yaw],
      "detections": [
        {"xy": [x, y], "yaw": 0.9, "size_wlh": [2.9, 11.2, 3.5], "name": "bus", "score": 0.71}
      ],
      "target": 0
    },
    {
      "t_us": 1533151603560112, "sensor": "radar", "channel": "RADAR_FRONT",
      "sd_token": "...", "sensor_xy_yaw": [x, y, yaw],
      "detections": [
        {"z": [range_m, bearing_rad, range_rate_mps], "xy": [x, y], "n_points": 5}
      ],
      "target": null
    },
    {
      "t_us": 1533151603612404, "sensor": "camera", "channel": "CAM_FRONT",
      "sd_token": "...", "sensor_xy_yaw": [x, y, yaw],
      "detections": [
        {"box_xyxy": [x1, y1, x2, y2], "cls": "bus", "score": 0.83, "bearing_span_rad": [b1, b2]}
      ],
      "target": 0
    }
  ]
}
```

`target` is the index of the detection gated to the target instance, or `null`. All
detections are kept for display; only the target one feeds the filter. JSON is used on
purpose for these files: they are small and students can open them.

`bearing_span_rad` is the horizontal extent of the camera box turned into two bearings
with the intrinsics. It lets the BEV draw the camera detection as a wedge from the
camera, since a 2D image box has no BEV position.

### frames/&lt;sample_token&gt;.npz

| Array | dtype | shape | content |
|---|---|---|---|
| `lidar_xyz` | float16 | (N, 3) | keyframe LiDAR sweep in scene frame, cropped to 60 m, voxel 0.2 m |
| `radar_xy` | float32 | (M, 2) | all raw radar points of the interval, scene frame |
| `radar_vel` | float32 | (M, 2) | `vx_comp`, `vy_comp` rotated to scene frame |
| `radar_meta` | int32 | (M, 2) | sensor id, event index in the JSON |

Only keyframe LiDAR sweeps are stored (40 per scene). Storing all 20 Hz sweeps would
multiply the size by 10 for no filter benefit, since LiDAR boxes exist only at 2 Hz.
Rough budget: a few MB per scene; to be measured in prep.

### Gating

Association is out of scope, so gating is done in `prep` against ground truth (oracle
gating), and documented as such in the UI:

- LiDAR box: center within 2 m of the target GT center at that keyframe.
- RADAR cluster: inside the target GT box grown by 1 m, with GT linearly interpolated
  between keyframes (the only place interpolation is used, and only to gate).
- Camera box: IoU > 0.3 with the projected, interpolated GT box.

## 6. Display

- BEV, ego-centered, fixed 100 m window, zoom slider.
- LiDAR points grey, RADAR points colored by sensor with a short velocity tick.
- Detection boxes: LiDAR rectangles, RADAR cluster markers, camera wedges. The gated
  detection is highlighted.
- Ground truth trajectory, estimated trajectory.
- Covariance ellipse: filled, thick outline, 95 % (chi-square 2 dof, 5.99), redrawn at
  every event, colored by the sensor that last updated the state. Optional ghost of
  the predicted ellipse just before the update, so the shrink at an update is visible.
- Side panel: state vector with names and units, sqrt of the diagonal of P, last sensor,
  time since last update, current camera class.
- Bottom strip: position error and NEES over time, with the 95 % NEES band.

Rendering with matplotlib into a `gr.Image`, one frame per tick of a `gr.Timer`, plus a
time slider for scrubbing.

## 7. Docker and startup

- `app` image: `python:3.11-slim` plus gradio, numpy, matplotlib, pyyaml. No torch, no
  nuScenes devkit, no CUDA. It refuses to start with a clear message if `cache/` is
  missing or has the wrong `schema_version`.
- `prep` image: adds nuscenes-devkit, scikit-learn and the YOLO runtime. Run once:
  `docker compose run --rm prep --accept-nuscenes-license`.
- Download: resumable, to `cache/_downloads/`, then selective extraction of only the
  chosen scenes' files, then optional deletion of the 4 GB archive (`--keep-archive`).

## 8. Decisions needed before writing code

**D1. LiDAR detection source.** CenterPoint predictions are hosted on SharePoint and I
could not find a stable scriptable URL. Proposal: `prep` accepts
`--lidar-detections path/to/centerpoint_val.json` if the instructor downloads the file
once, and otherwise downloads `megvii_val.json` from nuscenes.org automatically. Both
are in the same nuScenes detection format, so the parser is shared.

**D2. Camera contributes nothing to the state.** As specified, toggling the camera
never changes the trajectory or the ellipse, which students may read as a bug. Options:
(a) keep it as specified, camera shows the class and a wedge only; (b) add an optional
camera **bearing-only** update (azimuth of the box center), a classic nonlinear EKF case
that makes the camera visibly useful at long range. Recommendation: (b) as a toggle that is off by
default, keeping (a) as the default behavior.

**D3. YOLO runtime in prep.** (a) Ultralytics YOLOv8n with CPU torch: simple, but
AGPL-3.0 and a heavy prep image. (b) OpenCV DNN, like `yolo_for_tracking_2.py` in this
repo: no torch, consistent with the course, but needs a model file whose download link
must be checked. Either way it runs only in `prep`.

**D4. Licensing.** nuScenes is released under CC BY-NC-SA 4.0 for non-commercial use.
Not shipping the data in the image is right, and `prep` makes each user accept the
terms. Whether a paid course counts as commercial use is a question for the course
owner to confirm with Motional, not something the code can settle.

**D5. Location.** This proposal puts everything under `fusion_lab/` in this repo, next
to the existing notebooks. A separate repository is also reasonable.
