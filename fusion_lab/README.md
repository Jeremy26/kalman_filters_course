# Fusion Lab

A teaching tool for Kalman filtering and multi-sensor fusion on real driving data
(nuScenes mini). Pick a scene, switch LiDAR, RADAR and camera on or off, choose a
motion model and a filter, mistune Q and R, and replay the scene in a bird's eye view
where the covariance ellipse is the main character.

Runs on a laptop CPU, in Docker, in the browser. No GPU and no CUDA.

## Run it

```bash
# 1. Once: download nuScenes mini (4.2 GB) and public detections, build the cache (~23 MB).
#    Takes about 5 minutes after the downloads, most of it running YOLOX on the CPU.
docker compose run --rm prep --accept-nuscenes-license

# 2. Start the app, then open http://localhost:7860
docker compose up app
```

The downloads stay in `cache/_downloads` so a rerun does not fetch them again. Add
`--delete-archives` to the prep command to free the 4.6 GB once the cache is built.

Without Docker (Python 3.11):

```bash
pip install -r requirements-prep.txt && python -m prep --accept-nuscenes-license
pip install -r requirements-app.txt && pip install --no-deps gradio_rerun==0.38.1
python -m app
```

The two steps need separate environments: `nuscenes-devkit` pins numpy below 2 and
`rerun-sdk` needs numpy 2.

## What is in the cache

| Scene | Target | What it shows |
|---|---|---|
| scene-0553 | bus | Occlusion, then a sharp turn, then the bus drives beyond LiDAR range while RADAR still sees it |
| scene-0916 | bus | A clean 94 deg turn close to the ego car: compare CV and CTRV |
| scene-0103 | car | A car driving away from 19 to 86 m while the sensors fade out |

All three come from the official nuScenes **val** split, so the public detector outputs
on them are real predictions, not predictions on training data.

| Sensor | Source | Rate | What the filter gets |
|---|---|---|---|
| LiDAR | MEGVII 3D boxes (public nuScenes detection results) | 2 Hz (keyframes) | position `[px, py]` |
| RADAR | raw nuScenes radar points, DBSCAN per sweep and per radar | about 13 Hz per radar | `[range, bearing, range_rate]` in that radar's frame |
| Camera | YOLOX-s (ONNX, OpenCV DNN, run once in prep) | about 12 Hz per camera | class only, or bearing if enabled |

Detections are gated to the target with ground truth. Association is out of scope, and
the app says so.

## Code layout

```
kflab/          the filters. Pure numpy, no UI. This is what students read.
  motion.py       CV, CA, CTRV: f, jacobian F, process noise Q
  measurement.py  LiDAR (linear), RADAR (polar, EKF/UKF), camera bearing
  kf.py           linear Kalman filter
  ekf.py          extended Kalman filter
  ukf.py          unscented Kalman filter
  fusion.py       the event loop: sort by time, predict to t, update with that sensor
  metrics.py      position error and NEES against ground truth
  scene.py        loads a prepared scene
prep/           data preparation (separate image): download, extract, detect, gate, write
app/            Gradio controls + Rerun viewer
tests/          filter tests (no data needed) and a recording smoke test (uses the cache)
```

`kflab` works without the app, for example in a notebook:

```python
from kflab import Config, load_scene, run, evaluate
scene = load_scene("cache", "scene-0916")
track = run(scene, Config(model="CTRV", nonlinear="UKF"))
print(evaluate(scene, track).rmse)
```

## Things to try with students

Numbers below are with the default tuning, measured on the prepared cache.

1. **LiDAR only, then add RADAR** on scene-0553: 6.05 m RMSE with LiDAR alone, 2.33 m
   with both. RADAR fills the 2 Hz gaps and keeps the track alive past 100 m, where
   LiDAR returns nothing.
2. **CV against CTRV** on scene-0916 (use "Also show"): 0.63 m against 0.44 m. The gain
   is real but modest, and comes from the turn.
3. **EKF against UKF** with CTRV on scene-0553: 4.33 m against 2.25 m, and the UKF's
   NEES is half the EKF's.
4. **Mistune with the presets.** "Q too small" on scene-0553: RMSE 25 m and a mean NEES
   above 3000, the filter is sure of itself and wrong. "R too small": the RMSE barely
   moves (2.5 m) but the NEES jumps to 1700. The error alone does not reveal an
   overconfident filter; the NEES does. "Q too large" is not always worse on RMSE
   (2.07 m on scene-0553): it follows the turn better at the cost of a jumpier estimate.
5. **Camera bearing on.** The ellipse narrows across the line of sight only, since a
   camera gives no range. It helps on scene-0553 (2.33 to 1.99 m) but hurts on
   scene-0916 (0.63 to 0.73 m, NEES 4.5 to 20.8): 1 deg is probably too optimistic for the center
   of a 12 m bus seen from 20 m. Ask students to find a better value.
6. **Why the RADAR defaults are so large.** A radar cluster sits on the near face of
   the vehicle, not at its center: about 1.8 m short in range on the car of
   scene-0103, and around 7 deg off in bearing on the bus of scene-0916. Set RADAR to
   0.8 m and 2 deg: the mean NEES goes above 150 on both bus scenes, and on scene-0916
   adding RADAR triples the error (2.35 m against 0.76 m for LiDAR alone). The
   measurement model assumes a point target.

With the defaults, the filter is consistent on scene-0103 (mean NEES 1.2), a bit
overconfident on scene-0916 (4.5) and clearly overconfident on scene-0553 (27), where
the bus turns while occluded and is then tracked by RADAR alone at long range. That is
worth discussing rather than hiding with more tuning.

## Known limits

- Ground truth exists only at the 2 Hz annotations. Error and NEES are computed there.
- Detections are gated with ground truth (oracle gating).
- The LiDAR boxes come from MEGVII, because its results are downloadable from
  nuscenes.org without a login. A CenterPoint file in the same nuScenes format can be
  passed with `python -m prep --lidar-detections centerpoint_val.json`.
- The app image is about 1.2 GB, mostly the Rerun native bindings and pyarrow.
- nuScenes is CC BY-NC-SA 4.0, for non-commercial use. The images never contain the
  data: each user downloads it and accepts the terms with `--accept-nuscenes-license`.

See [DESIGN.md](DESIGN.md) for the cache format and the design decisions.
