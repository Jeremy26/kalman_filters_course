# Sensor Fusion EKF — Real Lidar + Radar on nuScenes

> **Reference project for the modernized Kalman Filters course.**
> One Extended Kalman Filter. One state. Two real sensors. Real data.

This module tracks a real vehicle by fusing **genuine lidar and radar returns
from the [nuScenes](https://www.nuscenes.org) dataset** — no simulation. A
**single** EKF holds a **single** state (the vehicle's `[px, py, vx, vy]` in the
global frame) and is corrected by whichever real sensor speaks next.

| Sensor | Real measurement used | Model | Why it's here |
|--------|-----------------------|-------|---------------|
| **Lidar** | centroid of the real LIDAR_TOP points inside the object's box → `(x, y)` | **linear** | precise, but biased to the visible face (~1.5 m) |
| **Radar** | real BEV-gated radar return → `(range, bearing, Doppler range-rate)` | **non-linear** (Jacobian) | noisier position, but gives **velocity** and sees at range |

The radar's polar model about the **moving ego vehicle** is non-linear, which is
exactly what forces the *Extended* Kalman Filter. That is the core lesson.

**Where fusion fits:** a Kalman filter *is* late (measurement-level) fusion. It
fuses (a) across **sensors** — lidar's good position with radar's good velocity —
and (b) across **time**, via the motion model. The detection/association step
("these returns are that car") is **given by the nuScenes annotation**, so 100%
of the course's attention stays on the *estimator*, not on training a detector.

## What the filter produces (that the raw data doesn't have)

The input is one noisy measurement per frame. The output is the object's full
motion state:

- **velocity** — never measured directly by lidar; recovered by fusing radar
  Doppler with the motion model;
- a **denoised** position track;
- a **1-second-ahead prediction** (drawn in Foxglove);
- calibrated **uncertainty** (the covariance).

## The results (real nuScenes tracks, committed in `data/`)

```
                       position RMSE     velocity RMSE
  track_437fe13d  (5 lidar + 22 radar over 11 s — lidar is sparse)
    LIDAR only        12.9 m              4.6 m/s     <- lidar alone is lost
    RADAR only         1.8 m              2.4 m/s
    FUSED              1.7 m              3.4 m/s     <- fusion holds the track

  track_ed634e83  (41 lidar + 41 radar — dense)
    LIDAR only         1.8 m              1.7 m/s     <- hurt by centroid bias
    RADAR only         1.3 m              0.8 m/s
    FUSED              1.4 m              1.1 m/s
```

Two honest lessons from **real** data: (1) when a sensor goes sparse, fusion is
what keeps you alive (the 437fe13d case); (2) raw lidar centroids are biased, so
more sensors ≠ automatically smaller error — you fuse for **robustness**, which
is the real reason AVs do it.

## Quick start

```bash
cd ekf_sensor_fusion
pip install -e ".[viz,dev]"

# The two real tracks are already committed, so you can run immediately:
python -m kf_fusion.run_ekf --track data/track_437fe13d.npz --ablation
python -m kf_fusion.run_ekf --track data/track_ed634e83.npz \
    --mcap outputs/ekf.mcap

pytest -q          # 6 tests, all on real data, no dataset needed
```

### Re-extracting tracks from the raw dataset (optional)

Only needed to pick different objects. Downloads ~4 GB from the **public** AWS
bucket (anonymous, no account):

```bash
python scripts/download_nuscenes.py
python -m kf_fusion.nuscenes_extract --instance ed634e83 \
    --out data/track_ed634e83.npz
```

### Colab

```python
# NOTE: the course code currently lives on a feature branch, so clone THAT branch
# (a plain clone checks out master, which does not have this project yet).
!git clone --branch claude/course-modernization-review-x6ytmr \
    https://github.com/Jeremy26/kalman_filters_course.git
%cd kalman_filters_course/ekf_sensor_fusion
!pip install -e ".[viz]" -q
!python -m kf_fusion.run_ekf --track data/track_437fe13d.npz --ablation
```

The committed tracks mean Colab needs **zero** dataset download to run the
filter. (Downloading the full nuScenes split for re-extraction also works in
Colab.)

## Visualize in Foxglove

There is **one** recording and **one** layout. Nothing else to open.

**Step 1 — get the recording.** Either use the one already in `outputs/`, or
generate it (needs the dataset, `pip install -e ".[viz,nuscenes]"`):

```bash
python scripts/download_nuscenes.py                                   # ~4 GB, one time
python scripts/make_scene_mcap.py --instance ed634e83 --out outputs/scene_ed634e83.mcap
```

**Step 2 — open it in Foxglove.**

1. Go to **[app.foxglove.dev](https://app.foxglove.dev)** (no install needed).
2. Click **Open local file** → choose **`outputs/scene_ed634e83.mcap`**
   *(this is the only file — do not look for any `ekf_*.mcap`)*.
3. Top-right layout menu → **Import from file…** → choose **`layouts/scene.json`**.
4. Press **▶ play** (bottom bar).

**Step 3 — what you're looking at.**

| Panel | Shows |
|-------|-------|
| **Left — Image** | the real camera, with **green** = the tracked car, **yellow** = other YOLO detections |
| **Right — 3D** | the real **lidar** cloud (height-colored), **red** radar points, and the **blue box** = the fused estimate tracking the car, with its velocity arrow and 1 s prediction. The view follows the ego car. |
| **Bottom — Plot** | the fused **speed** estimate over time |

If the 3D panel looks empty: make sure the layout is imported (step 3) — it sets
the view to follow the `ego` frame; without it the camera sits at the map origin,
far from the scene.

## Layout

```
ekf_sensor_fusion/
├── 02_fusion.ipynb           # THE teaching notebook (detection → the filter → fusion)
├── data/
│   ├── track_ed634e83.npz    # REAL dense track (committed, ~36 KB)
│   ├── track_437fe13d.npz    # REAL sparse-lidar track (committed, ~10 KB)
│   ├── sample_frames/        # a few real camera frames for the detection demo
│   └── nuscenes/             # raw dataset (gitignored; via download script)
├── scripts/
│   ├── download_nuscenes.py  # pull the split from public AWS (anonymous)
│   └── make_scene_mcap.py    # build the Foxglove recording
├── src/kf_fusion/
│   ├── nuscenes_extract.py   # real lidar+radar extraction from nuScenes
│   ├── detection.py          # real detectors: YOLO 2D, lidar clustering, 2D→3D, gating
│   ├── scene.py              # camera-pick single-object tracking + rich MCAP logging
│   ├── models.py             # F, Q, lidar/radar models + radar Jacobian
│   ├── ekf.py                # the EKF (predict / update_lidar / update_radar)
│   ├── dataset.py            # .npz track loader
│   ├── pipeline.py           # run_fusion(): end-to-end loop + sensor ablation
│   └── run_ekf.py            # CLI
├── tests/test_ekf.py         # 6 tests on the real tracks
└── layouts/scene.json        # the one Foxglove layout
```

## Where this sits in the course

```
Bayes filter → linear KF → EKF (this, real data) → UKF → "one big filter" capstone
                                     │
                                     └──▶ next course: N of these + data association = tracking
```
