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
!git clone https://github.com/Jeremy26/kalman_filters_course.git
%cd kalman_filters_course/ekf_sensor_fusion
!pip install -e ".[viz]" -q
!python -m kf_fusion.run_ekf --track data/track_437fe13d.npz --ablation
```

The committed tracks mean Colab needs **zero** dataset download to run the
filter. (Downloading the full nuScenes split for re-extraction also works in
Colab.)

## Visualize in Foxglove

1. Open [app.foxglove.dev](https://app.foxglove.dev) → **Open local file** →
   `outputs/ekf_ed634e83.mcap`.
2. Import the layout `layouts/ekf_fusion.json`.
3. Press play and watch, in the map frame:
   - **white** dots = the real lidar points on the car;
   - **yellow/red** = the incoming lidar/radar measurement;
   - **blue** path + translucent sphere = the estimate and its 1-σ covariance;
   - **orange** = the velocity arrow and the 1-second-ahead prediction;
   - **green** = ground truth;
   - the **NIS plot** under its 95% chi-square line = the filter proving it's
     honest about its own uncertainty.

## Layout

```
ekf_sensor_fusion/
├── data/
│   ├── track_ed634e83.npz    # REAL dense track (committed, ~36 KB)
│   ├── track_437fe13d.npz    # REAL sparse-lidar track (committed, ~10 KB)
│   └── nuscenes/             # raw dataset (gitignored; via download script)
├── scripts/download_nuscenes.py
├── src/kf_fusion/
│   ├── nuscenes_extract.py   # real lidar+radar extraction from nuScenes
│   ├── models.py             # F, Q, lidar/radar models + radar Jacobian
│   ├── ekf.py                # the EKF (predict / update_lidar / update_radar)
│   ├── dataset.py            # .npz track loader
│   ├── pipeline.py           # run_fusion(): end-to-end loop + sensor ablation
│   ├── metrics.py            # RMSE + NIS consistency
│   ├── viz_foxglove.py       # MCAP logging for Foxglove
│   └── run_ekf.py            # CLI
├── tests/test_ekf.py         # 6 tests on the real tracks
└── layouts/ekf_fusion.json   # Foxglove layout
```

## Where this sits in the course

```
Bayes filter → linear KF → EKF (this, real data) → UKF → "one big filter" capstone
                                     │
                                     └──▶ next course: N of these + data association = tracking
```
