# Sensor Fusion EKF — Lidar + Radar

> **Reference project for the modernized Kalman Filters course.**
> One Extended Kalman Filter. One state. Two sensors. No object detector.

This module is the "big Kalman filter" idea made concrete: a **single** filter
holds a **single** state — a tracked vehicle's `[px, py, vx, vy]` — and it is
corrected by whichever sensor speaks next:

| Sensor | Measures | Model | Why it's here |
|--------|----------|-------|---------------|
| **Lidar** | `(x, y)` Cartesian | **linear** (`z = Hx`) | precise position |
| **Radar** | `(range, bearing, range-rate)` polar | **non-linear** (`z = h(x)`, needs a Jacobian) | velocity via Doppler, sees through occlusion |

The radar's non-linear polar model is exactly what forces the *Extended* Kalman
Filter — you linearize `h(x)` with its Jacobian at every update. That single
idea is the whole lesson.

**There is no detection here on purpose.** Measurements are given as data. This
course teaches *estimation*; 3D detectors belong to the perception course
downstream. That keeps this project GPU-free, weights-free, and Colab-trivial.

## What students produce

- A from-scratch EKF (no `filterpy`) they can read line by line.
- A **Foxglove `.mcap` recording** — a shareable, scrubbable portfolio artifact,
  not a baked-in `.mp4`.
- A **metrics report**: RMSE vs. ground truth **and** NIS consistency (the check
  that separates an engineered filter from a toy one).

## Quick start (local)

```bash
cd ekf_sensor_fusion
pip install -e ".[viz,dev]"

# 1. Generate the dataset (realistic curved trajectory + a lidar occlusion)
python data/generate_dataset.py --out data/fusion_log.txt

# 2. Run the filter, write a Foxglove recording, and print the ablation
python -m kf_fusion.run_ekf --data data/fusion_log.txt \
    --mcap outputs/ekf_fusion.mcap --ablation

# 3. Check the tests pass (RMSE + consistency gates)
pytest -q
```

## Quick start (Colab)

The whole thing runs in Colab with zero local setup — that's deliberate:

```python
!git clone https://github.com/Jeremy26/kalman_filters_course.git
%cd kalman_filters_course/ekf_sensor_fusion
!pip install -e ".[viz]" -q

!python data/generate_dataset.py --out data/fusion_log.txt
!python -m kf_fusion.run_ekf --data data/fusion_log.txt --mcap outputs/ekf_fusion.mcap --ablation

from google.colab import files
files.download("outputs/ekf_fusion.mcap")   # drag into app.foxglove.dev
```

## Visualize in Foxglove

1. Open [app.foxglove.dev](https://app.foxglove.dev) → **Open local file** →
   `outputs/ekf_fusion.mcap`.
2. Import the layout `layouts/ekf_fusion.json`.
3. Press play and watch:
   - **green** = ground truth, **blue** = the estimate, translucent blue = the
     1-σ covariance (it *grows* during the occlusion, then snaps tight);
   - yellow/red dots = incoming lidar/radar measurements;
   - the **NIS plot** riding under its 95% chi-square line = the filter proving
     it's honest about its own uncertainty.

## The experiment that makes fusion matter

Around the middle of the run the **lidar is occluded** (the target passes behind
an obstacle). Running the same timeline three ways:

| Config | Position RMSE | What happens |
|--------|---------------|--------------|
| **Fused** | **~0.32 m** | coasts on radar through the occlusion, stays locked |
| Lidar only | ~1.10 m | goes blind and drifts during the occlusion |
| Radar only | ~0.89 m | never blocked, but always noisy in bearing |

Fusion wins not because "more sensors = smaller number," but because the sensors
**cover each other's failure modes** — the real reason autonomous vehicles fuse.

## Layout

```
ekf_sensor_fusion/
├── data/
│   ├── generate_dataset.py   # simulate GT trajectory + noisy lidar/radar + occlusion
│   └── fusion_log.txt        # generated measurement log (Udacity-compatible format)
├── src/kf_fusion/
│   ├── models.py             # F, Q, lidar/radar models + radar Jacobian
│   ├── ekf.py                # the EKF (predict / update_lidar / update_radar)
│   ├── dataset.py            # log reader
│   ├── pipeline.py           # run_fusion(): the end-to-end loop + ablation flags
│   ├── metrics.py            # RMSE + NIS consistency
│   ├── viz_foxglove.py       # MCAP logging for Foxglove
│   └── run_ekf.py            # CLI
├── tests/test_ekf.py         # Jacobian check, PSD covariance, accuracy, consistency
└── layouts/ekf_fusion.json   # Foxglove layout
```

## Where this sits in the course

This is the bridge project at the end of the estimation track:

```
Bayes filter → linear KF → EKF (this) → UKF → "one big filter" capstone
                                    │
                                    └──▶ next course: N of these + data association = tracking
```
