# Slides brief: Sensor Fusion bootcamp (10 days), theory decks

For the agent building the theory slides. Everything below comes from the course software
(Fusion Stacker: the Kalman Reactor and Mark I), its documentation, and measurements made on the
course's data. Do not add numbers, algorithms, company claims or examples that are not here: if a
slide needs something missing, leave a placeholder and ask Jeremy.

## 0. Context

- **Program**: 10 days. Two live sessions (Day 1 kick-off, Day 7 sabotage). On the other days the
  students get a workbook (exercises in the app) and these theory slides.
- **The software**, Fusion Stacker, hosted (one lab per student):
  - **Mark I**: the students assemble a fusion pipeline from blocks (Detect, Cluster, Fit box,
    Project, To 2D, To ground, Merge, ROI, Paint, IoU match, Distance match, Fuse, BEVFusion's To BEV /
    Fuse BEV / Detect, Kalman filter) and run it on a real scene. Four example pipelines: "Merge &
    Cluster", "Box Matching", "PointPainting", "BEVFusion" (named by what they do, so the students
    guess the kind of fusion).
  - **Kalman Reactor**: 20 steps, the students write a Kalman filter and make it fuse LiDAR, RADAR
    and camera.
  - **Sabotage Learning**: a bug is hidden in the pipeline's code, the student finds and fixes it.
- **The data**: CMHT Autonomous Dataset (McMaster University, CC0 1.0). Four scenes at dusk:
  cmht-dusk-01 (a car 8 to 24 m away, it stops at one point), cmht-dusk-02 (a car at 11 to 14 m/s,
  14 to 25 m away), cmht-dusk-03 (a truck 30 to 50 m away at 23 to 28 m/s), cmht-dusk-04 (a car).
  Sensors: a Velodyne HDL-32E LiDAR (9.4 to 9.9 Hz measured on the scenes, with dropped sweeps),
  one 4D imaging RADAR (Retina-4FN, about 600 points per frame, azimuth -48 to +44 degrees
  measured), one front camera (1280 x 720, 82 x 52 degrees from K, looks 8.7 degrees below the
  LiDAR's axis, sees our hood).
- **The score everywhere**: the position RMSE. At each instant with a ground-truth label, the
  distance between the filter's estimated position and the true center of the target, on the ground;
  squared, averaged over the scene, square root. In meters.
- **Audience**: engineers (perception, ML, robotics), technically literate.
- **Writing rules**: English. No em dashes. No one-word punch sentences. No "Not X. Not Y." lines.
  Plain, concrete sentences.

Figures: `slides_figures/` holds 15 figures made from the real data (the raw CMHT frames and the
course's cache), the same ones the students see in the app under Settings (⋯) → **Data
Preparation**. Use them as they are; section 1 says which figure goes with which day. Do not redraw
them with other numbers.

Number tags: **[app]** = shown or tested in the software today; **[measured 9 Nov]** = measured on the
current version; **[verify]** = measured on 2 or 3 Nov, before the last versions: check in the app
before putting it on a slide.

---

## 1. The Data Preparation figures, and where each one goes

What the students have in the app (Settings → Data Preparation): five tabs (The dataset, The
sensors, The preparation, The detectors, The errors), 18 slides. The slides of the bootcamp should
send the students there rather than repeat it; the figures can be reused on the theory slides.

| Figure | What it shows | Key numbers [app] | Day |
|---|---|---|---|
| `dataset_frame.jpg` | Frame 1480 as downloaded: camera image, LiDAR sweep (coloured by z), RADAR points (coloured by raw Doppler) | 47,386 LiDAR points, 810 RADAR points | 1, 3 |
| `scenes.jpg` | The 4 scenes: a camera image and the paths of our car and of the target | 20.7, 15.6, 19.0, 15.6 s | 1 |
| `rig.png` | Top view: fields of view to scale (farthest point in 9 raw frames, no crop), and the sensors' positions from the calibration files | LiDAR 360°, 106 m; RADAR -48° to +44°, 114 m; camera 82°, 0.16 m ahead, 0.21 m right, 0.15 m below the LiDAR | 1, 3 |
| `timeline.png` | The samples of each sensor on their own clock, a dropped LiDAR sweep; histogram of the LiDAR intervals | median 100 ms; 2.9 % over 150 ms; longest 0.29 s | 4 |
| `odometry.png` | Our path from KISS-ICP on cmht-dusk-02 and our speed | check against the RADAR: 0.25, 0.36, 0.41, 0.73 m/s RMS (scenes 01 to 04) | 3 |
| `ground.png` | Side view before (LiDAR frame, 1.66 m up, tilted) and after (levelled on the road, ground removed) | | 3 |
| `calibration.jpg` | LiDAR points on three poles, with the given calibration and with the refined one | given: 15 to 25 px off; refined by +0.23°, -0.40°, +0.16° | 3 |
| `radar_doppler.png` | RADAR points coloured by raw Doppler, then by the compensated range rate (cmht-dusk-04, frame 1700) | static world -18.5 → 0.1 m/s; the target car 1.8 → 22.6 m/s | 4 |
| `labels.png` | Truck 32 (cmht-dusk-03, frame 3420): the downloaded label floats 3.5 m in front of the truck; the corrected label sits on its points | | 2, 8 |
| `lidar_detector.png` | TransFusion-L boxes (class, score) against the corrected labels, dusk-02 and dusk-03 | the truck scored 0.20 | 2 |
| `camera_detector.jpg` | D-FINE-X as it comes out (10 boxes, two on our hood) and after score ≥ 0.3, NMS 0.45 and the hood rule (5 boxes) | | 2 |
| `segmentation.jpg` | A camera image and Mask2Former's class map; the hood has no class | | 2 |
| `rms_mae.png` | Histogram of 19,676 RADAR range-rate errors (calibration segments) with RMS, MAE and robust σ | RMS 3.16, MAE 1.13, robust σ 0.68 m/s; the 2.5 % of points off by more than 5 m/s make 86 % of the sum of squares; without them the RMS is 1.19 m/s | 8 |
| `radar_errors.png` | Histograms of the RADAR's range error and offset across the line of sight, same points | range: mean -0.38, RMS 1.74 m; across: mean -0.26, RMS 1.07 m | 8, 9 |
| `lidar_error_range.png` | TransFusion-L box centre error against the range to the target, 4 scenes | up to 30 m (493 sweeps): RMS 0.31 m in 2D, i.e. 0.22 m per axis; beyond 30 m the truck, 1.5 to 4 m | 2, 8 |

Formulas shown in the app, which the slides can reuse:
- Projection: [u, v, 1]ᵀ ∝ K [R | t] [x, y, z, 1]ᵀ, with K = [[738.3, 0, 619.2], [0, 737.8, 371.0], [0, 0, 1]].
- Doppler: a static point reads v_r = -u · v_ego, so the range rate over the ground is ṙ = v_r + u · v_ego.
- Errors: e_k = x̂_k - x_k; bias = mean(e); RMSE = sqrt(mean(e²)); MAE = mean(|e|);
  σ_robust = 1.4826 · median|e - median(e)|.
- The LiDAR's R from a published mean error: for a 2D Gaussian error of σ per axis,
  E‖e‖ = σ·sqrt(π/2), so σ = 0.28 / sqrt(π/2) = 0.22 m.

What the preparation did, in the app's words (for a slide or a reference page):
1. Our motion: KISS-ICP (Vizzo et al., 2023), map voxels 0.8 m, points 3 to 80 m, deskew off,
   constant-velocity prior scaled by each real time step, first velocity from the RADAR's Doppler.
2. The road: RANSAC plane 4 to 20 m around the car (inliers within 8 cm), z = 0 on the road under
   the car at each sweep (dusk-03 climbs 11.8 m); ground points (under 0.3 m above the local ground)
   removed.
3. The camera: the given extrinsic turned by +0.23°, -0.40°, +0.16°, kept only because it improved
   four criteria on held-out frames.
4. The RADAR: our velocity from the RADAR itself (RANSAC, 200 tries of 3 points, inliers within
   0.25 m/s), Doppler compensated; every point kept, with its height; one detection per point.
5. The labels: height smoothed over ±0.5 s, one size per object, the box moved so its visible faces
   lie on the LiDAR points (truck 32: +3.71 m → -0.03 m).
6. The detectors, run once on Kaggle: TransFusion-L (score ≥ 0.1), D-FINE-X (score ≥ 0.3, NMS 0.45,
   hood boxes dropped), Mask2Former (hood masked).

The 80 m crop exists only for the viewer's cloud: do not present it as a sensor range.

---

## Day 1: Kick-off (live). "The 5 design decisions behind every fusion"

Backbone: Jeremy's article https://www.thinkautonomous.ai/blog/9-types-of-sensor-fusion-algorithms
(its headings: Which, Why, When, Where, How). Each decision gets a theory slide, then a live demo in
Mark I on cmht-dusk-02.

1. **Title**: Sensor Fusion: the 5 design decisions behind every fusion.
2. **The question of the bootcamp**: you have a LiDAR, a RADAR and a camera. How do you build one
   view of the world from them? Visual: the car's three sensors and what each returns.
3. **What each sensor is good at**:
   - LiDAR: position and shape. No speed. Expensive.
   - RADAR: speed (Doppler) and range, works at night and in rain. No height, a rough angle, ghosts.
   - Camera: class and direction. Very good angle, poor distance (the distance comes from where the
     box touches the road, and far away one pixel is meters).
4. **Decision 1, WHICH sensors**. Demo, each sensor alone into the Kalman filter, dusk-02: LiDAR
   detector 0.24 m, RADAR 1.95 m, camera 2.55 m [measured 9 Nov].
5. **Decision 2, WHY fuse**: competitive (two sensors confirm each other: fewer false detections,
   and a rule for disagreements), complementary (see more), coordinated (create a measurement no
   sensor has alone).
6. **Demo, competitive**: Mark I "Box Matching", Fuse row "Seen by one sensor only": Keep it 64.4 m,
   Drop it 0.96 m [measured 9 Nov]. Keeping single-sensor objects keeps the ghosts; dropping them
   loses an object only one sensor sees.
7. **Demo, coordinated**: the camera's direction and the RADAR's distance make one measurement
   ("RADAR along camera"): camera + RADAR fused in the image, 0.59 m, no LiDAR [measured 9 Nov].
8. **Decision 3, WHEN do the sensors meet**: early (raw data merged before any detection), mid
   (features meet in a shared bird's eye view grid, BEVFusion), late (each sensor detects alone, the
   objects are merged at the end). Note from the article: projecting LiDAR points into camera boxes
   is often called early fusion but is a mix of early and late.
9. **Demo, early / mid / late** on dusk-02: Merge & Cluster 0.84 m [measured 9 Nov], BEVFusion
   0.23 m [measured 9 Nov], Box Matching 64.4 m [measured 9 Nov].
10. **Decision 4, WHERE do they meet**: in the image (reuse 2D detectors, depth is ambiguous), in 3D
    (real distances, camera boxes less precise), in the feature space (inside the network, nothing
    thrown away, hard to debug).
11. **Demo, image vs ground**: camera + RADAR fused in the image (ROI) 0.59 m, on the ground
    (distance match) 1.20 m [measured 9 Nov].
12. **Decision 5, HOW is it done**: rule-based (every step written by hand, easier to certify),
    hybrid (a network understands each sensor, geometry fuses them: PointPainting), learned
    (BEVFusion, end-to-end).
13. **Demo, how**: our rule pipelines (Merge & Cluster 0.84 m), PointPainting 0.88 m (hybrid)
    [measured 9 Nov], BEVFusion 0.23 m (learned) [measured 9 Nov].
14. **No architecture wins everywhere**: dusk-01 Merge & Cluster 6.83 m vs Box Matching 1.00 m;
    dusk-02 0.84 m vs 64.4 m [measured 9 Nov]. The ranking flips from one scene to the next. You must
    test.
15. **The 10 days**: the plan (Detect, Transforms, Associate & Fuse, Exploration, Sabotage, Kalman x2,
    Portfolio) and how a workbook day works.

---

## Day 2: Detect

Conclusions the students must reach: a detector trained elsewhere makes visible mistakes on your
data; a classical clustering pipeline is sometimes the safer choice; a false detection can be
harmless or can ruin the track, and the difference is association.

1. **Two ways to get boxes**: a trained network (TransFusion-L on LiDAR, D-FINE-X on the camera,
   Mask2Former for a class per pixel) or a classical pipeline (remove the ground, cluster the points,
   fit an oriented box, guess the class from the size).
2. **The models were not trained on CMHT**: run once on Kaggle; on CMHT TransFusion-L's scores are low
   (0.2 to 0.8) and it called the dusk-02 car a truck [verify]. TransFusion-L's own speed is right at
   low speed and wrong on the highway (median error 12.4 m/s on dusk-04, 22.5 m/s on dusk-03)
   (docs/cmht.md).
3. **When clustering helps**: dusk-03, the truck at 40 m: LiDAR detector into the filter 21.5 m
   (followed 19 % of the time), cluster + fit box 6.25 m [measured 9 Nov]. TransFusion-L gives the truck
   8.6 m of length against 5.6 m labelled (tests).
4. **Clustering's own failures**: vegetation, rails and walls make false boxes: 10.9 extra boxes per
   image, 4.3 after filtering by point density, shape, height and ground contact (classical detector,
   prep). One sweep cannot tell the front from the back: 21 boxes of 31 pointed the wrong way; a round
   shape skews the angle by 40 to 45 degrees.
5. **A false detection with no impact**: the camera network (D-FINE-X) labels our own hood a "car"
   (scores 0.34 to 0.48): a box over the whole bottom of the image, 10 images on dusk-03, 4 on dusk-04
   [measured 9 Nov]. The filter's error does not change:
   the box lands 5 m from us and 27 m from the target, the association gate rejects it.
6. **A false detection with impact**: lone LiDAR boxes of roadside objects in Box Matching: the filter
   takes one at t = 2.4 s on dusk-02 and follows a fixed object while the car drives away (64.4 m)
   [measured 9 Nov].
7. **RADAR clutter**: walls, guardrails and parked cars return many points. "Filter static" removes
   them by speed, and fails when the target stops: dusk-01, RADAR into the filter 12.4 m, with Filter
   static 138.8 m [measured 9 Nov].
8. **Measuring a detector**: recall at 2 m, position error, false detections (with incomplete labels,
   false detections are only an upper bound).

---

## Day 3: Transforms

Conclusions: projecting is the inverse pose, then K, then the division by the depth; the frames
(sensor, car, scene) must be known for every piece of data; calibration limits what fusion can do.
Format: give the formula and let them code it, or give the code and let them derive the formula.

1. **The frames**: sensor, car, scene. Every array lives in one of them.
2. **The pinhole camera**: pixel = K · [R | t] · X, then divide u and v by the depth z. K holds the
   focal lengths and the image center.
3. **Our projection** (mark1/solutions/project.py): P = K · inverse(camera pose)[:3, :]; each point as
   [x, y, z, 1] times P; divide by the depth; keep the points in front of the camera (depth > 0.5 m)
   and inside the image.
4. **The same formula in KITTI**: pixel = P_rect · R0_rect · Tr_velo_to_cam · X, then divide by z. Tr
   moves the LiDAR into the camera; R0 is the stereo rectification; P holds K and camera 2's offset.
   Question for the students: what is R0 for, and why does a single-camera dataset not need it?
5. **A 3D box in the image** (box_to_2d): the envelope of the 8 projected corners; always a bit larger
   than the vehicle seen, because the near corners are closer to us.
6. **A camera box on the ground** (to_ground): the ray through the bottom of the box meets the road
   plane; sensitive to the camera's tilt; far away one pixel is meters.
7. **The RADAR has no reliable height**: its measured heights spread from -1.5 m to +6.3 m (quantiles
   5 and 95 %, dusk-02, 123,008 points) [measured 9 Nov]. In the image it is drawn as a pillar from the
   ground to 2 m, or as a point at its measured height. Setting it at 1 m does as well as the measured
   height in the merged cloud (dusk-03: 2.50 m vs 2.97 m) [measured 9 Nov].
8. **Calibration**: check it by projecting LiDAR and RADAR onto the image (poles and traffic lights
   are the best witnesses). A rotation error is constant, a position error grows close by, a lens
   error grows at the edges. The dataset's calibration put the poles' points 15 to 25 px beside the
   poles (figure `calibration.jpg`); ours turns it by +0.23°, -0.40°, +0.16° and is still off by up
   to 1 degree at the left edge. At 25 m, 1 degree is 0.4 m to the side [app, Reactor
   step Run 4].
9. **How the dataset was built**: send the students to Settings → Data Preparation (section 1
   of this brief). On a slide: `calibration.jpg` (before / after on the poles) and `rig.png`
   (positions from the calibration files).

---

## Day 4: Associate and Fuse

Conclusions: association decides more than the filter; each attribute comes from the best sensor; the
RADAR's Doppler is only the radial speed; sensors never measure at the same instant.

1. **The ways to meet**: points in a box (ROI), points painted with the image's classes (Paint), one
   cloud of both sensors (Merge), boxes paired by overlap (IoU) or by distance.
2. **The gate**: a measurement is accepted when its Mahalanobis distance d² = νᵀ S⁻¹ ν is under the
   99 % chi-square threshold of its dimension.
3. **What to do with an object only one sensor sees**: keep it (you do not miss a car the camera did
   not see, you keep the ghosts) or drop it (no ghosts, you lose objects one sensor sees). Box
   Matching on dusk-02: 64.4 m vs 0.96 m [measured 9 Nov].
4. **Fuse: each attribute from the best sensor**: position from the LiDAR, speed from the RADAR, class
   from the camera (the Fuse row: position, size, class, speed, RADAR speed, single-sensor objects).
5. **The Doppler is radial** (the key slide of the day): the RADAR measures only the speed along its
   line of sight. A car crossing in front of you has almost no radial speed. Visual: the velocity
   vector and its projection on the line of sight.
6. **Compensating our own motion**: the measured speed includes ours. Figure `radar_doppler.png`
   (dusk-04, frame 1700): before, the static world reads -18.5 m/s and the target car 1.8 m/s; after
   ṙ = v_r + u · v_ego, the world reads 0.1 m/s and the car 22.6 m/s [app].
7. **Several RADAR points on one object**: which one gives the speed: the nearest to the center, the
   mean or the median (the median resists outliers).
8. **Time**: the LiDAR, RADAR and camera never measure at the same instant. The RADAR points are moved
   to the LiDAR's instant by their own velocity (window 0.05 s); the filter predicts to each
   measurement's time, then updates.

---

## Days 5, 6, 7

To be defined with Jeremy (open exploration, live Sabotage). Material that exists:
- The 10 architecture and tracking questions: Do you fuse early or late? In the camera space or on the
  ground? How do you put a RADAR point, which has no height, into a camera image? How much do you trust
  the RADAR's position? What do you do with an object only one sensor sees? Which association
  algorithm? What happens when your vehicle passes close to another one? The sensors never measure at
  the same time: how do you fuse them? Which motion model? How do you filter out ghosts?
- Sabotage: 83 bugs over the functions of Mark I, six kinds: flip, swap, units, frame, index,
  geometry. A bug does not always show in the numbers: one in box_corners only changed the height of
  the boxes in the camera; one in fit_box even lowered the mis-associations (50 to 15). Look at the
  step's view before reading the code.

---

## Days 8 and 9: Kalman filters (the whole Kalman Reactor over two days)

Conclusions: predict trusts the motion model (Q), update trusts the measurement (R); an uncertainty is
not an error; how to choose R; RMSE vs MAE; adding a sensor can make things worse; a filter can be
wrong about its own precision.

### Day 8: Kalman Filters
1. **The state**: x = [px, py, vx, vy], its covariance P.
2. **Constant velocity**: F moves the position by v·dt; Q = σa² G Gᵀ, a random acceleration.
3. **Predict**: x = F x, P = F P Fᵀ + Q.
4. **Update**: innovation, S = H P Hᵀ + R, gain K = P Hᵀ S⁻¹, then x and P.
5. **The loop with asynchronous sensors**: predict with the real time step to each measurement, then
   update with that sensor.
6. **Q and R in action** [app, Reactor step "Noise"]: defaults (σa 3.43, R x 1): 0.21 m. σa 0.1: the
   filter trusts its model too much, 9.17 m over the scene, 17.7 m at 14.6 s. σa 10: 0.19 m. R x 10
   (2.2 m): 0.35 m.
7. **σ is an uncertainty, not an error**: the error is measured against the truth; the uncertainty is
   what the filter believes.
8. **Where R comes from**: datasheet (the noise of one point on an ideal target), a published error,
   or a measurement against the truth on data kept apart from the scored scenes. Example, the RADAR:
   the datasheet gives 0.40 m in range and 0.013 m/s in Doppler; measured as a measurement of the
   car's center: 1.74 m in range, 3.16 m/s in Doppler (RMS) [app, Reactor step Run 3; docs].
9. **RMSE vs MAE**: RMSE = sqrt(mean(e²)), MAE = mean(|e|). Same σ for clean bell-shaped noise; with
   outliers the RMSE explodes. Example: nine errors of 0.5 m/s and one of 10: RMSE 3.2, MAE x 1.25
   = 1.8. The RADAR's Doppler on the calibration segments (figure `rms_mae.png`, 19,676 points):
   RMS 3.16 m/s, MAE 1.13 m/s, robust (median-based) σ 0.68 m/s. The 2.5 % of points off by more
   than 5 m/s make 86 % of the sum of squares; without them the RMS is 1.19 m/s [app]. "MAE-based
   1.41 m/s" elsewhere in this brief is the MAE turned into a Gaussian σ (1.13 x sqrt(π/2)); say
   which one a slide shows. The LiDAR's R (0.22 m) comes from TransFusion-L's published mATE, a mean
   error (figure `lidar_error_range.png` checks it on the scenes: 0.31 m in 2D up to 30 m).

### Day 9: Advanced Kalman Filters
1. **Why an EKF**: the RADAR measures range, bearing and range rate, non linear in the state. The
   Jacobian H.
2. **The camera's direction**: a bearing only; it corrects the angle, not the distance [app, Run 4:
   0.21 m without, 0.20 m with].
3. **Adding a sensor can make it worse**: dusk-03, LiDAR alone 2.64 m, + RADAR 4.65 m with the RADAR's
   Doppler R at 3.16 m/s; with the RADAR's R computed as the LiDAR's (MAE: 1.41 m/s, 1.90 m, 1.06 m)
   1.93 m [measured 9 Nov, Reactor settings]. The Doppler is only radial: when the LiDAR already
   knows the radial speed to about 0.3 m/s, a 1 m/s Doppler adds little.
4. **When the RADAR saves you**: after 13.7 s on the Reactor's scene the LiDAR misses the car and the
   RADAR still sees it: 0.31 m on average with the RADAR, 0.35 m without [app, Run 3].
5. **A bias is not in R**: the RADAR's points sit on the car's body, about 1.25 m short of its center;
   the filter does not know it [app, Run 3].
6. **Turning: CTRV**: 0.17 m with CTRV vs 0.20 m with constant velocity, the car turns about 200
   degrees at 12 m/s [app, Run 5].
7. **NEES, is the filter honest about itself?** NEES = (x̂ - x)ᵀ P⁻¹ (x̂ - x) on the position; about 2
   on average if the filter is consistent. In Mark I's examples it ranges from 6 to 59 [measured 9 Nov].
8. **Why: correlated errors**. The filter assumes each measurement's error is independent; the LiDAR
   sees the same faces of the car for dozens of frames, so the box center is off the same way each
   time. Frame-to-frame correlation of the box error 0.77 (Merge & Cluster on dusk-01, Box Matching on
   dusk-04); the mean of 10 boxes is 0.69 m off where independent errors would give 0.24 m [measured 9
   Nov]. The filter shrinks P as if it averaged, and becomes overconfident: its gate gets too narrow and
   the right box falls out of it (Merge & Cluster on dusk-01, 6.83 m). The Reactor's "Update every N
   LiDAR scans" shows the effect.
9. **One metric lies**: a lost track on dusk-02 showed a mean NEES of 3856 while 64 m away [verify];
   RMSE alone does not say why. Look at RMSE, the share of time followed, NEES and the wrong picks.
10. **Association matters more than the filter**: the same filter on dusk-02 with the right
    measurement chosen by hand: 0.84 m; with its own choice: 64.4 m [measured 9 Nov].
11. **How industry handles box size and partial views** (optional): Autoware keeps length and width in
    the tracker's state, updates them only when the measured extent is trusted, and moves the center
    from the nearest corner when the size changes; Apollo recomputes the size each frame from the
    current points and measures the speed on several reference points (anchor point, barycenter, box
    center, each corner) and keeps the most consistent. Neither uses a fixed box size. Sources:
    Autoware multi_object_tracker docs and PR #12560; Apollo mlf_shape_filter.cc and
    mlf_motion_measurement.cc.
12. **What comes next** (outlook): multi-object tracking (Hungarian assignment, track creation and
    deletion), IMM to mix CV, CA and CTRV; Kalman filters vs factor graphs (factor graphs win for
    localising your own car; a Kalman filter is a factor graph that keeps only the last state).

---

## Day 10: Closing (portfolio)

1. What a fusion portfolio piece shows: the architecture, the 5 decisions it answers, the RMSE per
   scene, one failure explained with its measured cause, what you would change.
2. Placeholder for Jeremy: examples, format, where to publish.

---

## Open points for Jeremy before the slides are final
- Every [verify] number must be re-measured on the current version.
- The dataset construction, the calibration files and the datasheet values are now visible to
  students (Settings → Data Preparation). The Kaggle notebook is not.
- Day 9 item 5 says the RADAR's points sit about 1.25 m short of the car's centre (Reactor, Run 3);
  the calibration segments give a mean range error of -0.38 m with two groups of points (figure
  `radar_errors.png`). Both are measured, on different data: check which one the slide needs.
