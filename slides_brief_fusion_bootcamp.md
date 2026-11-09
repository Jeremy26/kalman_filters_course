# Sensor Fusion bootcamp: slides and exercises, day by day

For the agent building the theory slides and the daily workbooks. Everything here comes from the
course software (Fusion Stacker) and from runs made on it. Do not add numbers, algorithms or claims
that are not here: if something is missing, leave a placeholder and ask Jeremy.

How to read each day:
- **What they must be able to say** at the end of the day: 3 to 5 sentences. The slides and the
  exercises exist to get them there.
- **Slides**: the theory deck of the day, slide by slide.
- **Exercises**: what the student does in the app. Each one goes predict, run, compare, explain.
  "Answer" is the result the student should find, for Jeremy and for the workbook's answer key; it
  never goes in the student's version.
- **To present**: what the student brings to the breakout room or posts in the community.

Number tags: **[app 9 Nov]** = run on the current version of the app on 9 Nov; **[app]** = shown in
the app's own texts; **[verify]** = older, check in the app before using it.

Writing rules for slides and workbooks: English. No em dashes. No one-word punch sentences. No "Not
X. Not Y." lines. Plain, concrete sentences.

---

## The app in one page (for the agent)

The students have Fusion Stacker in their browser. Three things in it:

**Mark I** builds a fusion pipeline from blocks, one row per sensor, then runs it on a scene.
- Top right: **Examples** (four ready pipelines: Merge & Cluster, Box Matching, PointPainting,
  BEVFusion) and the **scene menu** (cmht-dusk-01 to 04).
- Blocks: Detect (a network), Segment, Filter static (min speed), Cluster (gap), Fit box, Project,
  To 2D, To ground, Merge, ROI, Paint, IoU (min IoU), Distance (max distance), Fuse, To BEV,
  Fuse BEV, Kalman filter. LiDAR Detect has a **min score** (default 0.1).
- The **Fuse** card has choices: Position, Size, Class (which sensor first), Speed (From the RADAR
  or None), RADAR speed (point nearest the center, average, median), **Seen by one sensor only**
  (Keep it or Drop it).
- The **Project** card has "RADAR in the image": Pillar or Points.
- The **Kalman filter** card: motion model (Constant velocity, Constant acceleration, CTRV), and
  the noise: Q (acceleration σa: 0.5, 1, 2, 4, 8 m/s², default 2) and R for the LiDAR, the RADAR
  and the camera (x 0.25, 0.5, 1, 2, 4, 10).
- After **Run**: Position RMSE, the share of time the target is followed, the speed error, the
  wrong picks.
- **From Scratch** (blue switch): the student writes the 9 fusion functions themselves (project,
  box_to_2d, to_ground, merge_points, roi, paint, box_iou_match, box_distance_match, fuse).
- **Sabotage Learning** (red, after a Run): one step gets a realistic bug; the student finds it and
  fixes it.

**The Kalman Reactor**: 20 steps in which the student writes a Kalman filter and makes it fuse the
three sensors, on cmht-dusk-04 (and cmht-dusk-02 for the turning model). Its run step has three
sliders: Q (σa), R of the LiDAR (as a factor) and "Update every N LiDAR scans".

| # | Step | # | Step |
|---|---|---|---|
| 1 | LiDAR data | 11 | Camera class |
| 2 | RADAR data | 12 | Run 2: LiDAR + camera class |
| 3 | Camera data | 13 | RADAR setup |
| 4 | Filter setup | 14 | Update (extended), code |
| 5 | Motion model: F and Q, code | 15 | Expected RADAR measurement, code |
| 6 | Predict, code | 16 | Run 3: + RADAR |
| 7 | Update (linear), code | 17 | Camera direction, code |
| 8 | Main loop, code | 18 | Run 4: + camera direction |
| 9 | Run 1: LiDAR | 19 | Turning model (CTRV), code |
| 10 | Noise: Q and R | 20 | Run 5: CTRV |

**Settings (⋯) → Data Preparation**: a reference page on how the dataset was prepared. It is not
a day of the course; the workbooks can point to it when a student asks where the data comes from.

**The data**: CMHT Autonomous Dataset (McMaster University, CC0 1.0), four scenes at dusk.

| Scene | Target | What happens |
|---|---|---|
| cmht-dusk-01 | a car 9 to 24 m ahead | it stops at a red light for 11 s, then turns left |
| cmht-dusk-02 | a car 14 to 26 m ahead | a loop ramp, a sharp curve |
| cmht-dusk-03 | a truck 30 to 50 m ahead | highway, the truck keeps our speed |
| cmht-dusk-04 | a car | it overtakes us on the left |

**The score everywhere**: the Position RMSE, in meters: at each instant, the distance between the
filter's estimate and the true center of the target; squared, averaged over the scene, square root.

---

## Day 1: Kick-off (live). "The 5 design decisions behind every fusion"

**What they must be able to say**
1. Every fusion answers five questions: which sensors, why fuse them, when they meet, where they
   meet, how it is done.
2. Early, mid and late fusion are about when the sensors meet, and I can tell which one a pipeline
   is by looking at it.
3. No architecture wins on every scene: I have to test.

**Slides** (backbone: Jeremy's article
https://www.thinkautonomous.ai/blog/9-types-of-sensor-fusion-algorithms, headings Which, Why,
When, Where, How; each decision gets one theory slide, then a live demo in Mark I on cmht-dusk-02)
1. Title: Sensor Fusion: the 5 design decisions behind every fusion.
2. The question of the bootcamp: a LiDAR, a RADAR and a camera. How do you build one view of the
   world from them?
3. What each sensor gives: LiDAR, position and shape, no speed; RADAR, speed along its line of
   sight and range, no height, many false points; camera, class and direction, poor distance.
4. **Which**. Demo, each sensor alone into the Kalman filter, dusk-02: LiDAR detector 0.24 m,
   RADAR 1.95 m, camera 2.55 m [app 9 Nov].
5. **Why**: competitive (two sensors confirm each other), complementary (see more), coordinated
   (make a measurement no sensor has alone).
6. Demo, competitive: Box Matching, Fuse card, "Seen by one sensor only": Keep it 64.4 m, Drop it
   0.96 m [app 9 Nov].
7. Demo, coordinated: the camera's direction and the RADAR's distance as one measurement, fused in
   the image: 0.59 m with no LiDAR [verify].
8. **When**: early (raw data merged before detection), mid (features meet in a bird's eye view
   grid: BEVFusion), late (each sensor detects alone, objects merged at the end).
9. Demo: Merge & Cluster 0.84 m, BEVFusion 0.23 m, Box Matching 64.4 m on dusk-02 [app 9 Nov].
10. **Where**: in the image, in 3D on the ground, in the feature space.
11. Demo: camera + RADAR in the image 0.59 m, on the ground 1.20 m [verify].
12. **How**: rule-based, hybrid (PointPainting), learned (BEVFusion).
13. Demo: PointPainting 0.88 m, BEVFusion 0.23 m [verify for PointPainting].
14. No architecture wins everywhere: Merge & Cluster 6.83 m on dusk-01 and 0.84 m on dusk-02;
    Box Matching 64.4 m on dusk-02 [app 9 Nov].
15. The 10 days and how a workbook day works.

**Live exercise** (5 minutes, in the chat)
- Jeremy opens the four Examples one by one without saying their kind. The students type "early",
  "mid" or "late" for each before the answer. Answer: Merge & Cluster early, BEVFusion mid, Box
  Matching late. PointPainting paints the LiDAR points with the camera's classes before any box is
  made, but it needs the camera's network first: a good discussion point.

**To present**: nothing; the first workbook starts on Day 2.

---

## Day 2: Detect

**What they must be able to say**
1. A detector trained on another dataset makes visible mistakes on mine, and its score threshold
   can decide everything.
2. A classical pipeline (cluster, fit a box) is sometimes safer than a network, and sometimes
   worse.
3. A false detection can be harmless or can ruin the track; what decides is whether the filter
   takes it.
4. Removing static RADAR points removes the clutter, and also a car that stops.

**Slides**
1. Two ways to get boxes: a trained network (TransFusion-L on the LiDAR, D-FINE-X on the camera,
   Mask2Former for a class per pixel) or a classical pipeline (remove the ground, cluster the
   points, fit an oriented box).
2. The networks were trained on other datasets (nuScenes, COCO, Cityscapes) and run once on ours.
   On CMHT TransFusion-L's scores are low: the truck of dusk-03 gets 0.1 to 0.3 and is guessed
   8.6 m long instead of 5.6 m [app].
3. The score threshold: what it trades (fewer false boxes, more missed objects).
4. Clustering: how it works (points closer than a gap belong together), and what it gets wrong
   (vegetation, rails and walls make boxes; one sweep cannot tell the front of a car from its
   back).
5. A false detection: harmless when the filter's gate rejects it, harmful when the filter takes
   it.
6. RADAR clutter and Filter static: a wall reads 0 m/s once our own speed is removed, so does a
   car that stops.
7. How to measure a detector: recall, position error, false detections (with incomplete labels,
   an upper bound).

**Exercises** (all in Mark I)

*E1. The threshold.* Clear the pipeline, build LiDAR → Detect → Kalman filter, scene cmht-dusk-02.
- Predict: Detect's min score goes from 0.1 to 0.3. Better or worse?
- Run both and write the Position RMSE and the share of time followed.
- Answer: 0.1 gives 0.24 m, followed 100 %; 0.3 gives 90.9 m, followed 11 % [app 9 Nov]. The
  network gives low scores on this dataset: at 0.3 the target is often gone, and the filter
  follows something else.

*E2. Network or clustering.* Same pipeline on cmht-dusk-03 (the truck), then replace Detect with
Cluster → Fit box.
- Predict which one follows the truck better.
- Answer: Detect 21.5 m, followed 19 %; Cluster + Fit box 6.25 m, followed 87 % [app 9 Nov].
  Then the same two on cmht-dusk-02: Detect 0.24 m, Cluster + Fit box 64.4 m [app 9 Nov]. Each
  wins on one scene.
- Explain: look at the truck's boxes in the viewer on dusk-03. Why does the network miss it?

*E3. A false detection that ruins the track.* Load the example Box Matching on cmht-dusk-02. Fuse
card, "Seen by one sensor only": Keep it, then Drop it.
- Predict the difference.
- Answer: Keep it 64.4 m (followed 55 %, 50 wrong picks), Drop it 0.96 m (followed 99 %) [app 9
  Nov]. With Keep it, a lone LiDAR box on a roadside object is taken at about 2.4 s and the
  filter follows a fixed object while the car drives away.
- Find the instant in the viewer where the estimate leaves the car.

*E4. A false detection that changes nothing.* Camera → Detect on cmht-dusk-03 or 04: look at the
images where a box covers our own hood.
- Why does the filter not care? Answer: the box lands about 5 m from us and 27 m from the target;
  the gate rejects it [verify].

*E5. Filter static.* RADAR → Kalman filter, then RADAR → Filter static → Kalman filter, on
cmht-dusk-01 (the car that stops at a red light), then on cmht-dusk-02.
- Predict: removing static points should help. Does it?
- Answer: dusk-01: 12.4 m without, 138.8 m with Filter static (followed 1 %); dusk-02: 1.95 m
  both ways [app 9 Nov]. When the target stops, Filter static deletes it.

**To present**: one screenshot of a detection mistake that changed the RMSE and one that did not,
with one sentence each on why.

---

## Day 3: Transforms

**What they must be able to say**
1. Projecting a point into the image is: move it into the camera's frame, multiply by K, divide by
   the depth.
2. Every array lives in one frame (sensor, car, scene), and I must know which.
3. A camera box can be put on the ground, but the farther it is, the rougher.
4. Calibration limits what fusion can do, and I can check it by eye.

**Slides**
1. The frames: sensor, car, scene.
2. The pinhole camera: pixel = K · [R | t] · X, then divide u and v by the depth.
3. Our projection (the course's project.py): P = K · inverse(camera pose)[:3, :]; each point as
   [x, y, z, 1] times P; divide by the depth; keep the points in front of the camera and inside
   the image.
4. The same formula in KITTI: pixel = P_rect · R0_rect · Tr_velo_to_cam · X, then divide by z.
   Tr moves the LiDAR into the camera, R0 is the stereo rectification, P holds K and the camera's
   offset. Question: what is R0 for, and why does a single-camera dataset not need it?
5. A 3D box in the image (To 2D): the rectangle around the 8 projected corners.
6. A camera box on the ground (To ground): the ray through the bottom of the box meets the road;
   far away, one pixel is meters.
7. The RADAR has no reliable height: in the image, a pillar from the ground to 2 m, or a point at
   its measured height.
8. Calibration: project the LiDAR onto the image; poles and traffic lights show the error. A
   rotation error is the same everywhere, a position error grows close by. The dataset's
   calibration put the poles' points 15 to 25 pixels beside them; ours is turned by less than half
   a degree and is still up to 1 degree off at the left edge (at 25 m, 1 degree is 0.4 m) [app].

**Exercises**

*E1. Formula to code.* In Mark I, switch on From Scratch and open project.py. The workbook gives
the formula (slide 2) and nothing else.
- Write project(). Run its tests (Run in the editor: "n of m tests pass").
- Load PointPainting and Run: the points must fall on the objects in the camera view.

*E2. Code to formula.* The workbook gives KITTI's projection code (P2 · R0_rect · Tr_velo_to_cam).
- Write the formula it computes. Point to the line that does the division by the depth.
- Compare with your project(): which matrix of KITTI has no equivalent in ours, and why?

*E3. The camera alone on the ground.* Camera → Detect → To ground → Kalman filter on cmht-dusk-02.
- Predict the RMSE compared with the LiDAR's 0.24 m.
- Answer: 2.55 m [app 9 Nov]. Look at the boxes on the ground in the viewer: where are they wrong,
  near or far?

*E4. Pillar or Points.* A pipeline with the RADAR projected into the image (Project card, "RADAR
in the image"). Switch Pillar and Points and look at the camera view.
- Which one sits on the car? The RMSE of both is not measured yet: measure it in the app before
  the day.

*E5. Check the calibration.* In the camera view of cmht-dusk-01 (the red light), look at the LiDAR
points on the poles.
- Are they on the poles everywhere in the image? Where is the error largest?

**To present**: your project() passing its tests, and the formula KITTI's code computes, written by
hand.

---

## Day 4: Associate and Fuse

**What they must be able to say**
1. The RADAR measures only the speed along its line of sight, and that speed can save a track or
   hurt it.
2. Which sensor gives which attribute is a decision, and it changes the result.
3. What to do with an object only one sensor sees is a trade: ghosts against missed objects.
4. The sensors never measure at the same instant.

**Slides**
1. The ways to meet: points in a box (ROI), points painted with the image's classes (Paint), one
   cloud of both sensors (Merge), boxes paired by overlap (IoU) or by distance.
2. The gate: a measurement is accepted when d² = νᵀ S⁻¹ ν is under the 99 % chi-square threshold.
3. Fuse: each attribute from the sensor that measures it best (position from the LiDAR, speed from
   the RADAR, class from the camera).
4. **The Doppler is radial** (the key slide): the RADAR measures only the speed along its line of
   sight. A car crossing in front of you has almost none. Visual: the velocity vector and its
   projection on the line of sight.
5. Our own motion: the raw Doppler includes our speed; on cmht-dusk-04 the static world reads
   -18.5 m/s before compensation and 0.1 m/s after, the target car 1.8 m/s then 22.6 m/s [app].
6. Several RADAR points on one object: nearest to the center, average or median.
7. Time: each sensor has its own clock; the RADAR points are moved to the LiDAR's instant with
   their own speed (window 0.05 s); the filter predicts to each measurement's time.

**Exercises** (Mark I, example Merge & Cluster: LiDAR and RADAR merged)

*E1. Does the RADAR's speed help?* Fuse card, Speed: From the RADAR, then None. Scene cmht-dusk-02,
then cmht-dusk-04.
- Predict for each scene.
- Answer: dusk-02 (the curve): From the RADAR 0.84 m, None 92.4 m (followed 30 %). dusk-04 (the
  overtaking car, straight): From the RADAR 1.05 m, None 0.88 m [app 9 Nov]. The same sensor saves
  one scene and makes the other a bit worse.
- Explain with the line of sight: on dusk-04 the car is beside us part of the time.

*E2. Which RADAR points give the speed.* Fuse card, RADAR speed: nearest, average, median, on
cmht-dusk-04.
- Answer: 1.05 m, 1.03 m, 0.94 m [app 9 Nov]. On dusk-02 all three give 0.84 to 0.87 m.
- Why would the median resist better?

*E3. Ghosts against missed objects.* Box Matching, cmht-dusk-02, "Seen by one sensor only".
Already run on Day 2 (64.4 m against 0.96 m): this time, write which kind of object Drop it would
lose, and on which scene that would hurt.

*E4. Write the meeting yourself.* From Scratch: write box_distance_match() (or box_iou_match()).
Run its tests, then load Box Matching and compare the RMSE with the course's code.

**To present**: the table of E1 (two scenes, two settings) and two sentences on why the Doppler
helps on one and not the other.

---

## Days 5 and 6: Open exploration (proposal, to confirm with Jeremy)

**What they must be able to say**
1. I designed a fusion pipeline for a reason, and I can defend it with numbers on four scenes.
2. I know on which scene it fails and why.

**Slides**: short. The 10 architecture questions as a menu: early or late; in the camera space or
on the ground; how to put a RADAR point, with no height, into an image; how much to trust the
RADAR's position; what to do with an object only one sensor sees; which association; what happens
when we pass close to another vehicle; how to fuse sensors that never measure at the same time;
which motion model; how to filter out ghosts.

**Exercise**
1. Pick one question. Write your prediction in one sentence.
2. Build two pipelines in Mark I that answer it differently.
3. Run both on the four scenes. Fill a 2 x 4 table of Position RMSE and share of time followed.
4. Pick the scene where your choice fails and find the instant in the viewer.

**To present**: the table, the screenshot of the failure, three sentences: what you chose, where
it fails, what you would change.

---

## Day 7: Sabotage (live)

**What they must be able to say**
1. A bug can hide behind good numbers: look at what each step gives before reading code.
2. I can find a broken step from its view, then fix it.

**Live format**
1. Jeremy runs Sabotage Learning on a pipeline on screen; the students say which step looks wrong
   from the viewer only, before any code is opened.
2. Then each student runs Sabotage Learning on their own pipeline from Days 5 and 6, finds the
   step, fixes it, and Verify must pass.

**Slides**: the six kinds of bugs (flip, swap, units, frame, index, geometry), with one picture
each of what it does to a view; the method: look at the view, then the numbers, then the code.

**To present**: the bug you found, the view that gave it away, the fix (one line of code).

---

## Day 8: Kalman Filters (Reactor steps 1 to 10, then Mark I's Kalman filter card)

**What they must be able to say**
1. Predict trusts the motion model (Q), update trusts the measurement (R).
2. An uncertainty is what the filter believes; an error is measured against the truth.
3. A filter too sure of its sensor gets worse, and so does one too sure of its model.
4. RMSE and MAE say different things when there are outliers.

**Slides**
1. The state x = [px, py, vx, vy] and its covariance P.
2. Constant velocity: F moves the position by v·dt; Q = σa² G Gᵀ, a random acceleration.
3. Predict: x = F x, P = F P Fᵀ + Q.
4. Update: innovation, S = H P Hᵀ + R, gain K = P Hᵀ S⁻¹, then x and P.
5. The loop with sensors that measure at different times.
6. Where R comes from: a datasheet, a published error, or a measurement against the truth on data
   kept apart. The LiDAR's R: TransFusion-L's published mean error of 0.28 m gives 0.22 m per axis
   (0.28 / sqrt(π/2)). The RADAR's R, measured on 9 segments of the drive kept apart from the four
   scenes: 1.74 m in range, 1.07 m across, 3.16 m/s in speed [app].
7. RMSE vs MAE: RMSE = sqrt(mean(e²)), MAE = mean(|e|). Example: nine errors of 0.5 and one of 10
   give RMSE 3.2 and MAE 1.45. The RADAR's speed errors: RMSE 3.16 m/s, MAE 1.13 m/s; 2.5 % of the
   points are off by more than 5 m/s and make 86 % of the sum of squares [app].

**Exercises**

*E1. Write the filter.* Kalman Reactor, steps 1 to 9: look at each sensor, choose the cards, write
F and Q, predict, update and the main loop, then Run 1.

*E2. Q and R in the Reactor.* Step 10 (Noise). Before each run, predict better or worse.
- Answer [verify, measured before 9 Nov on dusk-04]: defaults 0.21 m; σa 0.1: 9.17 m (17.7 m at
  14.6 s); σa 10: 0.19 m; R of the LiDAR x 10: 0.35 m.

*E3. Too sure of the sensor, too sure of the model.* Mark I, LiDAR → Detect → Kalman filter,
cmht-dusk-02. Kalman filter card, R · LiDAR at x 0.25, 1, 4, 10.
- Answer: 0.38 m (followed 93 %), 0.24 m, 0.92 m, 2.36 m [app 9 Nov]. Too much trust in the box
  (x 0.25) is worse than the default: the gate gets too narrow and drops good boxes.
- Same scene, Q · acceleration σa at 0.5, 2, 8: 98.9 m (followed 13 %), 0.24 m, 0.18 m [app 9
  Nov]. With σa 0.5 the filter cannot follow the curve.
- Now on cmht-dusk-04 (straight): σa 0.5 gives 0.31 m, 2 gives 0.23 m, 8 gives 0.19 m [app 9 Nov].
  The same Q is a disaster on one scene and fine on the other: why?

*E4. RMSE vs MAE by hand.* The workbook gives ten errors; the student computes both, then answers:
which one would you report for the RADAR's speed, and why?

**To present**: the curve of RMSE against R · LiDAR (four points) with one sentence on why it is
not lowest at the smallest R.

---

## Day 9: Advanced Kalman Filters (Reactor steps 11 to 20, then Mark I)

**What they must be able to say**
1. The RADAR and the camera's direction are not linear in the state: the EKF linearizes them.
2. Adding a sensor can make the filter worse when its R does not tell the truth about it.
3. A turning target needs a turning model.
4. A filter can be wrong about its own precision (NEES), and RMSE alone does not show it.

**Slides**
1. Why an EKF: the RADAR measures range, bearing and range rate. The Jacobian H.
2. The camera's direction: a bearing only, it corrects the angle and not the distance.
3. Adding a sensor can make it worse (the table of E3 below).
4. A bias is not in R: the RADAR's echo comes from the face it sees, not the car's center.
5. Turning: CTRV.
6. NEES = eᵀ P⁻¹ e on the position: about 2 if the filter is honest about itself; in Mark I's
   examples it ranges from 6 to 59 [verify].
7. Why: the LiDAR sees the same faces of the car for dozens of frames, so its box errors are
   correlated (frame-to-frame correlation 0.77) [verify]; the filter averages them as if they were
   independent and becomes overconfident.
8. Outlook: multi-object tracking, IMM, factor graphs.

**Exercises**

*E1. Fuse the RADAR and the camera.* Kalman Reactor, steps 11 to 18 (camera class, RADAR setup,
the extended update, the expected RADAR measurement, the camera direction), with Runs 2, 3 and 4.
- Answers from the Reactor [verify]: Run 3, after 13.7 s the LiDAR misses the car and the RADAR
  still sees it: 0.31 m with the RADAR, 0.35 m without. Run 4: 0.21 m without the camera
  direction, 0.20 m with.

*E2. Turning.* Steps 19 and 20 (CTRV on cmht-dusk-02). Then in Mark I: LiDAR → Detect → Kalman
filter on cmht-dusk-02 with the three motion models.
- Answer: Constant velocity 0.24 m, Constant acceleration 0.22 m, CTRV 0.18 m [app 9 Nov]; the
  speed error drops from 1.10 to 0.38 m/s.

*E3. How much to trust the RADAR.* Mark I, Merge & Cluster, Kalman filter card, R · RADAR at x 1
and x 4, on the four scenes. Predict first: trusting the RADAR less should be safer.
- Answer [app 9 Nov]:

  | Scene | R · RADAR x 1 | x 4 |
  |---|---|---|
  | cmht-dusk-01 | 6.83 m | 7.49 m |
  | cmht-dusk-02 | 0.84 m | 64.9 m |
  | cmht-dusk-03 | 2.50 m | 7.56 m |
  | cmht-dusk-04 | 1.05 m | 0.91 m |

  Trusting the RADAR less loses the car on dusk-02 and helps a little on dusk-04.

*E4. Change the update rate.* Reactor, "Update every N LiDAR scans": 1, then 5. Watch the error
and the uncertainty ellipse. Why can using fewer LiDAR boxes make the filter more honest about
itself?

**To present**: the table of E3 and three sentences: which R you would ship, on which scene it
hurts, and why.

---

## Day 10: Closing (portfolio)

**What they must be able to say**
1. My pipeline answers the five decisions of Day 1, and I can show it with numbers.

**Slides**
1. What a fusion portfolio piece shows: the architecture, the five decisions it answers, the
   Position RMSE per scene, one failure explained with its measured cause, what you would change.
2. Placeholder for Jeremy: examples, format, where to publish.

**Exercise**: write the portfolio page from the Days 5 and 6 pipeline, using the tables of Days 2,
4, 8 and 9.

**To present**: the portfolio page.

---

## Open points for Jeremy

- Days 5, 6 and 7 are a proposal: confirm or change.
- To measure in the app before the day: Day 1 ROI and ground numbers (0.59 m, 1.20 m) and
  PointPainting (0.88 m); Day 3 Pillar against Points; every number tagged [verify].
- The Reactor exposes only σa, R of the LiDAR and the update rate. Mark I's Kalman filter card has
  Q and R for every sensor: the Q and R exercises use Mark I. Exposing every Q and R in the
  Reactor too is a code change waiting for your decision.
- Day 9, slide 4: the RADAR's echo is about 1.25 m short of the car's center in the Reactor's
  scene [verify]; on the 9 calibration segments the mean range error is -0.38 m, with two groups
  of points. Choose which one the slide shows.
