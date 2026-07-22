"""Generate the teaching notebook 02_fusion.ipynb (fusion, taught visibly)."""
import json

def md(*l): return {"cell_type": "markdown", "metadata": {}, "source": [x if x.endswith("\n") else x+"\n" for x in l]}
def code(*l): return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": [x if x.endswith("\n") else x+"\n" for x in l]}

C = []

C.append(md(
"# Sensor Fusion, seen — two sensors, one belief\n", "\n",
"By the end of this notebook you will *see*:\n", "\n",
"1. **where measurements come from** — a real object detector on a real camera,\n",
"2. **the Kalman filter itself** — predict then correct, written out and stepped through here,\n",
"3. **fusion working** — lidar + radar folded into one moving estimate, on real nuScenes data.\n", "\n",
"No black boxes: the filter is in this notebook, not hidden in a library."
))

# ---- setup ----
C.append(md("## Setup"))
C.append(code(
"# --- Colab setup (idempotent + cache-safe) ---\n",
"import os, sys\n",
"BRANCH = 'claude/course-modernization-review-x6ytmr'\n",
"if os.path.isdir('/content'):\n",
"    os.system('cd /content && rm -rf kalman_filters_course && '\n",
"              f'git clone --branch {BRANCH} https://github.com/Jeremy26/kalman_filters_course.git')\n",
"    os.chdir('/content/kalman_filters_course/ekf_sensor_fusion')\n",
"    os.system('pip install -q numpy matplotlib ultralytics')\n",
"for _m in [m for m in list(sys.modules) if 'kf_fusion' in m]:\n",
"    del sys.modules[_m]\n",
"sys.path.insert(0, 'src')\n",
"import numpy as np, matplotlib.pyplot as plt\n",
"from kf_fusion import load_track\n",
"print('setup OK')"
))

# ---- Step 1: detection ----
C.append(md(
"## Step 1 — Perception: where does a measurement come from?\n", "\n",
"A tracker doesn't get coordinates for free. Something has to **detect** the object first. "
"Here's a real object detector (YOLO) on a real camera frame from the scene. "
"The green box is the car we'll track; the yellow boxes are other detections (real clutter)."
))
C.append(code(
"import json\n",
"from ultralytics import YOLO\n",
"from kf_fusion.detection import ensure_yolo_weights\n",
"yolo = YOLO(ensure_yolo_weights())            # weights auto-download from Hugging Face\n",
"tgt = json.load(open('data/sample_frames/target_boxes.json'))['frame_0']['target_box']\n",
"img = 'data/sample_frames/frame_0.jpg'\n",
"res = yolo(img, verbose=False, classes=[2,5,7])[0]   # cars/buses/trucks\n",
"\n",
"im = plt.imread(img); fig, ax = plt.subplots(figsize=(11,6)); ax.imshow(im); ax.axis('off')\n",
"def draw(b, color, lw=2):\n",
"    ax.add_patch(plt.Rectangle((b[0],b[1]), b[2]-b[0], b[3]-b[1], fill=False, color=color, lw=lw))\n",
"tc = [(tgt[0]+tgt[2])/2, (tgt[1]+tgt[3])/2]\n",
"for box in res.boxes.xyxy.cpu().numpy():\n",
"    is_target = box[0] <= tc[0] <= box[2] and box[1] <= tc[1] <= box[3]\n",
"    draw(box, 'lime' if is_target else 'gold', 3 if is_target else 2)\n",
"ax.set_title('Real 2D detection — green = the car we track, yellow = other detections')\n",
"plt.show()"
))

C.append(md(
"That detection, done every frame by lidar and radar, is the **input** to the filter. "
"Let's look at those raw measurements for our car over ~20 seconds of real driving."
))
C.append(code(
"track = load_track('data/track_ed634e83.npz')\n",
"lidar = np.array([m.z[:2] for m in track.measurements if m.sensor=='lidar'])\n",
"# radar measures range/bearing about the moving ego -> convert to x,y just to plot it\n",
"radar = np.array([[m.sensor_pos[0]+m.z[0]*np.cos(m.z[1]), m.sensor_pos[1]+m.z[0]*np.sin(m.z[1])]\n",
"                  for m in track.measurements if m.sensor=='radar'])\n",
"gt = track.gt_state[:, :2]\n",
"plt.figure(figsize=(9,6))\n",
"plt.plot(gt[:,0], gt[:,1], 'k-', lw=2, label='true path')\n",
"plt.scatter(lidar[:,0], lidar[:,1], c='tab:blue', s=25, label='lidar detections')\n",
"plt.scatter(radar[:,0], radar[:,1], c='tab:red', s=25, marker='x', label='radar detections')\n",
"plt.axis('equal'); plt.legend(); plt.title('Raw detections are noisy and disagree — that is why we fuse'); plt.show()"
))

# ---- Step 2: the filter itself ----
C.append(md(
"## Step 2 — The Kalman filter, written out\n", "\n",
"The state is the car's position **and velocity**: `x = [px, py, vx, vy]`.\n", "\n",
"Two moves, forever:\n", "\n",
"- **predict** — push the state forward with constant velocity; uncertainty `P` **grows**.\n",
"- **update** — a measurement pulls the state back; uncertainty `P` **shrinks**.\n", "\n",
"Fusion is just: *update from whichever sensor spoke.* Lidar updates position; radar updates "
"position **and** velocity (Doppler). Same state, two sources. Here is the whole filter — read it."
))
C.append(code(
"class TinyEKF:\n",
"    def __init__(self, noise_a=4.0):\n",
"        self.x = np.zeros(4); self.P = np.eye(4); self.q = noise_a\n",
"\n",
"    def predict(self, dt):\n",
"        F = np.array([[1,0,dt,0],[0,1,0,dt],[0,0,1,0],[0,0,0,1]], float)\n",
"        g = np.array([0.5*dt*dt, 0.5*dt*dt, dt, dt])\n",
"        Q = np.outer(g, g) * self.q                 # process noise from acceleration\n",
"        self.x = F @ self.x\n",
"        self.P = F @ self.P @ F.T + Q               # <-- uncertainty GROWS\n",
"\n",
"    def _update(self, y, H, R):\n",
"        S = H @ self.P @ H.T + R\n",
"        K = self.P @ H.T @ np.linalg.inv(S)         # Kalman gain: trust measurement vs prediction\n",
"        self.x = self.x + K @ y\n",
"        self.P = (np.eye(4) - K @ H) @ self.P       # <-- uncertainty SHRINKS\n",
"\n",
"    def update_lidar(self, z):                      # z = [px, py] (linear)\n",
"        H = np.array([[1,0,0,0],[0,1,0,0]], float)\n",
"        self._update(z - H @ self.x, H, np.diag([1.0, 1.0]))\n",
"\n",
"    def update_radar(self, z, s):                   # z = [range, bearing, range-rate] about sensor s\n",
"        px,py,vx,vy = self.x; dx,dy = px-s[0], py-s[1]\n",
"        rho = max(np.hypot(dx,dy), 1e-6)\n",
"        h = np.array([rho, np.arctan2(dy,dx), (dx*vx+dy*vy)/rho])   # nonlinear -> the 'Extended' bit\n",
"        H = np.array([[dx/rho, dy/rho, 0, 0],\n",
"                      [-dy/rho**2, dx/rho**2, 0, 0],\n",
"                      [dy*(vx*dy-vy*dx)/rho**3, dx*(vy*dx-vx*dy)/rho**3, dx/rho, dy/rho]])\n",
"        y = z - h; y[1] = (y[1]+np.pi)%(2*np.pi) - np.pi            # wrap the angle\n",
"        self._update(y, H, np.diag([0.5, 0.01, 4.0]))\n",
"print('TinyEKF defined — that is the entire filter')"
))

C.append(md(
"### See one predict and one update\n",
"Watch the position uncertainty (the ellipse) balloon on **predict**, then snap tight on **update**."
))
C.append(code(
"def ellipse(P, xy, **kw):\n",
"    vals, vecs = np.linalg.eigh(P[:2,:2]); ang = np.degrees(np.arctan2(*vecs[:,1][::-1]))\n",
"    from matplotlib.patches import Ellipse\n",
"    return Ellipse(xy, 2*np.sqrt(vals[0]), 2*np.sqrt(vals[1]), angle=ang, fill=False, **kw)\n",
"\n",
"kf = TinyEKF(); m0 = track.measurements[0]\n",
"kf.x = np.array([m0.z[0], m0.z[1], 0, 0.]); kf.P = np.diag([2,2,25,25.])\n",
"fig, ax = plt.subplots(figsize=(7,7)); ax.plot(*kf.x[:2], 'ko', label='start')\n",
"ax.add_patch(ellipse(kf.P, kf.x[:2], color='gray', ls=':', label='start'))\n",
"kf.predict(2.0)\n",
"ax.plot(*kf.x[:2], 'r^'); ax.add_patch(ellipse(kf.P, kf.x[:2], color='red', label='after PREDICT (grew)'))\n",
"z = track.measurements[2].z[:2]; ax.plot(*z, 'b*', ms=14, label='lidar measurement')\n",
"kf.update_lidar(z)\n",
"ax.plot(*kf.x[:2], 'bo'); ax.add_patch(ellipse(kf.P, kf.x[:2], color='blue', label='after UPDATE (shrank)'))\n",
"ax.axis('equal'); ax.legend(); ax.set_title('predict grows uncertainty, update shrinks it'); plt.show()"
))

# ---- Step 3: fuse whole track ----
C.append(md(
"## Step 3 — Fuse the whole track, and watch it work\n", "\n",
"Now run that same filter across every measurement: predict to each timestamp, then update with "
"lidar or radar. The smooth line is the fused estimate riding through the noisy detections."
))
C.append(code(
"def run(measurements, use_lidar=True, use_radar=True):\n",
"    kf = TinyEKF(); est = []; last = None; started = False\n",
"    for m in measurements:\n",
"        if not started:\n",
"            kf.x = np.concatenate([m.initial_state()[:2], [0,0]]); kf.P = np.diag([2,2,25,25.])\n",
"            last = m.timestamp; started = True\n",
"        else:\n",
"            kf.predict(m.timestamp - last); last = m.timestamp\n",
"            if m.sensor=='lidar' and use_lidar: kf.update_lidar(m.z)\n",
"            if m.sensor=='radar' and use_radar: kf.update_radar(m.z, m.sensor_pos)\n",
"        est.append(kf.x.copy())\n",
"    return np.array(est)\n",
"\n",
"est = run(track.measurements)\n",
"plt.figure(figsize=(10,7))\n",
"plt.scatter(lidar[:,0], lidar[:,1], c='tab:blue', s=18, alpha=.5, label='lidar detections')\n",
"plt.scatter(radar[:,0], radar[:,1], c='tab:red', s=18, alpha=.5, marker='x', label='radar detections')\n",
"plt.plot(gt[:,0], gt[:,1], 'k-', lw=2, label='true path')\n",
"plt.plot(est[:,0], est[:,1], 'lime', lw=2.5, label='FUSED estimate')\n",
"plt.axis('equal'); plt.legend(); plt.title('One filter fuses noisy lidar + radar into a clean track'); plt.show()"
))

C.append(md(
"### Why bother with two sensors?\n",
"Lidar gives crisp position but no speed; radar's Doppler gives speed directly. Fusion gets both. "
"Here's the **estimated speed** — flat and sensible only because radar is in the mix."
))
C.append(code(
"speed_fused = np.hypot(est[:,2], est[:,3])\n",
"speed_lidar = np.hypot(*run(track.measurements, use_radar=False)[:, 2:].T)\n",
"t = track.timestamps if hasattr(track,'timestamps') else np.arange(len(est))\n",
"tt = [m.timestamp for m in track.measurements]\n",
"plt.figure(figsize=(10,4))\n",
"plt.plot(tt, speed_lidar, 'tab:orange', label='lidar only (velocity guessed)')\n",
"plt.plot(tt, speed_fused, 'green', lw=2, label='fused with radar Doppler')\n",
"plt.xlabel('time (s)'); plt.ylabel('speed (m/s)'); plt.legend()\n",
"plt.title('Radar Doppler makes the speed estimate trustworthy'); plt.show()"
))

# ---- Step 4: foxglove ----
C.append(md(
"## Step 4 — See it in the real 3D scene (Foxglove)\n", "\n",
"The plots above are the idea. The **full experience** — the camera with live detections, the real "
"lidar point cloud, the radar returns, and the tracked box moving through it — is a Foxglove "
"recording. Generate it (needs the dataset) or open the one shipped in `outputs/`:\n", "\n",
"1. go to [app.foxglove.dev](https://app.foxglove.dev) → **Open local file**,\n",
"2. choose `outputs/scene_ed634e83.mcap`,\n",
"3. **Import layout** → `layouts/scene.json`.\n", "\n",
"You'll see the camera (left) with detection boxes, and the 3D scene (right) with lidar, radar, and "
"the fused box tracking the car. To regenerate it yourself:\n", "\n",
"```bash\n",
"pip install -e \".[viz,nuscenes]\"\n",
"python scripts/download_nuscenes.py           # ~4 GB, one time\n",
"python scripts/make_scene_mcap.py --instance ed634e83 --out outputs/scene_ed634e83.mcap\n",
"```"
))

C.append(md(
"## Recap\n", "\n",
"- **Detection** produces noisy measurements (you saw YOLO do it).\n",
"- The **Kalman filter** predicts, then corrects — you read the whole thing and stepped it.\n",
"- **Fusion** is just updating one shared state from lidar *and* radar; you watched it clean up the "
"track and pin down the speed.\n", "\n",
"> Next: run *many* of these at once and you must decide which detection belongs to which car — "
"that's data association, and the start of multi-object tracking."
))

nb = {"cells": C, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
      "language_info": {"name": "python", "version": "3.11"}}, "nbformat": 4, "nbformat_minor": 5}
json.dump(nb, open("02_fusion.ipynb", "w"), indent=1)
print("wrote 02_fusion.ipynb")
