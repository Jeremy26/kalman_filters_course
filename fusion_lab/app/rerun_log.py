"""Turn a scene and filter runs into Rerun recordings (.rrd bytes).

    sensor_batches(scene)   LiDAR, RADAR, camera and ground truth as column batches,
                            built once per scene (the slow part).
    build_rrd(...)          a fresh recording with those batches plus the filter runs:
                            estimates, ellipses, metrics, state text and the view layout.

Everything lives in the scene frame, z up, on one timeline named "time" (seconds).
"""

import numpy as np
import rerun as rr
import rerun.blueprint as rrb

from kflab.metrics import CHI2_95
from kflab.motion import heading_of, speed_of

APP_ID = "fusion_lab"
TIMELINE = "time"
PREDICT_HZ = 20          # the ellipse is also logged between updates, to show it grow

# Colors on the dark Rerun background. LiDAR uses the Think Autonomous blue.
LIDAR = (0, 146, 248)
RADAR = (251, 146, 60)
CAMERA = (192, 132, 252)
GT = (226, 232, 240)
EGO = (148, 163, 184)
RUN_COLORS = [(34, 197, 94), (250, 204, 21)]   # main run, comparison run
SENSOR_RGB = {"lidar": LIDAR, "radar": RADAR, "camera": CAMERA}
BACKGROUND = (11, 17, 23)


def _stream(recording_id):
    return rr.RecordingStream(APP_ID, recording_id=recording_id)


def _t(rec, t):
    rec.set_time(TIMELINE, duration=float(t))


def _yaw_quat(yaw):
    return rr.Quaternion(xyzw=[0.0, 0.0, np.sin(yaw / 2), np.cos(yaw / 2)])


def _boxes(dets, z=1.0):
    return dict(
        centers=[[d["xy"][0], d["xy"][1], z] for d in dets],
        half_sizes=[[d["size_wlh"][1] / 2, d["size_wlh"][0] / 2, d["size_wlh"][2] / 2]
                    for d in dets],
        quaternions=[_yaw_quat(d["yaw"]) for d in dets],
    )


class Batches:
    """Recorded calls, replayed into a new recording: static logs and column sends."""

    def __init__(self):
        self.static, self.columns = [], []

    def log_static(self, entity, archetype):
        self.static.append((entity, archetype))

    def send(self, entity, times, columns, lengths):
        if len(times):
            order = np.argsort(times, kind="stable")
            if not np.all(order == np.arange(len(order))):
                raise ValueError(f"{entity}: rows must be sent in time order")
            self.columns.append((entity, np.asarray(times, float), columns.partition(lengths)))

    def replay(self, rec):
        for entity, archetype in self.static:
            rec.log(entity, archetype, static=True)
        for entity, times, cols in self.columns:
            rec.send_columns(entity, indexes=[rr.TimeColumn(TIMELINE, duration=times)],
                             columns=cols)


def sensor_batches(scene):
    """Everything the sensors and the ground truth show, as column batches (built once per scene)."""
    b = Batches()
    b.log_static("world", rr.ViewCoordinates.RIGHT_HAND_Z_UP)

    # Ground truth: the 2 Hz annotations as a path, and the annotated box at each keyframe.
    g = scene.gt
    path = np.column_stack([g["xy"], np.full(len(g["t"]), 0.05)])
    b.log_static("world/ground_truth/path", rr.LineStrips3D([path], colors=[GT], radii=0.06))
    b.log_static("world/ground_truth/annotations", rr.Points3D(path, colors=[GT], radii=0.15))
    w, l, h = scene.target["size_wlh"]
    n = len(g["t"])
    b.send("world/ground_truth/box", g["t"], rr.Boxes3D.columns(
        centers=[[x, y, h / 2] for x, y in g["xy"]], half_sizes=[[l / 2, w / 2, h / 2]] * n,
        quaternions=[_yaw_quat(y) for y in g["yaw"]], colors=[GT] * n, radii=[0.04] * n,
        labels=[f"ground truth, visibility {v}" for v in g["visibility"]]), [1] * n)

    # Ego car, interpolated at 10 Hz between keyframe poses.
    ts = np.arange(0.0, scene.duration, 0.1)
    poses = [scene.ego_pose_at(t) for t in ts]
    b.send("world/ego", ts, rr.Boxes3D.columns(
        centers=[[xy[0], xy[1], 0.8] for xy, _ in poses], half_sizes=[[2.4, 0.95, 0.8]] * len(ts),
        quaternions=[_yaw_quat(yaw) for _, yaw in poses], colors=[EGO] * len(ts),
        radii=[0.05] * len(ts), labels=["ego"] * len(ts)), [1] * len(ts))

    # LiDAR: points and boxes at each keyframe.
    lidar = [e for e in scene.events if e.sensor == "lidar"]
    pts = [scene.points(e.frame)["lidar_xyz"] for e in lidar]
    allp = np.concatenate(pts)
    shade = np.clip(allp[:, 2] / 3.0, 0, 1)[:, None]
    colors = (np.array([70, 85, 100]) + shade * np.array([150, 150, 150])).astype(np.uint8)
    b.log_static("world/lidar/points", rr.Points3D.from_fields(radii=rr.Radius.ui_points(1.2)))
    b.send("world/lidar/points", [e.t for e in lidar],
           rr.Points3D.columns(positions=allp, colors=colors), [len(p) for p in pts])
    others = [[d for k, d in enumerate(e.detections) if k != e.target] for e in lidar]
    flat = [d for o in others for d in o]
    b.log_static("world/lidar/boxes", rr.Boxes3D.from_fields(colors=[(*LIDAR, 110)], radii=0.03))
    b.send("world/lidar/boxes", [e.t for e in lidar], rr.Boxes3D.columns(**_boxes(flat)),
           [len(o) for o in others])
    gated = [e.target_detection for e in lidar if e.target is not None]
    b.log_static("world/lidar/gated", rr.Boxes3D.from_fields(colors=[LIDAR], radii=0.09))
    b.send("world/lidar/gated", [e.t for e in lidar], rr.Boxes3D.columns(
        **_boxes(gated), labels=[f"LiDAR {d['name']} {d['score']:.2f}" for d in gated]),
        [int(e.target is not None) for e in lidar])

    # RADAR: per radar, raw points with velocity, and the gated cluster.
    for ch in sorted({e.channel for e in scene.events if e.sensor == "radar"}):
        sweeps = [e for e in scene.events if e.channel == ch]
        base = f"world/radar/{ch.lower()}"
        xy, v = [], []
        for e in sweeps:
            rp = scene.points(e.frame)
            sel = rp["radar_meta"][:, 1] == e.index_in_frame
            xy.append(rp["radar_xy"][sel])
            v.append(rp["radar_vel"][sel])
        lens = [len(p) for p in xy]
        p3 = np.column_stack([np.concatenate(xy), np.full(sum(lens), 0.5)])
        v3 = np.column_stack([np.concatenate(v), np.zeros(sum(lens))])
        ts = [e.t for e in sweeps]
        b.log_static(f"{base}/points", rr.Points3D.from_fields(
            colors=[(*RADAR, 170)], radii=rr.Radius.ui_points(3.0)))
        b.send(f"{base}/points", ts, rr.Points3D.columns(positions=p3), lens)
        b.log_static(f"{base}/velocity", rr.Arrows3D.from_fields(colors=[(*RADAR, 150)],
                                                                 radii=0.03))
        b.send(f"{base}/velocity", ts, rr.Arrows3D.columns(origins=p3, vectors=v3), lens)
        hit = [e.target_detection for e in sweeps if e.target is not None]
        b.log_static(f"{base}/gated", rr.Points3D.from_fields(colors=[RADAR], radii=0.45))
        b.send(f"{base}/gated", ts, rr.Points3D.columns(
            positions=[[d["xy"][0], d["xy"][1], 0.6] for d in hit],
            labels=[f"RADAR r {d['z'][0]:.1f} m, rate {d['z'][2]:+.1f} m/s" for d in hit]),
            [int(e.target is not None) for e in sweeps])

    # Camera: a 2D box has no BEV position, so it is drawn as a wedge of directions.
    for ch in sorted({e.channel for e in scene.events if e.sensor == "camera"}):
        images = [e for e in scene.events if e.channel == ch]
        times, strips, labels, lens = [], [], [], []
        for j, e in enumerate(images):
            d = e.target_detection
            times.append(e.t)
            if d is None:
                lens.append(0)
            else:
                reach = 1.25 * float(np.hypot(*(scene.gt_at(e.t) - e.sensor_xy))) + 5
                a1, a2 = np.array(d["bearing_span"]) + e.sensor_yaw
                arc = np.linspace(a1, a2, 12)
                o = np.array([*e.sensor_xy, 1.5])
                ring = np.column_stack([o[0] + reach * np.cos(arc), o[1] + reach * np.sin(arc),
                                        np.full(len(arc), 1.5)])
                strips.append(np.vstack([o, ring, o]))
                labels.append(f"camera: {d['cls']} {d['score']:.2f}")
                lens.append(1)
            # Remove the wedge when this camera stops seeing the target for a while.
            nxt = images[j + 1].t if j + 1 < len(images) else np.inf
            if nxt - e.t > 0.15:
                times.append(e.t + 0.15)
                lens.append(0)
        base = f"world/camera/{ch.lower()}"
        b.log_static(base, rr.LineStrips3D.from_fields(colors=[CAMERA], radii=0.05))
        b.send(base, times, rr.LineStrips3D.columns(strips=strips, labels=labels), lens)
    return b


def ellipse_outline(xy, P2, n=48, z=0.3):
    """95 % confidence ellipse of a 2x2 covariance: outline points, semi-axes, yaw of major axis."""
    vals, vecs = np.linalg.eigh(P2)
    radii = np.sqrt(CHI2_95[2] * np.clip(vals, 1e-9, None))
    a = np.linspace(0, 2 * np.pi, n)
    pts = (vecs @ (radii[:, None] * np.vstack([np.cos(a), np.sin(a)]))).T + xy
    return np.column_stack([pts, np.full(n, z)]), radii, float(np.arctan2(vecs[1, 1], vecs[0, 1]))


def _state_markdown(track, step, name):
    m = track.model
    sd = np.sqrt(np.diag(step.P))
    rows = []
    for label, unit, v, e in zip(m.labels, m.units, step.x, sd):
        if unit.startswith("rad"):
            v, e, unit = np.rad2deg(v), np.rad2deg(e), unit.replace("rad", "deg")
        rows.append(f"| `{label}` | {v:8.2f} | ± {e:.2f} | {unit} |")
    nis = "" if step.nis is None else f", NIS {step.nis:.1f}"
    return (f"### {name}\n\n**Last update: {step.sensor.upper()}** ({step.channel}) "
            f"at t = {step.t:.2f} s{nis}\n\n| state | value | 1 sd | unit |\n|---|---|---|---|\n"
            + "\n".join(rows)
            + f"\n\nspeed {speed_of(m, step.x):.1f} m/s, heading "
              f"{np.rad2deg(heading_of(m, step.x)):.0f} deg")


def run_name(cfg):
    return f"{cfg.model} {cfg.nonlinear}"


def _slug(name):
    return name.replace(" ", "_").lower()


def _send(rec, entity, times, columns, lengths=None):
    """Send a whole history in one call: one row per time."""
    n = len(times)
    if n == 0:
        return
    rec.send_columns(entity, indexes=[rr.TimeColumn(TIMELINE, duration=np.asarray(times))],
                     columns=columns.partition(lengths if lengths is not None else [1] * n))


def _log_run(rec, scene, k, name, track, ev):
    rgb = RUN_COLORS[k % len(RUN_COLORS)]
    base = f"world/estimate/{k}_{_slug(name)}"
    for metric, values in (("error", ev.error), ("nees", ev.nees)):
        rec.log(f"metrics/{metric}/{_slug(name)}",
                rr.SeriesLines(colors=[rgb], names=[name], widths=2), static=True)
        _send(rec, f"metrics/{metric}/{_slug(name)}", ev.t, rr.Scalars.columns(scalars=values))
    if not track.steps:
        return
    steps = track.steps
    main = k == 0

    # The ellipse at every update, and predicted at PREDICT_HZ in between.
    times, states, colors = [], [], []
    grid = np.arange(steps[0].t, scene.duration, 1.0 / PREDICT_HZ)
    gi = 0
    for j, st in enumerate(steps):
        t_next = steps[j + 1].t if j + 1 < len(steps) else scene.duration
        color = SENSOR_RGB[st.sensor] if main else rgb
        times.append(st.t), states.append((st.x, st.P)), colors.append(color)
        while gi < len(grid) and grid[gi] <= st.t:
            gi += 1
        while gi < len(grid) and grid[gi] < t_next:
            x, P, _ = track.state_at(grid[gi])
            times.append(grid[gi]), states.append((x, P)), colors.append(color)
            gi += 1
    outlines, fills = [], []
    for x, P in states:
        outline, radii, yaw = ellipse_outline(x[:2], P[:2, :2])
        outlines.append(outline)
        fills.append(([x[0], x[1], 0.25], [radii[1], radii[0], 0.02], _yaw_quat(yaw)))
    n = len(times)
    _send(rec, f"{base}/ellipse", times, rr.LineStrips3D.columns(
        strips=outlines, colors=colors, radii=[0.14 if main else 0.08] * n))
    rec.log(f"{base}/ellipse_fill", rr.Ellipsoids3D.from_fields(fill_mode="solid"), static=True)
    _send(rec, f"{base}/ellipse_fill", times, rr.Ellipsoids3D.columns(
        centers=[f[0] for f in fills], half_sizes=[f[1] for f in fills],
        quaternions=[f[2] for f in fills], colors=[(*c, 70) for c in colors]))

    # Updates only: path so far, heading arrow, prior ellipse, LiDAR measurement, state text.
    ts = [st.t for st in steps]
    xy = np.array([[st.x[0], st.x[1], 0.3] for st in steps])
    _send(rec, f"{base}/path", ts, rr.LineStrips3D.columns(
        strips=[xy[:j + 1] for j in range(len(steps))], colors=[rgb] * len(steps),
        radii=[0.1] * len(steps)))
    heading = [heading_of(track.model, st.x) for st in steps]
    speed = [speed_of(track.model, st.x) for st in steps]
    _send(rec, f"{base}/state", ts, rr.Arrows3D.columns(
        origins=xy + [0, 0, 0.05], vectors=[[v * np.cos(h), v * np.sin(h), 0] for v, h in
                                             zip(speed, heading)],
        colors=[rgb] * len(steps), radii=[0.12] * len(steps), labels=[name] * len(steps)))
    if not main:
        return
    priors = [ellipse_outline(st.x_prior[:2], st.P_prior[:2, :2], z=0.28)[0] for st in steps]
    _send(rec, f"{base}/prior_ellipse", ts, rr.LineStrips3D.columns(
        strips=priors, colors=[(*SENSOR_RGB[st.sensor], 140) for st in steps],
        radii=[0.04] * len(steps)))
    lidar = [st for st in steps if st.sensor == "lidar"]
    _send(rec, f"{base}/measurement", ts, rr.Points3D.columns(
        positions=[[st.z[0], st.z[1], 0.4] for st in lidar], colors=[LIDAR] * len(lidar),
        radii=[0.25] * len(lidar)), lengths=[int(st.sensor == "lidar") for st in steps])
    # Only the media type is static: static text would hide the per-update text.
    rec.log("state", rr.TextDocument.from_fields(media_type=rr.MediaType.MARKDOWN), static=True)
    _send(rec, "state", ts, rr.TextDocument.columns(
        text=[_state_markdown(track, st, name) for st in steps]))
    for sensor, level in (("lidar", 3), ("radar", 2), ("camera", 1)):
        t_s = [st.t for st in steps if st.sensor == sensor]
        _send(rec, f"metrics/updates/{sensor}", t_s, rr.Scalars.columns(scalars=[level] * len(t_s)))


def build_rrd(scene, sensors, runs, sensors_shown, recording_id):
    """One complete recording: cached sensor batches, then the filter runs.

    runs: list of (name, track, evaluation), the first one is the main run. A new
    recording_id per call makes the viewer drop the previous run instead of merging.
    """
    rec = _stream(recording_id)
    stream = rec.binary_stream()
    sensors.replay(rec)
    for k, (name, track, ev) in enumerate(runs):
        _log_run(rec, scene, k, name, track, ev)
    for sensor, rgb in SENSOR_RGB.items():
        rec.log(f"metrics/updates/{sensor}", rr.SeriesPoints(colors=[rgb], names=[sensor],
                                                             marker_sizes=2), static=True)
    rec.log("metrics/nees/bound_95", rr.SeriesLines(colors=[(239, 68, 68)], names=["95 % bound"]),
            static=True)
    _send(rec, "metrics/nees/bound_95", [0.0, scene.duration],
          rr.Scalars.columns(scalars=[CHI2_95[2]] * 2))
    rec.send_blueprint(blueprint(scene, sensors_shown), make_active=True, make_default=True)
    return stream.read()


def blueprint(scene, sensors_shown):
    hidden = {"lidar": "- /world/lidar/**", "radar": "- /world/radar/**",
              "camera": "- /world/camera/**"}
    contents = ["+ /world/**"] + [rule for s, rule in hidden.items() if s not in sensors_shown]
    # Look straight down on the area covered by the target and the ego car.
    pts = np.vstack([scene.gt["xy"], [k.ego_xy for k in scene.keyframes]])
    lo, hi = pts.min(0), pts.max(0)
    c = (lo + hi) / 2
    height = float(max(hi - lo) * 1.1 + 20)
    bev = rrb.Spatial3DView(
        origin="/world", contents=contents, name="Bird's eye view", background=BACKGROUND,
        eye_controls=rrb.EyeControls3D(kind=rrb.components.Eye3DKind.Orbital,
                                       position=[c[0], c[1] - 0.001 * height, height],
                                       look_target=[c[0], c[1], 0.0], eye_up=[0.0, 1.0, 0.0]),
    )
    return rrb.Blueprint(
        rrb.Horizontal(
            bev,
            rrb.Vertical(
                rrb.TextDocumentView(origin="/state", name="State"),
                rrb.TimeSeriesView(origin="/metrics/error", name="Position error (m)"),
                rrb.TimeSeriesView(origin="/metrics/nees",
                                   name="NEES (95 % of points under the red line)"),
                rrb.TimeSeriesView(origin="/metrics/updates",
                                   name="Updates (LiDAR 3, RADAR 2, camera 1)"),
                row_shares=[1.6, 1, 1, 0.7],
            ),
            column_shares=[2.4, 1],
        ),
        # Start playing and loop, so the replay begins from t = 0 when the page opens.
        rrb.TimePanel(state="expanded", timeline=TIMELINE, play_state="playing", loop_mode="all"),
        collapse_panels=True,
    )
