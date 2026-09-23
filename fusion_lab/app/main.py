"""Fusion Lab: Gradio controls on the left, Rerun viewer on the right.

All the filtering lives in kflab. This file only turns the controls into a
kflab.Config, runs the filter over the whole scene and hands the result to Rerun.
"""

import html
import os
import tempfile
import uuid
from dataclasses import astuple, replace
from functools import lru_cache
from pathlib import Path

import gradio as gr
import numpy as np
from gradio_rerun import Rerun

from kflab import Config, available_scenes, evaluate, load_scene, run

from . import theme
from .rerun_log import build_rrd, run_name, sensor_batches

ROOT = Path(__file__).resolve().parents[1]
CACHE = Path(os.environ.get("FUSION_LAB_CACHE", ROOT / "cache"))
RRD_DIR = Path(tempfile.gettempdir()) / "fusion_lab_rrd"
SENSORS = ["LiDAR", "RADAR", "Camera"]
DEFAULT = Config()
COMPARE = ["Nothing", "Other nonlinear filter (EKF / UKF)", "CV", "CA", "CTRV"]

PRESETS = {
    "Default": {},
    "Q too small (overconfident model)": {"sigma_a": 0.1, "sigma_yawacc": 0.05},
    "Q too large (jumpy estimate)": {"sigma_a": 8.0, "sigma_yawacc": 3.0},
    "R too small (trusts every detection)": {"lidar_std": 0.05, "radar_range_std": 0.3,
                                             "radar_bearing_std": np.deg2rad(0.5),
                                             "radar_range_rate_std": 0.1},
    "R too large (ignores the sensors)": {"lidar_std": 4.0, "radar_range_std": 10.0,
                                          "radar_bearing_std": np.deg2rad(20),
                                          "radar_range_rate_std": 4.0},
}


def slider_values(preset):
    c = Config(**PRESETS[preset])
    return [c.sigma_a, c.sigma_yawacc, c.lidar_std, c.radar_range_std,
            float(np.rad2deg(c.radar_bearing_std)), c.radar_range_rate_std,
            float(np.rad2deg(c.camera_bearing_std))]


@lru_cache(maxsize=8)
def scene(name):
    return load_scene(CACHE, name)


@lru_cache(maxsize=8)
def sensors(name):
    return sensor_batches(scene(name))


def write_rrd(data):
    """Write one recording for the viewer, keeping only the most recent files."""
    RRD_DIR.mkdir(parents=True, exist_ok=True)
    for old in sorted(RRD_DIR.glob("*.rrd"), key=lambda p: p.stat().st_mtime)[:-20]:
        old.unlink(missing_ok=True)
    path = RRD_DIR / f"{uuid.uuid4().hex}.rrd"
    path.write_bytes(data)
    return str(path)


def make_config(model, nonlinear, sensors, cam_bearing, sigma_a, sigma_yawacc, lidar_std,
                radar_range_std, radar_bearing_deg, radar_rate_std, camera_bearing_deg):
    return Config(model=model, nonlinear=nonlinear, use_lidar="LiDAR" in sensors,
                  use_radar="RADAR" in sensors, use_camera="Camera" in sensors,
                  camera_bearing_update=bool(cam_bearing), sigma_a=float(sigma_a),
                  sigma_yawacc=float(sigma_yawacc), lidar_std=float(lidar_std),
                  radar_range_std=float(radar_range_std),
                  radar_bearing_std=float(np.deg2rad(radar_bearing_deg)),
                  radar_range_rate_std=float(radar_rate_std),
                  camera_bearing_std=float(np.deg2rad(camera_bearing_deg)))


@lru_cache(maxsize=32)
def filter_run(scene_name, config_key):
    s = scene(scene_name)
    track = run(s, Config(*config_key))
    return track, evaluate(s, track)


def compare_config(cfg, compare):
    if compare == COMPARE[1]:
        return replace(cfg, nonlinear="UKF" if cfg.nonlinear == "EKF" else "EKF")
    if compare in ("CV", "CA", "CTRV") and compare != cfg.model:
        return replace(cfg, model=compare)
    return None


def metrics_html(s, runs):
    rows = []
    for name, track, ev in runs:
        counts = {k: sum(st.sensor == k for st in track.steps) for k in ("lidar", "radar", "camera")}
        rows.append(
            f"<tr><td><b>{html.escape(name)}</b></td><td>{ev.rmse:.2f} m</td>"
            f"<td>{ev.mean_nees:.1f}</td><td>{100 * ev.inside_95:.0f} %</td>"
            f"<td>{counts['lidar']} / {counts['radar']} / {counts['camera']}</td></tr>")
    return (
        "<div class='ta-metrics'><table><tr><th>run</th><th>RMSE</th><th>mean NEES</th>"
        "<th>NEES under 95 % bound</th><th>updates L / R / C</th></tr>" + "".join(rows)
        + "</table><p class='ta-note'>A consistent filter has a mean NEES near 2 and about 95 % "
          "of its NEES values under the bound. Much higher: overconfident. Much lower: too "
          "cautious.</p></div>")


def update(scene_name, compare, *controls):
    s = scene(scene_name)
    cfg = make_config(*controls)
    runs = [(run_name(cfg), *filter_run(scene_name, astuple(cfg)))]
    other = compare_config(cfg, compare)
    if other is not None:
        runs.append((run_name(other), *filter_run(scene_name, astuple(other))))
    shown = {k for k, on in (("lidar", cfg.use_lidar), ("radar", cfg.use_radar),
                             ("camera", cfg.use_camera)) if on}
    data = build_rrd(s, sensors(scene_name), runs, shown, recording_id=uuid.uuid4().hex)
    return write_rrd(data), metrics_html(s, runs)


def scene_info(name):
    s = scene(name)
    return (f"**{s.name}**, target `{s.target['category']}`. {s.lesson}\n\n"
            f"<span class='ta-note'>nuScenes: {html.escape(s.description)}</span>")


def missing_cache_app():
    with gr.Blocks(title="Fusion Lab") as demo:
        gr.HTML(theme.HEADER)
        gr.Markdown(
            f"No prepared data found in `{CACHE}`.\n\nRun the preparation once (it downloads "
            "nuScenes mini, about 5 GB, then builds a small cache):\n\n"
            "```\ndocker compose run --rm prep --accept-nuscenes-license\n```\n\nthen restart "
            "the app.")
    return demo


def build_app():
    names = available_scenes(CACHE)
    if not names:
        return missing_cache_app()

    with gr.Blocks(title="Fusion Lab", fill_width=True) as demo:
        gr.HTML(theme.HEADER)
        with gr.Row(equal_height=False):
            with gr.Column(scale=1, min_width=340):
                scene_dd = gr.Dropdown(names, value=names[0], label="Scene")
                info = gr.Markdown(scene_info(names[0]))
                sensors = gr.CheckboxGroup(SENSORS, value=SENSORS, label="Active sensors")
                cam_bearing = gr.Checkbox(False, label="Camera also updates the bearing",
                                          info="Off: the camera gives the class only and never "
                                               "changes the estimate.")
                with gr.Row():
                    model = gr.Radio(["CV", "CA", "CTRV"], value="CV", label="Motion model")
                    nonlinear = gr.Radio(["EKF", "UKF"], value="EKF", label="Nonlinear filter",
                                         info="Used by RADAR, camera bearing and CTRV.")
                compare = gr.Dropdown(COMPARE, value="Nothing", label="Also show, in yellow")
                preset = gr.Dropdown(list(PRESETS), value="Default", label="Tuning preset")
                with gr.Accordion("Process noise Q", open=True):
                    sigma_a = gr.Slider(0.05, 10, DEFAULT.sigma_a, step=0.05,
                                        label="sigma_a (m/s2, jerk m/s3 for CA)")
                    sigma_yawacc = gr.Slider(0.02, 3, DEFAULT.sigma_yawacc, step=0.02,
                                             label="sigma yaw acceleration (rad/s2, CTRV)")
                with gr.Accordion("Measurement noise R", open=True):
                    lidar_std = gr.Slider(0.02, 5, DEFAULT.lidar_std, step=0.02,
                                          label="LiDAR position (m)")
                    radar_range = gr.Slider(0.1, 10, DEFAULT.radar_range_std, step=0.1,
                                            label="RADAR range (m)")
                    radar_bearing = gr.Slider(0.2, 20, float(np.rad2deg(DEFAULT.radar_bearing_std)),
                                              step=0.2, label="RADAR bearing (deg)")
                    radar_rate = gr.Slider(0.05, 5, DEFAULT.radar_range_rate_std, step=0.05,
                                           label="RADAR range rate (m/s)")
                    cam_std = gr.Slider(0.1, 10, float(np.rad2deg(DEFAULT.camera_bearing_std)),
                                        step=0.1, label="Camera bearing (deg)")

            with gr.Column(scale=4):
                viewer = Rerun(height=820, panel_states={"blueprint": "collapsed",
                                                         "selection": "collapsed",
                                                         "time": "expanded"})
                metrics = gr.HTML()
                gr.Markdown(
                    "Press play in the viewer's time panel, or drag its cursor. The thick "
                    "ellipse is the 95 % confidence region now, colored by the sensor that last "
                    "updated the state (blue LiDAR, orange RADAR, violet camera); the thin one is "
                    "the same region just before that update. Detections are gated to the target "
                    "with ground truth, since association is out of scope. Ground truth exists "
                    "only at the 2 Hz annotations.", elem_classes="ta-note")

        controls = [model, nonlinear, sensors, cam_bearing, sigma_a, sigma_yawacc, lidar_std,
                    radar_range, radar_bearing, radar_rate, cam_std]
        inputs = [scene_dd, compare] + controls
        for c in [compare] + controls:
            c.change(update, inputs, [viewer, metrics], trigger_mode="always_last")
        scene_dd.change(scene_info, scene_dd, info).then(update, inputs, [viewer, metrics])
        preset.change(slider_values, preset, [sigma_a, sigma_yawacc, lidar_std, radar_range,
                                              radar_bearing, radar_rate, cam_std])
        demo.load(update, inputs, [viewer, metrics])
    return demo


def main():
    build_app().launch(server_name=os.environ.get("HOST", "0.0.0.0"),
                       server_port=int(os.environ.get("PORT", "7860")),
                       theme=theme.theme, css=theme.CSS, head=theme.HEAD,
                       allowed_paths=[str(RRD_DIR)])


if __name__ == "__main__":
    main()
