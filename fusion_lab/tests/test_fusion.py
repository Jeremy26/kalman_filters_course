import numpy as np

from kflab.fusion import Config, run
from kflab.metrics import evaluate
from kflab.scene import Event, Keyframe, Scene


def synthetic_scene(noise=True, seed=0):
    """Target moving at (5, 1) m/s from (10, 0). LiDAR at 2 Hz, RADAR at 13 Hz."""
    rng = np.random.default_rng(seed)
    pos = lambda t: np.array([10 + 5 * t, 1 * t])
    vel = np.array([5.0, 1.0])
    radar_xy, radar_yaw = np.array([0.0, 0.0]), 0.0
    events = []
    for t in np.arange(0, 10, 0.5):
        z = pos(t) + (rng.normal(0, 0.3, 2) if noise else 0)
        events.append(Event(t, "lidar", "LIDAR_TOP", np.zeros(2), 0.0, [{"xy": list(z)}], 0, 0))
    for t in np.arange(0.03, 10, 1 / 13):
        p = pos(t)
        r = np.hypot(*p)
        z = [r, np.arctan2(p[1], p[0]), p @ vel / r]
        if noise:
            z = list(np.array(z) + rng.normal(0, [0.5, 0.01, 0.2]))
        events.append(Event(t, "radar", "RADAR_FRONT", radar_xy, radar_yaw, [{"z": z}], 0, 0))
    events.append(Event(0.2, "camera", "CAM_FRONT", np.zeros(2), 0.0,
                        [{"cls": "car", "bearing": 0.0}], 0, 0))
    events.sort(key=lambda e: e.t)
    ts = np.arange(0, 10, 0.5)
    gt = {"t": ts, "xy": np.array([pos(t) for t in ts]), "yaw": np.zeros(len(ts)),
          "speed": np.full(len(ts), np.hypot(*vel)), "visibility": [""] * len(ts),
          "num_lidar_pts": np.zeros(len(ts)), "num_radar_pts": np.zeros(len(ts))}
    kfs = [Keyframe(float(t), f"s{i}", np.zeros(2), 0.0) for i, t in enumerate(ts)]
    return Scene("synthetic", "", "", {}, 0, kfs, gt, events, None)


def test_steps_are_in_time_order_and_camera_class_only_does_not_update():
    track = run(synthetic_scene(), Config())
    times = [s.t for s in track.steps]
    assert times == sorted(times)
    assert "camera" not in {s.sensor for s in track.steps}


def test_toggling_camera_in_class_only_mode_changes_nothing():
    scene = synthetic_scene()
    a = run(scene, Config(use_camera=True))
    b = run(scene, Config(use_camera=False))
    np.testing.assert_allclose(a.steps[-1].x, b.steps[-1].x)


def test_fusion_converges_for_every_model_and_filter():
    scene = synthetic_scene()
    for model in ["CV", "CA", "CTRV"]:
        for nl in ["EKF", "UKF"]:
            ev = evaluate(scene, run(scene, Config(model=model, nonlinear=nl, sigma_a=0.5)))
            assert ev.error[-5:].max() < 0.5, (model, nl, ev.error[-5:])


def test_radar_alone_tracks():
    scene = synthetic_scene()
    ev = evaluate(scene, run(scene, Config(use_lidar=False)))
    assert ev.error[-5:].max() < 1.5


def test_nees_is_consistent_when_noise_matches():
    scene = synthetic_scene()
    ev = evaluate(scene, run(scene, Config(sigma_a=0.3, lidar_std=0.3, radar_range_std=0.5,
                                           radar_bearing_std=0.01, radar_range_rate_std=0.2)))
    assert ev.inside_95 > 0.8


def test_state_at_does_not_modify_track():
    track = run(synthetic_scene(), Config())
    before = track.steps[5].x.copy()
    track.state_at(track.steps[5].t + 0.3)
    np.testing.assert_array_equal(before, track.steps[5].x)
