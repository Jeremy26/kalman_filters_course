"""Smoke test of the Rerun recording, on the real cache when it exists."""

import uuid
from pathlib import Path

import pytest

from kflab import Config, available_scenes, evaluate, load_scene, run

CACHE = Path(__file__).resolve().parents[1] / "cache"
rerun_log = pytest.importorskip("app.rerun_log")


@pytest.mark.skipif(not available_scenes(CACHE), reason="no prepared cache")
def test_recording_builds_for_every_scene_and_model():
    for name in available_scenes(CACHE):
        s = load_scene(CACHE, name)
        batches = rerun_log.sensor_batches(s)
        runs = []
        for model in ("CV", "CTRV"):
            track = run(s, Config(model=model, nonlinear="UKF"))
            runs.append((model, track, evaluate(s, track)))
        data = rerun_log.build_rrd(s, batches, runs, {"lidar", "radar"}, uuid.uuid4().hex)
        assert len(data) > 100_000
