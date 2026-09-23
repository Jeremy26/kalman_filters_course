"""Load one prepared scene from the cache (see DESIGN.md, section "Cache format").

Times are converted to seconds since the first keyframe. Positions are in the
scene frame: nuScenes global coordinates minus the ego position at the first
keyframe, axes not rotated.
"""

import json
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import numpy as np

SCHEMA_VERSION = 1


@dataclass
class Event:
    """One sensor reading: a LiDAR keyframe, a RADAR sweep or a camera image."""

    t: float
    sensor: str                 # "lidar", "radar" or "camera"
    channel: str                # e.g. "RADAR_FRONT_LEFT"
    sensor_xy: np.ndarray
    sensor_yaw: float
    detections: list
    target: int | None          # index in detections of the one gated to the target
    frame: int                  # index of the keyframe interval this event belongs to
    index_in_frame: int = -1    # position in that frame's JSON file (links RADAR raw points)

    @property
    def target_detection(self):
        return None if self.target is None else self.detections[self.target]


@dataclass
class Keyframe:
    t: float
    sample_token: str
    ego_xy: np.ndarray
    ego_yaw: float


@dataclass
class Scene:
    name: str
    description: str
    lesson: str
    target: dict
    t0_us: int
    keyframes: list
    gt: dict                    # arrays: t, xy, yaw, speed, visibility, num_lidar_pts, num_radar_pts
    events: list = field(repr=False)
    directory: Path = field(repr=False)

    @property
    def duration(self):
        return max(self.keyframes[-1].t, self.events[-1].t if self.events else 0.0)

    def frame_at(self, t):
        """Index of the last keyframe at or before t."""
        times = [k.t for k in self.keyframes]
        return max(0, int(np.searchsorted(times, t, side="right")) - 1)

    def points(self, frame):
        return _load_points(str(self.directory), self.keyframes[frame].sample_token)

    def ego_pose_at(self, t):
        """Ego position and yaw at t, interpolated between keyframes."""
        times = np.array([k.t for k in self.keyframes])
        xy = np.array([k.ego_xy for k in self.keyframes])
        yaw = np.unwrap([k.ego_yaw for k in self.keyframes])
        return (np.array([np.interp(t, times, xy[:, 0]), np.interp(t, times, xy[:, 1])]),
                float(np.interp(t, times, yaw)))

    def gt_at(self, t):
        """Ground truth position interpolated between keyframes, for display only."""
        g = self.gt
        return np.array([np.interp(t, g["t"], g["xy"][:, 0]), np.interp(t, g["t"], g["xy"][:, 1])])


@lru_cache(maxsize=256)
def _load_points(directory, sample_token):
    with np.load(Path(directory) / "frames" / f"{sample_token}.npz") as data:
        return {k: data[k] for k in data.files}


def available_scenes(cache_dir):
    manifest = Path(cache_dir) / "manifest.json"
    if not manifest.exists():
        return []
    m = json.loads(manifest.read_text())
    if m.get("schema_version") != SCHEMA_VERSION:
        raise RuntimeError(
            f"Cache schema {m.get('schema_version')} does not match {SCHEMA_VERSION}. Rerun prep.")
    return m["scenes"]


def load_scene(cache_dir, name):
    d = Path(cache_dir) / name
    meta = json.loads((d / "scene.json").read_text())
    t0 = meta["keyframes"][0]["t_us"]
    sec = lambda t_us: (t_us - t0) * 1e-6

    keyframes = [Keyframe(sec(k["t_us"]), k["sample_token"], np.array(k["ego_xy_yaw"][:2]),
                          k["ego_xy_yaw"][2]) for k in meta["keyframes"]]
    g = meta["ground_truth"]
    gt = {
        "t": np.array([sec(a["t_us"]) for a in g]),
        "xy": np.array([a["xy"] for a in g]),
        "yaw": np.array([a["yaw"] for a in g]),
        "speed": np.array([a["speed"] for a in g]),
        "visibility": [a["visibility"] for a in g],
        "num_lidar_pts": np.array([a["num_lidar_pts"] for a in g]),
        "num_radar_pts": np.array([a["num_radar_pts"] for a in g]),
    }

    events = []
    for i, k in enumerate(meta["keyframes"]):
        frame = json.loads((d / "frames" / f"{k['sample_token']}.json").read_text())
        for j, e in enumerate(frame["events"]):
            events.append(Event(sec(e["t_us"]), e["sensor"], e["channel"],
                                np.array(e["sensor_xy_yaw"][:2]), e["sensor_xy_yaw"][2],
                                e["detections"], e["target"], i, j))
    events.sort(key=lambda e: e.t)

    return Scene(meta["name"], meta["description"], meta.get("lesson", ""), meta["target"], t0,
                 keyframes, gt, events, d)
