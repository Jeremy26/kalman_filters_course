"""Extract one object's real lidar + radar track from nuScenes.

This is where the *real data* enters the project. Given a nuScenes instance
(one physically tracked vehicle), we walk its keyframes and, at each one, pull
**genuine sensor returns** — not the annotation, the actual measurements:

* **lidar**  -> the centroid of the real LIDAR_TOP points that fall inside the
  object's 3D box, expressed in the global frame. A real, slightly noisy
  position fix.
* **radar**  -> the real radar returns (from all 5 radars) gated to the object
  in bird's-eye view, giving a position **and a Doppler range-rate** — the
  velocity information lidar simply does not have.

Ground truth (for scoring only) is the annotation centre and the devkit's
``box_velocity``. Each sensor keeps its own real timestamp, so the filter sees a
true asynchronous, multi-rate stream.

The output is a small ``.npz`` (a few hundred KB) so the filter, tests and
notebook all run **without** the 4 GB dataset. Re-extraction needs the dataset;
everything downstream does not.

Usage::

    python -m kf_fusion.nuscenes_extract --instance ed634e83 \
        --dataroot data/nuscenes --out data/track_ed634e83.npz
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

RADARS = [
    "RADAR_FRONT",
    "RADAR_FRONT_LEFT",
    "RADAR_FRONT_RIGHT",
    "RADAR_BACK_LEFT",
    "RADAR_BACK_RIGHT",
]


def _to_global(nusc, xyz: np.ndarray, sd_token: str) -> np.ndarray:
    """Transform sensor-frame points (3xN) to the global frame."""
    from pyquaternion import Quaternion

    sd = nusc.get("sample_data", sd_token)
    cs = nusc.get("calibrated_sensor", sd["calibrated_sensor_token"])
    ego = nusc.get("ego_pose", sd["ego_pose_token"])
    p = Quaternion(cs["rotation"]).rotation_matrix @ xyz + np.array(cs["translation"])[:, None]
    p = Quaternion(ego["rotation"]).rotation_matrix @ p + np.array(ego["translation"])[:, None]
    return p


def _sensor_origin_global(nusc, sd_token: str) -> np.ndarray:
    """Global (x, y) position of a sensor at a given sample_data."""
    return _to_global(nusc, np.zeros((3, 1)), sd_token)[:2, 0]


def extract_instance(nusc, instance_token: str, radar_gate_pad: float = 1.0) -> dict:
    """Return a dict of real measurement arrays for one tracked instance."""
    from nuscenes.utils.data_classes import LidarPointCloud, RadarPointCloud, Box
    from nuscenes.utils.geometry_utils import points_in_box
    from pyquaternion import Quaternion

    inst = nusc.get("instance", instance_token)

    events = []          # (timestamp, sensor, z, extra) in time order
    gt_states = []       # (timestamp, px, py, vx, vy)
    obj_lidar_pts = []   # subsampled real object points for visualization
    box_size = None

    ann_tok = inst["first_annotation_token"]
    while ann_tok:
        ann = nusc.get("sample_annotation", ann_tok)
        sample = nusc.get("sample", ann["sample_token"])
        cx, cy, cz = ann["translation"]
        box = Box(ann["translation"], ann["size"], Quaternion(ann["rotation"]))
        if box_size is None:
            box_size = np.array(ann["size"])  # w, l, h

        # -- ground truth (scoring only) --
        v = nusc.box_velocity(ann_tok)  # global m/s, may be nan at ends
        gt_states.append([_sd_time(nusc, sample["data"]["LIDAR_TOP"]),
                          cx, cy, float(v[0]), float(v[1])])

        # -- real LIDAR measurement: centroid of points inside the box --
        lsd = sample["data"]["LIDAR_TOP"]
        lpc = LidarPointCloud.from_file(nusc.get_sample_data_path(lsd))
        lg = _to_global(nusc, lpc.points[:3], lsd)
        mask = points_in_box(box, lg)
        if mask.sum() >= 3:
            centroid = lg[:2, mask].mean(axis=1)
            events.append((_sd_time(nusc, lsd), "lidar", centroid, np.zeros(2)))
            # keep up to 40 real points (object-relative) for the 3D view
            pts = lg[:2, mask] - np.array([cx, cy])[:, None]
            idx = np.linspace(0, pts.shape[1] - 1, min(40, pts.shape[1])).astype(int)
            obj_lidar_pts.append((_sd_time(nusc, lsd), pts[:, idx]))

        # -- real RADAR measurement: BEV-gated returns + Doppler range-rate --
        radius = max(ann["size"][0], ann["size"][1]) / 2 + radar_gate_pad
        for rch in RADARS:
            rsd = sample["data"][rch]
            rpc = RadarPointCloud.from_file(nusc.get_sample_data_path(rsd))
            if rpc.points.shape[1] == 0:
                continue
            rg = _to_global(nusc, rpc.points[:3], rsd)
            sel = np.hypot(rg[0] - cx, rg[1] - cy) < radius
            if not sel.any():
                continue
            # Nearest gated return to the box centre.
            j = np.argmin(np.hypot(rg[0] - cx, rg[1] - cy)[sel])
            gsel = rg[:, sel][:, j]
            # Compensated (ego-motion-removed) velocity -> global, then radial.
            vcomp = rpc.points[8:10, sel][:, j]
            cs = nusc.get("calibrated_sensor",
                          nusc.get("sample_data", rsd)["calibrated_sensor_token"])
            ego = nusc.get("ego_pose",
                           nusc.get("sample_data", rsd)["ego_pose_token"])
            R = (Quaternion(ego["rotation"]).rotation_matrix[:2, :2]
                 @ Quaternion(cs["rotation"]).rotation_matrix[:2, :2])
            vg = R @ vcomp
            s = _sensor_origin_global(nusc, rsd)
            rho = float(np.hypot(gsel[0] - s[0], gsel[1] - s[1]))
            phi = float(np.arctan2(gsel[1] - s[1], gsel[0] - s[0]))
            rho_dot = float(((gsel[0] - s[0]) * vg[0] + (gsel[1] - s[1]) * vg[1]) / max(rho, 1e-6))
            # extra carries the sensor origin so the EKF can build h(x) about it.
            events.append((_sd_time(nusc, rsd), "radar",
                           np.array([rho, phi, rho_dot]), s))

    events.sort(key=lambda e: e[0])
    t0 = events[0][0]

    return {
        "instance": instance_token,
        "category": nusc.get("category", inst["category_token"])["name"],
        "box_size": box_size,
        "t0": t0,
        "ev_time": np.array([e[0] - t0 for e in events]),
        "ev_sensor": np.array([e[1] for e in events]),
        "ev_z": _ragged(np.array([e[2] for e in events], dtype=object)),
        "ev_extra": _ragged(np.array([e[3] for e in events], dtype=object)),
        "gt_time": np.array([g[0] - t0 for g in gt_states]),
        "gt_state": np.array([g[1:] for g in gt_states]),
        "lidar_pts_time": np.array([t - t0 for t, _ in obj_lidar_pts]),
        "lidar_pts": np.array([p for _, p in obj_lidar_pts], dtype=object),
    }


def _sd_time(nusc, sd_token: str) -> float:
    return nusc.get("sample_data", sd_token)["timestamp"] / 1e6  # s


def _ragged(arr):
    return arr  # kept as object array; np.savez pickles it


def save(track: dict, out: Path) -> None:
    np.savez(out, allow_pickle=True, **track)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--instance", required=True, help="instance token (prefix ok)")
    ap.add_argument("--dataroot", default="data/nuscenes")
    ap.add_argument("--version", default="v1.0-mini")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    import warnings
    warnings.filterwarnings("ignore")
    from nuscenes.nuscenes import NuScenes

    nusc = NuScenes(version=args.version, dataroot=args.dataroot, verbose=False)
    token = next(i["token"] for i in nusc.instance if i["token"].startswith(args.instance))
    track = extract_instance(nusc, token)
    save(track, args.out)
    n_l = int((track["ev_sensor"] == "lidar").sum())
    n_r = int((track["ev_sensor"] == "radar").sum())
    print(f"instance {token[:8]} ({track['category']}): "
          f"{len(track['ev_time'])} events  ({n_l} lidar, {n_r} radar)  -> {args.out}")


if __name__ == "__main__":
    main()
