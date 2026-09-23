"""Resumable downloads and selective extraction of the nuScenes mini archive."""

import json
import shutil
import sys
import tarfile
import urllib.request
import zipfile
from pathlib import Path

NUSCENES_MINI_URL = "https://www.nuscenes.org/data/v1.0-mini.tgz"
MEGVII_URL = "https://www.nuscenes.org/data/detection-megvii.zip"
YOLOX_URL = "https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_s.onnx"


def download(url, dest):
    """Download url to dest, resuming a partial file. Skips a complete file."""
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(urllib.request.Request(url, method="HEAD")) as r:
        total = int(r.headers["Content-Length"])
    have = dest.stat().st_size if dest.exists() else 0
    if have == total:
        print(f"  {dest.name}: already downloaded")
        return dest
    if have > total:
        have = 0
    req = urllib.request.Request(url, headers={"Range": f"bytes={have}-"} if have else {})
    with urllib.request.urlopen(req) as r, open(dest, "ab" if have else "wb") as f:
        done, last = have, -1
        while chunk := r.read(1 << 20):
            f.write(chunk)
            done += len(chunk)
            pct = int(100 * done / total)
            if pct != last:
                sys.stdout.write(f"\r  {dest.name}: {pct:3d} % of {total / 1e9:.2f} GB")
                sys.stdout.flush()
                last = pct
    print()
    if dest.stat().st_size != total:
        raise RuntimeError(f"Incomplete download for {dest}, rerun prep to resume.")
    return dest


ALL_TABLES = ["attribute", "calibrated_sensor", "category", "ego_pose", "instance", "log", "map",
              "sample", "sample_annotation", "sample_data", "scene", "sensor", "visibility"]


def _read_tables(tgz, root):
    """First pass: extract the JSON tables. They sit at the start of the archive,
    so this stops after reading a few MB."""
    seen = set()
    with tarfile.open(tgz, "r|gz") as tar:
        for m in tar:
            if m.name.startswith("v1.0-mini/") and m.name.endswith(".json"):
                tar.extract(m, root, filter="data")
                seen.add(Path(m.name).stem)
                if seen >= set(ALL_TABLES):
                    break
    missing = set(ALL_TABLES) - seen
    if missing:
        raise RuntimeError(f"Tables missing from the archive: {sorted(missing)}")
    return {t: _load(root, t) for t in ["scene", "sample", "sample_data", "calibrated_sensor",
                                        "sensor"]}


def _load(root, table):
    return json.loads((Path(root) / "v1.0-mini" / f"{table}.json").read_text())


def extract_scenes(tgz, root, scene_names):
    """Extract the tables plus the sensor files of the chosen scenes only.

    Kept: LIDAR_TOP keyframes, every RADAR sweep, every camera image, and the map
    masks (the devkit refuses to start without them).
    Returns the number of sensor files extracted.
    """
    root = Path(root)
    tables = _read_tables(tgz, root)

    scenes = {s["name"]: s for s in tables["scene"]}
    unknown = set(scene_names) - set(scenes)
    if unknown:
        raise ValueError(f"Scenes not in nuScenes mini: {sorted(unknown)}")
    samples = {s["token"]: s for s in tables["sample"]}
    keep_samples = set()
    for name in scene_names:
        t = scenes[name]["first_sample_token"]
        while t:
            keep_samples.add(t)
            t = samples[t]["next"]
    channel = {c["token"]: c["channel"] for c in tables["sensor"]}
    cs_channel = {c["token"]: channel[c["sensor_token"]] for c in tables["calibrated_sensor"]}

    needed = set()
    for sd in tables["sample_data"]:
        if sd["sample_token"] not in keep_samples:
            continue
        ch = cs_channel[sd["calibrated_sensor_token"]]
        if ch == "LIDAR_TOP" and not sd["is_key_frame"]:
            continue
        needed.add(sd["filename"])

    needed |= {f"maps/{m['filename'].split('/')[-1]}" for m in _load(root, "map")}
    present = {f for f in needed if (root / f).exists()}
    todo = needed - present
    if not todo:
        print(f"  {len(needed)} sensor files already extracted")
        return len(needed)
    print(f"  extracting {len(todo)} sensor files (one pass over the archive, a few minutes)")
    with tarfile.open(tgz, "r|gz") as tar:
        for m in tar:
            if m.name in todo:
                tar.extract(m, root, filter="data")
                todo.discard(m.name)
                if not todo:
                    break
    if todo:
        raise RuntimeError(f"{len(todo)} files not found in the archive, e.g. {next(iter(todo))}")
    return len(needed)


def extract_megvii_val(zip_path, dest):
    dest = Path(dest)
    if dest.exists():
        return dest
    with zipfile.ZipFile(zip_path) as z, z.open("megvii_val.json") as src, open(dest, "wb") as out:
        shutil.copyfileobj(src, out, 1 << 20)
    return dest
