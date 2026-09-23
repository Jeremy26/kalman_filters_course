"""python -m prep --accept-nuscenes-license

Downloads nuScenes mini, the public MEGVII LiDAR detections and the YOLOX-s model,
extracts only the scenes listed in scenes.yaml and writes the compact cache used by
the app. Safe to rerun: finished downloads and extracted files are reused.
"""

import argparse
import shutil
import sys
import time
from pathlib import Path

import yaml

from . import download

LICENSE_TEXT = """\
nuScenes is released by Motional under CC BY-NC-SA 4.0, for non-commercial use.
Terms of use: https://www.nuscenes.org/terms-of-use
By passing --accept-nuscenes-license you confirm you accept these terms."""


def main():
    root = Path(__file__).resolve().parents[1]
    ap = argparse.ArgumentParser(prog="python -m prep", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cache", default=str(root / "cache"), help="output directory")
    ap.add_argument("--scenes", default=str(root / "scenes.yaml"))
    ap.add_argument("--accept-nuscenes-license", action="store_true")
    ap.add_argument("--lidar-detections", default=None,
                    help="nuScenes-format detection JSON (e.g. CenterPoint val). "
                         "Default: MEGVII val, downloaded from nuscenes.org")
    ap.add_argument("--camera-stride", type=int, default=1,
                    help="run YOLO on one image out of N (faster prep)")
    ap.add_argument("--delete-archives", action="store_true",
                    help="delete the 4.6 GB of downloads when done")
    args = ap.parse_args()

    print(LICENSE_TEXT)
    if not args.accept_nuscenes_license:
        sys.exit("\nRerun with --accept-nuscenes-license to continue.")

    spec = yaml.safe_load(Path(args.scenes).read_text())
    names = [s["name"] for s in spec["scenes"]]
    cache = Path(args.cache)
    dl, raw = cache / "_downloads", cache / "_nuscenes"
    t0 = time.time()

    print("\n[1/4] Downloads")
    tgz = download.download(download.NUSCENES_MINI_URL, dl / "v1.0-mini.tgz")
    onnx = download.download(download.YOLOX_URL, dl / "yolox_s.onnx")
    if args.lidar_detections:
        det_path, det_source = Path(args.lidar_detections), Path(args.lidar_detections).name
    else:
        z = download.download(download.MEGVII_URL, dl / "detection-megvii.zip")
        det_path, det_source = download.extract_megvii_val(z, dl / "megvii_val.json"), "megvii_val.json"

    print("\n[2/4] Extracting the chosen scenes")
    download.extract_scenes(tgz, raw, names)

    print("\n[3/4] Loading nuScenes tables and detections")
    # Imported here so that `--help` works without the heavy dependencies.
    from nuscenes.nuscenes import NuScenes

    from . import build, camera, lidar
    nusc = NuScenes(version="v1.0-mini", dataroot=str(raw), verbose=False)
    tokens = [s["token"] for sc in nusc.scene if sc["name"] in names
              for s in build.scene_samples(nusc, sc)]
    detections = lidar.load_detections(det_path, tokens)
    yolo = camera.YoloX(onnx)

    print("\n[4/4] Building the cache (YOLOX on CPU takes a few minutes)")
    for s in spec["scenes"]:
        build.build_scene(nusc, s, detections, yolo, cache, args.camera_stride)
    build.write_manifest(cache, names, det_source)

    shutil.rmtree(raw)
    if args.delete_archives:
        shutil.rmtree(dl)
    size = sum(f.stat().st_size for n in names for f in (cache / n).rglob("*") if f.is_file())
    print(f"\nDone in {time.time() - t0:.0f} s. Cache: {size / 1e6:.1f} MB in {cache}")


if __name__ == "__main__":
    main()
