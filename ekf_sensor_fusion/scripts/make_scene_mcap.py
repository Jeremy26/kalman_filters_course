"""Generate the rich single-object-tracking MCAP for a nuScenes scene.

Needs the raw dataset (run scripts/download_nuscenes.py first). Produces a
Foxglove recording with the camera + 2D detections, the full lidar cloud, radar
returns, the tf tree, and the tracking output.

    python scripts/make_scene_mcap.py --instance ed634e83 --out outputs/scene.mcap

Camera pick: pass --pick U V (pixel in CAM_FRONT of the object you want) to
select a different target; otherwise the demo instance is auto-selected.
"""
from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--instance", default="ed634e83")
    ap.add_argument("--dataroot", default="data/nuscenes")
    ap.add_argument("--version", default="v1.0-mini")
    ap.add_argument("--weights", default="models/yolo_det.pt")
    ap.add_argument("--out", type=Path, default=Path("outputs/scene.mcap"))
    ap.add_argument("--pick", type=float, nargs=2, default=None,
                    metavar=("U", "V"), help="pixel in CAM_FRONT to select the target")
    args = ap.parse_args()

    from nuscenes.nuscenes import NuScenes
    from kf_fusion import scene, detection

    nusc = NuScenes(version=args.version, dataroot=args.dataroot, verbose=False)
    token = next(i["token"] for i in nusc.instance if i["token"].startswith(args.instance))
    det = detection.Yolo2DDetector(weights=args.weights)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    res = scene.run_scene_sot(nusc, token, args.out, detector=det,
                              select=tuple(args.pick) if args.pick else None)
    print(f"\n{res['frames']} frames, position RMSE {res['rmse_pos']:.2f} m")
    print(f"wrote {res['mcap']} — open in Foxglove with layouts/scene.json")


if __name__ == "__main__":
    main()
