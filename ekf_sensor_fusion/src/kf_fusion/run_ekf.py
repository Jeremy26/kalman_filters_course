"""Command-line entry point: run the fusion filter on a real nuScenes track.

Examples
--------
Run on the bundled real track and write a Foxglove recording::

    python -m kf_fusion.run_ekf --track data/track_ed634e83.npz --mcap outputs/ekf.mcap

Compare lidar-only / radar-only / fused (the "what does radar add?" experiment)::

    python -m kf_fusion.run_ekf --track data/track_437fe13d.npz --ablation
"""

from __future__ import annotations

import argparse
from pathlib import Path

from .dataset import load_track
from .pipeline import run_fusion


def _print_summary(title: str, s: dict) -> None:
    print(f"\n=== {title} ===")
    print(f"  RMSE  pos={s['rmse_pos']:.2f} m   vel={s['rmse_vel']:.2f} m/s   "
          f"(px={s['rmse_px']:.2f} py={s['rmse_py']:.2f} vx={s['rmse_vx']:.2f} vy={s['rmse_vy']:.2f})")
    print(f"  NIS<95%  lidar={s['nis_lidar_below_95']:.2f}  "
          f"radar={s['nis_radar_below_95']:.2f}   (target ~0.95)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--track", type=Path, required=True, help="extracted .npz track")
    parser.add_argument("--noise-ax", type=float, default=4.0)
    parser.add_argument("--noise-ay", type=float, default=4.0)
    parser.add_argument("--ablation", action="store_true",
                        help="Also report lidar-only and radar-only for comparison.")
    args = parser.parse_args()

    track = load_track(args.track)
    print(f"track: {track.instance[:8]} ({track.category}) — "
          f"{len(track.measurements)} real measurements")

    fused = run_fusion(track, args.noise_ax, args.noise_ay)
    _print_summary("FUSED (lidar + radar)", fused.summary())
    print("\n  For a Foxglove recording of the full scene (camera + lidar + radar +")
    print("  tracking), run: python scripts/make_scene_mcap.py")

    if args.ablation:
        lidar_only = run_fusion(track, args.noise_ax, args.noise_ay,
                                use_lidar=True, use_radar=False)
        radar_only = run_fusion(track, args.noise_ax, args.noise_ay,
                                use_lidar=False, use_radar=True)
        _print_summary("LIDAR ONLY", lidar_only.summary())
        _print_summary("RADAR ONLY", radar_only.summary())


if __name__ == "__main__":
    main()
