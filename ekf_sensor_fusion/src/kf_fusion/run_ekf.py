"""Command-line entry point: run the fusion filter and print a metrics report.

Examples
--------
Run on the bundled dataset and write a Foxglove recording::

    python -m kf_fusion.run_ekf --data data/fusion_log.txt --mcap outputs/ekf.mcap

Compare lidar-only / radar-only / fused (the classic "why fuse?" experiment)::

    python -m kf_fusion.run_ekf --data data/fusion_log.txt --ablation
"""

from __future__ import annotations

import argparse
from pathlib import Path

from .dataset import read_log
from .pipeline import run_fusion


def _print_summary(title: str, summary: dict) -> None:
    print(f"\n=== {title} ===")
    print(
        f"  RMSE  px={summary['rmse_px']:.3f}  py={summary['rmse_py']:.3f}  "
        f"vx={summary['rmse_vx']:.3f}  vy={summary['rmse_vy']:.3f}  "
        f"(pos={summary['rmse_pos']:.3f} m)"
    )
    print(
        f"  NIS<95%  lidar={summary['nis_lidar_below_95']:.2f}  "
        f"radar={summary['nis_radar_below_95']:.2f}   (target ~0.95)"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--mcap", type=Path, default=None)
    parser.add_argument("--noise-ax", type=float, default=9.0)
    parser.add_argument("--noise-ay", type=float, default=9.0)
    parser.add_argument("--ablation", action="store_true",
                        help="Also report lidar-only and radar-only for comparison.")
    args = parser.parse_args()

    measurements = read_log(args.data)

    fused = run_fusion(measurements, args.noise_ax, args.noise_ay, mcap_path=args.mcap)
    _print_summary("FUSED (lidar + radar)", fused.summary())
    if args.mcap:
        print(f"\n  Wrote Foxglove recording -> {args.mcap}")
        print("  Open it at app.foxglove.dev and load layouts/ekf_fusion.json")

    if args.ablation:
        lidar_only = run_fusion(measurements, args.noise_ax, args.noise_ay,
                                use_lidar=True, use_radar=False)
        radar_only = run_fusion(measurements, args.noise_ax, args.noise_ay,
                                use_lidar=False, use_radar=True)
        _print_summary("LIDAR ONLY (coasts through occlusion)", lidar_only.summary())
        _print_summary("RADAR ONLY (noisy but never blocked)", radar_only.summary())
        print("\n  -> Fusion tracks through the lidar occlusion that sinks lidar-only,")
        print("     while staying far more precise than radar-only.")


if __name__ == "__main__":
    main()
