"""Generate a realistic lidar + radar measurement log for the EKF project.

We simulate a single vehicle driving a smooth, curved trajectory (a constant
turn-rate / constant-velocity "CTRV" path), then produce noisy measurements
from two sensors mounted at the origin:

* lidar  -> Cartesian (px, py)              + Gaussian noise
* radar  -> polar (rho, phi, rho_dot)       + Gaussian noise

The output file uses the well-known fusion-log format so it can be swapped for
the classic Udacity file or a nuScenes export without touching the filter::

    L  px  py  timestamp  gt_px gt_py gt_vx gt_vy
    R  rho phi rho_dot  timestamp  gt_px gt_py gt_vx gt_vy

Ground truth is included on every line so students can score RMSE directly.

The random stream is fully seeded, so the dataset is identical on every machine
-- important for a graded course. Run::

    python data/generate_dataset.py --out data/fusion_log.txt
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def ground_truth_trajectory(n: int, dt: float) -> np.ndarray:
    """A smooth curved path: velocity turns at a constant rate.

    Returns an (n, 4) array of [px, py, vx, vy] ground-truth states.
    """
    speed = 8.0        # m/s, ~29 km/h
    yaw_rate = 0.18    # rad/s, a gentle continuous curve
    px, py, yaw = 0.0, 0.0, 0.3
    states = []
    for _ in range(n):
        vx = speed * np.cos(yaw)
        vy = speed * np.sin(yaw)
        states.append([px, py, vx, vy])
        px += vx * dt
        py += vy * dt
        yaw += yaw_rate * dt
    return np.asarray(states)


def simulate(
    n: int = 500,
    dt: float = 0.05,
    seed: int = 26,
    occlusion: tuple[int, int] | None = (200, 260),
) -> list[str]:
    """Simulate an interleaved lidar/radar stream.

    ``occlusion`` is a ``(start, end)`` index window during which the **lidar is
    blocked** (e.g. the target passes behind a truck, or heavy rain). Radar,
    which penetrates far better, keeps returning. This is the scenario that
    makes fusion genuinely worth it: lidar-only goes blind and drifts, while the
    fused filter coasts on radar and stays locked on. Pass ``None`` to disable.
    """
    rng = np.random.default_rng(seed)
    gt = ground_truth_trajectory(n, dt)

    # Sensor noise (std dev). Matches the R matrices used by the filter.
    lidar_std = np.array([0.15, 0.15])            # m
    radar_std = np.array([0.3, 0.03, 0.3])        # m, rad, m/s

    lines: list[str] = []
    t = 1_477_010_443_000_000  # microseconds, arbitrary epoch (Udacity-style)
    for i in range(n):
        px, py, vx, vy = gt[i]
        t += int(dt * 1e6)
        lidar_blocked = occlusion is not None and occlusion[0] <= i < occlusion[1]
        # Alternate the two sensors, as real async sensors interleave.
        if i % 2 == 0:
            if lidar_blocked:
                continue  # occluded: no lidar return this step
            zx = px + rng.normal(0, lidar_std[0])
            zy = py + rng.normal(0, lidar_std[1])
            lines.append(
                f"L {zx:.6f} {zy:.6f} {t} {px:.6f} {py:.6f} {vx:.6f} {vy:.6f}"
            )
        else:
            rho = np.hypot(px, py)
            phi = np.arctan2(py, px)
            rho_dot = (px * vx + py * vy) / max(rho, 1e-6)
            rho += rng.normal(0, radar_std[0])
            phi += rng.normal(0, radar_std[1])
            rho_dot += rng.normal(0, radar_std[2])
            lines.append(
                f"R {rho:.6f} {phi:.6f} {rho_dot:.6f} {t} "
                f"{px:.6f} {py:.6f} {vx:.6f} {vy:.6f}"
            )
    return lines


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path(__file__).with_name("fusion_log.txt"))
    parser.add_argument("--n", type=int, default=500)
    parser.add_argument("--dt", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=26)
    args = parser.parse_args()

    lines = simulate(args.n, args.dt, args.seed)
    args.out.write_text("\n".join(lines) + "\n")
    print(f"Wrote {len(lines)} measurements to {args.out}")


if __name__ == "__main__":
    main()
