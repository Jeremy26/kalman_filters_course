"""Reader for the lidar + radar fusion log.

Each line is one measurement, tagged ``L`` (lidar) or ``R`` (radar), followed by
the sensor reading, a microsecond timestamp, and the ground-truth state. The
same format is produced by ``data/generate_dataset.py`` and by the classic
Udacity fusion dataset, so this reader works for both.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np


@dataclass
class Measurement:
    sensor: str            # "lidar" or "radar"
    z: np.ndarray          # the raw sensor reading
    timestamp: float       # seconds
    gt: np.ndarray         # ground-truth [px, py, vx, vy]

    def initial_state(self) -> np.ndarray:
        """Best first-guess state from this measurement (to seed the filter)."""
        if self.sensor == "lidar":
            return np.array([self.z[0], self.z[1], 0.0, 0.0])
        rho, phi, rho_dot = self.z
        px, py = rho * np.cos(phi), rho * np.sin(phi)
        # Radar range-rate is along the line of sight; a rough velocity seed.
        vx, vy = rho_dot * np.cos(phi), rho_dot * np.sin(phi)
        return np.array([px, py, vx, vy])


def read_log(path: str | Path) -> list[Measurement]:
    return list(iter_log(path))


def iter_log(path: str | Path) -> Iterator[Measurement]:
    with open(path, "r") as fh:
        for raw in fh:
            raw = raw.strip()
            if not raw:
                continue
            parts = raw.split()
            tag = parts[0].upper()
            if tag == "L":
                z = np.array([float(parts[1]), float(parts[2])])
                ts = float(parts[3]) / 1e6
                gt = np.array([float(p) for p in parts[4:8]])
                yield Measurement("lidar", z, ts, gt)
            elif tag == "R":
                z = np.array([float(parts[1]), float(parts[2]), float(parts[3])])
                ts = float(parts[4]) / 1e6
                gt = np.array([float(p) for p in parts[5:9]])
                yield Measurement("radar", z, ts, gt)
            else:  # pragma: no cover - defensive
                raise ValueError(f"Unknown sensor tag {tag!r} in line: {raw}")
