from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass(slots=True)
class CanonicalTrajectory:
    """Normalized intermediate representation for all trajectory sources.

    All trajectory inputs (waypoints, token-aligned traj, future sources)
    converge to this single representation before being compiled into
    FloodNet model inputs.
    """

    times: np.ndarray       # (T,) seconds, strictly monotonic
    xz: np.ndarray          # (T, 2) world x, z
    yaw: np.ndarray         # (T,) radians, heading direction
    valid_mask: np.ndarray  # (T,) bool, True where trajectory data exists
    fps: int                # trajectory sample rate
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.times = np.asarray(self.times, dtype=np.float32)
        self.xz = np.asarray(self.xz, dtype=np.float32)
        self.yaw = np.asarray(self.yaw, dtype=np.float32)
        self.valid_mask = np.asarray(self.valid_mask, dtype=bool)
        n = len(self.times)
        if self.xz.shape != (n, 2):
            raise ValueError(f"xz must have shape ({n}, 2), got {self.xz.shape}")
        if self.yaw.shape != (n,):
            raise ValueError(f"yaw must have shape ({n},), got {self.yaw.shape}")
        if self.valid_mask.shape != (n,):
            raise ValueError(f"valid_mask must have shape ({n},), got {self.valid_mask.shape}")

    @property
    def num_frames(self) -> int:
        return int(self.times.shape[0])

    @property
    def duration(self) -> float:
        return float(self.times[-1] - self.times[0]) if self.num_frames > 1 else 0.0
