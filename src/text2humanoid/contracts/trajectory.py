from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from text2humanoid.contracts.commands import TrajectoryPoint


class TrajectorySourceType(str, Enum):
    """External trajectory source categories.

    WAYPOINTS:    hand-authored or UI-generated waypoints (high-level).
    TOKEN_ALIGNED: pre-computed FloodNet token-aligned features (low-level compat).
    CANONICAL:    already-normalized CanonicalTrajectory (pass-through).
    """

    WAYPOINTS = "waypoints"
    TOKEN_ALIGNED = "token_aligned"
    CANONICAL = "canonical"


@dataclass(slots=True)
class TrajectorySource:
    """Unified external trajectory input — the single entry point before
    CanonicalTrajectory.

    All trajectory sources (waypoints, token-aligned features, pre-built
    canonical trajectories) are wrapped in this container so the planner
    layer only needs a single dispatch path.
    """

    source_type: str
    waypoints: list[TrajectoryPoint] = field(default_factory=list)
    token_aligned_traj: list[list[float]] | None = None
    token_mask: list[float] | None = None
    canonical: CanonicalTrajectory | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


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
