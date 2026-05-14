from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from text2humanoid.contracts.trajectory import TrajectorySource


class TransitionMode(str, Enum):
    APPEND = "append"
    REPLACE = "replace"
    CROSSFADE = "crossfade"


@dataclass(slots=True)
class TrajectoryPoint:
    t: float
    x: float
    y: float
    z: float

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


@dataclass(slots=True)
class TrajectoryCondition:
    """External trajectory input — the API-level contract.

    waypoints:        hand-authored spatial waypoints (high-level source).
    token_aligned_traj: pre-computed FloodNet token features (low-level compat).
    token_mask:       per-token validity mask for token_aligned_traj.

    Use to_source() to convert into the unified TrajectorySource before
    feeding into the planner pipeline.
    """

    waypoints: list[TrajectoryPoint] = field(default_factory=list)
    token_aligned_traj: list[list[float]] | None = None
    token_mask: list[float] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_source(self) -> TrajectorySource:
        from text2humanoid.contracts.trajectory import TrajectorySource, TrajectorySourceType

        if self.token_aligned_traj is not None:
            return TrajectorySource(
                source_type=TrajectorySourceType.TOKEN_ALIGNED.value,
                token_aligned_traj=self.token_aligned_traj,
                token_mask=self.token_mask,
                metadata=self.metadata,
            )
        if self.waypoints:
            return TrajectorySource(
                source_type=TrajectorySourceType.WAYPOINTS.value,
                waypoints=self.waypoints,
                metadata=self.metadata,
            )
        return TrajectorySource(
            source_type=TrajectorySourceType.WAYPOINTS.value,
            metadata=self.metadata,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "waypoints": [p.to_dict() for p in self.waypoints],
            "token_aligned_traj": self.token_aligned_traj,
            "token_mask": self.token_mask,
            "metadata": self.metadata,
        }


@dataclass(slots=True)
class PromptCommand:
    text: str
    trajectory: TrajectoryCondition | None = None
    submit_time: float = 0.0
    transition_mode: str = TransitionMode.APPEND.value
    command_id: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "trajectory": None if self.trajectory is None else self.trajectory.to_dict(),
            "submit_time": self.submit_time,
            "transition_mode": self.transition_mode,
            "command_id": self.command_id,
            "metadata": self.metadata,
        }
