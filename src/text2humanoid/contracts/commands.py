from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any


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
    waypoints: list[TrajectoryPoint] = field(default_factory=list)
    token_aligned_traj: list[list[float]] | None = None
    token_mask: list[float] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

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
