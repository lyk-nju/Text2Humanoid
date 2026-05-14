from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any


class SessionPhase(str, Enum):
    IDLE = "idle"
    WARMING = "warming"
    RUNNING = "running"
    DEGRADED = "degraded"
    RESETTING = "resetting"
    STOPPED = "stopped"
    ERROR = "error"


@dataclass(slots=True)
class RuntimeStatus:
    session_id: str
    phase: str = SessionPhase.IDLE.value
    buffer_frames: int = 0
    sim_time: float = 0.0
    latest_chunk_id: str = ""
    planner_latency_ms: float = 0.0
    retarget_latency_ms: float = 0.0
    runtime_latency_ms: float = 0.0
    falls: int = 0
    errors: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
