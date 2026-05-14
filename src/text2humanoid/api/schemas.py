from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class TrajectoryPointSchema(BaseModel):
    t: float
    x: float
    y: float
    z: float


class TrajectoryConditionSchema(BaseModel):
    waypoints: list[TrajectoryPointSchema] = Field(default_factory=list)
    token_aligned_traj: list[list[float]] | None = None
    token_mask: list[float] | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class PromptCommandSchema(BaseModel):
    text: str
    trajectory: TrajectoryConditionSchema | None = None
    submit_time: float = 0.0
    transition_mode: str = "append"
    command_id: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)


class CreateSessionResponse(BaseModel):
    session_id: str


class StatusResponse(BaseModel):
    session_id: str
    phase: str
    buffer_frames: int
    sim_time: float
    latest_chunk_id: str
    planner_latency_ms: float
    retarget_latency_ms: float
    runtime_latency_ms: float
    falls: int
    errors: list[str]
    metadata: dict[str, Any]
