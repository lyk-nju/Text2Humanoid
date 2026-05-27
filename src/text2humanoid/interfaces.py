from __future__ import annotations

from collections.abc import Iterator
from typing import Protocol, runtime_checkable

from text2humanoid.contracts.pipeline import (
    GenerateRequest,
    GeneratedMotion,
    RetargetInput,
    RobotMotion,
    TrackerInput,
    TrackerStatus,
)


@runtime_checkable
class GenerationBackend(Protocol):
    """Interface implemented by local, HTTP, ROS, or mock generation backends."""

    def generate_chunk(self, request: GenerateRequest) -> GeneratedMotion: ...

    def stream_chunks(self, request: GenerateRequest) -> Iterator[GeneratedMotion]: ...


@runtime_checkable
class RetargetInputBridge(Protocol):
    """Pure conversion from generated motion representation to retarget input."""

    def convert(self, motion: GeneratedMotion) -> RetargetInput: ...


@runtime_checkable
class RetargetBackend(Protocol):
    """Retarget model/runtime interface, e.g. MakeTrackingEasy."""

    def retarget(self, input_chunk: RetargetInput) -> RobotMotion: ...


@runtime_checkable
class TrackerInputBridge(Protocol):
    """Pure conversion from canonical robot motion to tracker-specific input."""

    def convert(self, motion: RobotMotion) -> TrackerInput: ...


@runtime_checkable
class TrackerBackend(Protocol):
    """Runtime/tracker interface, e.g. BFM-Zero or motion_tracking."""

    def push(self, tracker_input: TrackerInput) -> TrackerStatus: ...
