from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from text2humanoid.contracts.commands import PromptCommand
    from text2humanoid.contracts.chunks import HumanMotionChunk
    from text2humanoid.planner.floodnet_service import FloodNetPlannerService


@dataclass
class PlannerSession:
    """Planner-layer streaming state for continuous generation.

    Maintains the session-scoped prompt, trajectory, chunk cursor, and
    timeline so that the orchestrator does not need to re-derive context
    on each refill cycle.
    """

    command: PromptCommand
    chunk_index: int = 0
    next_start_time: float = 0.0
    metadata: dict = field(default_factory=dict)

    def advance(self, chunk_end_time: float) -> None:
        self.chunk_index += 1
        self.next_start_time = max(self.next_start_time, chunk_end_time)


class StreamPlannerDriver:
    """Thin driver that wraps FloodNetPlannerService with a PlannerSession."""

    def __init__(self, planner: FloodNetPlannerService) -> None:
        self.planner = planner
        self._session: PlannerSession | None = None

    @property
    def session(self) -> PlannerSession | None:
        return self._session

    def start_session(self, command: PromptCommand) -> PlannerSession:
        self._session = PlannerSession(command=command)
        return self._session

    def transition(self, command: PromptCommand) -> None:
        """Replace the active command while preserving timeline continuity.

        When a new command arrives in a running session, this replaces the
        planner's active command without resetting chunk_index or
        next_start_time, so the next generated chunk continues the timeline.
        """
        if self._session is None:
            self.start_session(command)
        else:
            self._session.command = command

    def reset(self) -> None:
        self._session = None

    def generate_next_chunk(self, feature_length: int | None = None) -> HumanMotionChunk | None:
        """Generate the next chunk in the current session.

        Returns None if no session is active.
        """
        if self._session is None:
            return None
        chunk = self.planner.generate_chunk(
            command=self._session.command,
            start_time=self._session.next_start_time,
            feature_length=feature_length,
        )
        self._session.advance(chunk.end_time)
        return chunk
