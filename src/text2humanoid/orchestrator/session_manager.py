from __future__ import annotations

import uuid
from dataclasses import dataclass, field

from text2humanoid.contracts.commands import PromptCommand
from text2humanoid.contracts.status import RuntimeStatus, SessionPhase
from text2humanoid.orchestrator.pipeline_coordinator import PipelineCoordinator
from text2humanoid.orchestrator.timeline import SessionTimeline
from text2humanoid.orchestrator.state_machine import can_transition


@dataclass(slots=True)
class SessionContext:
    timeline: SessionTimeline
    status: RuntimeStatus
    next_start_time: float = 0.0
    metadata: dict = field(default_factory=dict)


class SessionManager:
    def __init__(self, coordinator: PipelineCoordinator | None = None) -> None:
        self._sessions: dict[str, SessionContext] = {}
        self._coordinator = coordinator

    def create_session(self) -> str:
        session_id = uuid.uuid4().hex
        timeline = SessionTimeline(session_id=session_id)
        status = RuntimeStatus(session_id=session_id)
        self._sessions[session_id] = SessionContext(timeline=timeline, status=status)
        if self._coordinator is not None:
            self._coordinator.runtime.ensure_session(session_id)
        return session_id

    def get_status(self, session_id: str) -> RuntimeStatus:
        return self._sessions[session_id].status

    def push_command(self, session_id: str, command: PromptCommand) -> None:
        ctx = self._sessions[session_id]
        ctx.timeline.append(command)
        if self._coordinator is None:
            return
        if can_transition(ctx.status.phase, SessionPhase.WARMING.value) and ctx.status.phase == SessionPhase.IDLE.value:
            ctx.status = self._coordinator.warmup(session_id, command.text)
        ctx.status = self._coordinator.run_once(session_id, command, start_time=ctx.next_start_time)
        latest_chunk_end = float(ctx.status.metadata.get("latest_chunk_end_time", ctx.status.sim_time))
        ctx.next_start_time = max(ctx.next_start_time, latest_chunk_end)

    def reset_session(self, session_id: str) -> None:
        ctx = self._sessions[session_id]
        ctx.timeline.commands.clear()
        ctx.next_start_time = 0.0
        if self._coordinator is not None:
            self._coordinator.planner.reset()
            ctx.status = self._coordinator.runtime.reset_session(session_id)
        ctx.status.phase = SessionPhase.RESETTING.value

    def stop_session(self, session_id: str) -> None:
        self._sessions[session_id].status.phase = SessionPhase.STOPPED.value

    def run_refill_cycle(self, session_id: str, watermark_frames: int = 20, max_chunks: int = 10) -> int:
        """Produce additional chunks if the runtime buffer is below watermark.

        Reuses the last command from the session timeline as the template for
        each pipeline run.  Returns the number of chunks produced.
        """
        ctx = self._sessions[session_id]
        if self._coordinator is None:
            return 0
        chunks_produced = 0
        for _ in range(max_chunks):
            status = self._coordinator.runtime.get_status(session_id)
            if status.buffer_frames >= watermark_frames:
                break
            last_cmd = ctx.timeline.latest_command
            if last_cmd is None:
                break
            ctx.status = self._coordinator.run_once(session_id, last_cmd, start_time=ctx.next_start_time)
            latest_chunk_end = float(ctx.status.metadata.get("latest_chunk_end_time", ctx.status.sim_time))
            ctx.next_start_time = max(ctx.next_start_time, latest_chunk_end)
            chunks_produced += 1
            if ctx.status.phase in (SessionPhase.ERROR.value, SessionPhase.STOPPED.value):
                break
        return chunks_produced
