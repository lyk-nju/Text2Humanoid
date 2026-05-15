from __future__ import annotations

import threading
import time
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
    refill_active: bool = False
    refill_stop_event: threading.Event | None = None
    metadata: dict = field(default_factory=dict)


class SessionManager:
    def __init__(self, coordinator: PipelineCoordinator | None = None) -> None:
        self._sessions: dict[str, SessionContext] = {}
        self._coordinator = coordinator

    def _mark_stream_done(self, session_id: str) -> None:
        if self._coordinator is None:
            return
        backend = getattr(self._coordinator.runtime, "backend", None)
        if backend is not None and hasattr(backend, "mark_stream_done"):
            backend.mark_stream_done(session_id)

    def _mark_stream_error(self, session_id: str) -> None:
        if self._coordinator is None:
            return
        backend = getattr(self._coordinator.runtime, "backend", None)
        if backend is not None and hasattr(backend, "mark_stream_error"):
            backend.mark_stream_error(session_id)

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
        trans_rec = ctx.timeline.append(command)
        if self._coordinator is None:
            return

        is_first = ctx.status.phase == SessionPhase.IDLE.value
        if can_transition(ctx.status.phase, SessionPhase.WARMING.value) and is_first:
            ctx.status = self._coordinator.warmup(session_id, command.text)

        # Planner session: first command starts, subsequent commands transition
        is_replace = False
        if self._coordinator.planner is not None:
            driver = getattr(self._coordinator.planner, "_driver", None)
            if driver is not None:
                if driver.session is None:
                    driver.start_session(command)
                elif trans_rec is not None:
                    from text2humanoid.planner.prompt_transition import (
                        should_replace_immediately,
                        should_crossfade,
                        crossfade_overlap_frames,
                    )
                    if should_replace_immediately(command.transition_mode):
                        driver.transition(command)
                        is_replace = True
                    else:
                        # APPEND / CROSSFADE: queue as pending, refill handles it
                        driver.session.pending_command = command
                        if should_crossfade(command.transition_mode):
                            driver.session.metadata["crossfade_overlap"] = crossfade_overlap_frames(command.transition_mode)
                    trans_rec.boundary_chunk_index = driver.session.chunk_index
                    import time as _time
                    trans_rec.transition_time = _time.time()

        # REPLACE: run once immediately with new command.
        # APPEND/CROSSFADE/first command: run once to keep chunk flowing.
        if is_replace or is_first or trans_rec is None:
            ctx.status = self._coordinator.run_once(session_id, command, start_time=ctx.next_start_time)
            latest_chunk_end = float(ctx.status.metadata.get("latest_chunk_end_time", ctx.status.sim_time))
            ctx.next_start_time = max(ctx.next_start_time, latest_chunk_end)

        # Push transition + pending info to status
        ctx.status.metadata["active_command_id"] = ctx.timeline.active_command_id
        ctx.status.metadata["transition_mode"] = command.transition_mode
        ctx.status.metadata["transition_count"] = len(ctx.timeline.transitions)
        if driver is not None and driver.session is not None:
            ctx.status.metadata["pending_command_id"] = driver.session.pending_command.command_id if driver.session.has_pending else ""
        if trans_rec is not None:
            ctx.status.metadata["transition"] = {
                "previous_command_id": trans_rec.previous_command_id,
                "new_command_id": trans_rec.new_command_id,
                "boundary_chunk_index": trans_rec.boundary_chunk_index,
            }

    def reset_session(self, session_id: str) -> None:
        ctx = self._sessions[session_id]
        self.stop_refill_loop(session_id)
        ctx.timeline.commands.clear()
        ctx.next_start_time = 0.0
        if self._coordinator is not None:
            self._coordinator.planner.reset()
            ctx.status = self._coordinator.runtime.reset_session(session_id)
        ctx.status.phase = SessionPhase.RESETTING.value

    def stop_session(self, session_id: str) -> None:
        self.stop_refill_loop(session_id)
        self._mark_stream_done(session_id)
        self._sessions[session_id].status.phase = SessionPhase.STOPPED.value

    def run_refill_cycle(self, session_id: str, watermark_frames: int = 20, max_chunks: int = 10) -> int:
        """Produce additional chunks if the runtime buffer is below watermark.

        Uses the planner's session streaming state (StreamPlannerDriver) when
        available.  Falls back to replaying the last command if no planner
        session is active.
        """
        ctx = self._sessions[session_id]
        if self._coordinator is None:
            return 0

        # Prefer planner-native session streaming
        driver = getattr(self._coordinator.planner, "_driver", None)
        driver_session = driver.session if driver is not None else None

        # Promote pending command if APPEND/CROSSFADE transition is waiting
        pending_crossfade_overlap = 0
        if driver_session is not None and driver_session.has_pending:
            pending_crossfade_overlap = driver_session.metadata.pop("crossfade_overlap", 0)
            driver_session.promote_pending()

        chunks_produced = 0
        for _ in range(max_chunks):
            status = self._coordinator.runtime.get_status(session_id)
            if status.buffer_frames >= watermark_frames:
                break

            if driver is not None and driver_session is not None:
                chunk = driver.generate_next_chunk()
                if chunk is None:
                    break
                from text2humanoid.retarget.bridge_263_to_140 import human_chunk_to_nmr_input
                nmr_chunk = human_chunk_to_nmr_input(chunk, tgt_fps=self._coordinator.retarget.output_fps)
                result = self._coordinator.retarget.retarget_chunk(nmr_chunk)
                ref_chunk = self._coordinator.adapter.from_nmr_result(
                    chunk_id=chunk.chunk_id,
                    start_time=nmr_chunk.start_time,
                    fps=self._coordinator.retarget.output_fps,
                    result=result,
                )
                overlap = 4
                if pending_crossfade_overlap > 0:
                    overlap = pending_crossfade_overlap
                    pending_crossfade_overlap = 0  # only apply to first chunk after transition
                self._coordinator.runtime.push_reference_chunk(session_id, ref_chunk, overlap_frames=overlap)
                ctx.status = self._coordinator.runtime.get_status(session_id)
                ctx.next_start_time = driver_session.next_start_time
            else:
                # Fallback: replay last command
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

    def _refill_thread(self, session_id: str, watermark_frames: int, max_chunks: int, interval_sec: float) -> None:
        ctx = self._sessions.get(session_id)
        if ctx is None:
            return
        while ctx.refill_active and not (ctx.refill_stop_event and ctx.refill_stop_event.is_set()):
            try:
                self.run_refill_cycle(session_id, watermark_frames=watermark_frames, max_chunks=max_chunks)
            except Exception:
                self._mark_stream_error(session_id)
                ctx.status.phase = SessionPhase.ERROR.value
                break
            if ctx.status.phase in (SessionPhase.ERROR.value, SessionPhase.STOPPED.value):
                break
            # Wait, checking stop event periodically
            if ctx.refill_stop_event:
                ctx.refill_stop_event.wait(timeout=interval_sec)
            else:
                time.sleep(interval_sec)

    def start_refill_loop(self, session_id: str, watermark_frames: int = 20, max_chunks: int = 1, interval_sec: float = 0.5) -> None:
        ctx = self._sessions[session_id]
        if ctx.refill_active:
            return
        ctx.refill_active = True
        ctx.refill_stop_event = threading.Event()
        t = threading.Thread(
            target=self._refill_thread,
            args=(session_id, watermark_frames, max_chunks, interval_sec),
            daemon=True,
        )
        t.start()

    def stop_refill_loop(self, session_id: str) -> None:
        ctx = self._sessions.get(session_id)
        if ctx is None or not ctx.refill_active:
            return
        ctx.refill_active = False
        if ctx.refill_stop_event:
            ctx.refill_stop_event.set()
