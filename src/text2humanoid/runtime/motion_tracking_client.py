from __future__ import annotations

from text2humanoid.contracts.clips import G1ReferenceChunk
from text2humanoid.contracts.status import RuntimeStatus
from text2humanoid.runtime.reference_buffer import ReferenceBuffer
from text2humanoid.runtime.sync_manager import SyncManager


class MotionTrackingClient:
    """In-memory reference buffer shim — NOT a real motion_tracking runtime.

    This class accepts G1ReferenceChunks, manages per-session ReferenceBuffers,
    and tracks frame consumption.  It does NOT connect to a real tracking
    policy or simulation.  It exists to fix the runtime interface contract
    so the real motion_tracking source plugin can implement the same protocol
    later without changing the orchestrator.

    Once the real motion_tracking source plugin lands, this class should be
    replaced by (or delegate to) the actual runtime client.
    """
    def __init__(self, control_hz: int = 50, future_horizon_frames: int = 16, xml_path: str | None = None) -> None:
        self._buffers: dict[str, ReferenceBuffer] = {}
        self.sync = SyncManager(control_hz=control_hz, future_horizon_frames=future_horizon_frames)
        self._statuses: dict[str, RuntimeStatus] = {}
        self._consumed_frames: dict[str, int] = {}
        self._xml_path = xml_path

    def ensure_session(self, session_id: str) -> None:
        if session_id not in self._statuses:
            self._statuses[session_id] = RuntimeStatus(session_id=session_id)
            self._consumed_frames[session_id] = 0
            self._buffers[session_id] = ReferenceBuffer(xml_path=self._xml_path)

    def _buffer(self, session_id: str) -> ReferenceBuffer:
        self.ensure_session(session_id)
        return self._buffers[session_id]

    def push_reference_chunk(self, session_id: str, chunk: G1ReferenceChunk, overlap_frames: int = 4) -> None:
        self.ensure_session(session_id)
        buffer = self._buffer(session_id)
        buffer.append_chunk(chunk, overlap_frames=overlap_frames)
        status = self._statuses[session_id]
        status.buffer_frames = buffer.buffer_frames
        status.latest_chunk_id = chunk.chunk_id
        status.sim_time = self.sync.sim_time_from_frames(self._consumed_frames[session_id])

    def consume_step(self, session_id: str, frames: int = 1) -> None:
        self.ensure_session(session_id)
        buffer = self._buffer(session_id)
        buffer.advance(frames)
        self._consumed_frames[session_id] += int(frames)
        self._statuses[session_id].buffer_frames = buffer.buffer_frames
        self._statuses[session_id].sim_time = self.sync.sim_time_from_frames(self._consumed_frames[session_id])

    def get_status(self, session_id: str) -> RuntimeStatus:
        self.ensure_session(session_id)
        buffer = self._buffer(session_id)
        self._statuses[session_id].buffer_frames = buffer.buffer_frames
        self._statuses[session_id].latest_chunk_id = buffer.latest_chunk_id
        return self._statuses[session_id]

    def reset_session(self, session_id: str) -> RuntimeStatus:
        self.ensure_session(session_id)
        self._buffers[session_id].reset()
        self._consumed_frames[session_id] = 0
        status = self._statuses[session_id]
        status.buffer_frames = 0
        status.sim_time = 0.0
        status.latest_chunk_id = ""
        return status
