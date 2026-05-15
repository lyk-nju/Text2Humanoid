"""Runtime client for motion_tracking — shim and real backend implementations.

MotionTrackingClient maintains the stable orchestrator-facing interface
(push_reference_chunk / get_status / reset_session) and delegates to a
RuntimeBackend.  Two backends are provided:

  ShimBackend      — in-memory reference buffer (testing / offline dev)
  FloodNetFileBackend — writes reference clips as NPZ files consumable by
                         motion_tracking's FloodNetMotionSource

Once the real motion_tracking source plugin is connected online, a new
backend (e.g. ZMQ / DDS / pipe) can be added without changing the
orchestrator.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np

from text2humanoid.contracts.clips import G1ReferenceChunk
from text2humanoid.contracts.status import RuntimeStatus
from text2humanoid.runtime.reference_buffer import ReferenceBuffer
from text2humanoid.runtime.source_protocol import chunk_to_runtime_dict, validate_clip_payload
from text2humanoid.runtime.sync_manager import SyncManager


class RuntimeBackend(ABC):
    """Abstract backend for consuming G1ReferenceChunks."""

    @abstractmethod
    def ensure_session(self, session_id: str) -> None: ...

    @abstractmethod
    def push_reference_chunk(self, session_id: str, chunk: G1ReferenceChunk, overlap_frames: int = 4) -> None: ...

    @abstractmethod
    def consume_step(self, session_id: str, frames: int = 1) -> None: ...

    @abstractmethod
    def get_status(self, session_id: str) -> RuntimeStatus: ...

    @abstractmethod
    def reset_session(self, session_id: str) -> RuntimeStatus: ...


class ShimBackend(RuntimeBackend):
    """In-memory reference buffer — NOT a real motion_tracking runtime.

    This backend is intended for offline development and testing.  It
    stores reference chunks in memory and simulates frame consumption
    without connecting to any tracking policy or simulation.
    """

    def __init__(self, control_hz: int = 50, future_horizon_frames: int = 16, xml_path: str | None = None):
        self.sync = SyncManager(control_hz=control_hz, future_horizon_frames=future_horizon_frames)
        self._buffers: dict[str, ReferenceBuffer] = {}
        self._statuses: dict[str, RuntimeStatus] = {}
        self._consumed_frames: dict[str, int] = {}
        self._xml_path = xml_path

    def ensure_session(self, session_id: str) -> None:
        if session_id not in self._statuses:
            self._statuses[session_id] = RuntimeStatus(session_id=session_id)
            self._consumed_frames[session_id] = 0
            self._buffers[session_id] = ReferenceBuffer(xml_path=self._xml_path)

    def push_reference_chunk(self, session_id: str, chunk: G1ReferenceChunk, overlap_frames: int = 4) -> None:
        self.ensure_session(session_id)
        self._buffers[session_id].append_chunk(chunk, overlap_frames=overlap_frames)
        status = self._statuses[session_id]
        status.buffer_frames = self._buffers[session_id].buffer_frames
        status.latest_chunk_id = chunk.chunk_id
        status.sim_time = self.sync.sim_time_from_frames(self._consumed_frames[session_id])

    def consume_step(self, session_id: str, frames: int = 1) -> None:
        self.ensure_session(session_id)
        self._buffers[session_id].advance(frames)
        self._consumed_frames[session_id] += int(frames)
        self._statuses[session_id].buffer_frames = self._buffers[session_id].buffer_frames
        self._statuses[session_id].sim_time = self.sync.sim_time_from_frames(self._consumed_frames[session_id])

    def get_status(self, session_id: str) -> RuntimeStatus:
        self.ensure_session(session_id)
        buf = self._buffers[session_id]
        self._statuses[session_id].buffer_frames = buf.buffer_frames
        self._statuses[session_id].latest_chunk_id = buf.latest_chunk_id
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


class FloodNetFileBackend(RuntimeBackend):
    """Writes reference chunks as NPZ files for motion_tracking FloodNetMotionSource.

    Each pushed chunk is saved as a standalone NPZ file that can be loaded
    by motion_tracking when configured with motion_source: floodnet.
    """

    def __init__(self, output_dir: str | Path, control_hz: int = 50):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.sync = SyncManager(control_hz=control_hz)
        self._statuses: dict[str, RuntimeStatus] = {}
        self._chunk_counts: dict[str, int] = {}

    def ensure_session(self, session_id: str) -> None:
        if session_id not in self._statuses:
            self._statuses[session_id] = RuntimeStatus(session_id=session_id)
            self._chunk_counts[session_id] = 0
            session_dir = self.output_dir / session_id
            session_dir.mkdir(parents=True, exist_ok=True)

    def push_reference_chunk(self, session_id: str, chunk: G1ReferenceChunk, overlap_frames: int = 4) -> None:
        self.ensure_session(session_id)
        errors = validate_clip_payload(chunk_to_runtime_dict(chunk))
        if errors:
            raise ValueError(f"Invalid clip payload for {chunk.chunk_id}: {errors}")

        idx = self._chunk_counts[session_id]
        path = self.output_dir / session_id / f"chunk_{idx:04d}.npz"
        payload = chunk_to_runtime_dict(chunk)
        arrays = {k: v for k, v in payload.items() if isinstance(v, np.ndarray)}
        np.savez(path, **arrays)

        self._chunk_counts[session_id] += 1
        status = self._statuses[session_id]
        status.buffer_frames += chunk.num_frames
        status.latest_chunk_id = chunk.chunk_id

    def consume_step(self, session_id: str, frames: int = 1) -> None:
        self.ensure_session(session_id)
        status = self._statuses[session_id]
        status.sim_time += float(frames) / float(self.sync.control_hz)
        status.buffer_frames = max(0, status.buffer_frames - int(frames))

    def get_status(self, session_id: str) -> RuntimeStatus:
        self.ensure_session(session_id)
        return self._statuses[session_id]

    def reset_session(self, session_id: str) -> RuntimeStatus:
        self.ensure_session(session_id)
        self._chunk_counts[session_id] = 0
        status = self._statuses[session_id]
        status.buffer_frames = 0
        status.sim_time = 0.0
        status.latest_chunk_id = ""
        return status


class MotionTrackingClient:
    """Orchestrator-facing runtime client — delegates to a RuntimeBackend.

    This class keeps the stable interface (push_reference_chunk /
    get_status / reset_session) regardless of which backend is active.
    """

    def __init__(self, backend: RuntimeBackend | None = None, control_hz: int = 50, future_horizon_frames: int = 16, xml_path: str | None = None):
        if backend is not None:
            self._backend = backend
        else:
            self._backend = ShimBackend(control_hz=control_hz, future_horizon_frames=future_horizon_frames, xml_path=xml_path)
        self.sync = getattr(self._backend, "sync", SyncManager(control_hz=control_hz, future_horizon_frames=future_horizon_frames))

    @property
    def backend(self) -> RuntimeBackend:
        return self._backend

    def ensure_session(self, session_id: str) -> None:
        self._backend.ensure_session(session_id)

    def push_reference_chunk(self, session_id: str, chunk: G1ReferenceChunk, overlap_frames: int = 4) -> None:
        self._backend.push_reference_chunk(session_id, chunk, overlap_frames=overlap_frames)

    def consume_step(self, session_id: str, frames: int = 1) -> None:
        self._backend.consume_step(session_id, frames=frames)

    def get_status(self, session_id: str) -> RuntimeStatus:
        return self._backend.get_status(session_id)

    def reset_session(self, session_id: str) -> RuntimeStatus:
        return self._backend.reset_session(session_id)
