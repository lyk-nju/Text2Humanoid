"""Online runtime bridge — pushes reference chunks over a local TCP socket.

SocketBackend replaces file polling with direct socket communication:
  Text2Humanoid SocketBackend  →  TCP  →  motion_tracking SocketFloodNetSource

Protocol: each message is a 4-byte big-endian length prefix followed by
a JSON-encoded dict with the standard clip payload fields.

Lifecycle phases (per session):
  running  — consumer connected, chunks flowing
  stopped  — normal stop (mark_stream_done)
  error    — consumer disconnect / send failure (mark_stream_error)
"""

from __future__ import annotations

import json
import socket
import struct
import threading
from typing import Any

import numpy as np

from text2humanoid.contracts.clips import G1ReferenceChunk
from text2humanoid.contracts.status import RuntimeStatus, SessionPhase
from text2humanoid.runtime.motion_tracking_client import RuntimeBackend
from text2humanoid.runtime.source_protocol import chunk_to_runtime_dict, validate_clip_payload
from text2humanoid.runtime.sync_manager import SyncManager


def _to_serializable(obj):
    if isinstance(obj, np.ndarray): return obj.tolist()
    if isinstance(obj, (np.integer,)): return int(obj)
    if isinstance(obj, (np.floating,)): return float(obj)
    if isinstance(obj, np.bool_): return bool(obj)
    return obj


def _send_message(sock, payload):
    data = json.dumps(payload, default=_to_serializable).encode("utf-8")
    sock.sendall(struct.pack(">I", len(data)) + data)


class SocketBackend(RuntimeBackend):
    """Sends reference chunks over a TCP socket to motion_tracking.

    This replaces the file-polling path with direct online communication.
    The floodnet_file backend remains available as a fallback path.

    Lifecycle semantics:
      - ensure_session: creates session in RUNNING phase
      - push_reference_chunk: sends over TCP; on send failure marks session ERROR
      - mark_stream_done: normal stop → STOPPED
      - mark_stream_error: consumer disconnect / failure → ERROR
      - close: close all connections, mark active sessions ERROR
    """

    STREAM_PHASE_RUNNING = "running"
    STREAM_PHASE_DONE = "done"
    STREAM_PHASE_ERROR = "error"

    def __init__(self, host: str = "127.0.0.1", port: int = 15555, control_hz: int = 50):
        self.host = host
        self.port = port
        self.sync = SyncManager(control_hz=control_hz)
        self._server: socket.socket | None = None
        self._client: socket.socket | None = None
        self._lock = threading.Lock()
        self._statuses: dict[str, RuntimeStatus] = {}
        self._chunk_counts: dict[str, int] = {}
        self._phases: dict[str, str] = {}

    # ---- lifecycle ----

    @property
    def connected(self) -> bool:
        with self._lock:
            return self._client is not None

    def ensure_session(self, session_id: str) -> None:
        if session_id not in self._statuses:
            st = RuntimeStatus(session_id=session_id)
            st.phase = SessionPhase.RUNNING.value
            self._statuses[session_id] = st
            self._chunk_counts[session_id] = 0
            self._phases[session_id] = self.STREAM_PHASE_RUNNING
        self._ensure_connection()

    def _ensure_connection(self) -> None:
        with self._lock:
            if self._server is None:
                self._server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                self._server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                self._server.bind((self.host, self.port))
                self._server.listen(1)
                self._server.settimeout(2.0)
            if self._client is None:
                try:
                    self._client, _addr = self._server.accept()
                    self._client.settimeout(2.0)
                except socket.timeout:
                    pass

    def mark_stream_done(self, session_id: str) -> None:
        """Normal stop: consumer is done consuming, phase → done."""
        if session_id in self._phases:
            self._phases[session_id] = self.STREAM_PHASE_DONE
        if session_id in self._statuses:
            self._statuses[session_id].phase = SessionPhase.STOPPED.value

    def mark_stream_error(self, session_id: str, reason: str = "") -> None:
        """Consumer disconnect or send failure — phase → error."""
        if session_id in self._phases:
            self._phases[session_id] = self.STREAM_PHASE_ERROR
        if session_id in self._statuses:
            st = self._statuses[session_id]
            st.phase = SessionPhase.ERROR.value
            if reason:
                st.errors.append(reason)

    def get_phase(self, session_id: str) -> str:
        return self._phases.get(session_id, self.STREAM_PHASE_RUNNING)

    # ---- RuntimeBackend interface ----

    def push_reference_chunk(self, session_id: str, chunk: G1ReferenceChunk, overlap_frames: int = 4) -> None:
        self.ensure_session(session_id)
        payload = chunk_to_runtime_dict(chunk)
        errors = validate_clip_payload(payload)
        if errors:
            raise ValueError(f"Invalid clip payload: {errors}")

        msg = {"type": "chunk", "session_id": session_id, "chunk_id": chunk.chunk_id,
               "overlap_frames": overlap_frames, "payload": payload}

        send_ok = True
        with self._lock:
            if self._client is not None:
                try:
                    _send_message(self._client, msg)
                except (BrokenPipeError, ConnectionResetError, OSError) as e:
                    self._client = None
                    send_ok = False
                    self.mark_stream_error(session_id, f"send failed: {e}")
            else:
                send_ok = False

        idx = self._chunk_counts[session_id]
        self._chunk_counts[session_id] = idx + 1
        status = self._statuses[session_id]
        status.buffer_frames += chunk.num_frames
        status.latest_chunk_id = chunk.chunk_id

        if not send_ok:
            status.errors.append("chunk queued but not sent (no consumer connected)")

    def consume_step(self, session_id: str, frames: int = 1) -> None:
        self.ensure_session(session_id)
        s = self._statuses[session_id]
        s.sim_time += float(frames) / float(self.sync.control_hz)
        s.buffer_frames = max(0, s.buffer_frames - int(frames))

    def get_status(self, session_id: str) -> RuntimeStatus:
        self.ensure_session(session_id)
        return self._statuses[session_id]

    def reset_session(self, session_id: str) -> RuntimeStatus:
        self.ensure_session(session_id)
        self._chunk_counts[session_id] = 0
        self._phases[session_id] = self.STREAM_PHASE_RUNNING
        s = self._statuses[session_id]
        s.buffer_frames = 0; s.sim_time = 0.0; s.latest_chunk_id = ""
        s.phase = SessionPhase.IDLE.value
        return s

    def close(self) -> None:
        """Close all connections. Active sessions are marked error."""
        with self._lock:
            for sid in list(self._phases.keys()):
                phase = self._phases[sid]
                if phase not in (self.STREAM_PHASE_DONE, self.STREAM_PHASE_ERROR):
                    self.mark_stream_error(sid, "producer socket closed")
            for s in (self._client, self._server):
                if s is not None:
                    try: s.close()
                    except OSError: pass
            self._client = None; self._server = None
