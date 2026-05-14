from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from text2humanoid.contracts.clips import G1ReferenceChunk


@dataclass(slots=True)
class BufferedReference:
    fps: int
    root_pos: np.ndarray
    root_rot: np.ndarray
    dof_pos: np.ndarray
    local_body_pos: np.ndarray
    local_body_rot: np.ndarray
    body_names: list[str]
    joint_names: list[str]


def _normalize_quat(q: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(q, axis=-1, keepdims=True).clip(min=1e-8)
    return (q / norm).astype(np.float32)


def _blend_quat(q0: np.ndarray, q1: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    q0 = np.asarray(q0, dtype=np.float32)
    q1 = np.asarray(q1, dtype=np.float32)
    dot = np.sum(q0 * q1, axis=-1, keepdims=True)
    q1 = np.where(dot < 0.0, -q1, q1)
    mixed = q0 * (1.0 - alpha) + q1 * alpha
    return _normalize_quat(mixed)


class ReferenceBuffer:
    def __init__(self) -> None:
        self._buffer: BufferedReference | None = None
        self._cursor = 0
        self._latest_chunk_id = ""

    def reset(self) -> None:
        self._buffer = None
        self._cursor = 0
        self._latest_chunk_id = ""

    @property
    def latest_chunk_id(self) -> str:
        return self._latest_chunk_id

    @property
    def buffer_frames(self) -> int:
        if self._buffer is None:
            return 0
        return int(max(0, self._buffer.root_pos.shape[0] - self._cursor))

    def append_chunk(self, chunk: G1ReferenceChunk, overlap_frames: int = 0) -> None:
        """Append a chunk with optional overlap cross-fade.

        Cross-fade blends root_pos / root_rot / dof_pos / local_body_pos /
        local_body_rot independently.  This is an approximation: after blending,
        local_body_* may not be strictly FK-consistent with the blended root/dof.
        For production use, the blended root_pos/root_rot/dof_pos should be
        re-run through forward kinematics to regenerate local_body_*.
        """
        if self._buffer is None:
            self._buffer = BufferedReference(
                fps=chunk.fps,
                root_pos=chunk.root_pos.copy(),
                root_rot=chunk.root_rot.copy(),
                dof_pos=chunk.dof_pos.copy(),
                local_body_pos=chunk.local_body_pos.copy(),
                local_body_rot=chunk.local_body_rot.copy(),
                body_names=list(chunk.body_names),
                joint_names=list(chunk.joint_names),
            )
            self._latest_chunk_id = chunk.chunk_id
            return

        if self._cursor > 0:
            self._buffer.root_pos = self._buffer.root_pos[self._cursor :].copy()
            self._buffer.root_rot = self._buffer.root_rot[self._cursor :].copy()
            self._buffer.dof_pos = self._buffer.dof_pos[self._cursor :].copy()
            self._buffer.local_body_pos = self._buffer.local_body_pos[self._cursor :].copy()
            self._buffer.local_body_rot = self._buffer.local_body_rot[self._cursor :].copy()
            self._cursor = 0

        overlap = int(max(0, min(overlap_frames, self.buffer_frames, chunk.num_frames)))
        if overlap > 0:
            alpha = np.linspace(0.0, 1.0, overlap, dtype=np.float32).reshape(-1, 1)
            tail0 = self._buffer.root_pos[-overlap:]
            tail1 = chunk.root_pos[:overlap]
            self._buffer.root_pos[-overlap:] = tail0 * (1.0 - alpha) + tail1 * alpha

            tail0 = self._buffer.dof_pos[-overlap:]
            tail1 = chunk.dof_pos[:overlap]
            self._buffer.dof_pos[-overlap:] = tail0 * (1.0 - alpha) + tail1 * alpha

            self._buffer.root_rot[-overlap:] = _blend_quat(
                self._buffer.root_rot[-overlap:],
                chunk.root_rot[:overlap],
                alpha,
            )

            # NOTE: blending local_body_* independently from root/dof is an
            # approximation.  The blended local_body_pos and local_body_rot are
            # only correct to first order.  For strict FK consistency, re-derive
            # local_body_* from the blended root/dof via forward kinematics.
            alpha_body = alpha.reshape(-1, 1, 1)
            self._buffer.local_body_pos[-overlap:] = (
                self._buffer.local_body_pos[-overlap:] * (1.0 - alpha_body)
                + chunk.local_body_pos[:overlap] * alpha_body
            )
            self._buffer.local_body_rot[-overlap:] = _blend_quat(
                self._buffer.local_body_rot[-overlap:],
                chunk.local_body_rot[:overlap],
                alpha_body,
            )

        self._buffer.root_pos = np.concatenate([self._buffer.root_pos, chunk.root_pos[overlap:]], axis=0)
        self._buffer.root_rot = np.concatenate([self._buffer.root_rot, chunk.root_rot[overlap:]], axis=0)
        self._buffer.dof_pos = np.concatenate([self._buffer.dof_pos, chunk.dof_pos[overlap:]], axis=0)
        self._buffer.local_body_pos = np.concatenate([self._buffer.local_body_pos, chunk.local_body_pos[overlap:]], axis=0)
        self._buffer.local_body_rot = np.concatenate([self._buffer.local_body_rot, chunk.local_body_rot[overlap:]], axis=0)
        self._latest_chunk_id = chunk.chunk_id

    def get_horizon(self, num_frames: int) -> dict[str, np.ndarray]:
        if self._buffer is None:
            return {
                "root_pos": np.zeros((0, 3), dtype=np.float32),
                "root_rot": np.zeros((0, 4), dtype=np.float32),
                "dof_pos": np.zeros((0, 29), dtype=np.float32),
            }
        end = min(self._cursor + int(num_frames), self._buffer.root_pos.shape[0])
        return {
            "root_pos": self._buffer.root_pos[self._cursor:end],
            "root_rot": self._buffer.root_rot[self._cursor:end],
            "dof_pos": self._buffer.dof_pos[self._cursor:end],
            "local_body_pos": self._buffer.local_body_pos[self._cursor:end],
            "local_body_rot": self._buffer.local_body_rot[self._cursor:end],
        }

    def advance(self, frames: int = 1) -> None:
        if self._buffer is None:
            return
        self._cursor = min(self._cursor + int(frames), self._buffer.root_pos.shape[0])
