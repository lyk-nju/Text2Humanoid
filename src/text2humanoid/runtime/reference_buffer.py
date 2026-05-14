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
    def __init__(self, xml_path: str | None = None) -> None:
        self._buffer: BufferedReference | None = None
        self._cursor = 0
        self._latest_chunk_id = ""
        self._xml_path = xml_path

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

        Cross-fade blends root_pos / root_rot / dof_pos in the overlap region.
        local_body_* are NOT blended; instead they are regenerated from the
        blended root/dof via forward kinematics in _refresh_local_body().
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

        self._buffer.root_pos = np.concatenate(
            [self._buffer.root_pos, chunk.root_pos[overlap:]], axis=0
        )
        self._buffer.root_rot = np.concatenate(
            [self._buffer.root_rot, chunk.root_rot[overlap:]], axis=0
        )
        self._buffer.dof_pos = np.concatenate(
            [self._buffer.dof_pos, chunk.dof_pos[overlap:]], axis=0
        )
        self._buffer.local_body_pos = np.concatenate(
            [self._buffer.local_body_pos, chunk.local_body_pos[overlap:]], axis=0
        )
        self._buffer.local_body_rot = np.concatenate(
            [self._buffer.local_body_rot, chunk.local_body_rot[overlap:]], axis=0
        )
        self._latest_chunk_id = chunk.chunk_id

        if self._xml_path is not None:
            self._refresh_local_body()

    def _refresh_local_body(self) -> None:
        if self._buffer is None or self._buffer.root_pos.shape[0] == 0:
            return
        from text2humanoid.retarget.fk_features import (
            build_local_body_features,
            build_world_features,
        )

        body_pos_w, body_rot_w, _ = build_world_features(
            self._buffer.root_pos,
            self._buffer.root_rot,
            self._buffer.dof_pos,
            self._xml_path,
        )
        local_body_pos, local_body_rot = build_local_body_features(
            self._buffer.root_pos,
            self._buffer.root_rot,
            body_pos_w,
            body_rot_w,
        )
        self._buffer.local_body_pos = local_body_pos
        self._buffer.local_body_rot = local_body_rot

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
