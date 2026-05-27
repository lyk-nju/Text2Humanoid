from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from text2humanoid.contracts.bfmzero import BFMZeroMotionChunk


def save_bfmzero_chunk_npz(path: str | Path, chunk: BFMZeroMotionChunk) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        chunk_id=np.asarray(chunk.chunk_id),
        fps=np.asarray(chunk.fps, dtype=np.int64),
        frame_start=np.asarray(chunk.frame_start, dtype=np.int64),
        joint_pos=chunk.joint_pos,
        joint_vel=chunk.joint_vel,
        root_pos=chunk.root_pos,
        root_quat=chunk.root_quat,
        root_lin_vel_w=chunk.root_lin_vel_w,
        root_ang_vel_w=chunk.root_ang_vel_w,
        metadata_json=np.asarray(json.dumps(chunk.metadata, ensure_ascii=False, default=str)),
    )


def load_bfmzero_chunk_npz(path: str | Path) -> BFMZeroMotionChunk:
    data = np.load(path, allow_pickle=False)
    metadata: dict[str, Any] = {}
    if "metadata_json" in data:
        raw = str(data["metadata_json"].item())
        metadata = json.loads(raw) if raw else {}
    return BFMZeroMotionChunk(
        chunk_id=str(data["chunk_id"].item()),
        fps=int(data["fps"].item()),
        frame_start=int(data["frame_start"].item()),
        joint_pos=data["joint_pos"],
        joint_vel=data["joint_vel"],
        root_pos=data["root_pos"],
        root_quat=data["root_quat"],
        root_lin_vel_w=data["root_lin_vel_w"],
        root_ang_vel_w=data["root_ang_vel_w"],
        metadata=metadata,
    )
