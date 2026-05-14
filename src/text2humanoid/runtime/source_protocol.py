from __future__ import annotations

from typing import Any

import numpy as np

from text2humanoid.contracts.clips import G1ReferenceChunk


def chunk_to_runtime_dict(chunk: G1ReferenceChunk) -> dict[str, Any]:
    return {
        "chunk_id": chunk.chunk_id,
        "start_time": chunk.start_time,
        "fps": chunk.fps,
        "root_pos": chunk.root_pos,
        "root_rot": chunk.root_rot,
        "dof_pos": chunk.dof_pos,
        "local_body_pos": chunk.local_body_pos,
        "local_body_rot": chunk.local_body_rot,
        "body_names": np.asarray(chunk.body_names),
        "joint_names": np.asarray(chunk.joint_names),
    }


def frame_payload(chunk: G1ReferenceChunk, frame_idx: int) -> dict[str, Any]:
    return {
        "root_pos": chunk.root_pos[frame_idx].astype(np.float32),
        "root_quat": chunk.root_rot[frame_idx].astype(np.float32),
        "dof_pos": chunk.dof_pos[frame_idx].astype(np.float32),
    }
