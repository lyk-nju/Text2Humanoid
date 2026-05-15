"""Clip-level payload contract between Text2Humanoid and motion_tracking.

All payloads flowing from Text2Humanoid into the motion_tracking runtime
MUST conform to the shapes and conventions declared here.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from text2humanoid.contracts.clips import G1ReferenceChunk

# Required keys in a clip-level payload consumed by motion_tracking sources.
_REQUIRED_CLIP_KEYS = [
    "root_pos",
    "root_rot",
    "dof_pos",
    "local_body_pos",
    "local_body_rot",
    "joint_names",
    "body_names",
]

# Required keys in a per-frame payload.
_REQUIRED_FRAME_KEYS = ["root_pos", "root_quat", "dof_pos"]


def chunk_to_runtime_dict(chunk: G1ReferenceChunk) -> dict[str, Any]:
    """Serialize a G1ReferenceChunk into the runtime-consumable clip dict.

    root_rot is stored as xyzw (matching motion_tracking NPZ convention).
    joint_names and body_names are stored as numpy string arrays.
    """
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
    """Extract a single-frame payload from a G1ReferenceChunk.

    Returns root_pos (3,), root_quat (4, xyzw), dof_pos (29,).
    """
    return {
        "root_pos": chunk.root_pos[frame_idx].astype(np.float32),
        "root_quat": chunk.root_rot[frame_idx].astype(np.float32),
        "dof_pos": chunk.dof_pos[frame_idx].astype(np.float32),
    }


def validate_clip_payload(payload: dict[str, Any]) -> list[str]:
    """Validate a clip-level payload against the cross-project contract.

    Returns a list of error messages (empty = valid).
    """
    errors: list[str] = []
    for key in _REQUIRED_CLIP_KEYS:
        if key not in payload:
            errors.append(f"missing required clip key: {key}")

    if "root_rot" in payload:
        rot = np.asarray(payload["root_rot"])
        if rot.ndim != 2 or rot.shape[1] != 4:
            errors.append(f"root_rot must be (T, 4) xyzw, got {rot.shape}")

    if "root_pos" in payload:
        pos = np.asarray(payload["root_pos"])
        if pos.ndim != 2 or pos.shape[1] != 3:
            errors.append(f"root_pos must be (T, 3), got {pos.shape}")

    if "dof_pos" in payload:
        dof = np.asarray(payload["dof_pos"])
        if dof.ndim != 2:
            errors.append(f"dof_pos must be (T, J), got {dof.shape}")

    if "joint_names" in payload:
        jn = np.asarray(payload["joint_names"])
        if len(jn) != payload.get("dof_pos", np.zeros((1, 0))).shape[1]:
            errors.append(
                f"joint_names length {len(jn)} != dof_pos dim "
                f"{payload['dof_pos'].shape[1]}"
            )

    return errors


def validate_frame_payload(payload: dict[str, Any], expected_joints: int = 29) -> list[str]:
    """Validate a single-frame payload."""
    errors: list[str] = []
    for key in _REQUIRED_FRAME_KEYS:
        if key not in payload:
            errors.append(f"missing required frame key: {key}")
    if "dof_pos" in payload:
        dof = np.asarray(payload["dof_pos"]).reshape(-1)
        if dof.shape[0] != expected_joints:
            errors.append(f"dof_pos dim {dof.shape[0]} != expected {expected_joints}")
    if "root_quat" in payload:
        q = np.asarray(payload["root_quat"]).reshape(-1)
        if q.shape[0] != 4:
            errors.append(f"root_quat must be (4,) xyzw, got {q.shape}")
    return errors
