from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np


KNOWN_MOTION_DIMS = {
    "humanml3d_263": 263,
    "nmr_smplx_140": 140,
}


def validate_fps(fps: int | float, *, field_name: str = "fps") -> int:
    value = int(fps)
    if value <= 0:
        raise ValueError(f"{field_name} must be positive, got {fps}")
    return value


def as_float32_array(value: Any, *, field_name: str) -> np.ndarray:
    return np.asarray(value, dtype=np.float32)


def as_float32_matrix(
    value: Any,
    *,
    field_name: str,
    width: int | None = None,
) -> np.ndarray:
    arr = as_float32_array(value, field_name=field_name)
    if arr.ndim != 2:
        raise ValueError(f"{field_name} must have shape (T, D), got {arr.shape}")
    if width is not None and arr.shape[1] != width:
        raise ValueError(f"{field_name} must have shape (T, {width}), got {arr.shape}")
    return arr


def validate_known_representation_shape(
    representation: str,
    shape: Sequence[int],
) -> None:
    expected_dim = KNOWN_MOTION_DIMS.get(str(representation))
    if expected_dim is None:
        return
    if len(shape) != 2 or int(shape[1]) != expected_dim:
        raise ValueError(
            f"{representation} motion must have shape (T, {expected_dim}), got {tuple(shape)}"
        )


def validate_quat_order(quat_order: str) -> str:
    if quat_order not in ("wxyz", "xyzw"):
        raise ValueError("quat_order must be 'wxyz' or 'xyzw'")
    return quat_order


def duration_sec(num_frames: int, fps: int | float) -> float:
    checked_fps = validate_fps(fps)
    return int(num_frames) / float(checked_fps)


def base_chunk_metadata(
    *,
    chunk_id: str,
    representation: str,
    fps: int | float,
    frame_count: int,
    start_time: float = 0.0,
    shape: Sequence[int] | None = None,
    source_chunk_id: str | None = None,
    joint_order: str | None = None,
    quat_order: str | None = None,
) -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "chunk_id": chunk_id,
        "representation": representation,
        "fps": validate_fps(fps),
        "frame_count": int(frame_count),
        "start_time": float(start_time),
        "duration_sec": duration_sec(frame_count, fps),
    }
    if shape is not None:
        metadata["motion_shape"] = tuple(int(x) for x in shape)
    if source_chunk_id:
        metadata["source_chunk_id"] = source_chunk_id
    if joint_order:
        metadata["joint_order"] = joint_order
    if quat_order:
        metadata["quat_order"] = validate_quat_order(quat_order)
    return metadata
