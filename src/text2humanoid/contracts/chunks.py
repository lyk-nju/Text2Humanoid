from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from text2humanoid.contracts.validation import (
    as_float32_matrix,
    base_chunk_metadata,
    validate_fps,
)


@dataclass(slots=True)
class HumanMotionChunk:
    chunk_id: str
    start_time: float
    fps: int
    motion_263: np.ndarray
    text: str
    trajectory_payload: dict[str, Any] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.motion_263 = as_float32_matrix(self.motion_263, field_name="motion_263", width=263)
        self.fps = validate_fps(self.fps)

    @property
    def num_frames(self) -> int:
        return int(self.motion_263.shape[0])

    @property
    def end_time(self) -> float:
        return self.start_time + self.num_frames / float(self.fps)

    @property
    def duration_sec(self) -> float:
        return self.num_frames / float(self.fps)

    @property
    def motion_shape(self) -> tuple[int, int]:
        return tuple(int(x) for x in self.motion_263.shape)

    def to_chunk_metadata(self) -> dict[str, Any]:
        data = base_chunk_metadata(
            chunk_id=self.chunk_id,
            representation="humanml3d_263",
            fps=self.fps,
            frame_count=self.num_frames,
            start_time=self.start_time,
            shape=self.motion_263.shape,
        )
        data.update(self.metadata)
        return data


@dataclass(slots=True)
class NMRInputChunk:
    chunk_id: str
    start_time: float
    fps: int
    motion_140: np.ndarray
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.motion_140 = as_float32_matrix(self.motion_140, field_name="motion_140", width=140)
        self.fps = validate_fps(self.fps)

    @property
    def num_frames(self) -> int:
        return int(self.motion_140.shape[0])

    @property
    def end_time(self) -> float:
        return self.start_time + self.num_frames / float(self.fps)

    @property
    def duration_sec(self) -> float:
        return self.num_frames / float(self.fps)

    @property
    def motion_shape(self) -> tuple[int, int]:
        return tuple(int(x) for x in self.motion_140.shape)

    def to_chunk_metadata(self) -> dict[str, Any]:
        data = base_chunk_metadata(
            chunk_id=self.chunk_id,
            representation="nmr_smplx_140",
            fps=self.fps,
            frame_count=self.num_frames,
            start_time=self.start_time,
            shape=self.motion_140.shape,
        )
        data.update(self.metadata)
        return data
