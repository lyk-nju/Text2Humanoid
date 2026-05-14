from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


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
        self.motion_263 = np.asarray(self.motion_263, dtype=np.float32)
        if self.motion_263.ndim != 2 or self.motion_263.shape[1] != 263:
            raise ValueError("motion_263 must have shape (T, 263)")

    @property
    def num_frames(self) -> int:
        return int(self.motion_263.shape[0])

    @property
    def end_time(self) -> float:
        return self.start_time + self.num_frames / float(self.fps)


@dataclass(slots=True)
class NMRInputChunk:
    chunk_id: str
    start_time: float
    fps: int
    motion_140: np.ndarray
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.motion_140 = np.asarray(self.motion_140, dtype=np.float32)
        if self.motion_140.ndim != 2 or self.motion_140.shape[1] != 140:
            raise ValueError("motion_140 must have shape (T, 140)")

    @property
    def num_frames(self) -> int:
        return int(self.motion_140.shape[0])

    @property
    def end_time(self) -> float:
        return self.start_time + self.num_frames / float(self.fps)
