from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass(slots=True)
class G1ReferenceChunk:
    chunk_id: str
    start_time: float
    fps: int
    root_pos: np.ndarray
    root_rot: np.ndarray
    dof_pos: np.ndarray
    local_body_pos: np.ndarray
    local_body_rot: np.ndarray
    body_names: list[str]
    joint_names: list[str]
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.root_pos = np.asarray(self.root_pos, dtype=np.float32)
        self.root_rot = np.asarray(self.root_rot, dtype=np.float32)
        self.dof_pos = np.asarray(self.dof_pos, dtype=np.float32)
        self.local_body_pos = np.asarray(self.local_body_pos, dtype=np.float32)
        self.local_body_rot = np.asarray(self.local_body_rot, dtype=np.float32)

        n = self.root_pos.shape[0]
        if self.root_pos.shape != (n, 3):
            raise ValueError("root_pos must have shape (T, 3)")
        if self.root_rot.shape != (n, 4):
            raise ValueError("root_rot must have shape (T, 4) in xyzw")
        if self.dof_pos.ndim != 2:
            raise ValueError("dof_pos must have shape (T, J)")
        if self.local_body_pos.ndim != 3 or self.local_body_pos.shape[0] != n:
            raise ValueError("local_body_pos must have shape (T, B, 3)")
        if self.local_body_rot.ndim != 3 or self.local_body_rot.shape[0] != n:
            raise ValueError("local_body_rot must have shape (T, B, 4)")
        if self.dof_pos.shape[0] != n:
            raise ValueError("dof_pos time axis must match root_pos")
        if self.local_body_rot.shape[2] != 4:
            raise ValueError("local_body_rot must store quaternions in xyzw")
        if self.local_body_pos.shape[2] != 3:
            raise ValueError("local_body_pos must store xyz coordinates")

    @property
    def num_frames(self) -> int:
        return int(self.root_pos.shape[0])

    @property
    def end_time(self) -> float:
        return self.start_time + self.num_frames / float(self.fps)
