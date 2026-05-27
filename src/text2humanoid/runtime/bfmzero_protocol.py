from __future__ import annotations

from dataclasses import dataclass
import struct
from collections.abc import Iterator

import numpy as np

from text2humanoid.contracts.bfmzero import BFMZeroMotionChunk


BFMZERO_NUM_DOF = 29
BFMZERO_MOTION_FRAME_FLAG_END = 1 << 0
BFMZERO_MOTION_FRAME_SIZE = struct.calcsize("<II") + (
    2 * BFMZERO_NUM_DOF + 3 + 4 + 3 + 3
) * 4


@dataclass(slots=True)
class BFMZeroMotionFrame:
    """Binary-compatible copy of BFM-Zero utils.common.MotionFrameMessage."""

    frame_idx: int
    joint_pos: np.ndarray
    joint_vel: np.ndarray
    root_pos: np.ndarray
    root_quat: np.ndarray
    root_lin_vel_w: np.ndarray
    root_ang_vel_w: np.ndarray
    flags: int = 0

    def __post_init__(self) -> None:
        self.frame_idx = int(self.frame_idx)
        self.flags = int(self.flags)
        self.joint_pos = np.asarray(self.joint_pos, dtype=np.float32).reshape(-1)
        self.joint_vel = np.asarray(self.joint_vel, dtype=np.float32).reshape(-1)
        self.root_pos = np.asarray(self.root_pos, dtype=np.float32).reshape(-1)
        self.root_quat = np.asarray(self.root_quat, dtype=np.float32).reshape(-1)
        self.root_lin_vel_w = np.asarray(self.root_lin_vel_w, dtype=np.float32).reshape(-1)
        self.root_ang_vel_w = np.asarray(self.root_ang_vel_w, dtype=np.float32).reshape(-1)
        if self.joint_pos.size != BFMZERO_NUM_DOF or self.joint_vel.size != BFMZERO_NUM_DOF:
            raise ValueError("joint_pos and joint_vel must each have 29 elements")
        if self.root_pos.size != 3:
            raise ValueError("root_pos must have 3 elements")
        if self.root_quat.size != 4:
            raise ValueError("root_quat must have 4 elements in wxyz order")
        if self.root_lin_vel_w.size != 3 or self.root_ang_vel_w.size != 3:
            raise ValueError("root velocities must have 3 elements")

    def to_bytes(self) -> bytes:
        header = struct.pack("<II", self.frame_idx, self.flags)
        payload = b"".join(
            arr.astype(np.float32, copy=False).tobytes()
            for arr in (
                self.joint_pos,
                self.joint_vel,
                self.root_pos,
                self.root_quat,
                self.root_lin_vel_w,
                self.root_ang_vel_w,
            )
        )
        return header + payload

    @classmethod
    def from_bytes(cls, data: bytes) -> "BFMZeroMotionFrame":
        if len(data) != BFMZERO_MOTION_FRAME_SIZE:
            raise ValueError(
                f"Motion frame data length {len(data)} != expected {BFMZERO_MOTION_FRAME_SIZE}"
            )
        header_size = struct.calcsize("<II")
        frame_idx, flags = struct.unpack("<II", data[:header_size])
        offset = header_size

        def take(n_floats: int) -> np.ndarray:
            nonlocal offset
            n_bytes = n_floats * 4
            out = np.frombuffer(data[offset : offset + n_bytes], dtype=np.float32).copy()
            offset += n_bytes
            return out

        return cls(
            frame_idx=frame_idx,
            flags=flags,
            joint_pos=take(BFMZERO_NUM_DOF),
            joint_vel=take(BFMZERO_NUM_DOF),
            root_pos=take(3),
            root_quat=take(4),
            root_lin_vel_w=take(3),
            root_ang_vel_w=take(3),
        )


def iter_bfmzero_motion_frames(
    chunk: BFMZeroMotionChunk,
    *,
    mark_end: bool = False,
) -> Iterator[BFMZeroMotionFrame]:
    last_idx = chunk.num_frames - 1
    for i in range(chunk.num_frames):
        flags = BFMZERO_MOTION_FRAME_FLAG_END if mark_end and i == last_idx else 0
        yield BFMZeroMotionFrame(
            frame_idx=chunk.frame_start + i,
            flags=flags,
            joint_pos=chunk.joint_pos[i],
            joint_vel=chunk.joint_vel[i],
            root_pos=chunk.root_pos[i],
            root_quat=chunk.root_quat[i],
            root_lin_vel_w=chunk.root_lin_vel_w[i],
            root_ang_vel_w=chunk.root_ang_vel_w[i],
        )
