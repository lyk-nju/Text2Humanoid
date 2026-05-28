from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from text2humanoid.contracts.validation import base_chunk_metadata, validate_fps


G1_ISAAC_JOINT_NAMES = [
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
]


# Copied from BFM-Zero/scripts/utils/motion_loader.py.
# Meaning: bmimic_dof[i] corresponds to isaac_dof[G1_JOINT_MAPPING[i]].
G1_BMIMIC_TO_ISAAC_INDEX = np.asarray(
    [
        0, 6, 12,
        1, 7, 13,
        2, 8, 14,
        3, 9, 15, 22,
        4, 10, 16, 23,
        5, 11, 17, 24,
        18, 25,
        19, 26,
        20, 27,
        21, 28,
    ],
    dtype=np.int64,
)

G1_ISAAC_TO_BMIMIC_INDEX = np.zeros(29, dtype=np.int64)
for _bmimic_idx, _isaac_idx in enumerate(G1_BMIMIC_TO_ISAAC_INDEX.tolist()):
    G1_ISAAC_TO_BMIMIC_INDEX[_isaac_idx] = _bmimic_idx


def bmimic_dof_to_isaac(dof: np.ndarray) -> np.ndarray:
    arr = np.asarray(dof, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[1] != 29:
        raise ValueError(f"dof must have shape (T, 29), got {arr.shape}")
    return arr[:, G1_ISAAC_TO_BMIMIC_INDEX].astype(np.float32, copy=False)


@dataclass(slots=True)
class BFMZeroMotionChunk:
    """Motion chunk in the direct BFM-Zero tracking_online ZMQ format.

    Joint arrays are already in BFM-Zero Isaac order. Quaternions are wxyz.
    """

    chunk_id: str
    fps: int
    frame_start: int
    joint_pos: np.ndarray
    joint_vel: np.ndarray
    root_pos: np.ndarray
    root_quat: np.ndarray
    root_lin_vel_w: np.ndarray
    root_ang_vel_w: np.ndarray
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.fps = validate_fps(self.fps)
        self.frame_start = int(self.frame_start)
        self.joint_pos = np.asarray(self.joint_pos, dtype=np.float32)
        self.joint_vel = np.asarray(self.joint_vel, dtype=np.float32)
        self.root_pos = np.asarray(self.root_pos, dtype=np.float32)
        self.root_quat = np.asarray(self.root_quat, dtype=np.float32)
        self.root_lin_vel_w = np.asarray(self.root_lin_vel_w, dtype=np.float32)
        self.root_ang_vel_w = np.asarray(self.root_ang_vel_w, dtype=np.float32)

        n = self.joint_pos.shape[0] if self.joint_pos.ndim == 2 else -1
        expected = {
            "joint_pos": (n, 29),
            "joint_vel": (n, 29),
            "root_pos": (n, 3),
            "root_quat": (n, 4),
            "root_lin_vel_w": (n, 3),
            "root_ang_vel_w": (n, 3),
        }
        actual = {
            "joint_pos": self.joint_pos.shape,
            "joint_vel": self.joint_vel.shape,
            "root_pos": self.root_pos.shape,
            "root_quat": self.root_quat.shape,
            "root_lin_vel_w": self.root_lin_vel_w.shape,
            "root_ang_vel_w": self.root_ang_vel_w.shape,
        }
        for key, shape in expected.items():
            if actual[key] != shape:
                raise ValueError(f"{key} must have shape {shape}, got {actual[key]}")
        norm = np.linalg.norm(self.root_quat, axis=1, keepdims=True)
        if np.any(norm < 1e-8):
            raise ValueError("root_quat contains zero-norm quaternion")
        self.root_quat = self.root_quat / norm

    @property
    def num_frames(self) -> int:
        return int(self.joint_pos.shape[0])

    @property
    def frame_end(self) -> int:
        return self.frame_start + self.num_frames

    @property
    def duration_sec(self) -> float:
        return self.num_frames / float(self.fps)

    def to_chunk_metadata(self) -> dict[str, Any]:
        data = base_chunk_metadata(
            chunk_id=self.chunk_id,
            representation="bfmzero_motion_frame_stream",
            fps=self.fps,
            frame_count=self.num_frames,
            shape=self.joint_pos.shape,
            joint_order=str(self.metadata.get("runtime_joint_order", "isaac")),
            quat_order=str(self.metadata.get("root_quat_order", "wxyz")),
        )
        data["frame_start"] = self.frame_start
        data["frame_end"] = self.frame_end
        data.update(self.metadata)
        return data


def bfmzero_motion_from_bmimic_data(
    data: dict[str, np.ndarray],
    *,
    chunk_id: str = "",
    frame_start: int = 0,
    source_joint_order: str = "bmimic",
) -> BFMZeroMotionChunk:
    """Convert BFM-Zero data-format npz fields into direct ZMQ motion format.

    The BFM-Zero file loader remaps bmimic joint order to Isaac order. ZMQ mode
    bypasses that loader, so this conversion performs the remap explicitly.
    """

    fps_raw = np.asarray(data["fps"])
    fps = int(fps_raw.item() if fps_raw.ndim > 0 else fps_raw)

    joint_pos = np.asarray(data["joint_pos"], dtype=np.float32)
    joint_vel = np.asarray(data["joint_vel"], dtype=np.float32)
    if source_joint_order == "bmimic":
        joint_pos = bmimic_dof_to_isaac(joint_pos)
        joint_vel = bmimic_dof_to_isaac(joint_vel)
    elif source_joint_order != "isaac":
        raise ValueError("source_joint_order must be 'bmimic' or 'isaac'")

    body_pos_w = np.asarray(data["body_pos_w"], dtype=np.float32)
    body_quat_w = np.asarray(data["body_quat_w"], dtype=np.float32)
    body_lin_vel_w = np.asarray(data["body_lin_vel_w"], dtype=np.float32)
    body_ang_vel_w = np.asarray(data["body_ang_vel_w"], dtype=np.float32)

    return BFMZeroMotionChunk(
        chunk_id=chunk_id,
        fps=fps,
        frame_start=frame_start,
        joint_pos=joint_pos,
        joint_vel=joint_vel,
        root_pos=body_pos_w[:, 0, :],
        root_quat=body_quat_w[:, 0, :],
        root_lin_vel_w=body_lin_vel_w[:, 0, :],
        root_ang_vel_w=body_ang_vel_w[:, 0, :],
        metadata={
            "source_format": "bfmzero_data_npz",
            "source_joint_order": source_joint_order,
            "runtime_joint_order": "isaac",
            "root_quat_order": "wxyz",
        },
    )
