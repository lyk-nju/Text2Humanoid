from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal
import uuid

import numpy as np

from text2humanoid.contracts.validation import (
    as_float32_matrix,
    base_chunk_metadata,
    validate_fps,
    validate_known_representation_shape,
    validate_quat_order,
)


def _new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex}"


@dataclass(slots=True)
class TextSegment:
    text: str
    start_time: float | None = None
    end_time: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class TextInput:
    prompt: str
    segments: list[TextSegment] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.prompt = str(self.prompt)

    def to_dict(self) -> dict[str, Any]:
        return {
            "prompt": self.prompt,
            "segments": [segment.to_dict() for segment in self.segments],
            "metadata": self.metadata,
        }


@dataclass(slots=True)
class TrajectoryPoint:
    t: float
    x: float
    y: float
    z: float

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


@dataclass(slots=True)
class TrajInput:
    waypoints: list[TrajectoryPoint] = field(default_factory=list)
    token_aligned_traj: np.ndarray | None = None
    token_mask: np.ndarray | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.token_aligned_traj is not None:
            self.token_aligned_traj = np.asarray(self.token_aligned_traj, dtype=np.float32)
        if self.token_mask is not None:
            self.token_mask = np.asarray(self.token_mask, dtype=np.float32)

    def to_dict(self) -> dict[str, Any]:
        return {
            "waypoints": [point.to_dict() for point in self.waypoints],
            "token_aligned_traj": None
            if self.token_aligned_traj is None
            else self.token_aligned_traj.tolist(),
            "token_mask": None if self.token_mask is None else self.token_mask.tolist(),
            "metadata": self.metadata,
        }


@dataclass(slots=True)
class MultimodalInput:
    input_id: str = field(default_factory=lambda: _new_id("input"))
    text_input: TextInput | None = None
    traj_input: TrajInput | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def require_text_prompt(self) -> str:
        if self.text_input is None or not self.text_input.prompt.strip():
            raise ValueError("text_input.prompt is required")
        return self.text_input.prompt

    def to_dict(self) -> dict[str, Any]:
        return {
            "input_id": self.input_id,
            "text_input": None if self.text_input is None else self.text_input.to_dict(),
            "traj_input": None if self.traj_input is None else self.traj_input.to_dict(),
            "metadata": self.metadata,
        }


@dataclass(slots=True)
class GenerateSpec:
    mode: Literal["offline", "stream"] = "offline"
    fps: int = 20
    num_frames: int | None = None
    chunk_frames: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.fps = int(self.fps)
        if self.fps <= 0:
            raise ValueError(f"fps must be positive, got {self.fps}")
        if self.mode == "offline" and self.num_frames is not None and self.num_frames <= 0:
            raise ValueError("num_frames must be positive when provided")
        if self.mode == "stream" and self.chunk_frames is not None and self.chunk_frames <= 0:
            raise ValueError("chunk_frames must be positive when provided")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class GenerateRequest:
    input: MultimodalInput
    spec: GenerateSpec
    request_id: str = field(default_factory=lambda: _new_id("genreq"))
    session_id: str | None = None
    state_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "request_id": self.request_id,
            "input": self.input.to_dict(),
            "spec": self.spec.to_dict(),
            "session_id": self.session_id,
            "state_id": self.state_id,
            "metadata": self.metadata,
        }


@dataclass(slots=True)
class GeneratedMotion:
    motion_id: str
    representation: str
    motion: np.ndarray
    fps: int
    start_time: float = 0.0
    source_input_id: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.motion = as_float32_matrix(self.motion, field_name="motion")
        validate_known_representation_shape(self.representation, self.motion.shape)
        self.fps = validate_fps(self.fps)

    @property
    def num_frames(self) -> int:
        return int(self.motion.shape[0])

    @property
    def dim(self) -> int:
        return int(self.motion.shape[1])

    @property
    def end_time(self) -> float:
        return self.start_time + self.num_frames / float(self.fps)

    @property
    def duration_sec(self) -> float:
        return self.num_frames / float(self.fps)

    @property
    def motion_shape(self) -> tuple[int, int]:
        return tuple(int(x) for x in self.motion.shape)

    def to_chunk_metadata(self) -> dict[str, Any]:
        data = base_chunk_metadata(
            chunk_id=self.motion_id,
            representation=self.representation,
            fps=self.fps,
            frame_count=self.num_frames,
            start_time=self.start_time,
            shape=self.motion.shape,
        )
        data.update(self.metadata)
        return data


@dataclass(slots=True)
class RetargetInput:
    input_id: str
    source_motion_id: str
    representation: str
    motion: np.ndarray
    fps: int
    start_time: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.motion = as_float32_matrix(self.motion, field_name="motion")
        validate_known_representation_shape(self.representation, self.motion.shape)
        self.fps = validate_fps(self.fps)

    @property
    def num_frames(self) -> int:
        return int(self.motion.shape[0])

    @property
    def duration_sec(self) -> float:
        return self.num_frames / float(self.fps)

    @property
    def motion_shape(self) -> tuple[int, int]:
        return tuple(int(x) for x in self.motion.shape)

    def to_chunk_metadata(self) -> dict[str, Any]:
        data = base_chunk_metadata(
            chunk_id=self.input_id,
            representation=self.representation,
            fps=self.fps,
            frame_count=self.num_frames,
            start_time=self.start_time,
            shape=self.motion.shape,
            source_chunk_id=self.source_motion_id,
        )
        data.update(self.metadata)
        return data


@dataclass(slots=True)
class RobotMotion:
    motion_id: str
    source_input_id: str
    robot: str
    representation: str
    root_pos: np.ndarray
    root_quat: np.ndarray
    dof_pos: np.ndarray
    fps: int
    joint_names: list[str]
    quat_order: str = "wxyz"
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.root_pos = np.asarray(self.root_pos, dtype=np.float32)
        self.root_quat = np.asarray(self.root_quat, dtype=np.float32)
        self.dof_pos = np.asarray(self.dof_pos, dtype=np.float32)
        n = self.dof_pos.shape[0] if self.dof_pos.ndim == 2 else -1
        if self.dof_pos.ndim != 2:
            raise ValueError(f"dof_pos must have shape (T, J), got {self.dof_pos.shape}")
        if self.root_pos.shape != (n, 3):
            raise ValueError(f"root_pos must have shape {(n, 3)}, got {self.root_pos.shape}")
        if self.root_quat.shape != (n, 4):
            raise ValueError(f"root_quat must have shape {(n, 4)}, got {self.root_quat.shape}")
        if len(self.joint_names) != self.dof_pos.shape[1]:
            raise ValueError("joint_names length must match dof_pos dimension")
        self.quat_order = validate_quat_order(self.quat_order)
        self.fps = validate_fps(self.fps)

    @property
    def num_frames(self) -> int:
        return int(self.dof_pos.shape[0])

    @property
    def duration_sec(self) -> float:
        return self.num_frames / float(self.fps)

    @property
    def motion_shape(self) -> dict[str, tuple[int, ...]]:
        return {
            "root_pos": tuple(int(x) for x in self.root_pos.shape),
            "root_quat": tuple(int(x) for x in self.root_quat.shape),
            "dof_pos": tuple(int(x) for x in self.dof_pos.shape),
        }

    def to_chunk_metadata(self) -> dict[str, Any]:
        data = base_chunk_metadata(
            chunk_id=self.motion_id,
            representation=self.representation,
            fps=self.fps,
            frame_count=self.num_frames,
            source_chunk_id=self.source_input_id,
            shape=self.dof_pos.shape,
            joint_order=",".join(self.joint_names),
            quat_order=self.quat_order,
        )
        data["robot"] = self.robot
        data.update(self.metadata)
        return data


@dataclass(slots=True)
class TrackerInput:
    input_id: str
    source_motion_id: str
    tracker: str
    representation: str
    payload: Any
    fps: int
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.fps = validate_fps(self.fps)


@dataclass(slots=True)
class TrackerStatus:
    tracker: str
    phase: str
    frames_pushed: int = 0
    last_frame_idx: int | None = None
    errors: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


GeneratedMotionChunk = GeneratedMotion
RetargetInputChunk = RetargetInput
RobotMotionChunk = RobotMotion


def human_motion_chunk_to_generated_motion(chunk: Any) -> GeneratedMotion:
    """Adapter: contracts.chunks.HumanMotionChunk -> GeneratedMotion.

    Lets legacy orchestrator code that still produces HumanMotionChunk
    flow into streaming-style consumers that expect GeneratedMotion.
    """
    return GeneratedMotion(
        motion_id=chunk.chunk_id,
        representation="humanml3d_263",
        motion=chunk.motion_263,
        fps=chunk.fps,
        start_time=chunk.start_time,
        metadata=dict(chunk.metadata),
    )


def nmr_input_chunk_to_retarget_input(chunk: Any) -> RetargetInput:
    """Adapter: contracts.chunks.NMRInputChunk -> RetargetInput.

    The orchestrator's PipelineCoordinator still builds NMRInputChunk via
    `human_chunk_to_nmr_input`.  Use this when feeding such a chunk into
    a streaming-style retarget consumer.
    """
    return RetargetInput(
        input_id=chunk.chunk_id,
        source_motion_id=chunk.metadata.get("source_chunk_id", chunk.chunk_id),
        representation="nmr_smplx_140",
        motion=chunk.motion_140,
        fps=chunk.fps,
        start_time=chunk.start_time,
        metadata=dict(chunk.metadata),
    )
