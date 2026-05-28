from __future__ import annotations

from dataclasses import dataclass
from typing import Any


def frames_for_duration(duration_sec: float, fps: float) -> int:
    if duration_sec <= 0:
        raise ValueError("duration_sec must be positive")
    if fps <= 0:
        raise ValueError("fps must be positive")
    return max(1, int(round(float(duration_sec) * float(fps))))


@dataclass(frozen=True, slots=True)
class StreamingTimingConfig:
    chunk_duration_sec: float
    generation_fps: float
    retarget_fps: float
    runtime_fps: float
    low_watermark_sec: float
    target_buffer_sec: float

    def __post_init__(self) -> None:
        for field_name in (
            "chunk_duration_sec",
            "generation_fps",
            "retarget_fps",
            "runtime_fps",
            "low_watermark_sec",
            "target_buffer_sec",
        ):
            value = float(getattr(self, field_name))
            if value <= 0:
                raise ValueError(f"{field_name} must be positive")

    @property
    def generation_chunk_frames(self) -> int:
        return frames_for_duration(self.chunk_duration_sec, self.generation_fps)

    @property
    def retarget_chunk_frames(self) -> int:
        return frames_for_duration(self.chunk_duration_sec, self.retarget_fps)

    @property
    def runtime_chunk_frames(self) -> int:
        return frames_for_duration(self.chunk_duration_sec, self.runtime_fps)

    @property
    def low_watermark_frames(self) -> int:
        return frames_for_duration(self.low_watermark_sec, self.runtime_fps)

    @property
    def target_buffer_frames(self) -> int:
        return frames_for_duration(self.target_buffer_sec, self.runtime_fps)

    def to_chunk_metadata(
        self,
        *,
        source_fps: float,
        target_fps: float,
        frame_count: int,
    ) -> dict[str, float | int]:
        if frame_count <= 0:
            raise ValueError("frame_count must be positive")
        if target_fps <= 0:
            raise ValueError("target_fps must be positive")
        return {
            "chunk_duration_sec": float(self.chunk_duration_sec),
            "source_fps": float(source_fps),
            "target_fps": float(target_fps),
            "frame_count": int(frame_count),
            "duration_sec": float(frame_count) / float(target_fps),
        }


def load_streaming_timing_config(cfg: dict[str, Any]) -> StreamingTimingConfig:
    streaming_cfg = cfg.get("streaming") or {}
    planner_cfg = cfg.get("planner") or {}
    retarget_cfg = cfg.get("retarget") or {}
    runtime_cfg = cfg.get("runtime") or {}

    generation_fps = float(streaming_cfg.get("generation_fps", 20.0))
    retarget_fps = float(streaming_cfg.get("retarget_fps", retarget_cfg.get("tgt_fps", 30.0)))
    runtime_fps = float(streaming_cfg.get("runtime_fps", runtime_cfg.get("control_hz", 50.0)))

    if "chunk_duration_sec" in streaming_cfg:
        chunk_duration_sec = float(streaming_cfg["chunk_duration_sec"])
    else:
        legacy_chunk_frames = int(planner_cfg.get("chunk_frames", 40))
        chunk_duration_sec = float(legacy_chunk_frames) / generation_fps

    if "low_watermark_sec" in streaming_cfg:
        low_watermark_sec = float(streaming_cfg["low_watermark_sec"])
    else:
        low_watermark_frames = int(runtime_cfg.get("low_watermark_frames", 20))
        low_watermark_sec = float(low_watermark_frames) / runtime_fps

    if "target_buffer_sec" in streaming_cfg:
        target_buffer_sec = float(streaming_cfg["target_buffer_sec"])
    else:
        target_buffer_frames = int(runtime_cfg.get("high_watermark_frames", 60))
        target_buffer_sec = float(target_buffer_frames) / runtime_fps

    return StreamingTimingConfig(
        chunk_duration_sec=chunk_duration_sec,
        generation_fps=generation_fps,
        retarget_fps=retarget_fps,
        runtime_fps=runtime_fps,
        low_watermark_sec=low_watermark_sec,
        target_buffer_sec=target_buffer_sec,
    )
