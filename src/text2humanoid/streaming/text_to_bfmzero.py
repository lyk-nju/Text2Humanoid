from __future__ import annotations

from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
import os
from pathlib import Path
import time
from typing import Any
import uuid

import numpy as np
import yaml

from text2humanoid.contracts.bfmzero import BFMZeroMotionChunk
from text2humanoid.contracts.chunks import NMRInputChunk
from text2humanoid.contracts.pipeline import GenerateRequest, GenerateSpec, GeneratedMotion, MultimodalInput, TextInput
from text2humanoid.demo.console_controller import StreamJobRequest
from text2humanoid.runtime.streaming_bfmzero_publisher import StreamingBFMZeroPublisher


def _as_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value, dtype=np.float32)


def _text2humanoid_dir() -> Path:
    return Path(__file__).resolve().parents[3]


def _resolve_config_path(value: str) -> Path:
    path = Path(os.path.expandvars(value)).expanduser()
    if path.is_absolute():
        return path
    return (_text2humanoid_dir() / path).resolve()


def _resolve_root_path(cfg: dict[str, Any]) -> Path:
    raw = cfg.get("root_path", "auto")
    if raw == "auto" or raw is None:
        return Path(os.environ.get("TEXT2MOTION_ROOT", _text2humanoid_dir().parent)).expanduser().resolve()
    return Path(os.path.expandvars(raw)).expanduser().resolve()


def _resolve_path(root: Path, value: str) -> Path:
    path = Path(os.path.expandvars(value)).expanduser()
    if path.is_absolute():
        return path
    return root / path


def _default_convert_263_to_140(*args, **kwargs):
    from text2humanoid.retarget.bridge_263_to_140 import floodnet_263_to_nmr_140

    return floodnet_263_to_nmr_140(*args, **kwargs)


def _default_convert_retarget_to_bfmzero(*args, **kwargs):
    from text2humanoid.retarget.bfmzero_adapter import make_tracking_easy_result_to_bfmzero_motion

    return make_tracking_easy_result_to_bfmzero_motion(*args, **kwargs)


def _slice_bfmzero_chunk(chunk: BFMZeroMotionChunk, start: int, frame_start: int, chunk_id: str) -> BFMZeroMotionChunk:
    return BFMZeroMotionChunk(
        chunk_id=chunk_id,
        fps=chunk.fps,
        frame_start=frame_start,
        joint_pos=chunk.joint_pos[start:],
        joint_vel=chunk.joint_vel[start:],
        root_pos=chunk.root_pos[start:],
        root_quat=chunk.root_quat[start:],
        root_lin_vel_w=chunk.root_lin_vel_w[start:],
        root_ang_vel_w=chunk.root_ang_vel_w[start:],
        metadata=dict(chunk.metadata),
    )


def _coalesce_generated_motions(chunks: list[GeneratedMotion]) -> GeneratedMotion:
    if not chunks:
        raise ValueError("chunks must not be empty")
    if len(chunks) == 1:
        return chunks[0]
    first = chunks[0]
    last = chunks[-1]
    return GeneratedMotion(
        motion_id=f"{first.motion_id}_to_{last.motion_id}",
        representation=first.representation,
        motion=np.concatenate([chunk.motion for chunk in chunks], axis=0),
        fps=first.fps,
        start_time=first.start_time,
        source_input_id=first.source_input_id,
        metadata={
            **last.metadata,
            "coalesced_chunks": len(chunks),
            "source_motion_ids": [chunk.motion_id for chunk in chunks],
            "frame_start": first.metadata.get("frame_start", 0),
        },
    )


def _empty_stream_metrics() -> dict[str, dict[str, float | int]]:
    return {
        "generation": {"fps": 0.0, "buffer_frames": 0},
        "retarget": {"fps": 0.0, "buffer_frames": 0},
        "motion_control": {"fps": 0.0, "buffer_frames": 0},
    }


@dataclass(slots=True)
class StreamingRetargetBridge:
    retarget: Any
    xml_path: str
    device: str = "cpu"
    context_frames: int = 30
    retarget_fps: float = 30.0
    output_fps: float = 50.0
    convert_263_to_140: Callable[..., Any] = _default_convert_263_to_140
    convert_retarget_to_bfmzero: Callable[..., BFMZeroMotionChunk] = _default_convert_retarget_to_bfmzero
    _context_263: np.ndarray = field(init=False, repr=False)
    _next_frame_idx: int = field(init=False, default=0)

    def __post_init__(self) -> None:
        self.context_frames = max(0, int(self.context_frames))
        self.retarget_fps = float(self.retarget_fps)
        self.output_fps = float(self.output_fps)
        self._context_263 = np.zeros((0, 263), dtype=np.float32)

    def convert(self, motion: GeneratedMotion) -> BFMZeroMotionChunk:
        new_263 = np.asarray(motion.motion, dtype=np.float32)
        if self._context_263.size:
            window_263 = np.concatenate([self._context_263, new_263], axis=0)
        else:
            window_263 = new_263

        motion_140 = self.convert_263_to_140(
            window_263,
            src_fps=float(motion.fps),
            tgt_fps=float(self.retarget_fps),
        )
        motion_140_np = _as_numpy(motion_140)
        nmr_chunk = NMRInputChunk(
            chunk_id=motion.motion_id,
            start_time=max(0.0, motion.start_time - self._context_263.shape[0] / float(motion.fps)),
            fps=int(round(self.retarget_fps)),
            motion_140=motion_140_np,
            metadata=dict(motion.metadata),
        )
        result = self.retarget.retarget_chunk(nmr_chunk)
        full_chunk = self.convert_retarget_to_bfmzero(
            result,
            xml_path=self.xml_path,
            device=self.device,
            chunk_id=motion.motion_id,
            frame_start=self._next_frame_idx,
            src_fps=float(self.retarget_fps),
            tgt_fps=float(self.output_fps),
        )

        new_seconds = new_263.shape[0] / float(motion.fps)
        new_output_frames = max(1, int(round(new_seconds * float(full_chunk.fps))))
        tail_start = max(0, full_chunk.num_frames - new_output_frames)
        out = _slice_bfmzero_chunk(full_chunk, tail_start, self._next_frame_idx, motion.motion_id)
        out.metadata.update(
            {
                "source_motion_id": motion.motion_id,
                "source_text": motion.metadata.get("text", ""),
                "context_263_frames": int(self._context_263.shape[0]),
                "window_263_frames": int(window_263.shape[0]),
                "tail_start": int(tail_start),
            }
        )
        self._next_frame_idx = out.frame_end
        if self.context_frames > 0:
            self._context_263 = window_263[-self.context_frames :].copy()
        else:
            self._context_263 = np.zeros((0, 263), dtype=np.float32)
        return out


@dataclass(slots=True)
class StreamingTextToBFMZeroRunner:
    generation_backend: Any
    retarget_bridge: StreamingRetargetBridge
    publisher: Any
    src_fps: int = 20
    progress_callback: Callable[[str, str], None] | None = None
    generated_motion_callback: Callable[[GeneratedMotion], None] | None = None
    metrics_callback: Callable[[dict[str, dict[str, float | int]]], None] | None = None
    _metrics: dict[str, dict[str, float | int]] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._metrics = _empty_stream_metrics()

    def _emit(self, stage: str, message: str) -> None:
        if self.progress_callback is not None:
            self.progress_callback(stage, message)

    def _emit_metrics(self, **sections: dict[str, float | int]) -> None:
        for section, values in sections.items():
            self._metrics.setdefault(section, {}).update(values)
        if self.metrics_callback is not None:
            self.metrics_callback({name: dict(values) for name, values in self._metrics.items()})

    def _publisher_fps(self, fallback: float) -> float:
        try:
            return float(getattr(self.publisher, "fps", fallback))
        except (TypeError, ValueError):
            return float(fallback)

    def _publisher_queued_frames(self) -> int:
        try:
            return int(getattr(self.publisher, "queued_frames", 0))
        except (TypeError, ValueError):
            return 0

    def __call__(
        self,
        request: StreamJobRequest,
        should_stop: Callable[[], bool],
        current_text: Callable[[], str],
    ) -> int:
        if hasattr(self.retarget_bridge, "context_frames"):
            self.retarget_bridge.context_frames = max(0, int(request.context_frames))
        if hasattr(self.retarget_bridge, "output_fps"):
            self.retarget_bridge.output_fps = float(request.output_fps)

        gen_request = GenerateRequest(
            input=MultimodalInput(text_input=TextInput(prompt=request.text)),
            spec=GenerateSpec(
                mode="stream",
                fps=int(self.src_fps),
                chunk_frames=int(request.chunk_frames),
                metadata={
                    "history_length": int(request.history_length),
                    "denoise_steps": int(request.denoise_steps),
                },
            ),
            request_id=f"stream_{uuid.uuid4().hex[:12]}",
        )

        self._emit("stream_initializing", "Initializing FloodDiffusion stream")
        chunks: Iterator[GeneratedMotion] = self.generation_backend.stream_chunks(gen_request, should_stop=should_stop)
        self._metrics = _empty_stream_metrics()
        self._emit_metrics(
            motion_control={
                "fps": self._publisher_fps(float(request.output_fps)),
                "buffer_frames": self._publisher_queued_frames(),
            }
        )
        started = False
        last_text = request.text
        pending: list[GeneratedMotion] = []
        pending_frames = 0
        min_retarget_frames = max(4, int(request.chunk_frames))
        try:
            while not should_stop():
                chunk_t0 = time.perf_counter()
                if started and hasattr(self.generation_backend, "update_text"):
                    next_text = current_text()
                    if next_text != last_text:
                        self.generation_backend.update_text(next_text)
                        last_text = next_text
                try:
                    generation_t0 = time.perf_counter()
                    motion = next(chunks)
                except StopIteration:
                    break
                generation_elapsed = time.perf_counter() - generation_t0
                if not started:
                    self._emit("stream_initialized", "FloodDiffusion stream initialized")
                started = True
                if self.generated_motion_callback is not None:
                    self.generated_motion_callback(motion)
                pending.append(motion)
                pending_frames += motion.num_frames
                generation_fps = motion.num_frames / generation_elapsed if generation_elapsed > 0 else 0.0
                self._emit_metrics(
                    generation={
                        "fps": float(generation_fps),
                        "buffer_frames": int(pending_frames),
                    },
                    motion_control={
                        "fps": self._publisher_fps(float(request.output_fps)),
                        "buffer_frames": self._publisher_queued_frames(),
                    },
                )
                if pending_frames < min_retarget_frames:
                    continue

                motion_to_retarget = _coalesce_generated_motions(pending)
                pending = []
                pending_frames = 0

                retarget_t0 = time.perf_counter()
                bfmzero_chunk = self.retarget_bridge.convert(motion_to_retarget)
                sent = self.publisher.publish(bfmzero_chunk)
                retarget_elapsed = time.perf_counter() - retarget_t0
                retarget_fps = sent / retarget_elapsed if retarget_elapsed > 0 else 0.0
                retarget_buffer_frames = int(
                    bfmzero_chunk.metadata.get(
                        "context_263_frames",
                        getattr(self.retarget_bridge, "context_frames", 0),
                    )
                )
                queued_frames = self._publisher_queued_frames()
                self._emit_metrics(
                    generation={"buffer_frames": int(pending_frames)},
                    retarget={
                        "fps": float(retarget_fps),
                        "buffer_frames": retarget_buffer_frames,
                    },
                    motion_control={
                        "fps": self._publisher_fps(float(request.output_fps)),
                        "buffer_frames": queued_frames,
                    },
                )
                elapsed_ms = (time.perf_counter() - chunk_t0) * 1000.0
                pipeline_fps = sent / (elapsed_ms / 1000.0) if elapsed_ms > 0 else 0.0
                queued_suffix = f" queued_frames={queued_frames}" if queued_frames is not None else ""
                self._emit(
                    "stream_chunk",
                    f"Queued stream chunk frames={sent} "
                    f"frame_start={bfmzero_chunk.frame_start} "
                    f"text=\"{bfmzero_chunk.metadata.get('source_text', '')}\" "
                    f"elapsed_ms={elapsed_ms:.1f} pipeline_fps={pipeline_fps:.1f}"
                    f"{queued_suffix}",
                )
        finally:
            if hasattr(self.publisher, "close"):
                self.publisher.close()
        return 0


def build_streaming_text_to_bfmzero_runner(
    *,
    system_config: str = "configs/system/local_dev.yaml",
    flooddiffusion_root: str | Path | None = None,
    flooddiffusion_config: str = "configs/stream.yaml",
    host: str = "*",
    port: int = 5592,
    startup_delay_sec: float = 0.5,
    device: str = "cpu",
    progress_callback: Callable[[str, str], None] | None = None,
    generated_motion_callback: Callable[[GeneratedMotion], None] | None = None,
    metrics_callback: Callable[[dict[str, dict[str, float | int]]], None] | None = None,
) -> StreamingTextToBFMZeroRunner:
    from text2humanoid.generation import FloodDiffusionStreamingBackend
    from text2humanoid.retarget.nmr_service import NMRRetargetService

    cfg_path = _resolve_config_path(system_config)
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    root = _resolve_root_path(cfg)
    retarget_cfg = cfg.get("retarget", {})
    runtime_cfg = cfg.get("runtime", {})
    retarget_fps = float(retarget_cfg.get("tgt_fps", 30))
    output_fps = float(runtime_cfg.get("control_hz", 50))
    xml_path = str(_resolve_path(root, retarget_cfg.get("xml_path", "MakeTrackingEasy/assets/g1_mocap_29dof.xml")))

    text_update_callback = None
    if progress_callback is not None:
        text_update_callback = lambda text, elapsed: progress_callback(
            "stream_text_ready",
            f'Text update ready: "{text}" update used {elapsed:.2f}s',
        )

    generation = FloodDiffusionStreamingBackend(
        flooddiffusion_root=flooddiffusion_root or (root / "FloodDiffusion"),
        config_path=flooddiffusion_config,
        text_update_callback=text_update_callback,
    )
    retarget = NMRRetargetService(
        apply_filter=bool(retarget_cfg.get("apply_filter", True)),
        tgt_fps=int(round(retarget_fps)),
    )
    bridge = StreamingRetargetBridge(
        retarget=retarget,
        xml_path=xml_path,
        device=device,
        context_frames=30,
        retarget_fps=retarget_fps,
        output_fps=output_fps,
    )
    publisher = StreamingBFMZeroPublisher(
        host=host,
        port=int(port),
        realtime=True,
        fps=output_fps,
        startup_delay_sec=float(startup_delay_sec),
    )
    return StreamingTextToBFMZeroRunner(
        generation_backend=generation,
        retarget_bridge=bridge,
        publisher=publisher,
        src_fps=20,
        progress_callback=progress_callback,
        generated_motion_callback=generated_motion_callback,
        metrics_callback=metrics_callback,
    )
