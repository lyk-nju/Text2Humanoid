from __future__ import annotations

from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
import os
import sys
import threading
from typing import Any

import numpy as np

from text2humanoid.contracts.pipeline import GeneratedMotion


def _text2motion_root() -> Path:
    return Path(os.environ.get("TEXT2MOTION_ROOT", Path(__file__).resolve().parents[4])).resolve()


def _default_recover_frame(frame_263: np.ndarray) -> np.ndarray:
    flooddiffusion_root = _text2motion_root() / "FloodDiffusion"
    root = str(flooddiffusion_root)
    if root not in sys.path:
        sys.path.insert(0, root)
    from utils.motion_process import StreamJointRecovery263

    recovery = getattr(_default_recover_frame, "_recovery", None)
    if recovery is None:
        recovery = StreamJointRecovery263(joints_num=22, smoothing_alpha=0.5)
        setattr(_default_recover_frame, "_recovery", recovery)
    return recovery.process_frame(np.asarray(frame_263, dtype=np.float32))


@dataclass(slots=True)
class Streaming263DPreviewBuffer:
    """Live 263D preview buffer for the demo console Artifacts panel."""

    recover_frame: Callable[[np.ndarray], np.ndarray] = _default_recover_frame
    max_frames: int = 240
    _frames: deque[dict[str, Any]] = field(init=False, repr=False)
    _lock: threading.Lock = field(init=False, repr=False)
    _next_frame_idx: int = field(init=False, default=0)
    frames_enqueued: int = field(init=False, default=0)
    frames_dequeued: int = field(init=False, default=0)

    def __post_init__(self) -> None:
        self.max_frames = max(1, int(self.max_frames))
        self._frames = deque(maxlen=self.max_frames)
        self._lock = threading.Lock()

    def reset(self) -> None:
        with self._lock:
            self._frames.clear()
            self._next_frame_idx = 0
            self.frames_enqueued = 0
            self.frames_dequeued = 0
        recovery = getattr(self.recover_frame, "_recovery", None)
        if recovery is not None and hasattr(recovery, "reset"):
            recovery.reset()

    def append_motion(self, motion: GeneratedMotion) -> int:
        motion_263 = np.asarray(motion.motion, dtype=np.float32)
        if motion_263.ndim != 2 or motion_263.shape[1] != 263:
            raise ValueError(f"Expected 263D motion with shape (T, 263), got {motion_263.shape}")
        text = str(motion.metadata.get("text", ""))
        appended = 0
        with self._lock:
            for frame_263 in motion_263:
                joints = np.asarray(self.recover_frame(frame_263), dtype=np.float32)
                if joints.shape != (22, 3):
                    raise ValueError(f"Expected recovered joints shape (22, 3), got {joints.shape}")
                self._frames.append(
                    {
                        "frame_idx": self._next_frame_idx,
                        "text": text,
                        "joints": joints.tolist(),
                    }
                )
                self._next_frame_idx += 1
                appended += 1
            self.frames_enqueued += appended
        return appended

    def pop_frame(self) -> dict[str, Any]:
        with self._lock:
            if not self._frames:
                return {
                    "status": "waiting",
                    "frames_enqueued": self.frames_enqueued,
                    "frames_dequeued": self.frames_dequeued,
                }
            frame = self._frames.popleft()
            self.frames_dequeued += 1
            return {
                "status": "ok",
                **frame,
                "queued_frames": len(self._frames),
                "frames_enqueued": self.frames_enqueued,
                "frames_dequeued": self.frames_dequeued,
            }

    def status(self) -> dict[str, Any]:
        with self._lock:
            return {
                "queued_frames": len(self._frames),
                "frames_enqueued": self.frames_enqueued,
                "frames_dequeued": self.frames_dequeued,
            }
