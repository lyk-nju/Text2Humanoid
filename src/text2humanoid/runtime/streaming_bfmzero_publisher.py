from __future__ import annotations

from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
import threading
import time
from typing import Any

from text2humanoid.contracts.bfmzero import BFMZeroMotionChunk
from text2humanoid.runtime.bfmzero_protocol import BFMZeroMotionFrame, iter_bfmzero_motion_frames
from text2humanoid.runtime.bfmzero_zmq_sink import BFMZeroZmqSink


@dataclass(slots=True)
class BFMZeroFrameBuffer:
    """Thread-safe FIFO for contiguous BFM-Zero motion frames."""

    _frames: deque[BFMZeroMotionFrame] = field(init=False, repr=False)
    _condition: threading.Condition = field(init=False, repr=False)
    _next_frame_idx: int | None = field(init=False, default=None)
    _closed: bool = field(init=False, default=False)
    frames_enqueued: int = field(init=False, default=0)
    frames_dequeued: int = field(init=False, default=0)

    def __post_init__(self) -> None:
        self._frames = deque()
        self._condition = threading.Condition()

    def append_chunk(self, chunk: BFMZeroMotionChunk) -> int:
        with self._condition:
            if self._closed:
                raise RuntimeError("cannot append to closed BFM-Zero frame buffer")
            if self._next_frame_idx is None:
                self._next_frame_idx = chunk.frame_start
            elif chunk.frame_start != self._next_frame_idx:
                raise ValueError(
                    f"discontinuous BFM-Zero stream: expected frame_start={self._next_frame_idx}, "
                    f"got {chunk.frame_start}"
                )

            count = 0
            for frame in iter_bfmzero_motion_frames(chunk, mark_end=False):
                self._frames.append(frame)
                count += 1
            self._next_frame_idx = chunk.frame_end
            self.frames_enqueued += count
            self._condition.notify_all()
            return count

    def pop_nowait(self) -> BFMZeroMotionFrame | None:
        with self._condition:
            if not self._frames:
                return None
            self.frames_dequeued += 1
            return self._frames.popleft()

    def pop(self, timeout: float | None = None) -> BFMZeroMotionFrame | None:
        with self._condition:
            if not self._frames and not self._closed:
                self._condition.wait(timeout=timeout)
            if not self._frames:
                return None
            self.frames_dequeued += 1
            return self._frames.popleft()

    def qsize(self) -> int:
        with self._condition:
            return len(self._frames)

    def close(self) -> None:
        with self._condition:
            self._closed = True
            self._condition.notify_all()


@dataclass(slots=True)
class StreamingBFMZeroPublisher:
    """Stateful BFM-Zero ZMQ publisher with an internal paced frame buffer."""

    sink: Any | None = None
    host: str = "*"
    port: int = 5592
    realtime: bool = True
    fps: float = 50.0
    startup_delay_sec: float = 0.5
    auto_start: bool = True
    sleep_fn: Callable[[float], None] = time.sleep
    time_fn: Callable[[], float] = time.perf_counter
    buffer: BFMZeroFrameBuffer | None = None
    _sink: Any = field(init=False, repr=False)
    _buffer: BFMZeroFrameBuffer = field(init=False, repr=False)
    _stop: threading.Event = field(init=False, repr=False)
    _thread: threading.Thread | None = field(init=False, default=None, repr=False)
    _has_published: bool = field(init=False, default=False)
    frames_published: int = field(init=False, default=0)

    def __post_init__(self) -> None:
        self._sink = self.sink or BFMZeroZmqSink(host=self.host, port=self.port)
        self._buffer = self.buffer or BFMZeroFrameBuffer()
        self._stop = threading.Event()
        if self.auto_start:
            self.start()

    @property
    def queued_frames(self) -> int:
        return self._buffer.qsize()

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._thread = threading.Thread(target=self._run, name="bfmzero-paced-publisher", daemon=True)
        self._thread.start()

    def publish(self, chunk: BFMZeroMotionChunk) -> int:
        if self.auto_start:
            self.start()
        return self._buffer.append_chunk(chunk)

    def pump_frame(self) -> bool:
        frame = self._buffer.pop_nowait()
        if frame is None:
            return False
        self._publish_frame(frame)
        return True

    def _run(self) -> None:
        dt = 1.0 / float(self.fps)
        next_t = self.time_fn()
        while not self._stop.is_set():
            frame = self._buffer.pop(timeout=0.05)
            if frame is None:
                next_t = self.time_fn()
                continue
            self._publish_frame(frame)
            if self.realtime:
                next_t += dt
                sleep_for = next_t - self.time_fn()
                if sleep_for > 0:
                    self.sleep_fn(sleep_for)
                else:
                    next_t = self.time_fn()

    def _publish_frame(self, frame: BFMZeroMotionFrame) -> None:
        if not self._has_published and self.startup_delay_sec > 0:
            self.sleep_fn(float(self.startup_delay_sec))
        self._sink.publish_frame(frame)
        self._has_published = True
        self.frames_published += 1

    def close(self) -> None:
        self._stop.set()
        self._buffer.close()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        self._sink.close()
