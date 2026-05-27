from __future__ import annotations

import time

from text2humanoid.contracts.bfmzero import BFMZeroMotionChunk
from text2humanoid.runtime.bfmzero_protocol import BFMZeroMotionFrame, iter_bfmzero_motion_frames


class BFMZeroZmqSink:
    """Publisher for BFM-Zero tracking_online source=zmq.

    This class imports pyzmq lazily so format tests can run without the BFM-Zero
    runtime environment installed.
    """

    def __init__(self, host: str = "*", port: int = 5592) -> None:
        import zmq

        self.host = host
        self.port = int(port)
        self._context = zmq.Context.instance()
        self._socket = self._context.socket(zmq.PUB)
        self._socket.bind(f"tcp://{host}:{self.port}")

    def publish_chunk(
        self,
        chunk: BFMZeroMotionChunk,
        *,
        realtime: bool = True,
        mark_end: bool = False,
        startup_delay_sec: float = 0.2,
    ) -> int:
        """Publish one motion chunk as BFM-Zero MotionFrameMessage frames."""

        if startup_delay_sec > 0:
            time.sleep(float(startup_delay_sec))
        dt = 1.0 / float(chunk.fps)
        next_t = time.perf_counter()
        count = 0
        for frame in iter_bfmzero_motion_frames(chunk, mark_end=mark_end):
            self._socket.send(frame.to_bytes())
            count += 1
            if realtime:
                next_t += dt
                sleep_for = next_t - time.perf_counter()
                if sleep_for > 0:
                    time.sleep(sleep_for)
                else:
                    next_t = time.perf_counter()
        return count

    def publish_frame(self, frame: BFMZeroMotionFrame) -> None:
        self._socket.send(frame.to_bytes())

    def close(self) -> None:
        self._socket.close(linger=200)
