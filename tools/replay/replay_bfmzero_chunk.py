from __future__ import annotations

import argparse
from collections.abc import Sequence

from text2humanoid.runtime.bfmzero_chunk_npz import load_bfmzero_chunk_npz
from text2humanoid.runtime.bfmzero_zmq_sink import BFMZeroZmqSink


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Replay a saved BFMZeroMotionChunk NPZ over ZMQ.")
    parser.add_argument("--input", required=True, help="Path to BFMZeroMotionChunk NPZ")
    parser.add_argument("--host", default="*", help="ZMQ bind host")
    parser.add_argument("--port", type=int, default=5592, help="ZMQ bind port")
    parser.add_argument("--mark-end", action="store_true", help="Mark final frame with END")
    parser.add_argument("--no-realtime", action="store_true", help="Publish without frame sleeps")
    parser.add_argument("--startup-delay-sec", type=float, default=0.5)
    parser.add_argument("--dry-run", action="store_true", help="Load payload but do not publish")
    args = parser.parse_args(argv)

    chunk = load_bfmzero_chunk_npz(args.input)
    print(
        f"chunk_id={chunk.chunk_id} frames={chunk.num_frames} fps={chunk.fps} "
        f"frame_start={chunk.frame_start}"
    )
    if args.dry_run:
        return 0

    sink = BFMZeroZmqSink(host=args.host, port=args.port)
    try:
        sent = sink.publish_chunk(
            chunk,
            realtime=not args.no_realtime,
            mark_end=args.mark_end,
            startup_delay_sec=args.startup_delay_sec,
        )
    finally:
        sink.close()
    print(f"published={sent} endpoint=tcp://{args.host}:{args.port}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
