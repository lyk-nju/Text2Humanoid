from __future__ import annotations

import argparse
from collections.abc import Sequence

from text2humanoid.runtime.bfmzero_zmq_sink import BFMZeroZmqSink
from text2humanoid.runtime.g1_reference_to_bfmzero import (
    G1ReferenceToBFMZeroInputBridge,
    load_g1_reference_npz,
)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Replay a Text2Humanoid G1 reference NPZ to BFM-Zero tracking_online ZMQ."
    )
    parser.add_argument("--input", required=True, help="Path to reference_*.npz")
    parser.add_argument("--input-fps", type=int, default=30, help="Reference artifact FPS")
    parser.add_argument("--output-fps", type=int, default=50, help="BFM-Zero stream FPS")
    parser.add_argument("--host", default="*", help="ZMQ bind host")
    parser.add_argument("--port", type=int, default=5592, help="ZMQ bind port")
    parser.add_argument("--frame-start", type=int, default=0)
    parser.add_argument("--chunk-id", default=None)
    parser.add_argument("--mark-end", action="store_true", help="Mark the final frame with END")
    parser.add_argument("--no-realtime", action="store_true", help="Publish without frame sleeps")
    parser.add_argument("--startup-delay-sec", type=float, default=0.5)
    parser.add_argument("--dry-run", action="store_true", help="Build payload but do not open ZMQ")
    args = parser.parse_args(argv)

    motion = load_g1_reference_npz(
        args.input,
        fps=args.input_fps,
        motion_id=args.chunk_id,
    )
    bridge = G1ReferenceToBFMZeroInputBridge(
        frame_start=args.frame_start,
        target_fps=args.output_fps,
    )
    tracker_input = bridge.convert(motion)
    chunk = tracker_input.payload

    print(
        f"tracker={tracker_input.tracker} representation={tracker_input.representation} "
        f"frames={chunk.num_frames} fps={chunk.fps} frame_start={chunk.frame_start}"
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
