from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

from text2humanoid.visualization.motion_debug import (
    HUMANML3D_CHAINS,
    load_motion_array,
    render_skeleton_video,
    write_json,
)


def _recover_joint_positions_263(motion_263):
    from text2humanoid.retarget.bridge_263_to_140 import _ensure_paths

    _ensure_paths()
    from utils.motion_process import recover_joint_positions_263

    return recover_joint_positions_263(motion_263, 22)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Render a 263D HumanML3D/FloodDiffusion motion as video.")
    parser.add_argument("--input", required=True, help="Path to (T,263) .npy or .npz motion")
    parser.add_argument("--output", required=True, help="Output .mp4 or .gif path")
    parser.add_argument("--fps", type=float, default=20.0, help="Input motion FPS")
    parser.add_argument("--stride", type=int, default=1, help="Render every Nth frame")
    parser.add_argument("--summary", default=None, help="Optional JSON summary path")
    args = parser.parse_args(argv)

    motion = load_motion_array(args.input, expected_dim=263)
    joints = _recover_joint_positions_263(motion)

    render_skeleton_video(
        joints,
        output_path=args.output,
        fps=args.fps,
        chains=HUMANML3D_CHAINS,
        stride=args.stride,
        title=f"263D {Path(args.input).name}",
    )

    if args.summary:
        write_json(
            args.summary,
            {
                "input": str(args.input),
                "output": str(args.output),
                "frames": int(motion.shape[0]),
                "fps": float(args.fps),
                "representation": "263D HumanML3D/FloodDiffusion",
            },
        )
    print(f"rendered={args.output} frames={motion.shape[0]} fps={args.fps}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
