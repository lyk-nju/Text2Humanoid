from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

import numpy as np

from text2humanoid.visualization.motion_debug import (
    HUMANML3D_CHAINS,
    load_motion_array,
    render_skeleton_video,
    world_joints_from_motion_140,
    write_json,
)


def _convert_263_to_140(motion_263: np.ndarray, *, src_fps: float, tgt_fps: float) -> np.ndarray:
    from text2humanoid.retarget.bridge_263_to_140 import floodnet_263_to_nmr_140

    return floodnet_263_to_nmr_140(motion_263, src_fps=src_fps, tgt_fps=tgt_fps).cpu().numpy()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Render a 140D NMR input motion as video.")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--input", help="Path to (T,140) .npy or .npz motion")
    source.add_argument("--motion-263", help="Path to (T,263) .npy or .npz motion; converted before rendering")
    parser.add_argument("--output", required=True, help="Output .mp4 or .gif path")
    parser.add_argument("--src-fps", type=float, default=20.0, help="263D source FPS when --motion-263 is used")
    parser.add_argument("--fps", type=float, default=30.0, help="140D FPS")
    parser.add_argument("--stride", type=int, default=1, help="Render every Nth frame")
    parser.add_argument("--local", action="store_true", help="Do not integrate root x/z velocity")
    parser.add_argument("--save-140", default=None, help="Optional path to save converted 140D .npy")
    parser.add_argument("--summary", default=None, help="Optional JSON summary path")
    args = parser.parse_args(argv)

    if args.motion_263:
        motion_263 = load_motion_array(args.motion_263, expected_dim=263)
        motion_140 = _convert_263_to_140(motion_263, src_fps=args.src_fps, tgt_fps=args.fps)
        source_name = Path(args.motion_263).name
    else:
        motion_140 = load_motion_array(args.input, expected_dim=140)
        source_name = Path(args.input).name

    if args.save_140:
        Path(args.save_140).parent.mkdir(parents=True, exist_ok=True)
        np.save(args.save_140, motion_140.astype(np.float32))

    joints = world_joints_from_motion_140(motion_140, integrate_root=not args.local)
    render_skeleton_video(
        joints,
        output_path=args.output,
        fps=args.fps,
        chains=HUMANML3D_CHAINS,
        stride=args.stride,
        title=f"140D {source_name}",
    )

    if args.summary:
        write_json(
            args.summary,
            {
                "input": str(args.input or args.motion_263),
                "output": str(args.output),
                "frames": int(motion_140.shape[0]),
                "fps": float(args.fps),
                "representation": "140D NMR input",
                "integrated_root_xz": not args.local,
            },
        )
    print(f"rendered={args.output} frames={motion_140.shape[0]} fps={args.fps}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
