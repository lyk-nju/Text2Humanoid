from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Sequence

from text2humanoid.visualization.motion_debug import (
    G1_SIMPLE_CHAINS,
    load_retarget_npz,
    render_retarget_diagnostics_video,
    render_skeleton_video,
    summarize_retarget_motion,
    write_json,
)


_TEXT2HUMANOID_DIR = Path(__file__).resolve().parents[1]
_TEXT2MOTION_ROOT = _TEXT2HUMANOID_DIR.parent


def _default_xml_path() -> Path:
    root = Path(os.environ.get("TEXT2MOTION_ROOT", _TEXT2MOTION_ROOT)).expanduser().resolve()
    return root / "MakeTrackingEasy" / "assets" / "g1_mocap_29dof.xml"


def _retarget_to_g1_body_positions(result, *, xml_path: str, device: str, fps: float):
    from text2humanoid.retarget.mte_imports import ensure_make_tracking_easy_paths

    ensure_make_tracking_easy_paths()
    from convert_bmimic import convert_to_bmimic

    bmimic = convert_to_bmimic(
        result,
        xml_path=xml_path,
        device=device,
        src_fps=fps,
        tgt_fps=fps,
    )
    return bmimic["body_pos_w"]


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Render MakeTrackingEasy retarget output as G1 video or diagnostics.")
    parser.add_argument("--input", required=True, help="Path to MTE result .npz with dof/root_trans/root_rot_quat")
    parser.add_argument("--output", required=True, help="Output .mp4 or .gif path")
    parser.add_argument("--fps", type=float, default=30.0, help="Input retarget FPS")
    parser.add_argument("--stride", type=int, default=1, help="Render every Nth frame")
    parser.add_argument("--mode", choices=("skeleton", "diagnostics"), default="skeleton")
    parser.add_argument("--up-axis", choices=("y", "z"), default="z", help="Vertical axis for skeleton rendering")
    parser.add_argument("--xml-path", default=str(_default_xml_path()), help="G1 MuJoCo XML path for skeleton mode")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--summary", default=None, help="Optional JSON summary path")
    args = parser.parse_args(argv)

    result = load_retarget_npz(args.input)
    if args.mode == "diagnostics":
        render_retarget_diagnostics_video(
            result,
            output_path=args.output,
            fps=args.fps,
            stride=args.stride,
            title=f"retarget diagnostics {Path(args.input).name}",
        )
    else:
        body_pos = _retarget_to_g1_body_positions(
            result,
            xml_path=args.xml_path,
            device=args.device,
            fps=args.fps,
        )
        render_skeleton_video(
            body_pos,
            output_path=args.output,
            fps=args.fps,
            chains=G1_SIMPLE_CHAINS,
            stride=args.stride,
            up_axis=args.up_axis,
            title=f"G1 retarget {Path(args.input).name}",
        )

    if args.summary:
        summary = summarize_retarget_motion(result)
        summary.update(
            {
                "input": str(args.input),
                "output": str(args.output),
                "fps": float(args.fps),
                "mode": args.mode,
                "up_axis": args.up_axis,
            }
        )
        write_json(args.summary, summary)

    print(f"rendered={args.output} frames={result['dof'].shape[0]} fps={args.fps} mode={args.mode}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
