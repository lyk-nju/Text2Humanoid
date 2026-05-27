from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path
import sys

from text2humanoid.contracts.pipeline import (
    GenerateRequest,
    GenerateSpec,
    MultimodalInput,
    TextInput,
)
from text2humanoid.generation import FloodDiffusionBackend


_TEXT2HUMANOID_DIR = Path(__file__).resolve().parents[1]
_TEXT2MOTION_ROOT = _TEXT2HUMANOID_DIR.parent

if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
if str(_TEXT2HUMANOID_DIR) not in sys.path:
    sys.path.insert(0, str(_TEXT2HUMANOID_DIR))

from retarget_to_bfmzero import main as retarget_to_bfmzero_main


def visualize_263d_main(argv: Sequence[str] | None = None) -> int:
    from tools.visualize_263D import main as render_main

    return render_main(argv)


def visualize_140d_main(argv: Sequence[str] | None = None) -> int:
    from tools.visualize_140D import main as render_main

    return render_main(argv)


def visualize_retarget_main(argv: Sequence[str] | None = None) -> int:
    from tools.visualize_retarget import main as render_main

    return render_main(argv)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate text motion with FloodDiffusion and retarget it to a BFM-Zero chunk."
    )
    parser.add_argument("--text", required=True, help="Prompt text for FloodDiffusion")
    parser.add_argument("--out-prefix", default="assets/saved/text2bfmzero")
    parser.add_argument("--frames", type=int, default=None, help="Trim generated 263D output to this frame count")
    parser.add_argument("--generation-steps", type=int, default=150)
    parser.add_argument("--fps", type=int, default=20, help="Generated 263D FPS")
    parser.add_argument("--flooddiffusion-root", default=str(_TEXT2MOTION_ROOT / "FloodDiffusion"))
    parser.add_argument("--flooddiffusion-config", default="configs/stream.yaml")
    parser.add_argument("--python", default=sys.executable, help="Python executable for FloodDiffusion")
    parser.add_argument(
        "--override",
        action="append",
        default=[],
        help="FloodDiffusion override, e.g. model.params.text_encoder_device=cpu",
    )
    parser.add_argument("--system-config", default="configs/system/local_dev.yaml")
    parser.add_argument("--retarget-fps", type=float, default=None)
    parser.add_argument("--output-fps", type=float, default=50.0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--publish", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--host", default="*")
    parser.add_argument("--port", type=int, default=5592)
    parser.add_argument("--mark-end", action="store_true")
    parser.add_argument("--no-realtime", action="store_true")
    parser.add_argument("--inspect", action="store_true", help="Save and render 263D/140D/retarget intermediates")
    parser.add_argument("--video-dir", default="assets/video", help="Directory for --inspect videos")
    parser.add_argument("--render-stride", type=int, default=1, help="Render every Nth frame for --inspect videos")
    return parser.parse_args(argv)


def _resolve_output_prefix(value: str) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    return (_TEXT2HUMANOID_DIR / path).resolve()


def _resolve_video_dir(value: str) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    return (_TEXT2HUMANOID_DIR / path).resolve()


def _run_inspection_renders(
    *,
    prefix: Path,
    video_dir: Path,
    motion_path: Path,
    motion_140_path: Path,
    retarget_path: Path,
    fps: int,
    retarget_fps: float | None,
    render_stride: int,
    device: str,
) -> None:
    video_dir.mkdir(parents=True, exist_ok=True)
    retarget_video = video_dir / f"{prefix.name}_retarget.mp4"
    retarget_diag_video = video_dir / f"{prefix.name}_retarget_diagnostics.mp4"
    visualize_263d_main(
        [
            "--input",
            str(motion_path),
            "--output",
            str(video_dir / f"{prefix.name}_motion_263.mp4"),
            "--fps",
            str(fps),
            "--stride",
            str(render_stride),
        ]
    )
    visualize_140d_main(
        [
            "--input",
            str(motion_140_path),
            "--output",
            str(video_dir / f"{prefix.name}_motion_140.mp4"),
            "--fps",
            str(retarget_fps or 30.0),
            "--stride",
            str(render_stride),
        ]
    )
    visualize_retarget_main(
        [
            "--input",
            str(retarget_path),
            "--output",
            str(retarget_video),
            "--fps",
            str(retarget_fps or 30.0),
            "--stride",
            str(render_stride),
            "--device",
            device,
        ]
    )
    visualize_retarget_main(
        [
            "--input",
            str(retarget_path),
            "--output",
            str(retarget_diag_video),
            "--fps",
            str(retarget_fps or 30.0),
            "--stride",
            str(render_stride),
            "--mode",
            "diagnostics",
        ]
    )
    print(f"Inspection videos ready: {video_dir}")


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    prefix = _resolve_output_prefix(args.out_prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    motion_path = prefix.with_name(f"{prefix.name}_motion_263.npz")
    motion_140_path = prefix.with_name(f"{prefix.name}_motion_140.npy")
    retarget_path = prefix.with_name(f"{prefix.name}_mte_filter_true.npz")
    bfmzero_path = prefix.with_name(f"{prefix.name}_bfmzero.npz")

    overrides = args.override or [
        "model.params.text_encoder_device=cpu",
        "model.params.low_cpu_mem_load=true",
    ]
    backend = FloodDiffusionBackend(
        flooddiffusion_root=args.flooddiffusion_root,
        config_path=args.flooddiffusion_config,
        output_dir=prefix.parent,
        python_executable=args.python,
        overrides=overrides,
    )
    request = GenerateRequest(
        input=MultimodalInput(text_input=TextInput(prompt=args.text)),
        spec=GenerateSpec(
            mode="offline",
            fps=args.fps,
            num_frames=args.frames,
            metadata={"generation_steps": args.generation_steps},
        ),
        metadata={"output_path": str(motion_path)},
    )
    generated = backend.generate_chunk(request)
    print(f"Generated motion_263: frames={generated.num_frames} fps={generated.fps} path={motion_path}")

    retarget_args = [
        "--motion",
        str(motion_path),
        "--config",
        args.system_config,
        "--src-fps",
        str(args.fps),
        "--output-fps",
        str(args.output_fps),
        "--output",
        str(bfmzero_path),
        "--device",
        args.device,
    ]
    if args.retarget_fps is not None:
        retarget_args.extend(["--retarget-fps", str(args.retarget_fps)])
    if args.publish:
        retarget_args.append("--publish")
    if args.dry_run:
        retarget_args.append("--dry-run")
    if args.mark_end:
        retarget_args.append("--mark-end")
    if args.no_realtime:
        retarget_args.append("--no-realtime")
    if args.inspect:
        retarget_args.extend(["--save-140", str(motion_140_path)])
        retarget_args.extend(["--save-retarget", str(retarget_path)])
    retarget_args.extend(["--host", args.host, "--port", str(args.port)])

    rc = retarget_to_bfmzero_main(retarget_args)
    if rc != 0:
        return rc
    if args.inspect:
        _run_inspection_renders(
            prefix=prefix,
            video_dir=_resolve_video_dir(args.video_dir),
            motion_path=motion_path,
            motion_140_path=motion_140_path,
            retarget_path=retarget_path,
            fps=args.fps,
            retarget_fps=args.retarget_fps,
            render_stride=args.render_stride,
            device=args.device,
        )
    print(f"BFM-Zero chunk ready: {bfmzero_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
