from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
import os
from pathlib import Path
import subprocess
import sys

import numpy as np

from text2humanoid.contracts.pipeline import GenerateRequest, GeneratedMotion


def _text2motion_root() -> Path:
    return Path(os.environ.get("TEXT2MOTION_ROOT", Path(__file__).resolve().parents[4])).resolve()


def _default_flooddiffusion_root() -> Path:
    return _text2motion_root() / "FloodDiffusion"


def _load_motion_263(path: Path) -> np.ndarray:
    if path.suffix == ".npy":
        data = np.load(path)
    elif path.suffix == ".npz":
        with np.load(path, allow_pickle=False) as npz:
            data = None
            for key in ("motion_263", "motion", "arr_0", "features"):
                if key in npz:
                    data = npz[key]
                    break
            if data is None:
                raise ValueError(f"NPZ does not contain a 263D motion key: {sorted(npz.files)}")
    else:
        raise ValueError(f"Unsupported FloodDiffusion output format: {path.suffix}")

    motion = np.asarray(data, dtype=np.float32)
    if motion.ndim == 3:
        motion = motion[0]
    if motion.ndim != 2 or motion.shape[1] != 263:
        raise ValueError(f"Expected FloodDiffusion motion shape (T, 263), got {motion.shape}")
    return motion


@dataclass(slots=True)
class FloodDiffusionBackend:
    """GenerationBackend that calls FloodDiffusion/generate_ldf.py as a subprocess."""

    flooddiffusion_root: Path | str = field(default_factory=_default_flooddiffusion_root)
    config_path: str = "configs/stream.yaml"
    output_dir: Path | str = "assets/saved"
    python_executable: str = sys.executable
    overrides: Sequence[str] = field(default_factory=lambda: (
        "model.params.text_encoder_device=cpu",
        "model.params.low_cpu_mem_load=true",
    ))

    def __post_init__(self) -> None:
        self.flooddiffusion_root = Path(self.flooddiffusion_root).expanduser().resolve()
        self.output_dir = Path(self.output_dir).expanduser().resolve()

    def generate_chunk(self, request: GenerateRequest) -> GeneratedMotion:
        if request.spec.mode != "offline":
            raise ValueError("FloodDiffusionBackend.generate_chunk only supports offline mode")

        prompt = request.input.require_text_prompt()
        output_path = Path(
            request.metadata.get("output_path")
            or self.output_dir / f"{request.request_id}_motion_263.npz"
        ).expanduser()
        if not output_path.is_absolute():
            output_path = (_text2motion_root() / output_path).resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)

        generation_steps = int(
            request.spec.metadata.get(
                "generation_steps",
                request.spec.metadata.get("steps", 150),
            )
        )
        cmd = [
            self.python_executable,
            "generate_ldf.py",
            "--config",
            self.config_path,
            "--text",
            prompt,
            "--steps",
            str(generation_steps),
            "--save-motion",
            str(output_path),
            "--no-render",
        ]
        if self.overrides:
            cmd.extend(["--override", *self.overrides])

        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        subprocess.run(cmd, cwd=self.flooddiffusion_root, env=env, check=True)

        raw_motion = _load_motion_263(output_path)
        motion = raw_motion
        if request.spec.num_frames is not None and motion.shape[0] > request.spec.num_frames:
            motion = motion[: request.spec.num_frames]

        return GeneratedMotion(
            motion_id=f"{request.request_id}_motion",
            representation="humanml3d_263",
            motion=motion,
            fps=request.spec.fps,
            source_input_id=request.input.input_id,
            metadata={
                "backend_type": "flooddiffusion_subprocess",
                "prompt": prompt,
                "artifact_path": str(output_path),
                "raw_num_frames": int(raw_motion.shape[0]),
                "generation_steps": generation_steps,
                "config_path": self.config_path,
            },
        )

    def stream_chunks(self, request: GenerateRequest) -> Iterator[GeneratedMotion]:
        yield self.generate_chunk(request)
