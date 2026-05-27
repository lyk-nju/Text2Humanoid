from __future__ import annotations

from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
import os
from pathlib import Path
import sys
import threading
import time
from typing import Any

import numpy as np

from text2humanoid.contracts.pipeline import GenerateRequest, GeneratedMotion


def _text2motion_root() -> Path:
    return Path(os.environ.get("TEXT2MOTION_ROOT", Path(__file__).resolve().parents[4])).resolve()


def _default_flooddiffusion_root() -> Path:
    return _text2motion_root() / "FloodDiffusion"


def _as_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value, dtype=np.float32)


@dataclass(slots=True)
class FloodDiffusionStreamingBackend:
    """In-process FloodDiffusion streaming backend.

    The backend mirrors FloodDiffusion/web_demo's state model: start/reset once,
    keep model and VAE caches alive, and let update_text affect future stream
    steps without resetting generated history.
    """

    flooddiffusion_root: Path | str = field(default_factory=_default_flooddiffusion_root)
    config_path: str = "configs/stream.yaml"
    model: Any | None = None
    vae: Any | None = None
    model_loader: Callable[[Path, str], tuple[Any, Any]] | None = None
    text_update_callback: Callable[[str, float], None] | None = None
    _lock: threading.RLock = field(init=False, repr=False)
    _current_text: str = field(init=False, default="")
    _pending_text: str | None = field(init=False, default=None)
    _started: bool = field(init=False, default=False)
    _first_chunk: bool = field(init=False, default=True)
    _chunk_index: int = field(init=False, default=0)
    _frame_cursor: int = field(init=False, default=0)
    _preencode_thread: threading.Thread | None = field(init=False, default=None, repr=False)

    def __post_init__(self) -> None:
        self.flooddiffusion_root = Path(self.flooddiffusion_root).expanduser().resolve()
        self._lock = threading.RLock()

    def _load_models(self) -> tuple[Any, Any]:
        if self.vae is not None and self.model is not None:
            return self.vae, self.model
        if self.model_loader is not None:
            self.vae, self.model = self.model_loader(self.flooddiffusion_root, self.config_path)
            return self.vae, self.model

        root = str(self.flooddiffusion_root)
        if root not in sys.path:
            sys.path.insert(0, root)

        import torch
        from torch_ema import ExponentialMovingAverage
        from utils.initialize import instantiate, load_config

        original_dir = os.getcwd()
        os.chdir(self.flooddiffusion_root)
        try:
            torch.set_float32_matmul_precision("high")
            device = "cuda" if torch.cuda.is_available() else "cpu"
            cfg = load_config(config_path=self.config_path)

            vae = instantiate(target=cfg.test_vae.target, cfg=None, hfstyle=False, **cfg.test_vae.params)
            vae_ckpt = torch.load(cfg.test_vae_ckpt, map_location="cpu", weights_only=False)
            vae.load_state_dict(vae_ckpt["state_dict"], strict=True)
            if "ema_state" in vae_ckpt:
                vae_ema = ExponentialMovingAverage(vae.parameters(), decay=cfg.test_vae.ema_decay)
                vae_ema.load_state_dict(vae_ckpt["ema_state"])
                vae_ema.copy_to(vae.parameters())
            vae.to(device)
            vae.eval()

            model = instantiate(target=cfg.model.target, cfg=None, hfstyle=False, **cfg.model.params)
            checkpoint = torch.load(cfg.test_ckpt, map_location="cpu", weights_only=False)
            model.load_state_dict(checkpoint["state_dict"], strict=True)
            if "ema_state" in checkpoint:
                ema = ExponentialMovingAverage(model.parameters(), decay=cfg.model.ema_decay)
                ema.load_state_dict(checkpoint["ema_state"])
                ema.copy_to(model.parameters())
            model.to(device)
            model.eval()
            self.vae = vae
            self.model = model
            return vae, model
        finally:
            os.chdir(original_dir)

    def start_stream(self, request: GenerateRequest) -> None:
        if request.spec.mode != "stream":
            raise ValueError("FloodDiffusionStreamingBackend requires GenerateSpec(mode='stream')")

        vae, model = self._load_models()
        history_length = int(request.spec.metadata.get("history_length", 30))
        denoise_steps = int(request.spec.metadata.get("denoise_steps", 10))
        prompt = request.input.require_text_prompt()
        if hasattr(vae, "clear_cache"):
            vae.clear_cache()
        model.init_generated(history_length, batch_size=1, num_denoise_steps=denoise_steps)
        with self._lock:
            self._current_text = prompt
            self._pending_text = None
            self._started = True
            self._first_chunk = True
            self._chunk_index = 0
            self._frame_cursor = 0

    def update_text(self, text: str) -> None:
        with self._lock:
            if not self._started:
                raise RuntimeError("stream has not been started")
            if text == self._current_text or text == self._pending_text:
                return
            _, model = self._load_models()
            if not hasattr(model, "encode_text_with_cache"):
                self._current_text = text
                return
            self._pending_text = text
            started_at = time.perf_counter()
            thread = threading.Thread(
                target=self._preencode_and_switch_text,
                args=(text, started_at),
                name="flooddiffusion-text-preencode",
                daemon=True,
            )
            self._preencode_thread = thread
            thread.start()

    def _preencode_and_switch_text(self, text: str, started_at: float) -> None:
        _, model = self._load_models()
        device = "cpu"
        try:
            if hasattr(model, "parameters"):
                first_param = next(model.parameters())
                device = getattr(first_param, "device", device)
        except Exception:
            device = "cpu"
        model.encode_text_with_cache([text], device)
        elapsed = time.perf_counter() - started_at
        with self._lock:
            if self._pending_text != text:
                return
            self._current_text = text
            self._pending_text = None
        if self.text_update_callback is not None:
            self.text_update_callback(text, elapsed)

    def current_text(self) -> str:
        with self._lock:
            return self._current_text

    def stream_chunks(
        self,
        request: GenerateRequest,
        *,
        should_stop: Callable[[], bool] | None = None,
    ) -> Iterator[GeneratedMotion]:
        self.start_stream(request)
        vae, model = self._load_models()
        should_stop = should_stop or (lambda: False)

        while not should_stop():
            with self._lock:
                text = self._current_text
                first_chunk = self._first_chunk
                chunk_index = self._chunk_index
                frame_start = self._frame_cursor

            output = model.stream_generate_step({"text": [text]}, first_chunk=first_chunk)
            generated = output["generated"][0]
            decoded = vae.stream_decode(generated[None, :], first_chunk=first_chunk)[0]
            motion = _as_numpy(decoded)
            if motion.ndim != 2 or motion.shape[1] != 263:
                raise ValueError(f"Expected decoded stream chunk shape (T, 263), got {motion.shape}")

            with self._lock:
                self._first_chunk = False
                self._chunk_index += 1
                self._frame_cursor += int(motion.shape[0])

            yield GeneratedMotion(
                motion_id=f"{request.request_id}_stream_{chunk_index:06d}",
                representation="humanml3d_263",
                motion=motion,
                fps=request.spec.fps,
                start_time=frame_start / float(request.spec.fps),
                source_input_id=request.input.input_id,
                metadata={
                    "backend_type": "flooddiffusion_streaming",
                    "chunk_index": chunk_index,
                    "frame_start": frame_start,
                    "text": text,
                },
            )
