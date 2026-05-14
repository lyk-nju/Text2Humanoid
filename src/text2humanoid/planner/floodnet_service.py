from __future__ import annotations

import os
import sys
import uuid
from pathlib import Path

import numpy as np
import torch

from text2humanoid.contracts.chunks import HumanMotionChunk
from text2humanoid.contracts.commands import PromptCommand
from text2humanoid.infra.logging import get_logger
from text2humanoid.planner.prompt_transition import resolve_chunk_frames
from text2humanoid.planner.traj_conditioning import build_floodnet_model_input

_LOG = get_logger("text2humanoid.planner")
_ROOT = Path("/home/yuankai/Text2Motion/FloodNet")


def _add_floodnet_path() -> None:
    root_str = str(_ROOT)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)


class FloodNetPlannerService:
    def __init__(self, config_path: str, chunk_frames: int = 40) -> None:
        self.config_path = config_path
        self.chunk_frames = int(chunk_frames)
        self._loaded = False
        self._vae = None
        self._model = None
        self._device = "cpu"

    def _load(self) -> None:
        if self._loaded:
            return
        _add_floodnet_path()
        from torch_ema import ExponentialMovingAverage
        from utils.initialize import instantiate, load_config

        cfg = load_config(self.config_path)
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        torch.set_float32_matmul_precision("high")

        vae = instantiate(target=cfg.test_vae.target, cfg=None, hfstyle=False, **cfg.test_vae.params)
        vae_ckpt = torch.load(cfg.test_vae_ckpt, map_location="cpu", weights_only=False)
        vae.load_state_dict(vae_ckpt["state_dict"], strict=True)
        if "ema_state" in vae_ckpt:
            vae_ema = ExponentialMovingAverage(vae.parameters(), decay=cfg.test_vae.ema_decay)
            vae_ema.load_state_dict(vae_ckpt["ema_state"])
            vae_ema.copy_to(vae.parameters())
        vae.to(self._device).eval()

        model = instantiate(target=cfg.model.target, cfg=None, hfstyle=False, **cfg.model.params)
        model_ckpt = torch.load(cfg.test_ckpt, map_location="cpu", weights_only=False)
        model.load_state_dict(model_ckpt["state_dict"], strict=True)
        if "ema_state" in model_ckpt:
            n_shadow = len(model_ckpt["ema_state"]["shadow_params"])
            ema_params = [p for p in model.parameters() if p.requires_grad]
            if len(ema_params) != n_shadow:
                ema_params = list(model.parameters())
            ema = ExponentialMovingAverage(ema_params, decay=cfg.model.ema_decay)
            ema.load_state_dict(model_ckpt["ema_state"])
            ema.copy_to(ema_params)
        model.to(self._device).eval()

        self._vae = vae
        self._model = model
        self._loaded = True
        _LOG.info("Loaded FloodNet planner on %s", self._device)

    def warmup(self, text: str) -> None:
        self._load()
        dummy = PromptCommand(text=text or "stand still")
        self.generate_chunk(dummy, start_time=0.0, feature_length=4)

    def reset(self) -> None:
        if self._model is not None and hasattr(self._model, "init_generated"):
            try:
                self._model.init_generated(30, batch_size=1)
            except Exception:
                _LOG.exception("Planner reset failed")

    def generate_chunk(
        self,
        command: PromptCommand,
        start_time: float,
        feature_length: int | None = None,
    ) -> HumanMotionChunk:
        self._load()
        assert self._vae is not None and self._model is not None

        num_frames = int(feature_length or resolve_chunk_frames(command, self.chunk_frames))
        model_input = build_floodnet_model_input(command, num_frames)
        with torch.no_grad():
            out = self._model.generate(model_input)
            latent = out["generated"][0]
            decoded = self._vae.decode(latent[None, ...].to(self._device))[0]
        motion_263 = decoded.detach().cpu().numpy().astype(np.float32)
        return HumanMotionChunk(
            chunk_id=command.command_id or uuid.uuid4().hex,
            start_time=start_time,
            fps=20,
            motion_263=motion_263,
            text=command.text,
            trajectory_payload=None if command.trajectory is None else command.trajectory.to_dict(),
            metadata={"device": self._device},
        )
