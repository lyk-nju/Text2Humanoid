from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

from text2humanoid.contracts.chunks import NMRInputChunk

_NMR_ROOT = Path("/home/yuankai/Text2Motion/MakeTrackingEasy")


def _add_nmr_path() -> None:
    path = str(_NMR_ROOT)
    if path not in sys.path:
        sys.path.insert(0, path)


class NMRRetargetService:
    def __init__(self, apply_filter: bool = True) -> None:
        self.apply_filter = bool(apply_filter)
        self._loaded = False
        self._model = None
        self._smplx_mean = None
        self._smplx_std = None
        self._g1_mean = None
        self._g1_std = None
        self._device = None
        self.output_fps = 30

    def _load(self) -> None:
        if self._loaded:
            return
        _add_nmr_path()
        from inference import load_all

        model, _, _, smplx_mean, smplx_std, g1_mean, g1_std, device = load_all()
        self._model = model
        self._smplx_mean = smplx_mean
        self._smplx_std = smplx_std
        self._g1_mean = g1_mean
        self._g1_std = g1_std
        self._device = device
        self._loaded = True

    def retarget_chunk(self, chunk: NMRInputChunk) -> dict[str, Any]:
        self._load()
        _add_nmr_path()
        from inference import infer_from_tensor

        tensor = torch.from_numpy(np.asarray(chunk.motion_140, dtype=np.float32))
        result = infer_from_tensor(
            tensor,
            self._model,
            self._smplx_mean,
            self._smplx_std,
            self._g1_mean,
            self._g1_std,
            self._device,
            apply_filter=self.apply_filter,
        )
        if result is None:
            raise RuntimeError("NMR retarget returned no result")
        return result
