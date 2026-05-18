from __future__ import annotations

import sys
from typing import Any

import numpy as np
import torch

from text2humanoid.contracts.chunks import NMRInputChunk
from text2humanoid.infra.paths import get_make_tracking_easy_root


def _add_nmr_path() -> None:
    path = str(get_make_tracking_easy_root())
    if path not in sys.path:
        sys.path.insert(0, path)


class NMRRetargetService:
    def __init__(self, apply_filter: bool = True, tgt_fps: int = 30) -> None:
        self.apply_filter = bool(apply_filter)
        self.tgt_fps = int(tgt_fps)
        self._loaded = False
        self._model = None
        self._smplx_mean = None
        self._smplx_std = None
        self._g1_mean = None
        self._g1_std = None
        self._device = None

    @property
    def output_fps(self) -> int:
        return self.tgt_fps

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
        """Run NMR retarget on a 140D motion tensor."""
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
