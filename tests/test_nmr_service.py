from __future__ import annotations

import sys
import types

import numpy as np
import torch

from text2humanoid.contracts.chunks import NMRInputChunk
from text2humanoid.retarget.nmr_service import NMRRetargetService


def _install_fake_inference(monkeypatch, filter_calls):
    fake_inference = types.ModuleType("inference")
    fake_inference.CHUNK_FRAMES = 64
    fake_inference.STRIDE_FRAMES = 32

    def fake_infer_chunk(smplx_motion, *args):
        return torch.zeros((smplx_motion.shape[0], 217), dtype=torch.float32)

    def fake_postprocess_g1(pred_motion, apply_filter=True):
        filter_calls.append(bool(apply_filter))
        frames = pred_motion.shape[0]
        return (
            torch.zeros((frames, 29), dtype=torch.float32),
            torch.zeros((frames, 4), dtype=torch.float32),
            torch.zeros((frames, 3), dtype=torch.float32),
        )

    fake_inference._infer_chunk = fake_infer_chunk
    fake_inference.postprocess_g1 = fake_postprocess_g1
    monkeypatch.setitem(sys.modules, "inference", fake_inference)


def _service() -> NMRRetargetService:
    service = NMRRetargetService(apply_filter=True, tgt_fps=30)
    service._loaded = True
    service._model = object()
    service._smplx_mean = torch.zeros(140)
    service._smplx_std = torch.ones(140)
    service._g1_mean = torch.zeros(217)
    service._g1_std = torch.ones(217)
    service._device = torch.device("cpu")
    return service


def _chunk(frames: int) -> NMRInputChunk:
    return NMRInputChunk(
        chunk_id=f"chunk_{frames}",
        start_time=0.0,
        fps=30,
        motion_140=np.zeros((frames, 140), dtype=np.float32),
    )


def test_nmr_retarget_disables_filter_for_chunks_that_are_too_short_for_filtfilt(monkeypatch):
    filter_calls = []
    _install_fake_inference(monkeypatch, filter_calls)

    result = _service().retarget_chunk(_chunk(15))

    assert filter_calls == [False]
    assert result["dof"].shape == (15, 29)


def test_nmr_retarget_keeps_filter_for_chunks_longer_than_filtfilt_padlen(monkeypatch):
    filter_calls = []
    _install_fake_inference(monkeypatch, filter_calls)

    result = _service().retarget_chunk(_chunk(16))

    assert filter_calls == [True]
    assert result["dof"].shape == (16, 29)
