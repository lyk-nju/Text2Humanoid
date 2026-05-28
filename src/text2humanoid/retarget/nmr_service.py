from __future__ import annotations

import sys
from typing import Any

import numpy as np
import torch

from text2humanoid.contracts.chunks import NMRInputChunk
from text2humanoid.infra.paths import get_make_tracking_easy_root

# scipy's filtfilt requires len(input) > padlen.  For the 4th-order Butterworth
# used in MakeTrackingEasy/inference.py:189 (`butter(4, 5/(30/2), btype='low')`),
# padlen defaults to 3 * max(len(a), len(b)) = 15.  We use 16 to keep one frame
# of headroom; chunks shorter than this run unfiltered.  MTE's postprocess_g1
# applies its own internal `T >= 13` gate (MakeTrackingEasy/inference.py:188),
# so passing a 13-15 frame chunk with apply_filter=True would still get
# filtered there — but our gate is the stricter / safer one.
_MIN_FILTER_FRAMES = 16


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

    def retarget_chunk(self, chunk: Any) -> dict[str, Any]:
        """Run NMR retarget on a 140D motion tensor.

        Accepts either a legacy `NMRInputChunk` (which carries `motion_140`)
        or a `RetargetInput` from contracts.pipeline (which carries `motion`).
        Streaming code is migrating to the latter; orchestrator code still
        uses the former.  See contracts/chunks.py for the deprecation
        roadmap.

        Prefer MakeTrackingEasy's tensor API when present.  Older checked-out
        copies may not have it, so the fallback mirrors the same per-chunk
        _infer_chunk path used by infer_single().
        """
        self._load()
        _add_nmr_path()
        motion_140 = getattr(chunk, "motion_140", None)
        if motion_140 is None:
            motion_140 = chunk.motion  # RetargetInput convention
        smplx_motion = torch.from_numpy(np.asarray(motion_140, dtype=np.float32))
        effective_apply_filter = self.apply_filter and smplx_motion.shape[0] >= _MIN_FILTER_FRAMES

        try:
            from inference import infer_from_tensor
        except ImportError:
            infer_from_tensor = None

        if infer_from_tensor is not None:
            result = infer_from_tensor(
                smplx_motion,
                self._model,
                self._smplx_mean,
                self._smplx_std,
                self._g1_mean,
                self._g1_std,
                self._device,
                apply_filter=effective_apply_filter,
            )
            if result is None:
                raise RuntimeError("NMR retarget returned no result")
            return result

        from inference import (
            _infer_chunk,
            postprocess_g1,
            CHUNK_FRAMES,
            STRIDE_FRAMES,
        )

        smplx_mean = self._smplx_mean
        smplx_std = self._smplx_std
        g1_mean = self._g1_mean
        g1_std = self._g1_std
        device = self._device

        T_orig = smplx_motion.shape[0]
        if T_orig < 4:
            raise RuntimeError(f"Motion too short: {T_orig} frames")

        T_pad = ((T_orig + 3) // 4) * 4
        if T_pad > T_orig:
            pad = smplx_motion[-1:].repeat(T_pad - T_orig, 1)
            smplx_motion = torch.cat([smplx_motion, pad], dim=0)

        T = T_pad
        if T <= CHUNK_FRAMES:
            pred_motion = _infer_chunk(
                smplx_motion, self._model, smplx_mean, smplx_std, g1_mean, g1_std, device,
            )
        else:
            chunks = []
            starts = []
            for start in range(0, T, STRIDE_FRAMES):
                end = min(start + CHUNK_FRAMES, T)
                seg_len = (end - start) // 4 * 4
                if seg_len < 4:
                    break
                seg = smplx_motion[start:start + seg_len]
                chunks.append(
                    _infer_chunk(
                        seg, self._model, smplx_mean, smplx_std, g1_mean, g1_std, device,
                    )
                )
                starts.append(start)

            pred_motion = chunks[0]
            for i in range(1, len(chunks)):
                overlap = starts[i - 1] + len(chunks[i - 1]) - starts[i]
                if overlap > 0:
                    w = torch.linspace(0, 1, overlap).unsqueeze(1)
                    prev_tail = pred_motion[-overlap:]
                    curr_head = chunks[i][:overlap]
                    blended = prev_tail * (1 - w) + curr_head * w
                    pred_motion = torch.cat(
                        [pred_motion[:-overlap], blended, chunks[i][overlap:]], dim=0,
                    )
                else:
                    pred_motion = torch.cat([pred_motion, chunks[i]], dim=0)

        pred_motion = pred_motion[:T_orig]
        pred_dof, pred_rot_quat, pred_trans = postprocess_g1(
            pred_motion, apply_filter=effective_apply_filter,
        )

        result: dict[str, Any] = {
            "dof": pred_dof.numpy(),
            "root_trans": pred_trans.numpy(),
            "root_rot_quat": pred_rot_quat.numpy(),
        }
        return result
