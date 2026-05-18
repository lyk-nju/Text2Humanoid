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
        """Run NMR retarget on a 140D motion tensor.

        Mirrors MTE infer_single pipeline: canonical alignment +
        standardize once for whole motion, model forward in chunks with
        overlap blending, then postprocess_g1 for dof/root output.
        """
        self._load()
        _add_nmr_path()
        from inference import (
            _extract_yaw,
            _make_y_rot,
            _rotate_motion_features,
            postprocess_g1,
            CHUNK_FRAMES,
            STRIDE_FRAMES,
        )

        smplx_motion = torch.from_numpy(np.asarray(chunk.motion_140, dtype=np.float32))
        smplx_mean = self._smplx_mean
        smplx_std = self._smplx_std
        g1_mean = self._g1_mean
        g1_std = self._g1_std
        device = self._device
        model = self._model

        # Canonical alignment once for the whole motion
        yaw = _extract_yaw(smplx_motion[0, 2:8])
        R_canon = _make_y_rot(-yaw)
        R_restore = _make_y_rot(yaw)
        smplx_motion = _rotate_motion_features(smplx_motion, R_canon, n_joints=22)

        # Standardize once
        smplx_motion = (smplx_motion - smplx_mean) / smplx_std

        T_orig = smplx_motion.shape[0]
        if T_orig < 4:
            raise RuntimeError(f"Motion too short: {T_orig} frames")

        T_pad = ((T_orig + 3) // 4) * 4
        if T_pad > T_orig:
            pad = smplx_motion[-1:].repeat(T_pad - T_orig, 1)
            smplx_motion = torch.cat([smplx_motion, pad], dim=0)

        def _run_model(seg: torch.Tensor) -> torch.Tensor:
            inp = seg.unsqueeze(0).float().to(device)
            ml = torch.tensor([seg.shape[0]]).to(device)
            with torch.no_grad():
                preds, _ = model(smplx_motion=inp, motion_length=ml, mode='predict')
            return preds[0].cpu() * g1_std + g1_mean

        T = T_pad
        if T <= CHUNK_FRAMES:
            pred_motion = _run_model(smplx_motion)
        else:
            chunks = []
            starts = []
            for start in range(0, T, STRIDE_FRAMES):
                end = min(start + CHUNK_FRAMES, T)
                seg_len = (end - start) // 4 * 4
                if seg_len < 4:
                    break
                seg = smplx_motion[start:start + seg_len]
                chunks.append(_run_model(seg))
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
        pred_motion = _rotate_motion_features(pred_motion, R_restore, n_joints=30)

        pred_dof, pred_rot_quat, pred_trans = postprocess_g1(
            pred_motion, apply_filter=self.apply_filter,
        )

        result: dict[str, Any] = {
            "dof": pred_dof.numpy(),
            "root_trans": pred_trans.numpy(),
            "root_rot_quat": pred_rot_quat.numpy(),
        }
        return result
