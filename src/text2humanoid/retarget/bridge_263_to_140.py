from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

from text2humanoid.contracts.chunks import HumanMotionChunk, NMRInputChunk
from text2humanoid.retarget.mte_imports import ensure_make_tracking_easy_paths

_FLOODNET_ROOT = Path("/home/yuankai/Text2Motion/FloodNet")


def _ensure_paths() -> None:
    root_path = str(_FLOODNET_ROOT)
    if root_path not in sys.path:
        sys.path.insert(0, root_path)
    ensure_make_tracking_easy_paths()


def _resample_linear(x: torch.Tensor, src_fps: float, tgt_fps: float) -> torch.Tensor:
    frames_new = max(1, round(x.shape[0] * tgt_fps / src_fps))
    out = torch.nn.functional.interpolate(
        x.T.unsqueeze(0),
        size=frames_new,
        mode="linear",
        align_corners=True,
    ).squeeze(0).T
    return out


def assemble_nmr_motion(
    joint_pos: torch.Tensor,
    root_quat_wxyz: torch.Tensor,
    src_fps: float = 20.0,
    tgt_fps: float = 30.0,
) -> torch.Tensor:
    _ensure_paths()
    from utils.rotation_conversions import matrix_to_rotation_6d

    t, n_joint, _ = joint_pos.shape
    cos_a = root_quat_wxyz[:, 0]
    sin_a = root_quat_wxyz[:, 2]
    zeros = torch.zeros_like(cos_a)
    ones = torch.ones_like(cos_a)
    rot = torch.stack(
        [
            cos_a,
            zeros,
            sin_a,
            zeros,
            ones,
            zeros,
            -sin_a,
            zeros,
            cos_a,
        ],
        dim=-1,
    ).view(t, 3, 3)
    root_6d = matrix_to_rotation_6d(rot)

    joint_vel = torch.zeros_like(joint_pos)
    joint_vel[1:] = joint_pos[1:] - joint_pos[:-1]

    y_min = joint_pos[:, :, 1].min()
    joint_pos_norm = joint_pos.clone()
    joint_pos_norm[:, :, 1] -= y_min

    root_vel = torch.zeros((t, 3), dtype=joint_pos.dtype)
    root_vel[1:] = joint_pos_norm[1:, 0] - joint_pos_norm[:-1, 0]

    joint_pos_centered = joint_pos_norm.clone()
    joint_pos_centered[:, :, 0] -= joint_pos_norm[:, 0:1, 0]
    joint_pos_centered[:, :, 2] -= joint_pos_norm[:, 0:1, 2]

    motion_140 = torch.zeros((t, 2 + 6 + n_joint * 3 + n_joint * 3), dtype=joint_pos.dtype)
    motion_140[:, 0] = root_vel[:, 0]
    motion_140[:, 1] = root_vel[:, 2]
    motion_140[:, 2:8] = root_6d
    motion_140[:, 8 : 8 + n_joint * 3] = joint_pos_centered.reshape(t, -1)
    motion_140[:, 8 + n_joint * 3 :] = joint_vel.reshape(t, -1)

    if abs(src_fps - tgt_fps) > 0.01:
        motion_140 = _resample_linear(motion_140, src_fps, tgt_fps)
    return motion_140


def floodnet_263_to_nmr_140(
    feature_263: np.ndarray,
    src_fps: float = 20.0,
    tgt_fps: float = 30.0,
) -> torch.Tensor:
    _ensure_paths()
    from utils.motion_process import recover_joint_positions_263, recover_root_rot_pos

    joint_pos = torch.from_numpy(recover_joint_positions_263(feature_263, 22)).float()
    feature_t = torch.from_numpy(feature_263).float().unsqueeze(0)
    root_quat, _ = recover_root_rot_pos(feature_t)
    root_quat = root_quat.squeeze(0)
    return assemble_nmr_motion(joint_pos, root_quat, src_fps=src_fps, tgt_fps=tgt_fps)


def human_chunk_to_nmr_input(
    chunk: HumanMotionChunk,
    tgt_fps: int = 30,
) -> NMRInputChunk:
    motion_140 = floodnet_263_to_nmr_140(chunk.motion_263, src_fps=float(chunk.fps), tgt_fps=float(tgt_fps))
    return NMRInputChunk(
        chunk_id=chunk.chunk_id,
        start_time=chunk.start_time,
        fps=tgt_fps,
        motion_140=motion_140.cpu().numpy(),
        metadata=dict(chunk.metadata),
    )
