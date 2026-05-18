from __future__ import annotations

import math
import sys

import numpy as np
import torch
import yaml

from text2humanoid.infra.paths import get_make_tracking_easy_root, get_motion_tracking_root
from text2humanoid.retarget.mte_imports import ensure_make_tracking_easy_paths

JOINT_MAPPING = [
    0, 6, 12,
    1, 7, 13,
    2, 8, 14,
    3, 9, 15, 22,
    4, 10, 16, 23,
    5, 11, 17, 24,
    18, 25,
    19, 26,
    20, 27,
    21, 28,
]

# NMR inference output joint names in MakeTrackingEasy model output order.
# This is the left/right-interleaved "policy" order — NOT the XML kinematic tree order.
# JOINT_MAPPING remaps from this order to dataset_joint_names order.
_NMR_DOF_NAMES: list[str] = [
    "left_hip_pitch_joint", "right_hip_pitch_joint", "waist_yaw_joint",
    "left_hip_roll_joint", "right_hip_roll_joint", "waist_roll_joint",
    "left_hip_yaw_joint", "right_hip_yaw_joint", "waist_pitch_joint",
    "left_knee_joint", "right_knee_joint",
    "left_shoulder_pitch_joint", "right_shoulder_pitch_joint",
    "left_ankle_pitch_joint", "right_ankle_pitch_joint",
    "left_shoulder_roll_joint", "right_shoulder_roll_joint",
    "left_ankle_roll_joint", "right_ankle_roll_joint",
    "left_shoulder_yaw_joint", "right_shoulder_yaw_joint",
    "left_elbow_joint", "right_elbow_joint",
    "left_wrist_roll_joint", "right_wrist_roll_joint",
    "left_wrist_pitch_joint", "right_wrist_pitch_joint",
    "left_wrist_yaw_joint", "right_wrist_yaw_joint",
]


def validate_joint_mapping(dataset_joint_names: list[str]) -> None:
    """Assert JOINT_MAPPING correctly remaps NMR names to dataset names.

    Raises AssertionError if the mapping would produce a silent mismatch.
    """
    if len(_NMR_DOF_NAMES) != len(JOINT_MAPPING):
        raise AssertionError(
            f"_NMR_DOF_NAMES length {len(_NMR_DOF_NAMES)} != JOINT_MAPPING length {len(JOINT_MAPPING)}"
        )
    if len(dataset_joint_names) != len(JOINT_MAPPING):
        raise AssertionError(
            f"dataset_joint_names length {len(dataset_joint_names)} != JOINT_MAPPING length {len(JOINT_MAPPING)}"
        )
    remapped = [None] * len(_NMR_DOF_NAMES)
    for i, name in enumerate(_NMR_DOF_NAMES):
        remapped[JOINT_MAPPING[i]] = name
    if remapped != dataset_joint_names:
        raise AssertionError(
            "JOINT_MAPPING does not correctly remap NMR DOF names to dataset joint names. "
            f"Got: {remapped}"
        )


def _add_paths() -> None:
    ensure_make_tracking_easy_paths()
    mte_src = str(get_make_tracking_easy_root() / "src")
    if mte_src in sys.path:
        sys.path.remove(mte_src)
    sys.path.insert(0, mte_src)
    sys.modules.pop("utils", None)
    path = str(get_motion_tracking_root())
    if path not in sys.path:
        sys.path.insert(0, path)


def load_dataset_joint_names(
    tracking_config_path: str | None = None,
) -> list[str]:
    if tracking_config_path is None:
        tracking_config_path = str(get_motion_tracking_root() / "sim2real/config/tracking.yaml")
    with open(tracking_config_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return list(data["dataset_joint_names"])


def load_kinematics_model(
    xml_path: str | None = None,
    device: str = "cpu",
):
    if xml_path is None:
        xml_path = str(get_make_tracking_easy_root() / "assets/g1_mocap_29dof.xml")
    _add_paths()
    from utils.kinematics_model import KinematicsModel

    return KinematicsModel(xml_path, device=device)


def remap_nmr_dof_to_dataset(dof_pos_nmr: np.ndarray) -> np.ndarray:
    dof_pos_nmr = np.asarray(dof_pos_nmr, dtype=np.float32)
    mapped = np.zeros_like(dof_pos_nmr)
    mapped[:, JOINT_MAPPING] = dof_pos_nmr
    return mapped


def _quat_inverse_xyzw(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float32)
    out = q.copy()
    out[..., :3] *= -1.0
    norm = np.sum(q * q, axis=-1, keepdims=True).clip(min=1e-8)
    return out / norm


def _quat_mul_xyzw(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    ax, ay, az, aw = np.moveaxis(a, -1, 0)
    bx, by, bz, bw = np.moveaxis(b, -1, 0)
    out = np.stack(
        [
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
            aw * bw - ax * bx - ay * by - az * bz,
        ],
        axis=-1,
    )
    return out.astype(np.float32)


def _quat_rotate_xyzw(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    q_xyz = q[..., :3]
    qw = q[..., 3:4]
    uv = np.cross(q_xyz, v)
    uuv = np.cross(q_xyz, uv)
    return v + 2.0 * (qw * uv + uuv)


def build_world_features(
    root_pos: np.ndarray,
    root_rot_xyzw: np.ndarray,
    dof_pos_dataset: np.ndarray,
    xml_path: str,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    km = load_kinematics_model(xml_path=xml_path, device="cpu")
    root_pos_t = torch.from_numpy(np.asarray(root_pos, dtype=np.float32))
    root_rot_t = torch.from_numpy(np.asarray(root_rot_xyzw, dtype=np.float32))
    dof_pos_t = torch.from_numpy(np.asarray(dof_pos_dataset, dtype=np.float32))
    body_pos_w, body_rot_w = km.forward_kinematics(root_pos_t, root_rot_t, dof_pos_t)
    return (
        body_pos_w.detach().cpu().numpy().astype(np.float32),
        body_rot_w.detach().cpu().numpy().astype(np.float32),
        list(km.body_names),
    )


def build_local_body_features(
    root_pos: np.ndarray,
    root_rot_xyzw: np.ndarray,
    body_pos_w: np.ndarray,
    body_rot_w: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    root_inv = _quat_inverse_xyzw(root_rot_xyzw)
    rel_pos = body_pos_w - root_pos[:, None, :]
    local_pos = _quat_rotate_xyzw(root_inv[:, None, :], rel_pos)
    local_rot = _quat_mul_xyzw(root_inv[:, None, :], body_rot_w)
    return local_pos.astype(np.float32), local_rot.astype(np.float32)
