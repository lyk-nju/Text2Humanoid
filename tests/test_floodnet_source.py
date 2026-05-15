"""Smoke tests verifying the Text2Humanoid → motion_tracking clip pipeline.

Tests that a clip exported by FloodNetFileBackend can be loaded and validated,
simulating the exact logic FloodNetMotionSource._load_floodnet_clip() uses,
without importing motion_tracking runtime dependencies.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np

from text2humanoid.contracts.clips import G1ReferenceChunk
from text2humanoid.runtime.motion_tracking_client import FloodNetFileBackend
from text2humanoid.runtime.source_protocol import validate_clip_payload

# Joint name orders — must match motion_tracking tracking.yaml and controller.yaml
# These are the same names listed in FloodNetMotionSource's remap call site.
OBS_JOINT_NAMES = [
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

DATASET_JOINT_NAMES = [
    "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
    "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
    "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
    "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
    "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint",
    "left_elbow_joint", "left_wrist_roll_joint", "left_wrist_pitch_joint", "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
    "right_elbow_joint", "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint",
]


def _remap_joint_array_by_names(data, source_joint_names, target_joint_names):
    """Inlined from motion_sources.remap_joint_array_by_names."""
    data = np.asarray(data, dtype=np.float32)
    name_to_idx = {name: i for i, name in enumerate(source_joint_names)}
    remap = np.zeros((data.shape[0], len(target_joint_names)), dtype=np.float32)
    for i, name in enumerate(target_joint_names):
        j = name_to_idx.get(name, None)
        if j is not None:
            remap[:, i] = data[:, j]
    return remap


def _make_chunk(n: int = 10) -> G1ReferenceChunk:
    root_pos = np.zeros((n, 3), dtype=np.float32)
    root_pos[:, 0] = np.linspace(0, 2, n, dtype=np.float32)
    rng = np.random.default_rng(42)
    return G1ReferenceChunk(
        chunk_id="smoke",
        start_time=0.0, fps=30,
        root_pos=root_pos,
        root_rot=np.tile(np.array([[0, 0, 0, 1]], dtype=np.float32), (n, 1)),
        dof_pos=rng.normal(0, 0.1, (n, 29)).astype(np.float32),
        local_body_pos=rng.normal(0, 0.1, (n, 30, 3)).astype(np.float32),
        local_body_rot=np.tile(np.array([[[0, 0, 0, 1]]], dtype=np.float32), (n, 30, 1)),
        body_names=[f"body_{i}" for i in range(30)],
        joint_names=DATASET_JOINT_NAMES,
        metadata={"root_quat_order": "xyzw", "body_rot_order": "xyzw"},
    )


# ---- Write → Load → Validate (exact FloodNetMotionSource logic) -------------

def _load_clip_like_floodnet_source(npz_path: str):
    """Simulate FloodNetMotionSource._load_floodnet_clip exactly."""
    data = np.load(npz_path, allow_pickle=True)
    joint_pos = data["dof_pos"].astype(np.float32)
    root_pos = data["root_pos"].astype(np.float32)
    root_rot_xyzw = data["root_rot"].astype(np.float32)
    # xyzw -> wxyz (matching motion_tracking internal convention)
    root_quat = np.concatenate([root_rot_xyzw[:, 3:4], root_rot_xyzw[:, :3]], axis=-1)
    jn_raw = data["joint_names"].tolist()
    source_joint_names = [
        n.decode("utf-8") if isinstance(n, bytes) else str(n) for n in jn_raw
    ]
    joint_pos_remapped = _remap_joint_array_by_names(
        joint_pos, source_joint_names, OBS_JOINT_NAMES,
    )
    return {
        "joint_pos": joint_pos_remapped,
        "root_quat": root_quat,
        "root_pos": root_pos,
    }


def test_backend_write_then_load_as_source():
    """Full round-trip: FloodNetFileBackend → NPZ → FloodNetMotionSource logic."""
    chunk = _make_chunk(10)
    with tempfile.TemporaryDirectory() as tmp:
        # Step 1: Text2Humanoid side writes
        backend = FloodNetFileBackend(output_dir=tmp)
        backend.push_reference_chunk("s1", chunk)
        npz_path = str(Path(tmp) / "s1" / "chunk_0000.npz")

        # Step 2: Validate payload
        raw = dict(np.load(npz_path, allow_pickle=True))
        errors = validate_clip_payload(raw)
        assert len(errors) == 0, f"validation errors: {errors}"

        # Step 3: Load as FloodNetMotionSource would
        motion = _load_clip_like_floodnet_source(npz_path)

        assert motion["joint_pos"].shape == (10, 29)
        assert motion["root_quat"].shape == (10, 4)
        assert motion["root_pos"].shape == (10, 3)
        assert np.isfinite(motion["joint_pos"]).all()
        assert np.isfinite(motion["root_quat"]).all()
        assert np.isfinite(motion["root_pos"]).all()


def test_multiple_chunks_roundtrip():
    """Two chunks → both loadable."""
    with tempfile.TemporaryDirectory() as tmp:
        backend = FloodNetFileBackend(output_dir=tmp)
        backend.push_reference_chunk("s1", _make_chunk(6))
        backend.push_reference_chunk("s1", _make_chunk(8))

        for fname in ["chunk_0000.npz", "chunk_0001.npz"]:
            npz_path = Path(tmp) / "s1" / fname
            result = dict(np.load(npz_path, allow_pickle=True))
            errors = validate_clip_payload(result)
            assert len(errors) == 0, f"{fname}: {errors}"


def test_loaded_clip_has_future_horizon():
    """Verify a loaded clip provides enough frames for future horizon."""
    chunk = _make_chunk(20)
    with tempfile.TemporaryDirectory() as tmp:
        backend = FloodNetFileBackend(output_dir=tmp)
        backend.push_reference_chunk("s1", chunk)
        npz_path = str(Path(tmp) / "s1" / "chunk_0000.npz")
        motion = _load_clip_like_floodnet_source(npz_path)

        # Simulate consuming frames at index 0
        # future_steps = [0, 1, 2, 3, 4, -1, -2, -4, -8, -12, -16]
        # max positive = 4, so need at least 5 frames from index 0
        ref_idx = 0
        future_indices = [ref_idx + s for s in [0, 1, 2, 3, 4]]
        for fi in future_indices:
            assert fi < motion["joint_pos"].shape[0], f"frame {fi} out of range"


def test_xyzw_to_wxyz_conversion():
    """Verify the quaternion order conversion in the source load path."""
    chunk = _make_chunk(5)
    with tempfile.TemporaryDirectory() as tmp:
        backend = FloodNetFileBackend(output_dir=tmp)
        backend.push_reference_chunk("s1", chunk)
        npz_path = str(Path(tmp) / "s1" / "chunk_0000.npz")

        raw = dict(np.load(npz_path, allow_pickle=True))
        root_rot_xyzw = raw["root_rot"].astype(np.float32)
        # xyzw: scalar at index 3
        assert abs(root_rot_xyzw[0, 3] - 1.0) < 0.01

        motion = _load_clip_like_floodnet_source(npz_path)
        root_quat_wxyz = motion["root_quat"]
        # wxyz: scalar at index 0
        assert abs(root_quat_wxyz[0, 0] - 1.0) < 0.01
