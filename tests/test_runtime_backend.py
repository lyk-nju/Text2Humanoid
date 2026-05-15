from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np

from text2humanoid.contracts.clips import G1ReferenceChunk
from text2humanoid.runtime.motion_tracking_client import (
    FloodNetFileBackend,
    MotionTrackingClient,
    ShimBackend,
)
from text2humanoid.runtime.source_protocol import (
    chunk_to_runtime_dict,
    validate_clip_payload,
    validate_frame_payload,
)


def _make_chunk(n: int = 6) -> G1ReferenceChunk:
    return G1ReferenceChunk(
        chunk_id="test",
        start_time=0.0,
        fps=30,
        root_pos=np.zeros((n, 3), dtype=np.float32),
        root_rot=np.tile(np.array([[0, 0, 0, 1]], dtype=np.float32), (n, 1)),
        dof_pos=np.zeros((n, 29), dtype=np.float32),
        local_body_pos=np.zeros((n, 30, 3), dtype=np.float32),
        local_body_rot=np.tile(np.array([[[0, 0, 0, 1]]], dtype=np.float32), (n, 30, 1)),
        body_names=[f"body_{i}" for i in range(30)],
        joint_names=[f"joint_{i}" for i in range(29)],
        metadata={"root_quat_order": "xyzw"},
    )


# ---- 006.3: ShimBackend -----------------------------------------------------

def test_shim_backend_basic():
    backend = ShimBackend(control_hz=50)
    chunk = _make_chunk(6)
    backend.push_reference_chunk("s1", chunk)
    status = backend.get_status("s1")
    assert status.buffer_frames == 6
    backend.consume_step("s1", frames=2)
    assert backend.get_status("s1").buffer_frames == 4


def test_shim_backend_reset():
    backend = ShimBackend()
    chunk = _make_chunk(6)
    backend.push_reference_chunk("s1", chunk)
    backend.reset_session("s1")
    assert backend.get_status("s1").buffer_frames == 0
    assert backend.get_status("s1").sim_time == 0.0


# ---- 006.3: FloodNetFileBackend ---------------------------------------------

def test_floodnet_file_backend_writes_npz():
    chunk = _make_chunk(6)
    with tempfile.TemporaryDirectory() as tmp:
        backend = FloodNetFileBackend(output_dir=tmp)
        backend.push_reference_chunk("s1", chunk)
        out_path = Path(tmp) / "s1" / "chunk_0000.npz"
        assert out_path.exists()
        data = np.load(out_path, allow_pickle=True)
        assert "root_pos" in data.files
        assert "root_rot" in data.files
        assert "dof_pos" in data.files
        assert "local_body_pos" in data.files
        assert "local_body_rot" in data.files
        assert "joint_names" in data.files
        assert "body_names" in data.files


def test_floodnet_file_backend_multiple_chunks():
    with tempfile.TemporaryDirectory() as tmp:
        backend = FloodNetFileBackend(output_dir=tmp)
        backend.push_reference_chunk("s1", _make_chunk(6))
        backend.push_reference_chunk("s1", _make_chunk(8))
        assert (Path(tmp) / "s1" / "chunk_0000.npz").exists()
        assert (Path(tmp) / "s1" / "chunk_0001.npz").exists()


def test_floodnet_file_backend_errors_on_invalid_payload():
    with tempfile.TemporaryDirectory() as tmp:
        backend = FloodNetFileBackend(output_dir=tmp)
        # Create a chunk with wrong joint_names count to trigger validation
        bad_chunk = G1ReferenceChunk(
            chunk_id="bad",
            start_time=0.0, fps=30,
            root_pos=np.zeros((4, 3), dtype=np.float32),
            root_rot=np.tile(np.array([[0, 0, 0, 1]], dtype=np.float32), (4, 1)),
            dof_pos=np.zeros((4, 10), dtype=np.float32),  # wrong joint dim
            local_body_pos=np.zeros((4, 30, 3), dtype=np.float32),
            local_body_rot=np.tile(np.array([[[0, 0, 0, 1]]], dtype=np.float32), (4, 30, 1)),
            body_names=[f"body_{i}" for i in range(30)],
            joint_names=[f"joint_{i}" for i in range(29)],  # 29 names != 10 dof
            metadata={"root_quat_order": "xyzw"},
        )
        try:
            backend.push_reference_chunk("s1", bad_chunk)
            raise AssertionError("should have raised ValueError")
        except ValueError:
            pass


# ---- 006.3: MotionTrackingClient delegates to backend -----------------------

def test_motion_tracking_client_with_shim():
    client = MotionTrackingClient(control_hz=50)
    assert isinstance(client.backend, ShimBackend)
    chunk = _make_chunk(6)
    client.push_reference_chunk("s1", chunk)
    assert client.get_status("s1").buffer_frames == 6


def test_motion_tracking_client_with_file_backend():
    with tempfile.TemporaryDirectory() as tmp:
        backend = FloodNetFileBackend(output_dir=tmp)
        client = MotionTrackingClient(backend=backend)
        chunk = _make_chunk(6)
        client.push_reference_chunk("s1", chunk)
        assert (Path(tmp) / "s1" / "chunk_0000.npz").exists()


# ---- 006.2: source_protocol validation --------------------------------------

def test_validate_clip_payload_clean():
    chunk = _make_chunk(6)
    payload = chunk_to_runtime_dict(chunk)
    errors = validate_clip_payload(payload)
    assert len(errors) == 0


def test_validate_clip_payload_missing_key():
    payload = {"root_pos": np.zeros((6, 3), dtype=np.float32)}
    errors = validate_clip_payload(payload)
    assert len(errors) > 0


def test_validate_clip_payload_wrong_shape():
    chunk = _make_chunk(6)
    payload = chunk_to_runtime_dict(chunk)
    payload["root_rot"] = np.zeros((6, 3), dtype=np.float32)  # should be (T,4)
    errors = validate_clip_payload(payload)
    assert any("root_rot" in e for e in errors)


def test_validate_frame_payload_clean():
    chunk = _make_chunk(6)
    from text2humanoid.runtime.source_protocol import frame_payload
    fp = frame_payload(chunk, 0)
    errors = validate_frame_payload(fp, expected_joints=29)
    assert len(errors) == 0


def test_validate_frame_payload_wrong_joints():
    payload = {"root_pos": np.zeros(3), "root_quat": np.zeros(4), "dof_pos": np.zeros(10)}
    errors = validate_frame_payload(payload, expected_joints=29)
    assert len(errors) > 0
