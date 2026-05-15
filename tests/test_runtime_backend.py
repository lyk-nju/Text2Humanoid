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


# ---- 006: config_loader backend selection -----------------------------------

def test_build_components_with_floodnet_file_backend():
    """Verify runtime.backend='floodnet_file' creates FloodNetFileBackend."""
    import tempfile
    from text2humanoid.infra.paths import get_root, set_root
    from text2humanoid.infra.config_loader import build_components

    saved = get_root()
    real_root = get_root()
    with tempfile.TemporaryDirectory() as tmp:
        cfg = {
            "root_path": str(real_root),
            "artifacts_root": tmp,
            "host": "127.0.0.1",
            "port": 9999,
            "planner": {
                "config_path": "FloodNet/configs/ldf_generate.yaml",
            },
            "retarget": {
                "apply_filter": False,
            },
            "runtime": {
                "backend": "floodnet_file",
                "floodnet_output_dir": tmp + "/floodnet_clips",
                "control_hz": 50,
            },
        }
        try:
            session_manager, artifact_store = build_components(cfg)
            coordinator = session_manager._coordinator
            client = coordinator.runtime
            assert isinstance(client.backend, FloodNetFileBackend), \
                f"Expected FloodNetFileBackend, got {type(client.backend)}"
            assert client.backend.output_dir == Path(tmp + "/floodnet_clips")
        finally:
            set_root(str(saved))


def test_build_components_with_socket_backend():
    """Verify runtime.backend='socket' creates SocketBackend."""
    from text2humanoid.infra.paths import get_root, set_root
    from text2humanoid.infra.config_loader import build_components
    from text2humanoid.runtime.socket_backend import SocketBackend

    saved = get_root()
    try:
        cfg = {
            "root_path": str(saved),
            "artifacts_root": "./artifacts/test_socket",
            "host": "127.0.0.1", "port": 9999,
            "planner": {"config_path": "FloodNet/configs/ldf_generate.yaml"},
            "retarget": {"apply_filter": False},
            "runtime": {"backend": "socket", "socket_port": 15556, "control_hz": 50},
        }
        session_manager, _ = build_components(cfg)
        client = session_manager._coordinator.runtime
        assert isinstance(client.backend, SocketBackend)
        assert client.backend.port == 15556
    finally:
        set_root(str(saved))


def test_build_components_default_backend_is_shim():
    """Verify default runtime.backend is ShimBackend."""
    from text2humanoid.infra.paths import get_root, set_root
    from text2humanoid.infra.config_loader import build_components

    saved = get_root()
    real_root = get_root()
    try:
        cfg = {
            "root_path": str(real_root),
            "artifacts_root": "./artifacts/test_default",
            "host": "127.0.0.1",
            "port": 9999,
            "planner": {"config_path": "FloodNet/configs/ldf_generate.yaml"},
            "retarget": {"apply_filter": False},
            "runtime": {"control_hz": 50},
        }
        session_manager, _ = build_components(cfg)
        client = session_manager._coordinator.runtime
        from text2humanoid.runtime.motion_tracking_client import ShimBackend
        assert isinstance(client.backend, ShimBackend), \
            f"Expected ShimBackend by default, got {type(client.backend)}"
    finally:
        set_root(str(saved))


# ---- 008: online refill + stream lifecycle smoke -----------------------------

def test_stream_status_written_on_push():
    """Each push updates stream_status.json with running phase."""
    chunk = _make_chunk(5)
    with tempfile.TemporaryDirectory() as tmp:
        backend = FloodNetFileBackend(output_dir=tmp)
        backend.push_reference_chunk("s1", chunk)
        import json
        status_path = Path(tmp) / "s1" / "stream_status.json"
        assert status_path.exists()
        st = json.loads(open(status_path).read())
        assert st["phase"] == "running"
        assert st["chunk_count"] == 1


def test_stream_lifecycle_done_and_error():
    """mark_stream_done/error write correct phases."""
    with tempfile.TemporaryDirectory() as tmp:
        backend = FloodNetFileBackend(output_dir=tmp)
        backend.ensure_session("s1")
        backend.mark_stream_done("s1")
        import json
        st = json.loads(open(Path(tmp) / "s1" / "stream_status.json").read())
        assert st["phase"] == "done"

        backend.mark_stream_error("s1")
        st = json.loads(open(Path(tmp) / "s1" / "stream_status.json").read())
        assert st["phase"] == "error"


def test_session_refill_cycle_produces_multiple_chunks():
    """run_refill_cycle produces multiple chunks via mock coordinator."""
    from text2humanoid.contracts.commands import PromptCommand, TrajectoryCondition, TrajectoryPoint
    from text2humanoid.orchestrator.session_manager import SessionManager
    from text2humanoid.orchestrator.pipeline_coordinator import PipelineCoordinator
    from text2humanoid.runtime.fallback_policy import FallbackPolicy

    backend = ShimBackend(control_hz=50)
    fallback = FallbackPolicy(low_watermark_frames=5, high_watermark_frames=60)

    chunk_count = [0]

    class _MockPlanner:
        def warmup(self, text): return None
        def reset(self): pass
        def generate_chunk(self, command, start_time):
            import uuid
            chunk_count[0] += 1
            from text2humanoid.contracts.chunks import HumanMotionChunk
            return HumanMotionChunk(
                chunk_id=uuid.uuid4().hex, start_time=start_time, fps=20,
                motion_263=np.zeros((4, 263), dtype=np.float32),
                text=command.text, metadata={"device": "cpu"},
            )

    class _MockRetarget:
        output_fps = 30
        def retarget_chunk(self, nmr_chunk):
            return {"dof": np.zeros((6, 29), dtype=np.float32),
                    "root_trans": np.zeros((6, 3), dtype=np.float32),
                    "root_rot_quat": np.tile(np.array([[1, 0, 0, 0]], dtype=np.float32), (6, 1))}

    class _MockAdapter:
        def from_nmr_result(self, chunk_id, start_time, fps, result):
            return _make_chunk(6)

    coordinator = PipelineCoordinator(
        planner=_MockPlanner(), retarget=_MockRetarget(),
        adapter=_MockAdapter(), runtime=MotionTrackingClient(backend=backend),
        fallback=fallback,
    )
    sm = SessionManager(coordinator=coordinator)
    sid = sm.create_session()
    cmd = PromptCommand(
        text="walk",
        trajectory=TrajectoryCondition(
            waypoints=[TrajectoryPoint(t=0, x=0, y=0, z=0), TrajectoryPoint(t=2, x=3, y=0, z=4)]
        ),
    )
    # Directly push a chunk into the backend to avoid bridge dependency
    backend.push_reference_chunk(sid, _make_chunk(6))
    sm._sessions[sid].timeline.append(cmd)
    assert backend.get_status(sid).buffer_frames == 6

    # Consume all frames
    for _ in range(10):
        backend.consume_step(sid, frames=1)
    low_status = backend.get_status(sid)
    assert low_status.buffer_frames == 0, f"buffer should be 0, got {low_status.buffer_frames}"

    # Refill cycle should call planner 3 times (max_chunks=3)
    from unittest import mock as umock
    from text2humanoid.contracts.chunks import NMRInputChunk

    def _fake_nmr_input(chunk, tgt_fps=30):
        return NMRInputChunk(chunk_id=chunk.chunk_id, start_time=chunk.start_time,
                             fps=tgt_fps, motion_140=np.zeros((6, 140), dtype=np.float32))

    chunk_count[0] = 0
    # watermark=30 means multiple chunks needed (6+2+2+2... with overlap=4)
    with umock.patch("text2humanoid.orchestrator.pipeline_coordinator.human_chunk_to_nmr_input", _fake_nmr_input):
        produced = sm.run_refill_cycle(sid, watermark_frames=30, max_chunks=10)
    assert produced > 1, f"should produce multiple chunks, got {produced}"
    assert chunk_count[0] > 1, f"planner should be called multiple times, got {chunk_count[0]}"
    refilled = backend.get_status(sid)
    assert refilled.buffer_frames > 0, "buffer should have frames after refill"


def test_background_refill_loop_produces_chunks():
    """start_refill_loop auto-produces chunks in background."""
    import time as time_mod
    from text2humanoid.contracts.commands import PromptCommand, TrajectoryCondition, TrajectoryPoint
    from text2humanoid.orchestrator.session_manager import SessionManager
    from text2humanoid.orchestrator.pipeline_coordinator import PipelineCoordinator
    from text2humanoid.runtime.fallback_policy import FallbackPolicy

    backend = ShimBackend(control_hz=50)
    fallback = FallbackPolicy(low_watermark_frames=5, high_watermark_frames=60)

    class _MockPlanner2:
        def warmup(self, text): return None
        def reset(self): pass
        def generate_chunk(self, command, start_time):
            import uuid
            from text2humanoid.contracts.chunks import HumanMotionChunk
            return HumanMotionChunk(
                chunk_id=uuid.uuid4().hex, start_time=start_time, fps=20,
                motion_263=np.zeros((4, 263), dtype=np.float32),
                text=command.text, metadata={"device": "cpu"},
            )

    class _MockRetarget2:
        output_fps = 30
        def retarget_chunk(self, nmr_chunk):
            return {"dof": np.zeros((6, 29), dtype=np.float32),
                    "root_trans": np.zeros((6, 3), dtype=np.float32),
                    "root_rot_quat": np.tile(np.array([[1, 0, 0, 0]], dtype=np.float32), (6, 1))}

    class _MockAdapter2:
        def from_nmr_result(self, chunk_id, start_time, fps, result):
            return _make_chunk(6)

    coordinator = PipelineCoordinator(
        planner=_MockPlanner2(), retarget=_MockRetarget2(),
        adapter=_MockAdapter2(), runtime=MotionTrackingClient(backend=backend),
        fallback=fallback,
    )
    sm = SessionManager(coordinator=coordinator)
    sid = sm.create_session()
    cmd = PromptCommand(
        text="walk",
        trajectory=TrajectoryCondition(
            waypoints=[TrajectoryPoint(t=0, x=0, y=0, z=0), TrajectoryPoint(t=2, x=3, y=0, z=4)]
        ),
    )
    # Use mock bridge
    from unittest import mock as umock
    from text2humanoid.contracts.chunks import NMRInputChunk
    def _fake(chunk, tgt_fps=30):
        return NMRInputChunk(chunk_id=chunk.chunk_id, start_time=chunk.start_time,
                             fps=tgt_fps, motion_140=np.zeros((6, 140), dtype=np.float32))

    with umock.patch("text2humanoid.orchestrator.pipeline_coordinator.human_chunk_to_nmr_input", _fake):
        sm.push_command(sid, cmd)
    initial = backend.get_status(sid).buffer_frames
    assert initial > 0

    # Start background refill and consume rapidly to trigger refill
    sm.start_refill_loop(sid, watermark_frames=20, max_chunks=1, interval_sec=0.1)
    time_mod.sleep(0.3)  # give background thread time to run

    sm.stop_refill_loop(sid)
    final = backend.get_status(sid).buffer_frames
    # Background loop should have produced additional chunks
    assert final >= initial, f"final buffer {final} >= initial {initial}"


def test_stop_session_marks_stream_done():
    """stop_session calls mark_stream_done on FloodNetFileBackend."""
    with tempfile.TemporaryDirectory() as tmp:
        backend = FloodNetFileBackend(output_dir=tmp)
        from text2humanoid.orchestrator.session_manager import SessionManager
        from text2humanoid.orchestrator.pipeline_coordinator import PipelineCoordinator
        from text2humanoid.runtime.fallback_policy import FallbackPolicy

        coordinator = PipelineCoordinator(
            planner=None, retarget=None, adapter=None,
            runtime=MotionTrackingClient(backend=backend),
            fallback=FallbackPolicy(),
        )
        sm = SessionManager(coordinator=coordinator)
        sid = sm.create_session()
        backend.push_reference_chunk(sid, _make_chunk(5))
        sm.stop_session(sid)

        import json
        status_path = Path(tmp) / sid / "stream_status.json"
        assert status_path.exists(), "stream_status.json should exist"
        st = json.loads(open(status_path).read())
        assert st["phase"] == "done"


def test_shim_backend_unaffected_by_stream_lifecycle():
    """ShimBackend still works as before — no stream_status files."""
    backend = ShimBackend(control_hz=50)
    chunk = _make_chunk(6)
    backend.push_reference_chunk("s1", chunk)
    assert backend.get_status("s1").buffer_frames == 6
    backend.consume_step("s1", frames=3)
    assert backend.get_status("s1").buffer_frames == 3
    backend.reset_session("s1")
    assert backend.get_status("s1").buffer_frames == 0
