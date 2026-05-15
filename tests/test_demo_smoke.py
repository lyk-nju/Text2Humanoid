"""End-to-end demo smoke: fixed text + fixed trajectory → continuous chunks.

Verifies the full demo lifecycle using mock planner/retarget/adapter so no
real models are needed.
"""

from __future__ import annotations

import json
import tempfile
import time
from pathlib import Path

import numpy as np

from text2humanoid.contracts.commands import PromptCommand, TrajectoryCondition, TrajectoryPoint
from text2humanoid.contracts.chunks import HumanMotionChunk, NMRInputChunk
from text2humanoid.contracts.clips import G1ReferenceChunk
from text2humanoid.runtime.motion_tracking_client import (
    FloodNetFileBackend,
    MotionTrackingClient,
)
from text2humanoid.orchestrator.pipeline_coordinator import PipelineCoordinator
from text2humanoid.orchestrator.session_manager import SessionManager
from text2humanoid.runtime.fallback_policy import FallbackPolicy
from text2humanoid.planner.stream_driver import StreamPlannerDriver


def _make_ref(n: int):
    return G1ReferenceChunk(
        chunk_id="mock", start_time=0.0, fps=30,
        root_pos=np.zeros((n, 3), dtype=np.float32),
        root_rot=np.tile(np.array([[0, 0, 0, 1]], dtype=np.float32), (n, 1)),
        dof_pos=np.zeros((n, 29), dtype=np.float32),
        local_body_pos=np.zeros((n, 30, 3), dtype=np.float32),
        local_body_rot=np.tile(np.array([[[0, 0, 0, 1]]], dtype=np.float32), (n, 30, 1)),
        body_names=[f"b{i}" for i in range(30)],
        joint_names=[f"j{i}" for i in range(29)],
        metadata={"root_quat_order": "xyzw"},
    )


class _DemoMockPlanner:
    def __init__(self):
        self._driver = None

    def warmup(self, text):
        pass

    def generate_chunk(self, command, start_time, feature_length=None):
        import uuid
        return HumanMotionChunk(
            chunk_id=uuid.uuid4().hex, start_time=start_time, fps=20,
            motion_263=np.zeros((4, 263), dtype=np.float32),
            text=command.text, metadata={"device": "mock"},
        )

    def reset(self):
        if self._driver is not None:
            self._driver._session = None


class _DemoMockRetarget:
    output_fps = 30
    def retarget_chunk(self, nmr_chunk):
        return {"dof": np.zeros((6, 29), dtype=np.float32),
                "root_trans": np.zeros((6, 3), dtype=np.float32),
                "root_rot_quat": np.tile(np.array([[1, 0, 0, 0]], dtype=np.float32), (6, 1))}


class _DemoMockAdapter:
    def from_nmr_result(self, chunk_id, start_time, fps, result):
        return _make_ref(6)


def _build_demo(tmp: str):
    """Shared setup for demo smoke tests."""
    from unittest import mock as umock

    backend = FloodNetFileBackend(output_dir=tmp)
    planner = _DemoMockPlanner()
    planner._driver = StreamPlannerDriver(planner)
    coordinator = PipelineCoordinator(
        planner=planner, retarget=_DemoMockRetarget(), adapter=_DemoMockAdapter(),
        runtime=MotionTrackingClient(backend=backend),
        fallback=FallbackPolicy(low_watermark_frames=10, high_watermark_frames=80),
    )
    sm = SessionManager(coordinator=coordinator)
    sid = sm.create_session()

    def _fake_nmr(chunk, tgt_fps=30):
        return NMRInputChunk(chunk_id=chunk.chunk_id, start_time=chunk.start_time,
                             fps=tgt_fps, motion_140=np.zeros((6, 140), dtype=np.float32))

    # Patch ALL bridge import sites so both push_command (pipeline_coordinator)
    # and run_refill_cycle (session_manager via local import) work.
    import contextlib
    patcher = contextlib.ExitStack()
    patcher.enter_context(umock.patch(
        "text2humanoid.orchestrator.pipeline_coordinator.human_chunk_to_nmr_input", _fake_nmr))
    patcher.enter_context(umock.patch(
        "text2humanoid.retarget.bridge_263_to_140.human_chunk_to_nmr_input", _fake_nmr))
    patcher.enter_context(umock.patch(
        "text2humanoid.retarget.bridge_263_to_140.floodnet_263_to_nmr_140",
        lambda *a, **kw: np.zeros((6, 140), dtype=np.float32)))
    return backend, planner, coordinator, sm, sid, patcher


def test_demo_lifecycle_single_command_multiple_chunks():
    """Full demo lifecycle: one push → auto refill → multiple chunks → done."""
    with tempfile.TemporaryDirectory() as tmp:
        backend, planner, coordinator, sm, sid, patcher = _build_demo(tmp)

        cmd = PromptCommand(
            text="walk forward slowly",
            trajectory=TrajectoryCondition(
                waypoints=[TrajectoryPoint(t=0, x=0, y=0, z=0),
                           TrajectoryPoint(t=5, x=5, y=0, z=0)],
            ),
        )

        sm.push_command(sid, cmd)

        # Verify planner session was started
        assert planner._driver.session is not None
        assert planner._driver.session.command is cmd

        # Start background refill
        sm.start_refill_loop(sid, watermark_frames=60, max_chunks=1, interval_sec=0.05)
        time.sleep(0.3)
        sm.stop_refill_loop(sid)

        # Verify multiple chunks in output dir
        session_dir = Path(tmp) / sid
        chunk_files = sorted(session_dir.glob("chunk_*.npz"))
        assert len(chunk_files) >= 2, f"expected >=2 chunks, got {len(chunk_files)}"

        # Verify chunk_index.json
        manifest = json.loads(open(session_dir / "chunk_index.json").read())
        assert manifest["chunk_count"] >= 2

        # Verify stream lifecycle: running → done
        st = json.loads(open(session_dir / "stream_status.json").read())
        assert st["phase"] == "running"

        sm.stop_session(sid)
        st_done = json.loads(open(session_dir / "stream_status.json").read())
        assert st_done["phase"] == "done"

        # Planner session chunk_index advanced
        assert planner._driver.session.chunk_index >= 1


def test_demo_output_dir_grows():
    """Output directory chunk count monotonically increases."""
    with tempfile.TemporaryDirectory() as tmp:
        backend, planner, coordinator, sm, sid, patcher = _build_demo(tmp)

        sm.push_command(sid, PromptCommand(
            text="walk",
            trajectory=TrajectoryCondition(
                waypoints=[TrajectoryPoint(t=0, x=0, y=0, z=0),
                           TrajectoryPoint(t=3, x=3, y=0, z=0)]
            ),
        ))

        counts = []
        session_dir = Path(tmp) / sid
        for _ in range(5):
            sm.run_refill_cycle(sid, watermark_frames=300, max_chunks=1)
            n = len(list(session_dir.glob("chunk_*.npz")))
            counts.append(n)

        assert counts == sorted(counts), f"chunk count non-decreasing: {counts}"
        assert counts[-1] > counts[0], "chunk count should grow"


def test_demo_no_manual_command_replay():
    """Refill works without re-pushing the command."""
    with tempfile.TemporaryDirectory() as tmp:
        backend, planner, coordinator, sm, sid, patcher = _build_demo(tmp)

        sm.push_command(sid, PromptCommand(
            text="walk",
            trajectory=TrajectoryCondition(
                waypoints=[TrajectoryPoint(t=0, x=0, y=0, z=0),
                           TrajectoryPoint(t=2, x=2, y=0, z=0)]
            ),
        ))

        total = 0
        for _ in range(5):
            total += sm.run_refill_cycle(sid, watermark_frames=500, max_chunks=1)
        assert total > 0, "refill should produce chunks without manual command"
