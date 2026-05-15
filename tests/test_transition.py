"""Online command transition smoke — running session accepts second command."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np

from text2humanoid.contracts.commands import PromptCommand, TrajectoryCondition, TrajectoryPoint
from text2humanoid.contracts.chunks import HumanMotionChunk, NMRInputChunk
from text2humanoid.contracts.clips import G1ReferenceChunk
from text2humanoid.runtime.motion_tracking_client import FloodNetFileBackend, MotionTrackingClient
from text2humanoid.orchestrator.pipeline_coordinator import PipelineCoordinator
from text2humanoid.orchestrator.session_manager import SessionManager
from text2humanoid.orchestrator.timeline import SessionTimeline
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
        body_names=[f"b{i}" for i in range(30)], joint_names=[f"j{i}" for i in range(29)],
        metadata={"root_quat_order": "xyzw"},
    )


class _MockPlanner:
    def __init__(self): self._driver = None
    def warmup(self, text): pass
    def generate_chunk(self, command, start_time, feature_length=None):
        import uuid
        return HumanMotionChunk(chunk_id=uuid.uuid4().hex, start_time=start_time, fps=20,
                                motion_263=np.zeros((4, 263), dtype=np.float32),
                                text=command.text, metadata={"device": "mock"})
    def reset(self):
        if self._driver is not None: self._driver._session = None


class _MockRetarget:
    output_fps = 30
    def retarget_chunk(self, nmr_chunk):
        return {"dof": np.zeros((6, 29), dtype=np.float32),
                "root_trans": np.zeros((6, 3), dtype=np.float32),
                "root_rot_quat": np.tile(np.array([[1, 0, 0, 0]], dtype=np.float32), (6, 1))}


class _MockAdapter:
    def from_nmr_result(self, chunk_id, start_time, fps, result): return _make_ref(6)


def _fake_nmr(chunk, tgt_fps=30):
    return NMRInputChunk(chunk_id=chunk.chunk_id, start_time=chunk.start_time,
                         fps=tgt_fps, motion_140=np.zeros((6, 140), dtype=np.float32))


def _setup(tmp):
    from unittest import mock as umock
    import contextlib
    backend = FloodNetFileBackend(output_dir=tmp)
    planner = _MockPlanner()
    planner._driver = StreamPlannerDriver(planner)
    coordinator = PipelineCoordinator(
        planner=planner, retarget=_MockRetarget(), adapter=_MockAdapter(),
        runtime=MotionTrackingClient(backend=backend), fallback=FallbackPolicy())
    sm = SessionManager(coordinator=coordinator)
    sid = sm.create_session()
    p = contextlib.ExitStack()
    p.enter_context(umock.patch("text2humanoid.orchestrator.pipeline_coordinator.human_chunk_to_nmr_input", _fake_nmr))
    p.enter_context(umock.patch("text2humanoid.retarget.bridge_263_to_140.human_chunk_to_nmr_input", _fake_nmr))
    p.enter_context(umock.patch("text2humanoid.retarget.bridge_263_to_140.floodnet_263_to_nmr_140",
                                 lambda *a, **kw: np.zeros((6, 140), dtype=np.float32)))
    return backend, planner, coordinator, sm, sid, p


# ---- Timeline tests ---------------------------------------------------------

def test_timeline_first_command_no_transition():
    tl = SessionTimeline(session_id="t")
    assert tl.append(PromptCommand(text="a", command_id="1")) is None
    assert tl.active_command_id == "1"
    assert tl.last_transition is None


def test_timeline_records_transition():
    tl = SessionTimeline(session_id="t")
    tl.append(PromptCommand(text="a", command_id="1"))
    rec = tl.append(PromptCommand(text="b", command_id="2"))
    assert rec is not None
    assert rec.previous_command_id == "1"
    assert rec.new_command_id == "2"
    assert tl.active_command_id == "2"


def test_timeline_multiple_transitions():
    tl = SessionTimeline(session_id="t")
    for i in range(3):
        tl.append(PromptCommand(text=str(i), command_id=str(i)))
    assert len(tl.transitions) == 2


# ---- Running session transition smoke ---------------------------------------

def test_running_session_accepts_second_command():
    with tempfile.TemporaryDirectory() as tmp:
        backend, planner, coordinator, sm, sid, p = _setup(tmp)
        with p:
            c1 = PromptCommand(text="walk", command_id="cmd1",
                               trajectory=TrajectoryCondition(
                                   waypoints=[TrajectoryPoint(t=0, x=0, y=0, z=0),
                                              TrajectoryPoint(t=2, x=2, y=0, z=0)]))
            sm.push_command(sid, c1)
            assert planner._driver.session.command is c1
            sm.run_refill_cycle(sid, watermark_frames=100, max_chunks=1)
            before = planner._driver.session.chunk_index

            c2 = PromptCommand(text="run", command_id="cmd2",
                               transition_mode="replace",
                               trajectory=TrajectoryCondition(
                                   waypoints=[TrajectoryPoint(t=0, x=0, y=0, z=0),
                                              TrajectoryPoint(t=5, x=10, y=0, z=0)]))
            sm.push_command(sid, c2)

            # REPLACE mode: planner session immediately has new command
            assert planner._driver.session.command is c2
            assert planner._driver.session.chunk_index >= before

            # Timeline records transition
            ctx = sm._sessions[sid]
            assert ctx.timeline.last_transition is not None
            assert ctx.timeline.active_command_id == "cmd2"

            # Status metadata reflects transition
            meta = ctx.status.metadata
            assert meta.get("active_command_id") == "cmd2"
            assert meta.get("transition", {}).get("previous_command_id") == "cmd1"


def test_refill_uses_new_command_after_transition():
    with tempfile.TemporaryDirectory() as tmp:
        backend, planner, coordinator, sm, sid, p = _setup(tmp)
        with p:
            sm.push_command(sid, PromptCommand(text="walk", command_id="1"))
            sm.push_command(sid, PromptCommand(text="run fast", command_id="2"))
            sm.run_refill_cycle(sid, watermark_frames=200, max_chunks=2)
            assert planner._driver.session.command.text == "run fast"

            session_dir = Path(tmp) / sid
            # 1st cmd chunk + refill chunks (APPEND doesn't generate immediate chunk)
            assert len(list(session_dir.glob("chunk_*.npz"))) >= 2


def test_transition_does_not_break_stream_lifecycle():
    with tempfile.TemporaryDirectory() as tmp:
        backend, planner, coordinator, sm, sid, p = _setup(tmp)
        with p:
            sm.push_command(sid, PromptCommand(text="a", command_id="1"))
            sm.push_command(sid, PromptCommand(text="b", command_id="2"))
            sm.run_refill_cycle(sid, watermark_frames=100, max_chunks=2)
        sm.stop_session(sid)
        st = json.loads(open(Path(tmp) / sid / "stream_status.json").read())
        assert st["phase"] == "done"


# ---- 012: multi-command sequence smoke --------------------------------------

def test_three_commands_replay_append():
    """running session accepts 3 commands with APPEND mode."""
    with tempfile.TemporaryDirectory() as tmp:
        backend, planner, coordinator, sm, sid, p = _setup(tmp)
        with p:
            sm.push_command(sid, PromptCommand(text="a", command_id="1",
                transition_mode="append",
                trajectory=TrajectoryCondition(
                    waypoints=[TrajectoryPoint(t=0, x=0, y=0, z=0),
                               TrajectoryPoint(t=1, x=1, y=0, z=0)])))
            sm.push_command(sid, PromptCommand(text="b", command_id="2",
                transition_mode="append"))
            sm.push_command(sid, PromptCommand(text="c", command_id="3",
                transition_mode="append"))

            ctx = sm._sessions[sid]
            assert len(ctx.timeline.transitions) == 2
            assert ctx.timeline.active_command_id == "3"

            # REFILL: APPEND mode → pending promoted on refill, then new command used
            sm.run_refill_cycle(sid, watermark_frames=400, max_chunks=5)
            # APPEND only queues pending; first cmd + refills produce chunks
            session_dir = Path(tmp) / sid
            assert len(list(session_dir.glob("chunk_*.npz"))) >= 2


def test_replace_mode_switches_immediately():
    """REPLACE mode: new command replaces active immediately."""
    with tempfile.TemporaryDirectory() as tmp:
        backend, planner, coordinator, sm, sid, p = _setup(tmp)
        with p:
            sm.push_command(sid, PromptCommand(text="old", command_id="1"))
            assert planner._driver.session.command.text == "old"

            sm.push_command(sid, PromptCommand(text="new", command_id="2",
                transition_mode="replace"))
            # REPLACE should immediately switch the planner session's active command
            assert planner._driver.session.command.text == "new"
            assert not planner._driver.session.has_pending


def test_append_mode_sets_pending():
    """APPEND mode: new command becomes pending, not active yet."""
    with tempfile.TemporaryDirectory() as tmp:
        backend, planner, coordinator, sm, sid, p = _setup(tmp)
        with p:
            sm.push_command(sid, PromptCommand(text="first", command_id="1"))
            sm.push_command(sid, PromptCommand(text="second", command_id="2",
                transition_mode="append"))
            # APPEND: old command still active, new command is pending
            assert planner._driver.session.command.text == "first"
            assert planner._driver.session.has_pending
            assert planner._driver.session.pending_command.text == "second"

            # Refill promotes pending
            sm.run_refill_cycle(sid, watermark_frames=200, max_chunks=1)
            assert planner._driver.session.command.text == "second"
            assert not planner._driver.session.has_pending


def test_refill_keeps_working_after_three_commands():
    """After 3 commands, refill still produces chunks without restart."""
    with tempfile.TemporaryDirectory() as tmp:
        backend, planner, coordinator, sm, sid, p = _setup(tmp)
        with p:
            sm.push_command(sid, PromptCommand(text="a", command_id="1"))
            sm.push_command(sid, PromptCommand(text="b", command_id="2"))
            sm.push_command(sid, PromptCommand(text="c", command_id="3"))

            # Multiple refill cycles should all succeed
            for _ in range(5):
                sm.run_refill_cycle(sid, watermark_frames=500, max_chunks=1)

            session_dir = Path(tmp) / sid
            assert len(list(session_dir.glob("chunk_*.npz"))) >= 5


def test_lifecycle_preserved_after_three_commands():
    """Three commands → stop → stream_status is done."""
    with tempfile.TemporaryDirectory() as tmp:
        backend, planner, coordinator, sm, sid, p = _setup(tmp)
        with p:
            sm.push_command(sid, PromptCommand(text="a", command_id="1"))
            sm.push_command(sid, PromptCommand(text="b", command_id="2"))
            sm.push_command(sid, PromptCommand(text="c", command_id="3"))
            sm.run_refill_cycle(sid, watermark_frames=200, max_chunks=3)
        sm.stop_session(sid)
        st = json.loads(open(Path(tmp) / sid / "stream_status.json").read())
        assert st["phase"] == "done"


def test_append_does_not_generate_immediate_chunk():
    """APPEND mode: new command doesn't generate a chunk immediately."""
    with tempfile.TemporaryDirectory() as tmp:
        backend, planner, coordinator, sm, sid, p = _setup(tmp)
        with p:
            sm.push_command(sid, PromptCommand(text="first", command_id="1"))
            n_before = len(list((Path(tmp) / sid).glob("chunk_*.npz")))

            # APPEND: should NOT call run_once, so no new chunk
            sm.push_command(sid, PromptCommand(text="second", command_id="2",
                transition_mode="append"))
            n_after_append = len(list((Path(tmp) / sid).glob("chunk_*.npz")))
            assert n_after_append == n_before, \
                f"APPEND should not generate immediate chunk: {n_before} → {n_after_append}"

            # Pending should be set
            assert planner._driver.session.has_pending
            assert planner._driver.session.pending_command.text == "second"

            # Refill promotes pending and generates
            sm.run_refill_cycle(sid, watermark_frames=200, max_chunks=1)
            n_after_refill = len(list((Path(tmp) / sid).glob("chunk_*.npz")))
            assert n_after_refill > n_before, "refill should generate new chunk after APPEND"


def test_replace_generates_immediate_chunk():
    """REPLACE mode: new command generates a chunk immediately."""
    with tempfile.TemporaryDirectory() as tmp:
        backend, planner, coordinator, sm, sid, p = _setup(tmp)
        with p:
            sm.push_command(sid, PromptCommand(text="old", command_id="1"))
            n_before = len(list((Path(tmp) / sid).glob("chunk_*.npz")))

            # REPLACE: calls run_once immediately → new chunk
            sm.push_command(sid, PromptCommand(text="new", command_id="2",
                transition_mode="replace"))
            n_after = len(list((Path(tmp) / sid).glob("chunk_*.npz")))
            assert n_after > n_before, \
                f"REPLACE should generate immediate chunk: {n_before} → {n_after}"
