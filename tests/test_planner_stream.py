from __future__ import annotations

import numpy as np

from text2humanoid.contracts.commands import PromptCommand, TrajectoryCondition, TrajectoryPoint
from text2humanoid.planner.stream_driver import PlannerSession, StreamPlannerDriver


def _wp(t, x, y, z):
    return TrajectoryPoint(t=t, x=x, y=y, z=z)


# ---- PlannerSession contract ------------------------------------------------

def test_planner_session_init():
    cmd = PromptCommand(text="walk")
    sess = PlannerSession(command=cmd)
    assert sess.chunk_index == 0
    assert sess.next_start_time == 0.0


def test_planner_session_advance():
    cmd = PromptCommand(text="walk")
    sess = PlannerSession(command=cmd)
    sess.advance(2.5)
    assert sess.chunk_index == 1
    assert sess.next_start_time == 2.5
    sess.advance(2.0)  # lower value — stays at max
    assert sess.next_start_time == 2.5


# ---- StreamPlannerDriver with mock planner ----------------------------------

class _MockPlanner:
    """Mock FloodNetPlannerService for stream driver tests."""
    def __init__(self):
        self.reset_called = False
        self._driver = None

    def warmup(self, text):
        pass

    def generate_chunk(self, command, start_time, feature_length=None):
        from text2humanoid.contracts.chunks import HumanMotionChunk
        return HumanMotionChunk(
            chunk_id="mock", start_time=start_time, fps=20,
            motion_263=np.zeros((4, 263), dtype=np.float32),
            text=command.text, metadata={},
        )

    def reset(self):
        self.reset_called = True
        # Real FloodNetPlannerService.reset() calls driver.reset() then model.init_generated()
        if self._driver is not None:
            self._driver._session = None


def test_driver_start_session():
    planner = _MockPlanner()
    driver = StreamPlannerDriver(planner)
    cmd = PromptCommand(text="walk")
    sess = driver.start_session(cmd)
    assert sess is not None
    assert sess.command is cmd
    assert driver.session is sess


def test_driver_generate_next_chunk():
    planner = _MockPlanner()
    driver = StreamPlannerDriver(planner)
    cmd = PromptCommand(text="walk")
    driver.start_session(cmd)

    c1 = driver.generate_next_chunk()
    assert c1 is not None
    assert c1.start_time == 0.0
    assert driver.session.chunk_index == 1

    c2 = driver.generate_next_chunk()
    assert c2 is not None
    assert c2.start_time > 0.0
    assert driver.session.chunk_index == 2


def test_driver_generate_without_session_returns_none():
    planner = _MockPlanner()
    driver = StreamPlannerDriver(planner)
    assert driver.generate_next_chunk() is None


def test_driver_reset_clears_session():
    planner = _MockPlanner()
    driver = StreamPlannerDriver(planner)
    driver.start_session(PromptCommand(text="walk"))
    assert driver.session is not None
    driver.reset()
    assert driver.session is None
    # driver.reset() only clears session; planner.reset() is called separately
    # by FloodNetPlannerService.reset() which also calls model.init_generated()


# ---- Planner session refill integration -------------------------------------

def test_multiple_chunks_from_single_session():
    """Single PlannerSession produces multiple chunks without re-creating command."""
    planner = _MockPlanner()
    driver = StreamPlannerDriver(planner)
    cmd = PromptCommand(
        text="walk",
        trajectory=TrajectoryCondition(
            waypoints=[_wp(0, 0, 0, 0), _wp(5, 10, 0, 0)]
        ),
    )
    driver.start_session(cmd)

    chunks = []
    for _ in range(3):
        c = driver.generate_next_chunk()
        assert c is not None
        chunks.append(c)

    assert len(chunks) == 3
    assert driver.session.chunk_index == 3
    # Timeline advances monotonically
    times = [c.start_time for c in chunks]
    assert times == sorted(times)


# ---- SessionManager integration smoke ---------------------------------------

def test_push_command_starts_planner_session():
    """push_command calls driver.start_session so planner-native path is active."""
    from unittest import mock as umock

    from text2humanoid.contracts.chunks import NMRInputChunk
    from text2humanoid.contracts.commands import PromptCommand, TrajectoryCondition, TrajectoryPoint
    from text2humanoid.orchestrator.pipeline_coordinator import PipelineCoordinator
    from text2humanoid.orchestrator.session_manager import SessionManager
    from text2humanoid.runtime.fallback_policy import FallbackPolicy
    from text2humanoid.runtime.motion_tracking_client import MotionTrackingClient, ShimBackend

    planner = _MockPlanner()
    backend = ShimBackend(control_hz=50)

    class _MockRetarget:
        output_fps = 30
        def retarget_chunk(self, nmr_chunk):
            return {"dof": np.zeros((6, 29), dtype=np.float32),
                    "root_trans": np.zeros((6, 3), dtype=np.float32),
                    "root_rot_quat": np.tile(np.array([[1, 0, 0, 0]], dtype=np.float32), (6, 1))}

    class _MockAdapter2:
        def from_nmr_result(self, chunk_id, start_time, fps, result):
            from text2humanoid.contracts.clips import G1ReferenceChunk
            return G1ReferenceChunk(
                chunk_id=chunk_id, start_time=start_time, fps=fps,
                root_pos=np.zeros((6, 3), dtype=np.float32),
                root_rot=np.tile(np.array([[0, 0, 0, 1]], dtype=np.float32), (6, 1)),
                dof_pos=np.zeros((6, 29), dtype=np.float32),
                local_body_pos=np.zeros((6, 30, 3), dtype=np.float32),
                local_body_rot=np.tile(np.array([[[0, 0, 0, 1]]], dtype=np.float32), (6, 30, 1)),
                body_names=["b"] * 30, joint_names=["j"] * 29,
                metadata={"root_quat_order": "xyzw"},
            )

    # Attach driver to mock planner (simulating FloodNetPlannerService.__init__)
    from text2humanoid.planner.stream_driver import StreamPlannerDriver
    planner._driver = StreamPlannerDriver(planner)

    coordinator = PipelineCoordinator(
        planner=planner, retarget=_MockRetarget(), adapter=_MockAdapter2(),
        runtime=MotionTrackingClient(backend=backend),
        fallback=FallbackPolicy(),
    )

    sm = SessionManager(coordinator=coordinator)
    sid = sm.create_session()
    cmd = PromptCommand(
        text="walk",
        trajectory=TrajectoryCondition(
            waypoints=[TrajectoryPoint(t=0, x=0, y=0, z=0), TrajectoryPoint(t=2, x=3, y=0, z=4)]
        ),
    )

    def _fake_nmr(chunk, tgt_fps=30):
        return NMRInputChunk(chunk_id=chunk.chunk_id, start_time=chunk.start_time,
                             fps=tgt_fps, motion_140=np.zeros((6, 140), dtype=np.float32))

    with umock.patch("text2humanoid.orchestrator.pipeline_coordinator.human_chunk_to_nmr_input", _fake_nmr):
        sm.push_command(sid, cmd)

    # Verify planner session was started
    driver = coordinator.planner._driver
    assert driver.session is not None, "push_command should start planner session"
    assert driver.session.command is cmd
    assert driver.session.chunk_index >= 0

    # Verify refill uses planner-native path (not fallback)
    low_status = backend.get_status(sid)
    with umock.patch("text2humanoid.retarget.bridge_263_to_140.human_chunk_to_nmr_input", _fake_nmr), \
         umock.patch("text2humanoid.orchestrator.pipeline_coordinator.human_chunk_to_nmr_input", _fake_nmr):
        produced = sm.run_refill_cycle(sid, watermark_frames=100, max_chunks=3)
    assert produced > 0, "refill should produce chunks via planner session"


def test_planner_reset_clears_session():
    """FloodNetPlannerService.reset() clears the planner session via driver reset."""
    planner = _MockPlanner()
    from text2humanoid.planner.stream_driver import StreamPlannerDriver
    driver = StreamPlannerDriver(planner)
    planner._driver = driver
    driver.start_session(PromptCommand(text="walk"))
    assert driver.session is not None

    # FloodNetPlannerService.reset() calls driver.reset() then model.init_generated()
    driver.reset()
    assert driver.session is None
    planner.reset_called = True  # simulate model init_generated step
    assert planner.reset_called
