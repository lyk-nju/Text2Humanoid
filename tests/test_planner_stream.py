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

    def generate_chunk(self, command, start_time, feature_length=None):
        from text2humanoid.contracts.chunks import HumanMotionChunk
        return HumanMotionChunk(
            chunk_id="mock", start_time=start_time, fps=20,
            motion_263=np.zeros((4, 263), dtype=np.float32),
            text=command.text, metadata={},
        )

    def reset(self):
        self.reset_called = True


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
    assert planner.reset_called


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
