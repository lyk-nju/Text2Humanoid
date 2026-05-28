import numpy as np

from text2humanoid.contracts.chunks import HumanMotionChunk, NMRInputChunk
from text2humanoid.contracts.clips import G1ReferenceChunk
from text2humanoid.contracts.commands import PromptCommand, TrajectoryCondition, TrajectoryPoint


def test_command_to_dict():
    cmd = PromptCommand(
        text="walk",
        trajectory=TrajectoryCondition(waypoints=[TrajectoryPoint(t=0.0, x=0.0, y=0.0, z=0.0)]),
    )
    data = cmd.to_dict()
    assert data["text"] == "walk"
    assert data["trajectory"]["waypoints"][0]["x"] == 0.0


def test_human_motion_chunk_to_generated_motion_preserves_fields():
    from text2humanoid.contracts.pipeline import (
        GeneratedMotion,
        human_motion_chunk_to_generated_motion,
    )

    legacy = HumanMotionChunk(
        chunk_id="walk_0",
        start_time=1.5,
        fps=20,
        motion_263=np.full((6, 263), 0.5, dtype=np.float32),
        text="walk forward",
        metadata={"prompt": "walk forward", "extra": "x"},
    )
    motion = human_motion_chunk_to_generated_motion(legacy)
    assert isinstance(motion, GeneratedMotion)
    assert motion.motion_id == "walk_0"
    assert motion.representation == "humanml3d_263"
    assert motion.motion.shape == (6, 263)
    assert motion.fps == 20
    assert motion.start_time == 1.5
    assert motion.metadata["extra"] == "x"


def test_nmr_input_chunk_to_retarget_input_propagates_source_id():
    from text2humanoid.contracts.pipeline import (
        RetargetInput,
        nmr_input_chunk_to_retarget_input,
    )

    legacy = NMRInputChunk(
        chunk_id="walk_0_nmr",
        start_time=1.5,
        fps=30,
        motion_140=np.zeros((9, 140), dtype=np.float32),
        metadata={"source_chunk_id": "walk_0"},
    )
    ri = nmr_input_chunk_to_retarget_input(legacy)
    assert isinstance(ri, RetargetInput)
    assert ri.input_id == "walk_0_nmr"
    assert ri.source_motion_id == "walk_0"
    assert ri.motion.shape == (9, 140)
    assert ri.fps == 30


def test_chunk_shapes():
    human = HumanMotionChunk("a", 0.0, 20, np.zeros((4, 263), dtype=np.float32), "walk")
    nmr = NMRInputChunk("b", 0.0, 30, np.zeros((6, 140), dtype=np.float32))
    ref = G1ReferenceChunk(
        "c",
        0.0,
        30,
        np.zeros((6, 3), dtype=np.float32),
        np.tile(np.array([[0, 0, 0, 1]], dtype=np.float32), (6, 1)),
        np.zeros((6, 29), dtype=np.float32),
        np.zeros((6, 30, 3), dtype=np.float32),
        np.tile(np.array([[[0, 0, 0, 1]]], dtype=np.float32), (6, 30, 1)),
        ["pelvis"] * 30,
        ["j"] * 29,
    )
    assert human.num_frames == 4
    assert nmr.num_frames == 6
    assert ref.num_frames == 6
