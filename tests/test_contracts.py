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
