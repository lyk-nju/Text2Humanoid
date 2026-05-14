import numpy as np

from text2humanoid.contracts.clips import G1ReferenceChunk
from text2humanoid.runtime.reference_buffer import ReferenceBuffer


def _chunk(chunk_id: str, start: float, n: int) -> G1ReferenceChunk:
    return G1ReferenceChunk(
        chunk_id=chunk_id,
        start_time=start,
        fps=30,
        root_pos=np.zeros((n, 3), dtype=np.float32),
        root_rot=np.tile(np.array([[0, 0, 0, 1]], dtype=np.float32), (n, 1)),
        dof_pos=np.zeros((n, 29), dtype=np.float32),
        local_body_pos=np.zeros((n, 30, 3), dtype=np.float32),
        local_body_rot=np.tile(np.array([[[0, 0, 0, 1]]], dtype=np.float32), (n, 30, 1)),
        body_names=["pelvis"] * 30,
        joint_names=["j"] * 29,
    )


def test_reference_buffer_append_and_advance():
    buf = ReferenceBuffer()
    buf.append_chunk(_chunk("a", 0.0, 10))
    assert buf.buffer_frames == 10
    horizon = buf.get_horizon(4)
    assert horizon["dof_pos"].shape == (4, 29)
    buf.advance(3)
    assert buf.buffer_frames == 7
    buf.append_chunk(_chunk("b", 10 / 30.0, 10), overlap_frames=2)
    assert buf.latest_chunk_id == "b"
