import numpy as np

from text2humanoid.contracts.clips import G1ReferenceChunk
from text2humanoid.runtime.motion_tracking_client import MotionTrackingClient


def test_runtime_client_accepts_reference_chunk():
    client = MotionTrackingClient(control_hz=50, future_horizon_frames=16)
    chunk = G1ReferenceChunk(
        chunk_id="demo",
        start_time=0.0,
        fps=30,
        root_pos=np.zeros((6, 3), dtype=np.float32),
        root_rot=np.tile(np.array([[0, 0, 0, 1]], dtype=np.float32), (6, 1)),
        dof_pos=np.zeros((6, 29), dtype=np.float32),
        local_body_pos=np.zeros((6, 30, 3), dtype=np.float32),
        local_body_rot=np.tile(np.array([[[0, 0, 0, 1]]], dtype=np.float32), (6, 30, 1)),
        body_names=["pelvis"] * 30,
        joint_names=["j"] * 29,
    )
    client.push_reference_chunk("s1", chunk)
    status = client.get_status("s1")
    assert status.buffer_frames == 6
    client.consume_step("s1", frames=2)
    assert client.get_status("s1").buffer_frames == 4
