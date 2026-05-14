import numpy as np

from text2humanoid.retarget.g1_reference_adapter import G1ReferenceAdapter


def test_reference_adapter_builds_chunk():
    adapter = G1ReferenceAdapter()
    result = {
        "dof": np.zeros((5, 29), dtype=np.float32),
        "root_trans": np.zeros((5, 3), dtype=np.float32),
        "root_rot_quat": np.tile(np.array([[1, 0, 0, 0]], dtype=np.float32), (5, 1)),
    }
    chunk = adapter.from_nmr_result("x", 0.0, 30, result)
    assert chunk.dof_pos.shape == (5, 29)
    assert chunk.root_rot.shape == (5, 4)
    assert chunk.local_body_pos.shape[0] == 5
    assert len(chunk.joint_names) == 29
