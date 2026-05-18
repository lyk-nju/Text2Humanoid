import numpy as np
import torch

from text2humanoid.retarget.bridge_263_to_140 import assemble_nmr_motion


def test_assemble_nmr_motion_shape():
    joint_pos = torch.zeros((8, 22, 3), dtype=torch.float32)
    joint_pos[:, :, 1] = 1.0
    root_quat = torch.zeros((8, 4), dtype=torch.float32)
    root_quat[:, 0] = 1.0
    out = assemble_nmr_motion(joint_pos, root_quat, src_fps=20.0, tgt_fps=30.0)
    assert out.shape[1] == 140
    assert out.shape[0] == 12
    assert np.isfinite(out.numpy()).all()


def test_assemble_nmr_motion_uses_yaw_quaternion_half_angle():
    joint_pos = torch.zeros((2, 22, 3), dtype=torch.float32)
    joint_pos[:, :, 1] = 1.0
    root_quat = torch.zeros((2, 4), dtype=torch.float32)
    half = np.pi / 4.0
    root_quat[:, 0] = float(np.cos(half))
    root_quat[:, 2] = float(np.sin(half))

    out = assemble_nmr_motion(joint_pos, root_quat, src_fps=20.0, tgt_fps=20.0)

    expected_first_two_rows = np.array([0.0, 0.0, 1.0, 0.0, 1.0, 0.0], dtype=np.float32)
    assert np.allclose(out[0, 2:8].numpy(), expected_first_two_rows, atol=1e-5)
