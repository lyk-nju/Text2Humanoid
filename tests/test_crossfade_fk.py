from __future__ import annotations

import numpy as np

from text2humanoid.contracts.clips import G1ReferenceChunk
from text2humanoid.infra.paths import get_make_tracking_easy_root
from text2humanoid.retarget.fk_features import (
    build_local_body_features,
    build_world_features,
)
from text2humanoid.runtime.reference_buffer import ReferenceBuffer


def _make_chunk(
    chunk_id: str,
    start: float,
    n: int,
    root_pos_offset: float = 0.0,
    dof_base: float = 0.0,
) -> G1ReferenceChunk:
    root_pos = np.zeros((n, 3), dtype=np.float32)
    root_pos[:, 0] = np.linspace(0.0, 1.0, n, dtype=np.float32) + root_pos_offset
    root_rot = np.tile(np.array([[0, 0, 0, 1]], dtype=np.float32), (n, 1))
    dof_pos = np.full((n, 29), dof_base, dtype=np.float32)
    body_pos_w, body_rot_w, body_names = build_world_features(
        root_pos, root_rot, dof_pos,
        str(get_make_tracking_easy_root() / "assets/g1_mocap_29dof.xml"),
    )
    local_body_pos, local_body_rot = build_local_body_features(
        root_pos, root_rot, body_pos_w, body_rot_w,
    )
    return G1ReferenceChunk(
        chunk_id=chunk_id,
        start_time=start,
        fps=30,
        root_pos=root_pos,
        root_rot=root_rot,
        dof_pos=dof_pos,
        local_body_pos=local_body_pos,
        local_body_rot=local_body_rot,
        body_names=body_names,
        joint_names=["j"] * 29,
    )


def test_crossfade_without_fk_recomputation():
    buf = ReferenceBuffer(xml_path=None)
    c1 = _make_chunk("a", 0.0, 8, root_pos_offset=0.0, dof_base=0.0)
    c2 = _make_chunk("b", 8 / 30.0, 8, root_pos_offset=1.0, dof_base=0.1)

    buf.append_chunk(c1)
    assert buf.buffer_frames == 8
    buf.append_chunk(c2, overlap_frames=3)
    # Without xml_path, local_body is concatenated (not FK-recomputed)
    assert buf.buffer_frames == 13
    horizon = buf.get_horizon(13)
    n_bodies = c1.local_body_pos.shape[1]
    assert horizon["local_body_pos"].shape == (13, n_bodies, 3)
    assert horizon["local_body_rot"].shape == (13, n_bodies, 4)


def test_crossfade_with_fk_recomputation():
    xml_path = str(get_make_tracking_easy_root() / "assets/g1_mocap_29dof.xml")
    buf = ReferenceBuffer(xml_path=xml_path)
    c1 = _make_chunk("a", 0.0, 8, root_pos_offset=0.0, dof_base=0.0)
    c2 = _make_chunk("b", 8 / 30.0, 8, root_pos_offset=1.0, dof_base=0.1)

    buf.append_chunk(c1)
    buf.append_chunk(c2, overlap_frames=3)

    # Verify FK consistency: recompute local_body from stored root/dof
    body_pos_w, body_rot_w, _ = build_world_features(
        buf._buffer.root_pos,
        buf._buffer.root_rot,
        buf._buffer.dof_pos,
        xml_path,
    )
    expected_local_pos, expected_local_rot = build_local_body_features(
        buf._buffer.root_pos,
        buf._buffer.root_rot,
        body_pos_w,
        body_rot_w,
    )

    assert np.allclose(buf._buffer.local_body_pos, expected_local_pos, atol=1e-5), (
        "local_body_pos is not FK-consistent with blended root/dof"
    )
    assert np.allclose(buf._buffer.local_body_rot, expected_local_rot, atol=1e-5), (
        "local_body_rot is not FK-consistent with blended root/dof"
    )


def test_crossfade_blended_root_values():
    """Verify that root/dof are actually blended in the overlap region."""
    xml_path = str(get_make_tracking_easy_root() / "assets/g1_mocap_29dof.xml")
    buf = ReferenceBuffer(xml_path=xml_path)
    c1 = _make_chunk("a", 0.0, 8, root_pos_offset=0.0, dof_base=0.0)
    c2 = _make_chunk("b", 8 / 30.0, 8, root_pos_offset=1.0, dof_base=0.5)

    buf.append_chunk(c1)
    # Record root_pos before blending (at overlap start)
    pre_blend_root = buf._buffer.root_pos[5:8].copy()
    buf.append_chunk(c2, overlap_frames=3)
    # After blending, overlap region (indices 5:8) should differ from pre_blend
    post_blend_root = buf._buffer.root_pos[5:8]
    # The blended values should be between c1 and c2 values
    assert not np.allclose(pre_blend_root, post_blend_root), (
        "root_pos was not blended in overlap"
    )
