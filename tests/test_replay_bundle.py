from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np

from text2humanoid.contracts.clips import G1ReferenceChunk
from text2humanoid.evaluation.replay_checks import (
    assert_reference_chunk_sane,
    assert_reference_metadata_complete,
    assert_reference_shapes_stable,
    validate_exported_bundle,
    validate_reference_chunk,
)
from text2humanoid.infra.artifact_store import ArtifactStore


def _make_reference(n: int = 6) -> G1ReferenceChunk:
    return G1ReferenceChunk(
        chunk_id="test",
        start_time=0.0,
        fps=30,
        root_pos=np.zeros((n, 3), dtype=np.float32),
        root_rot=np.tile(np.array([[0, 0, 0, 1]], dtype=np.float32), (n, 1)),
        dof_pos=np.zeros((n, 29), dtype=np.float32),
        local_body_pos=np.zeros((n, 30, 3), dtype=np.float32),
        local_body_rot=np.tile(np.array([[[0, 0, 0, 1]]], dtype=np.float32), (n, 30, 1)),
        body_names=["pelvis"] + [f"link_{i}" for i in range(29)],
        joint_names=[f"joint_{i}" for i in range(29)],
        metadata={"root_quat_order": "xyzw", "body_rot_order": "xyzw"},
    )


# ---- 005.3: reference chunk validation --------------------------------------

def test_reference_chunk_sane():
    ref = _make_reference()
    assert_reference_chunk_sane(ref)


def test_reference_chunk_nan_detected():
    ref = _make_reference()
    ref.root_pos[0, 0] = np.nan
    errors = validate_reference_chunk(ref)
    assert any("non-finite" in e for e in errors)


def test_reference_shapes_stable():
    ref = _make_reference(8)
    assert_reference_shapes_stable(ref)


def test_reference_metadata_complete():
    ref = _make_reference()
    assert_reference_metadata_complete(ref)


def test_reference_metadata_missing_quat_order():
    ref = _make_reference()
    ref.metadata.pop("root_quat_order", None)
    errors = validate_reference_chunk(ref)
    assert any("root_quat_order" in e for e in errors)


def test_validate_reference_chunk_clean():
    ref = _make_reference()
    errors = validate_reference_chunk(ref)
    assert len(errors) == 0


# ---- 005.2: artifact bundle export -------------------------------------------

def test_artifact_store_export_replay_bundle():
    with tempfile.TemporaryDirectory() as tmp:
        store = ArtifactStore(tmp)
        ref = _make_reference(6)
        export_dir = store.export_replay_bundle(
            replay_id="test_replay",
            command={"text": "walk", "trajectory": {"waypoints": [[0, 0, 0, 0], [1, 1, 0, 0]]}},
            human_motion_263=np.zeros((8, 263), dtype=np.float32),
            human_fps=20,
            reference_chunk=ref,
            pipeline_timing={"planner_ms": 100, "retarget_ms": 50, "adapter_ms": 10},
            metadata={"source_type": "waypoints"},
        )
        assert (export_dir / "command.json").exists()
        assert (export_dir / "human_chunk.npz").exists()
        assert (export_dir / "reference_chunk.npz").exists()
        assert (export_dir / "metadata.json").exists()


def test_validate_exported_bundle_clean():
    with tempfile.TemporaryDirectory() as tmp:
        store = ArtifactStore(tmp)
        ref = _make_reference(6)
        export_dir = store.export_replay_bundle(
            replay_id="test_validate",
            command={"text": "walk"},
            human_motion_263=np.zeros((8, 263), dtype=np.float32),
            human_fps=20,
            reference_chunk=ref,
            pipeline_timing={"planner_ms": 100, "retarget_ms": 50, "adapter_ms": 10},
        )
        errors = validate_exported_bundle(export_dir)
        assert len(errors) == 0, f"errors: {errors}"


def test_validate_exported_bundle_missing_file():
    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp) / "incomplete"
        d.mkdir()
        (d / "command.json").write_text("{}")
        errors = validate_exported_bundle(d)
        assert len(errors) > 0


def test_exported_metadata_contains_key_fields():
    with tempfile.TemporaryDirectory() as tmp:
        store = ArtifactStore(tmp)
        ref = _make_reference(6)
        export_dir = store.export_replay_bundle(
            replay_id="test_meta",
            command={"text": "walk"},
            human_motion_263=np.zeros((8, 263), dtype=np.float32),
            human_fps=20,
            reference_chunk=ref,
            pipeline_timing={"planner_ms": 100, "retarget_ms": 50, "adapter_ms": 10},
            metadata={"source_type": "waypoints", "trajectory_source": "test"},
        )
        meta = json.loads(open(export_dir / "metadata.json").read())
        assert meta["human_shape"] == [8, 263]
        assert meta["reference_fps"] == 30
        assert len(meta["body_names"]) == 30
        assert len(meta["joint_names"]) == 29
        assert meta["timing"]["planner_ms"] == 100
        assert meta["metadata"]["source_type"] == "waypoints"
