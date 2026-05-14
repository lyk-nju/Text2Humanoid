from __future__ import annotations

from pathlib import Path

import numpy as np

from text2humanoid.contracts.clips import G1ReferenceChunk


def assert_reference_chunk_sane(chunk: G1ReferenceChunk) -> None:
    for arr in (chunk.root_pos, chunk.root_rot, chunk.dof_pos, chunk.local_body_pos, chunk.local_body_rot):
        if not np.all(np.isfinite(arr)):
            raise AssertionError("reference chunk contains non-finite values")


def assert_reference_shapes_stable(chunk: G1ReferenceChunk) -> None:
    n = chunk.num_frames
    if chunk.root_pos.shape != (n, 3):
        raise AssertionError(f"root_pos shape {chunk.root_pos.shape} != ({n}, 3)")
    if chunk.root_rot.shape != (n, 4):
        raise AssertionError(f"root_rot shape {chunk.root_rot.shape} != ({n}, 4)")
    if chunk.dof_pos.shape[0] != n:
        raise AssertionError(f"dof_pos time dim {chunk.dof_pos.shape[0]} != {n}")
    if chunk.local_body_pos.shape[0] != n:
        raise AssertionError(f"local_body_pos time dim {chunk.local_body_pos.shape[0]} != {n}")
    if chunk.local_body_rot.shape[0] != n:
        raise AssertionError(f"local_body_rot time dim {chunk.local_body_rot.shape[0]} != {n}")


def assert_reference_metadata_complete(chunk: G1ReferenceChunk) -> None:
    if not chunk.joint_names or len(chunk.joint_names) == 0:
        raise AssertionError("joint_names is empty")
    if not chunk.body_names or len(chunk.body_names) == 0:
        raise AssertionError("body_names is empty")
    if "root_quat_order" not in chunk.metadata:
        raise AssertionError("metadata missing root_quat_order")


def validate_reference_chunk(chunk: G1ReferenceChunk) -> list[str]:
    """Run all reference chunk sanity checks. Returns list of error messages."""
    errors: list[str] = []
    try:
        assert_reference_chunk_sane(chunk)
    except AssertionError as e:
        errors.append(str(e))
    try:
        assert_reference_shapes_stable(chunk)
    except AssertionError as e:
        errors.append(str(e))
    try:
        assert_reference_metadata_complete(chunk)
    except AssertionError as e:
        errors.append(str(e))
    return errors


def validate_exported_bundle(export_dir: str | Path) -> list[str]:
    """Validate an exported replay bundle on disk."""
    errors: list[str] = []
    d = Path(export_dir)
    expected = ["command.json", "human_chunk.npz", "reference_chunk.npz", "metadata.json"]
    for name in expected:
        if not (d / name).exists():
            errors.append(f"missing file: {name}")

    if (d / "metadata.json").exists():
        import json
        meta = json.loads(open(d / "metadata.json", encoding="utf-8").read())
        for key in ["human_shape", "reference_fps", "dof_shape", "body_names", "joint_names"]:
            if key not in meta:
                errors.append(f"metadata.json missing key: {key}")

    if (d / "reference_chunk.npz").exists():
        data = np.load(d / "reference_chunk.npz", allow_pickle=True)
        for key in ["root_pos", "root_rot", "dof_pos", "local_body_pos", "local_body_rot"]:
            if key not in data.files:
                errors.append(f"reference_chunk.npz missing array: {key}")
            else:
                arr = data[key]
                if not np.all(np.isfinite(arr)):
                    errors.append(f"reference_chunk.npz {key} contains non-finite values")

    return errors
