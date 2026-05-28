from __future__ import annotations

from typing import Any

from text2humanoid.contracts.bfmzero import (
    BFMZeroMotionChunk,
    bfmzero_motion_from_bmimic_data,
)
from text2humanoid.retarget.mte_imports import ensure_make_tracking_easy_paths


def attach_bfmzero_contract_metadata(
    chunk: BFMZeroMotionChunk,
    *,
    source_chunk_id: str,
    source_representation: str,
    source_fps: float,
    target_fps: float,
) -> BFMZeroMotionChunk:
    metadata = chunk.to_chunk_metadata()
    metadata.update(
        {
            "source_chunk_id": source_chunk_id,
            "source_representation": source_representation,
            "source_fps": float(source_fps),
            "target_fps": float(target_fps),
            "runtime_joint_order": metadata.get("runtime_joint_order", "isaac"),
            "root_quat_order": metadata.get("root_quat_order", "wxyz"),
        }
    )
    chunk.metadata.update(metadata)
    return chunk


def make_tracking_easy_result_to_bfmzero_motion(
    result: dict[str, Any],
    *,
    xml_path: str,
    device: str,
    chunk_id: str = "",
    frame_start: int = 0,
    src_fps: float = 30.0,
    tgt_fps: float = 50.0,
) -> BFMZeroMotionChunk:
    """Convert MakeTrackingEasy retarget output into BFM-Zero ZMQ motion.

    The intermediate MakeTrackingEasy bmimic converter is reused because it
    computes FK-consistent root/body velocities in the exact format BFM-Zero's
    file loader already supports.
    """

    ensure_make_tracking_easy_paths()
    from convert_bmimic import convert_to_bmimic

    bmimic = convert_to_bmimic(
        result,
        xml_path=xml_path,
        device=device,
        tgt_fps=tgt_fps,
        src_fps=src_fps,
    )
    chunk = bfmzero_motion_from_bmimic_data(
        bmimic,
        chunk_id=chunk_id,
        frame_start=frame_start,
        source_joint_order="bmimic",
    )
    return attach_bfmzero_contract_metadata(
        chunk,
        source_chunk_id=chunk_id,
        source_representation="nmr_smplx_140",
        source_fps=src_fps,
        target_fps=tgt_fps,
    )
