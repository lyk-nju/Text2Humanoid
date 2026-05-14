from __future__ import annotations

import numpy as np

from text2humanoid.contracts.clips import G1ReferenceChunk


def assert_reference_chunk_sane(chunk: G1ReferenceChunk) -> None:
    for arr in (chunk.root_pos, chunk.root_rot, chunk.dof_pos, chunk.local_body_pos, chunk.local_body_rot):
        if not np.all(np.isfinite(arr)):
            raise AssertionError("reference chunk contains non-finite values")
