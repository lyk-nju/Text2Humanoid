from __future__ import annotations

from text2humanoid.contracts.commands import PromptCommand


def resolve_chunk_frames(command: PromptCommand, default_chunk_frames: int) -> int:
    requested = command.metadata.get("chunk_frames", default_chunk_frames)
    return max(4, int(requested))
