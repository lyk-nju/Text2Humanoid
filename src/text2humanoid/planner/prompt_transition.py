"""Transition mode semantics for online command switching.

APPEND    — new command takes effect after current chunk completes.
REPLACE   — new command replaces active command immediately.
CROSSFADE — reserved; currently behaves like APPEND.
"""

from __future__ import annotations

from text2humanoid.contracts.commands import PromptCommand, TransitionMode


def resolve_chunk_frames(command: PromptCommand, default_chunk_frames: int) -> int:
    requested = command.metadata.get("chunk_frames", default_chunk_frames)
    return max(4, int(requested))


def should_replace_immediately(mode: str) -> bool:
    return mode == TransitionMode.REPLACE.value


def should_append_after_current(mode: str) -> bool:
    return mode in (TransitionMode.APPEND.value, TransitionMode.CROSSFADE.value)
