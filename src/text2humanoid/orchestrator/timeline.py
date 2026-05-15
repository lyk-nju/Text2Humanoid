from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from text2humanoid.contracts.commands import PromptCommand


@dataclass(slots=True)
class TransitionRecord:
    """Records a command transition in a running session."""
    previous_command_id: str
    new_command_id: str
    transition_time: float = 0.0
    boundary_chunk_index: int = -1
    transition_mode: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "previous_command_id": self.previous_command_id,
            "new_command_id": self.new_command_id,
            "transition_time": self.transition_time,
            "boundary_chunk_index": self.boundary_chunk_index,
            "transition_mode": self.transition_mode,
            "metadata": self.metadata,
        }


@dataclass(slots=True)
class SessionTimeline:
    session_id: str
    commands: list[PromptCommand] = field(default_factory=list)
    transitions: list[TransitionRecord] = field(default_factory=list)
    sim_time: float = 0.0

    @property
    def active_command(self) -> PromptCommand | None:
        return self.commands[-1] if self.commands else None

    @property
    def latest_command(self) -> PromptCommand | None:
        return self.active_command

    @property
    def active_command_id(self) -> str:
        cmd = self.active_command
        return cmd.command_id if cmd else ""

    def append(self, command: PromptCommand) -> TransitionRecord | None:
        prev = self.active_command
        self.commands.append(command)
        if prev is not None and prev.command_id != command.command_id:
            rec = TransitionRecord(
                previous_command_id=prev.command_id,
                new_command_id=command.command_id,
                transition_mode=command.transition_mode,
            )
            self.transitions.append(rec)
            return rec
        return None

    @property
    def last_transition(self) -> TransitionRecord | None:
        return self.transitions[-1] if self.transitions else None

    @property
    def pending_transition(self) -> bool:
        if not self.transitions:
            return False
        last = self.transitions[-1]
        return last.boundary_chunk_index < 0
