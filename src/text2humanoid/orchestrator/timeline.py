from __future__ import annotations

from dataclasses import dataclass, field

from text2humanoid.contracts.commands import PromptCommand


@dataclass(slots=True)
class SessionTimeline:
    session_id: str
    commands: list[PromptCommand] = field(default_factory=list)
    sim_time: float = 0.0

    def append(self, command: PromptCommand) -> None:
        self.commands.append(command)

    @property
    def latest_command(self) -> PromptCommand | None:
        return self.commands[-1] if self.commands else None
