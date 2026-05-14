from __future__ import annotations

from dataclasses import dataclass

from text2humanoid.contracts.commands import PromptCommand
from text2humanoid.contracts.chunks import HumanMotionChunk
from text2humanoid.planner.floodnet_service import FloodNetPlannerService


@dataclass(slots=True)
class StreamPlannerDriver:
    planner: FloodNetPlannerService
    next_start_time: float = 0.0

    def step(self, command: PromptCommand) -> HumanMotionChunk:
        chunk = self.planner.generate_chunk(command, start_time=self.next_start_time)
        self.next_start_time = chunk.end_time
        return chunk
