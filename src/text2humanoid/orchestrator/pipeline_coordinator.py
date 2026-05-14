from __future__ import annotations

from dataclasses import dataclass
import time

from text2humanoid.contracts.commands import PromptCommand
from text2humanoid.contracts.status import RuntimeStatus, SessionPhase
from text2humanoid.planner.floodnet_service import FloodNetPlannerService
from text2humanoid.retarget.bridge_263_to_140 import human_chunk_to_nmr_input
from text2humanoid.retarget.g1_reference_adapter import G1ReferenceAdapter
from text2humanoid.retarget.nmr_service import NMRRetargetService
from text2humanoid.runtime.fallback_policy import FallbackPolicy
from text2humanoid.runtime.motion_tracking_client import MotionTrackingClient


@dataclass(slots=True)
class PipelineCoordinator:
    planner: FloodNetPlannerService
    retarget: NMRRetargetService
    adapter: G1ReferenceAdapter
    runtime: MotionTrackingClient
    fallback: FallbackPolicy

    def warmup(self, session_id: str, text: str) -> RuntimeStatus:
        t0 = time.perf_counter()
        self.planner.warmup(text)
        status = self.runtime.get_status(session_id)
        status.phase = SessionPhase.WARMING.value
        status.planner_latency_ms = (time.perf_counter() - t0) * 1000.0
        return status

    def run_once(self, session_id: str, command: PromptCommand, start_time: float) -> RuntimeStatus:
        t0 = time.perf_counter()
        human_chunk = self.planner.generate_chunk(command, start_time=start_time)
        t1 = time.perf_counter()
        nmr_chunk = human_chunk_to_nmr_input(human_chunk)
        result = self.retarget.retarget_chunk(nmr_chunk)
        t2 = time.perf_counter()
        ref_chunk = self.adapter.from_nmr_result(
            chunk_id=human_chunk.chunk_id,
            start_time=nmr_chunk.start_time,
            fps=self.retarget.output_fps,
            result=result,
        )
        self.runtime.push_reference_chunk(session_id, ref_chunk)
        t3 = time.perf_counter()
        status = self.runtime.get_status(session_id)
        status.phase = (
            SessionPhase.RUNNING.value if not self.fallback.should_degrade(status.buffer_frames) else SessionPhase.DEGRADED.value
        )
        status.latest_chunk_id = ref_chunk.chunk_id
        status.planner_latency_ms = (t1 - t0) * 1000.0
        status.retarget_latency_ms = (t2 - t1) * 1000.0
        status.runtime_latency_ms = (t3 - t2) * 1000.0
        status.metadata["latest_chunk_start_time"] = ref_chunk.start_time
        status.metadata["latest_chunk_end_time"] = ref_chunk.end_time
        status.metadata["latest_chunk_frames"] = ref_chunk.num_frames
        return status
