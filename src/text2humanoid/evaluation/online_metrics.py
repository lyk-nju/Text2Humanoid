from __future__ import annotations

from text2humanoid.contracts.status import RuntimeStatus


def summarize_runtime_status(status: RuntimeStatus) -> dict[str, float | int | str]:
    return {
        "phase": status.phase,
        "buffer_frames": status.buffer_frames,
        "sim_time": status.sim_time,
        "falls": status.falls,
        "planner_latency_ms": status.planner_latency_ms,
        "retarget_latency_ms": status.retarget_latency_ms,
        "runtime_latency_ms": status.runtime_latency_ms,
    }
