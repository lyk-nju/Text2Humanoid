from __future__ import annotations


class SyncManager:
    def __init__(self, control_hz: int = 50, future_horizon_frames: int = 16) -> None:
        self.control_hz = int(control_hz)
        self.future_horizon_frames = int(future_horizon_frames)

    def required_horizon(self) -> int:
        return self.future_horizon_frames

    def sim_time_from_frames(self, consumed_frames: int) -> float:
        return float(consumed_frames) / float(self.control_hz)
