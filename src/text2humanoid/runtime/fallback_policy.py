from __future__ import annotations


class FallbackPolicy:
    def __init__(self, low_watermark_frames: int = 20, high_watermark_frames: int = 60) -> None:
        self.low_watermark_frames = int(low_watermark_frames)
        self.high_watermark_frames = int(high_watermark_frames)

    def should_degrade(self, buffer_frames: int) -> bool:
        return int(buffer_frames) < self.low_watermark_frames

    def is_recovered(self, buffer_frames: int) -> bool:
        return int(buffer_frames) >= self.high_watermark_frames
