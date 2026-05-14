from __future__ import annotations


def buffer_watermark_ratio(buffer_frames: int, target_frames: int) -> float:
    if target_frames <= 0:
        return 0.0
    return float(buffer_frames) / float(target_frames)
