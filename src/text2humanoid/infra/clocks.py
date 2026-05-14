from __future__ import annotations

import time


def wall_time() -> float:
    return time.time()


def monotonic_time() -> float:
    return time.monotonic()
