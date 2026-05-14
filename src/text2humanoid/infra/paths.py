from __future__ import annotations

import os
from pathlib import Path


def _detect_root() -> Path:
    env = os.environ.get("TEXT2MOTION_ROOT")
    if env:
        return Path(env).expanduser().resolve()
    return Path(__file__).resolve().parent.parent.parent.parent.parent


_root: Path | None = None


def set_root(path: str | Path) -> None:
    global _root
    _root = Path(path).expanduser().resolve()


def get_root() -> Path:
    global _root
    if _root is None:
        _root = _detect_root()
    return _root


def get_floodnet_root() -> Path:
    return get_root() / "FloodNet"


def get_make_tracking_easy_root() -> Path:
    return get_root() / "MakeTrackingEasy"


def get_motion_tracking_root() -> Path:
    return get_root() / "motion_tracking"
