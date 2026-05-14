from __future__ import annotations

import sys
from pathlib import Path

_NMR_ROOT = Path("/home/yuankai/Text2Motion/MakeTrackingEasy")
_NMR_SRC = _NMR_ROOT / "src"


def ensure_make_tracking_easy_paths() -> None:
    for path in (str(_NMR_ROOT), str(_NMR_SRC)):
        if path not in sys.path:
            sys.path.insert(0, path)
