from __future__ import annotations

import sys

from text2humanoid.infra.paths import get_make_tracking_easy_root


def ensure_make_tracking_easy_paths() -> None:
    root = get_make_tracking_easy_root()
    for path in (str(root), str(root / "src")):
        if path not in sys.path:
            sys.path.insert(0, path)
