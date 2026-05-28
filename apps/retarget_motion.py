from __future__ import annotations

from pathlib import Path
import sys

_TEXT2HUMANOID_DIR = Path(__file__).resolve().parents[1]
if str(_TEXT2HUMANOID_DIR) not in sys.path:
    sys.path.insert(0, str(_TEXT2HUMANOID_DIR))

from tools.conversion.retarget_motion import *  # noqa: F401,F403
from tools.conversion.retarget_motion import main


if __name__ == "__main__":
    raise SystemExit(main())
