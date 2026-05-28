from __future__ import annotations

from pathlib import Path
import sys

_TEXT2HUMANOID_DIR = Path(__file__).resolve().parents[1]
if str(_TEXT2HUMANOID_DIR) not in sys.path:
    sys.path.insert(0, str(_TEXT2HUMANOID_DIR))

from tools.replay import replay_trajectory as _impl

PromptCommand = _impl.PromptCommand
TrajectoryCondition = _impl.TrajectoryCondition
TrajectoryPoint = _impl.TrajectoryPoint
human_chunk_to_nmr_input = _impl.human_chunk_to_nmr_input


def run_replay_pipeline(*args, **kwargs):
    _impl.human_chunk_to_nmr_input = human_chunk_to_nmr_input
    return _impl.run_replay_pipeline(*args, **kwargs)


def main() -> None:
    return _impl.main()


if __name__ == "__main__":
    raise SystemExit(main())
