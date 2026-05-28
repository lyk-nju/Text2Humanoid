from __future__ import annotations

from pathlib import Path
import sys

_TEXT2HUMANOID_DIR = Path(__file__).resolve().parents[1]
if str(_TEXT2HUMANOID_DIR) not in sys.path:
    sys.path.insert(0, str(_TEXT2HUMANOID_DIR))

from tools.conversion import retarget_to_bfmzero as _impl

floodnet_263_to_nmr_140 = _impl.floodnet_263_to_nmr_140
NMRRetargetService = _impl.NMRRetargetService
make_tracking_easy_result_to_bfmzero_motion = _impl.make_tracking_easy_result_to_bfmzero_motion
BFMZeroZmqSink = _impl.BFMZeroZmqSink


def main(*args, **kwargs):
    _impl.floodnet_263_to_nmr_140 = floodnet_263_to_nmr_140
    _impl.NMRRetargetService = NMRRetargetService
    _impl.make_tracking_easy_result_to_bfmzero_motion = make_tracking_easy_result_to_bfmzero_motion
    _impl.BFMZeroZmqSink = BFMZeroZmqSink
    return _impl.main(*args, **kwargs)


if __name__ == "__main__":
    raise SystemExit(main())
