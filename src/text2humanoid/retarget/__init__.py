from __future__ import annotations

from typing import Any

__all__ = ["G1ReferenceAdapter", "NMRRetargetService", "floodnet_263_to_nmr_140"]


def __getattr__(name: str) -> Any:
    if name == "floodnet_263_to_nmr_140":
        from .bridge_263_to_140 import floodnet_263_to_nmr_140

        return floodnet_263_to_nmr_140
    if name == "G1ReferenceAdapter":
        from .g1_reference_adapter import G1ReferenceAdapter

        return G1ReferenceAdapter
    if name == "NMRRetargetService":
        from .nmr_service import NMRRetargetService

        return NMRRetargetService
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
