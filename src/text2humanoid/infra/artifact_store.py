from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from text2humanoid.contracts.status import RuntimeStatus


class ArtifactStore:
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def session_dir(self, session_id: str) -> Path:
        out = self.root / session_id
        out.mkdir(parents=True, exist_ok=True)
        return out

    def save_json(self, session_id: str, name: str, payload: dict[str, Any]) -> Path:
        path = self.session_dir(session_id) / name
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        return path

    def save_npz(self, session_id: str, name: str, **arrays: np.ndarray) -> Path:
        path = self.session_dir(session_id) / name
        np.savez(path, **arrays)
        return path

    def export_status_bundle(self, session_id: str, status: RuntimeStatus) -> Path:
        return self.save_json(session_id, "status.json", status.to_dict())
