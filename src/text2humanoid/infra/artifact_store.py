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

    def replay_dir(self, replay_id: str) -> Path:
        out = self.root / "replays" / replay_id
        out.mkdir(parents=True, exist_ok=True)
        return out

    def save_json(self, session_id: str, name: str, payload: dict[str, Any]) -> Path:
        path = self.session_dir(session_id) / name
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, default=str)
        return path

    def save_npz(self, session_id: str, name: str, **arrays: np.ndarray) -> Path:
        path = self.session_dir(session_id) / name
        np.savez(path, **arrays)
        return path

    def save_reference_npz(self, session_id: str, name: str, ref: Any) -> Path:
        from text2humanoid.runtime.source_protocol import chunk_to_runtime_dict

        payload = chunk_to_runtime_dict(ref)
        arrays = {}
        for k, v in payload.items():
            if isinstance(v, np.ndarray):
                arrays[k] = v
        return self.save_npz(session_id, name, **arrays)

    def export_status_bundle(self, session_id: str, status: RuntimeStatus) -> Path:
        return self.save_json(session_id, "status.json", status.to_dict())

    def export_replay_bundle(
        self,
        replay_id: str,
        command: dict[str, Any],
        human_motion_263: np.ndarray,
        human_fps: int,
        reference_chunk: Any,
        pipeline_timing: dict[str, float],
        metadata: dict[str, Any] | None = None,
    ) -> Path:
        from text2humanoid.runtime.source_protocol import chunk_to_runtime_dict

        d = self.replay_dir(replay_id)
        with open(d / "command.json", "w", encoding="utf-8") as f:
            json.dump(command, f, indent=2, default=str)
        np.savez(d / "human_chunk.npz", motion_263=human_motion_263, fps=np.array([human_fps]))
        ref_payload = chunk_to_runtime_dict(reference_chunk)
        ref_arrays = {k: v for k, v in ref_payload.items() if isinstance(v, np.ndarray)}
        np.savez(d / "reference_chunk.npz", **ref_arrays)
        bundle_meta = {
            "replay_id": replay_id,
            "human_shape": list(human_motion_263.shape),
            "human_fps": human_fps,
            "reference_fps": reference_chunk.fps,
            "reference_frames": reference_chunk.num_frames,
            "dof_shape": list(reference_chunk.dof_pos.shape),
            "local_body_shape": list(reference_chunk.local_body_pos.shape),
            "body_names": reference_chunk.body_names,
            "joint_names": reference_chunk.joint_names,
            "timing": pipeline_timing,
        }
        if metadata:
            bundle_meta["metadata"] = metadata
        with open(d / "metadata.json", "w", encoding="utf-8") as f:
            json.dump(bundle_meta, f, indent=2, default=str)
        return d
