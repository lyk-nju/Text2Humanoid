from __future__ import annotations

from typing import Any

import torch

from text2humanoid.contracts.commands import PromptCommand


def build_floodnet_model_input(command: PromptCommand, feature_length: int) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "feature_length": torch.tensor([feature_length], dtype=torch.long),
        "text": [[command.text]],
        "feature_text_end": [[feature_length]],
    }
    if command.trajectory is None:
        return payload

    if command.trajectory.token_aligned_traj is not None:
        traj = torch.tensor(command.trajectory.token_aligned_traj, dtype=torch.float32).unsqueeze(0)
        payload["traj_features"] = traj
        payload["token_length"] = torch.tensor([traj.shape[1]], dtype=torch.long)
    if command.trajectory.token_mask is not None:
        payload["token_mask"] = torch.tensor(command.trajectory.token_mask, dtype=torch.float32).unsqueeze(0)
    payload["trajectory_metadata"] = command.trajectory.metadata
    return payload
