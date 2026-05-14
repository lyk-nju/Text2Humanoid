from __future__ import annotations

from typing import Any

import numpy as np
import torch

from text2humanoid.contracts.commands import PromptCommand, TrajectoryPoint
from text2humanoid.contracts.trajectory import CanonicalTrajectory, TrajectorySource, TrajectorySourceType

_FLOODNET_TOKEN_FPS = 5
_TRAJ_FEATURE_DIM = 4  # [x, z, cos(yaw), sin(yaw)]


def _derive_yaw(xz: np.ndarray) -> np.ndarray:
    """Derive yaw angle from xz velocity direction. First frame uses second frame's direction."""
    vel = np.zeros_like(xz)
    vel[1:] = xz[1:] - xz[:-1]
    vel[0] = vel[1]
    yaw = np.arctan2(vel[:, 1], vel[:, 0])
    return yaw.astype(np.float32)


def waypoints_to_canonical(
    waypoints: list[TrajectoryPoint],
    fps: int = 30,
    duration: float | None = None,
) -> CanonicalTrajectory:
    """Convert a list of TrajectoryPoints to a CanonicalTrajectory.

    Waypoints are sorted by time, then linearly interpolated onto a uniform
    grid at the given fps.  Yaw is derived from the velocity direction.
    """
    if not waypoints:
        return _empty_canonical(0.0, fps)

    sorted_pts = sorted(waypoints, key=lambda p: p.t)
    t_wp = np.array([p.t for p in sorted_pts], dtype=np.float32)
    x_wp = np.array([p.x for p in sorted_pts], dtype=np.float32)
    z_wp = np.array([p.z for p in sorted_pts], dtype=np.float32)

    t_start = float(t_wp[0])
    t_end = float(t_wp[-1]) if duration is None else t_start + float(duration)
    if t_end <= t_start:
        t_end = t_start + 1.0 / float(fps)

    n_frames = max(2, int(np.ceil((t_end - t_start) * float(fps))) + 1)
    times = np.linspace(t_start, t_end, n_frames, dtype=np.float32)

    x_interp = np.interp(times, t_wp, x_wp).astype(np.float32)
    z_interp = np.interp(times, t_wp, z_wp).astype(np.float32)
    xz = np.stack([x_interp, z_interp], axis=-1)

    yaw = _derive_yaw(xz)

    # valid where within waypoint time bounds
    valid_mask = np.ones(n_frames, dtype=bool)

    return CanonicalTrajectory(
        times=times,
        xz=xz,
        yaw=yaw,
        valid_mask=valid_mask,
        fps=int(fps),
        metadata={"source": "waypoints", "num_waypoints": len(waypoints)},
    )


def token_aligned_to_canonical(
    token_aligned_traj: list[list[float]],
    token_mask: list[float] | None = None,
    token_fps: int = _FLOODNET_TOKEN_FPS,
) -> CanonicalTrajectory:
    """Convert FloodNet's token_aligned_traj format to CanonicalTrajectory.

    token_aligned_traj: list of [x, z, cos(yaw), sin(yaw)] per token.
    """
    if not token_aligned_traj:
        return _empty_canonical(0.0, token_fps)

    arr = np.asarray(token_aligned_traj, dtype=np.float32)  # (T_token, 4)
    n_tokens = arr.shape[0]
    times = np.arange(n_tokens, dtype=np.float32) / float(token_fps)
    xz = arr[:, :2]
    cos_yaw = arr[:, 2]
    sin_yaw = arr[:, 3]
    yaw = np.arctan2(sin_yaw, cos_yaw).astype(np.float32)

    if token_mask is not None:
        valid_mask = np.asarray(token_mask, dtype=bool)
    else:
        valid_mask = np.ones(n_tokens, dtype=bool)

    return CanonicalTrajectory(
        times=times,
        xz=xz,
        yaw=yaw,
        valid_mask=valid_mask,
        fps=int(token_fps),
        metadata={"source": "token_aligned"},
    )


def canonical_to_floodnet_features(
    traj: CanonicalTrajectory,
    feature_length: int,
    token_fps: int = _FLOODNET_TOKEN_FPS,
) -> dict[str, Any]:
    """Compile a CanonicalTrajectory into FloodNet model input features.

    Returns a dict with keys:
      - traj_features: (1, feature_length, 4) tensor [x, z, cos(yaw), sin(yaw)]
      - token_length: tensor with number of valid trajectory tokens
      - token_mask: (1, feature_length) float tensor
    """
    # Resample canonical trajectory to token grid
    token_times = np.arange(feature_length, dtype=np.float32) / float(token_fps)

    if traj.num_frames == 0:
        return _empty_traj_payload(feature_length)

    x_interp = np.interp(token_times, traj.times, traj.xz[:, 0]).astype(np.float32)
    z_interp = np.interp(token_times, traj.times, traj.xz[:, 1]).astype(np.float32)
    yaw_interp = np.interp(token_times, traj.times, traj.yaw).astype(np.float32)

    traj_features = np.stack(
        [x_interp, z_interp, np.cos(yaw_interp), np.sin(yaw_interp)],
        axis=-1,
    ).astype(np.float32)

    token_mask = np.isin(token_times, traj.times).astype(np.float32) if traj.valid_mask.any() else np.zeros(feature_length, dtype=np.float32)

    return {
        "traj_features": torch.from_numpy(traj_features).unsqueeze(0),
        "token_length": torch.tensor([feature_length], dtype=torch.long),
        "token_mask": torch.from_numpy(token_mask).unsqueeze(0),
    }


def _empty_canonical(start_time: float, fps: int) -> CanonicalTrajectory:
    times = np.array([start_time, start_time + 1.0 / float(fps)], dtype=np.float32)
    return CanonicalTrajectory(
        times=times,
        xz=np.zeros((2, 2), dtype=np.float32),
        yaw=np.zeros(2, dtype=np.float32),
        valid_mask=np.zeros(2, dtype=bool),
        fps=int(fps),
        metadata={"source": "empty"},
    )


def _empty_traj_payload(feature_length: int) -> dict[str, Any]:
    return {
        "traj_features": torch.zeros((1, feature_length, _TRAJ_FEATURE_DIM), dtype=torch.float32),
        "token_length": torch.tensor([feature_length], dtype=torch.long),
        "token_mask": torch.zeros((1, feature_length), dtype=torch.float32),
    }


def trajectory_source_to_canonical(source: TrajectorySource) -> CanonicalTrajectory:
    """Convert a TrajectorySource to CanonicalTrajectory.

    This is the single dispatch point for all external trajectory sources.
    Every source type flows through this function before reaching FloodNet.
    """
    if source.source_type == TrajectorySourceType.CANONICAL.value and source.canonical is not None:
        return source.canonical

    if source.source_type == TrajectorySourceType.TOKEN_ALIGNED.value and source.token_aligned_traj is not None:
        return token_aligned_to_canonical(
            source.token_aligned_traj,
            source.token_mask,
        )

    if source.source_type == TrajectorySourceType.WAYPOINTS.value and source.waypoints:
        return waypoints_to_canonical(source.waypoints)

    # Empty source or unrecognized type — return empty canonical
    return _empty_canonical(0.0, 30)


def build_floodnet_model_input(command: PromptCommand, feature_length: int) -> dict[str, Any]:
    """Build FloodNet model input dict from a PromptCommand.

    The single pipeline is:
      TrajectoryCondition → TrajectorySource → CanonicalTrajectory → FloodNet

    TrajectoryCondition.to_source() converts the API-level fields (waypoints
    or token_aligned_traj) into a unified TrajectorySource.  The planner
    then compiles through canonical trajectory to FloodNet features.
    """
    payload: dict[str, Any] = {
        "feature_length": torch.tensor([feature_length], dtype=torch.long),
        "text": [[command.text]],
        "feature_text_end": [[feature_length]],
    }

    if command.trajectory is None:
        return payload

    source = command.trajectory.to_source()
    canonical = trajectory_source_to_canonical(source)
    traj_payload = canonical_to_floodnet_features(canonical, feature_length)
    payload.update(traj_payload)
    payload["trajectory_metadata"] = {
        "source_type": source.source_type,
        **command.trajectory.metadata,
        **canonical.metadata,
    }
    return payload
