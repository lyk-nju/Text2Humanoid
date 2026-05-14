from __future__ import annotations

import numpy as np

from text2humanoid.contracts.commands import PromptCommand, TrajectoryCondition, TrajectoryPoint
from text2humanoid.contracts.trajectory import (
    CanonicalTrajectory,
    TrajectorySource,
    TrajectorySourceType,
)
from text2humanoid.planner.traj_conditioning import (
    build_floodnet_model_input,
    trajectory_source_to_canonical,
    waypoints_to_canonical,
)


def _wp(t: float, x: float, y: float, z: float) -> TrajectoryPoint:
    return TrajectoryPoint(t=t, x=x, y=y, z=z)


# ---- 004.1: TrajectorySource contract ---------------------------------------

def test_trajectory_source_waypoints():
    src = TrajectorySource(
        source_type=TrajectorySourceType.WAYPOINTS.value,
        waypoints=[_wp(0, 0, 0, 0), _wp(1, 1, 0, 1)],
    )
    assert src.source_type == "waypoints"
    assert len(src.waypoints) == 2


def test_trajectory_source_token_aligned():
    src = TrajectorySource(
        source_type=TrajectorySourceType.TOKEN_ALIGNED.value,
        token_aligned_traj=[[0, 0, 1, 0], [1, 0, 1, 0]],
        token_mask=[1.0, 0.5],
    )
    assert src.source_type == "token_aligned"
    assert src.token_aligned_traj is not None


def test_trajectory_source_canonical():
    canonical = waypoints_to_canonical([_wp(0, 0, 0, 0), _wp(1, 1, 0, 1)], fps=10)
    src = TrajectorySource(
        source_type=TrajectorySourceType.CANONICAL.value,
        canonical=canonical,
    )
    assert src.source_type == "canonical"
    assert src.canonical is not None


# ---- 004.1: TrajectoryCondition.to_source() ---------------------------------

def test_condition_to_source_waypoints():
    cond = TrajectoryCondition(waypoints=[_wp(0, 0, 0, 0), _wp(2, 3, 0, 4)])
    src = cond.to_source()
    assert src.source_type == "waypoints"
    assert len(src.waypoints) == 2


def test_condition_to_source_token_aligned():
    cond = TrajectoryCondition(
        token_aligned_traj=[[0, 0, 1, 0]],
        token_mask=[1.0],
    )
    src = cond.to_source()
    assert src.source_type == "token_aligned"
    assert src.token_aligned_traj is not None


def test_condition_to_source_empty():
    cond = TrajectoryCondition()
    src = cond.to_source()
    assert src.source_type == "waypoints"
    assert src.waypoints == []


# ---- 004.2: trajectory_source_to_canonical() ---------------------------------

def test_source_to_canonical_waypoints():
    src = TrajectorySource(
        source_type=TrajectorySourceType.WAYPOINTS.value,
        waypoints=[_wp(0, 0, 0, 0), _wp(1, 1, 0, 1)],
    )
    canonical = trajectory_source_to_canonical(src)
    assert isinstance(canonical, CanonicalTrajectory)
    assert canonical.metadata.get("source") == "waypoints"


def test_source_to_canonical_token_aligned():
    src = TrajectorySource(
        source_type=TrajectorySourceType.TOKEN_ALIGNED.value,
        token_aligned_traj=[[0, 0, 1, 0], [0.5, 0, 1, 0]],
        token_mask=[1.0, 1.0],
    )
    canonical = trajectory_source_to_canonical(src)
    assert isinstance(canonical, CanonicalTrajectory)
    assert canonical.metadata.get("source") == "token_aligned"


def test_source_to_canonical_canonical_pass_through():
    original = waypoints_to_canonical([_wp(0, 0, 0, 0), _wp(1, 1, 0, 1)], fps=10)
    src = TrajectorySource(
        source_type=TrajectorySourceType.CANONICAL.value,
        canonical=original,
    )
    result = trajectory_source_to_canonical(src)
    assert result is original  # pass-through, not a copy


# ---- 004.2/3: build_floodnet_model_input uses unified pipeline ---------------

def test_build_input_waypoints_via_source():
    cmd = PromptCommand(
        text="walk",
        trajectory=TrajectoryCondition(
            waypoints=[_wp(0, 0, 0, 0), _wp(2, 3, 0, 4)]
        ),
    )
    payload = build_floodnet_model_input(cmd, feature_length=20)
    assert "traj_features" in payload
    assert payload["traj_features"].shape[2] == 4
    meta = payload.get("trajectory_metadata", {})
    assert "source_type" in meta


def test_build_input_token_aligned_via_source():
    cmd = PromptCommand(
        text="walk",
        trajectory=TrajectoryCondition(
            token_aligned_traj=[[0, 0, 1, 0], [0.5, 0, 1, 0]],
            token_mask=[1.0, 0.5],
        ),
    )
    payload = build_floodnet_model_input(cmd, feature_length=2)
    assert "traj_features" in payload
    meta = payload.get("trajectory_metadata", {})
    assert meta.get("source_type") == "token_aligned"


def test_build_input_does_not_depend_on_specific_field_names():
    """The planner should not care which field the source came from."""
    cmd_waypoints = PromptCommand(
        text="walk",
        trajectory=TrajectoryCondition(
            waypoints=[_wp(0, 0, 0, 0), _wp(1, 1, 0, 0)]
        ),
    )
    payload_wp = build_floodnet_model_input(cmd_waypoints, feature_length=10)

    cmd_tokens = PromptCommand(
        text="walk",
        trajectory=TrajectoryCondition(
            token_aligned_traj=[[0, 0, 1, 0]] * 2,
            token_mask=[1.0, 1.0],
        ),
    )
    payload_tk = build_floodnet_model_input(cmd_tokens, feature_length=2)

    # Both should produce structurally identical payloads
    for key in ["traj_features", "token_length", "token_mask"]:
        assert key in payload_wp
        assert key in payload_tk
