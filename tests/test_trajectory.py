from __future__ import annotations

import numpy as np
import torch

from text2humanoid.contracts.commands import PromptCommand, TrajectoryCondition, TrajectoryPoint
from text2humanoid.contracts.trajectory import CanonicalTrajectory
from text2humanoid.planner.traj_conditioning import (
    build_floodnet_model_input,
    canonical_to_floodnet_features,
    token_aligned_to_canonical,
    waypoints_to_canonical,
)


def _make_waypoint(t: float, x: float, y: float, z: float) -> TrajectoryPoint:
    return TrajectoryPoint(t=t, x=x, y=y, z=z)


# ---- 003.2: waypoint -> canonical -------------------------------------------

def test_waypoints_sorted_by_time():
    wp = [
        _make_waypoint(2.0, 2.0, 0.0, 2.0),
        _make_waypoint(0.0, 0.0, 0.0, 0.0),
        _make_waypoint(1.0, 1.0, 0.0, 1.0),
    ]
    traj = waypoints_to_canonical(wp, fps=10)
    assert np.all(np.diff(traj.times) >= 0), "times not monotonic"
    assert traj.xz[0, 0] == 0.0 and traj.xz[0, 1] == 0.0
    assert traj.xz[-1, 0] == 2.0 and traj.xz[-1, 1] == 2.0


def test_waypoints_straight_line_interpolation():
    wp = [
        _make_waypoint(0.0, 0.0, 0.0, 0.0),
        _make_waypoint(1.0, 1.0, 0.0, 2.0),
    ]
    traj = waypoints_to_canonical(wp, fps=10)
    # At t=0.5, x=0.5, z=1.0
    idx = np.argmin(np.abs(traj.times - 0.5))
    assert abs(traj.xz[idx, 0] - 0.5) < 0.05
    assert abs(traj.xz[idx, 1] - 1.0) < 0.05


def test_waypoints_yaw_derived_from_direction():
    wp = [
        _make_waypoint(0.0, 0.0, 0.0, 0.0),
        _make_waypoint(1.0, 1.0, 0.0, 0.0),  # moving +x → yaw ~0
    ]
    traj = waypoints_to_canonical(wp, fps=30)
    # yaw should be approximately 0 for +x motion
    assert np.all(np.abs(traj.yaw[1:]) < 0.1), f"expected yaw ~0, got {traj.yaw[:5]}"


def test_waypoints_yaw_for_z_motion():
    wp = [
        _make_waypoint(0.0, 0.0, 0.0, 0.0),
        _make_waypoint(1.0, 0.0, 0.0, 1.0),  # moving +z → yaw ~pi/2
    ]
    traj = waypoints_to_canonical(wp, fps=30)
    assert np.all(traj.yaw[1:] > 1.0), f"expected yaw ~pi/2, got {traj.yaw[:5]}"


def test_waypoints_empty():
    traj = waypoints_to_canonical([], fps=30)
    assert traj.num_frames > 0


# ---- 003.2: token_aligned -> canonical --------------------------------------

def test_token_aligned_to_canonical():
    tokens = [[0.0, 0.0, 1.0, 0.0], [1.0, 0.0, 1.0, 0.0], [2.0, 0.0, 1.0, 0.0]]
    traj = token_aligned_to_canonical(tokens, token_mask=[1.0, 1.0, 0.0])
    assert traj.num_frames == 3
    assert traj.fps == 5
    assert abs(traj.yaw[0]) < 0.01  # cos=1, sin=0 → yaw=0
    assert not traj.valid_mask[2]


# ---- 003.3: canonical -> FloodNet features ----------------------------------

def test_canonical_to_floodnet_features_shape():
    wp = [_make_waypoint(0.0, 0.0, 0.0, 0.0), _make_waypoint(2.0, 2.0, 0.0, 2.0)]
    canonical = waypoints_to_canonical(wp, fps=30)
    result = canonical_to_floodnet_features(canonical, feature_length=10)
    assert result["traj_features"].shape == (1, 10, 4)
    assert result["token_length"].item() == 10
    assert result["token_mask"].shape == (1, 10)


def test_canonical_to_floodnet_features_values():
    wp = [_make_waypoint(0.0, 0.0, 0.0, 0.0), _make_waypoint(1.0, 1.0, 0.0, 0.0)]
    canonical = waypoints_to_canonical(wp, fps=30)
    feature_length = 10
    result = canonical_to_floodnet_features(canonical, feature_length)
    feats = result["traj_features"][0].numpy()
    # First token (t=0): xz=(0,0), yaw~0 → cos≈1, sin≈0
    assert abs(feats[0, 0] - 0.0) < 0.1
    assert abs(feats[0, 2] - 1.0) < 0.1  # cos(0)
    assert abs(feats[0, 3] - 0.0) < 0.1  # sin(0)


# ---- 003.3/4: build_floodnet_model_input with waypoints ----------------------

def test_build_input_waypoints_only_produces_traj_features():
    cmd = PromptCommand(
        text="walk",
        trajectory=TrajectoryCondition(
            waypoints=[
                _make_waypoint(0.0, 0.0, 0.0, 0.0),
                _make_waypoint(2.0, 3.0, 0.0, 4.0),
            ]
        ),
    )
    payload = build_floodnet_model_input(cmd, feature_length=40)
    assert "traj_features" in payload
    assert payload["traj_features"].shape[1] == 40
    assert payload["traj_features"].shape[2] == 4
    assert "token_mask" in payload
    assert "trajectory_metadata" in payload


def test_build_input_no_trajectory():
    cmd = PromptCommand(text="stand")
    payload = build_floodnet_model_input(cmd, feature_length=10)
    assert "traj_features" not in payload
    assert "token_mask" not in payload


def test_build_input_token_aligned_still_works():
    """Backward compatibility: explicit token_aligned_traj still works."""
    cmd = PromptCommand(
        text="walk",
        trajectory=TrajectoryCondition(
            token_aligned_traj=[[0.0, 0.0, 1.0, 0.0], [0.5, 0.0, 1.0, 0.0]],
            token_mask=[1.0, 0.5],
        ),
    )
    payload = build_floodnet_model_input(cmd, feature_length=2)
    assert "traj_features" in payload
    assert payload["traj_features"].shape == (1, 2, 4)
    assert "token_mask" in payload


def test_build_input_waypoints_take_priority_over_empty_tokens():
    cmd = PromptCommand(
        text="walk",
        trajectory=TrajectoryCondition(
            waypoints=[
                _make_waypoint(0.0, 0.0, 0.0, 0.0),
                _make_waypoint(1.0, 1.0, 0.0, 1.0),
            ],
        ),
    )
    payload = build_floodnet_model_input(cmd, feature_length=20)
    assert payload["traj_features"] is not None
    assert payload["traj_features"].shape[2] == 4


# ---- CanonicalTrajectory contract validation ---------------------------------

def test_canonical_trajectory_validation():
    times = np.array([0.0, 0.5, 1.0], dtype=np.float32)
    ct = CanonicalTrajectory(
        times=times,
        xz=np.zeros((3, 2), dtype=np.float32),
        yaw=np.zeros(3, dtype=np.float32),
        valid_mask=np.ones(3, dtype=bool),
        fps=30,
    )
    assert ct.num_frames == 3
    assert ct.duration == 1.0


def test_canonical_trajectory_rejects_mismatched_shapes():
    import pytest
    with pytest.raises(ValueError):
        CanonicalTrajectory(
            times=np.array([0.0, 1.0], dtype=np.float32),
            xz=np.zeros((3, 2), dtype=np.float32),  # wrong length
            yaw=np.zeros(2, dtype=np.float32),
            valid_mask=np.ones(2, dtype=bool),
            fps=30,
        )
