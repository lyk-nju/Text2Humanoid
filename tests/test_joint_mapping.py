from __future__ import annotations

import pytest

from text2humanoid.retarget.fk_features import (
    _NMR_DOF_NAMES,
    JOINT_MAPPING,
    validate_joint_mapping,
)


def test_validate_joint_mapping_passes_with_correct_names():
    # Use the known-correct dataset joint names from tracking.yaml
    dataset_names = [
        "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
        "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
        "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
        "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
        "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
        "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint",
        "left_elbow_joint", "left_wrist_roll_joint", "left_wrist_pitch_joint", "left_wrist_yaw_joint",
        "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
        "right_elbow_joint", "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint",
    ]
    validate_joint_mapping(dataset_names)


def test_validate_joint_mapping_fails_on_order_drift():
    """If dataset joint names are reordered, validation must catch it."""
    correct_names = [
        "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
        "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
        "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
        "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
        "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
        "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint",
        "left_elbow_joint", "left_wrist_roll_joint", "left_wrist_pitch_joint", "left_wrist_yaw_joint",
        "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
        "right_elbow_joint", "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint",
    ]
    # Swap two entries to simulate order drift
    drifted = list(correct_names)
    drifted[0], drifted[1] = drifted[1], drifted[0]
    with pytest.raises(AssertionError):
        validate_joint_mapping(drifted)


def test_validate_joint_mapping_fails_on_wrong_length():
    with pytest.raises(AssertionError):
        validate_joint_mapping(["only_one_joint"])


def test_joint_mapping_lengths_match():
    assert len(JOINT_MAPPING) == 29
    assert len(_NMR_DOF_NAMES) == 29


def test_remap_produces_correct_order():
    """Validate that JOINT_MAPPING remaps NMR names to dataset names correctly."""
    dataset_names = [
        "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
        "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
        "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
        "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
        "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
        "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint",
        "left_elbow_joint", "left_wrist_roll_joint", "left_wrist_pitch_joint", "left_wrist_yaw_joint",
        "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
        "right_elbow_joint", "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint",
    ]
    remapped = [None] * len(_NMR_DOF_NAMES)
    for i, name in enumerate(_NMR_DOF_NAMES):
        remapped[JOINT_MAPPING[i]] = name
    assert remapped == dataset_names
