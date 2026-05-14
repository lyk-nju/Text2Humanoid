from __future__ import annotations

import os
from pathlib import Path
from unittest import mock

from text2humanoid.infra.paths import get_root, set_root
from text2humanoid.infra.config_loader import (
    build_components,
    resolve_config_path,
    resolve_path,
    resolve_root_path,
)


def test_set_root():
    saved = get_root()
    try:
        set_root("/tmp/test_text2motion_root")
        assert str(get_root()) == "/tmp/test_text2motion_root"
        from text2humanoid.infra.paths import (
            get_floodnet_root,
            get_make_tracking_easy_root,
            get_motion_tracking_root,
        )
        assert str(get_floodnet_root()) == "/tmp/test_text2motion_root/FloodNet"
        assert str(get_make_tracking_easy_root()) == "/tmp/test_text2motion_root/MakeTrackingEasy"
        assert str(get_motion_tracking_root()) == "/tmp/test_text2motion_root/motion_tracking"
    finally:
        set_root(str(saved))


def test_env_var_override():
    saved = get_root()
    with mock.patch.dict(os.environ, {"TEXT2MOTION_ROOT": "/tmp/env_root"}, clear=False):
        try:
            from text2humanoid.infra import paths as pmod
            pmod._root = None
            assert str(pmod.get_root()) == "/tmp/env_root"
        finally:
            set_root(str(saved))


def test_resolve_path_absolute():
    root = Path("/fake/root")
    assert str(resolve_path(root, "/absolute/path")) == "/absolute/path"


def test_resolve_path_relative():
    root = Path("/fake/root")
    assert str(resolve_path(root, "FloodNet/configs/x.yaml")) == "/fake/root/FloodNet/configs/x.yaml"


def test_resolve_config_path_absolute():
    assert str(resolve_config_path("/etc/config.yaml")) == "/etc/config.yaml"


def test_resolve_config_path_relative():
    result = resolve_config_path("configs/system/local_dev.yaml")
    assert result.is_absolute()
    assert result.name == "local_dev.yaml"
    assert "configs/system" in str(result)


def test_resolve_root_path_auto_injects():
    saved = get_root()
    try:
        set_root(str(saved))
        root = resolve_root_path({"root_path": "auto"})
        from text2humanoid.infra import paths as pmod
        assert pmod._root is not None
        assert root == pmod._root
    finally:
        set_root(str(saved))


def test_resolve_root_path_explicit_injects():
    saved = get_root()
    try:
        root = resolve_root_path({"root_path": "/tmp/explicit_root"})
        assert str(root) == "/tmp/explicit_root"
        from text2humanoid.infra import paths as pmod
        assert str(pmod._root) == "/tmp/explicit_root"
    finally:
        set_root(str(saved))


def test_build_components_paths():
    real_root = get_root()
    cfg = {
        "root_path": str(real_root),
        "artifacts_root": "./artifacts/test_paths",
        "host": "127.0.0.1",
        "port": 9999,
        "planner": {
            "config_path": "FloodNet/configs/ldf_generate.yaml",
            "chunk_frames": 20,
        },
        "retarget": {
            "apply_filter": False,
            "tgt_fps": 25,
            "xml_path": "MakeTrackingEasy/assets/g1_mocap_29dof.xml",
        },
        "runtime": {
            "tracking_config": "motion_tracking/sim2real/config/tracking.yaml",
            "control_hz": 50,
            "low_watermark_frames": 10,
            "high_watermark_frames": 30,
            "future_horizon_frames": 8,
        },
    }
    try:
        session_manager, artifact_store = build_components(cfg)
        coordinator = session_manager._coordinator

        assert "test_paths" in str(artifact_store.root)
        assert "FloodNet" in coordinator.planner.config_path
        assert coordinator.planner.chunk_frames == 20
        assert coordinator.retarget.tgt_fps == 25
        assert coordinator.retarget.output_fps == 25
        assert coordinator.adapter.xml_path is not None
        assert "g1_mocap_29dof.xml" in str(coordinator.adapter.xml_path)
        assert coordinator.runtime.sync.control_hz == 50
        assert coordinator.fallback.low_watermark_frames == 10

        from text2humanoid.infra import paths as pmod
        assert pmod._root is not None
    finally:
        set_root(str(real_root))


def test_build_components_defaults():
    real_root = get_root()
    cfg = {
        "root_path": str(real_root),
        "artifacts_root": "./artifacts/defaults",
        "host": "127.0.0.1",
        "port": 9999,
        "planner": {
            "config_path": "FloodNet/configs/ldf_generate.yaml",
        },
    }
    try:
        session_manager, artifact_store = build_components(cfg)
        coordinator = session_manager._coordinator
        assert coordinator.planner.chunk_frames == 40
        assert coordinator.retarget.apply_filter is True
        assert coordinator.retarget.tgt_fps == 30
        assert coordinator.runtime.sync.control_hz == 50
        assert coordinator.fallback.low_watermark_frames == 20
    finally:
        set_root(str(real_root))
