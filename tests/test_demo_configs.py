"""Smoke: demo config files are well-formed and select correct backends."""

from __future__ import annotations

from pathlib import Path

import yaml

_PROJECT_DIR = Path(__file__).resolve().parent.parent


def _load_yaml(rel_path: str) -> dict:
    path = _PROJECT_DIR / rel_path
    assert path.exists(), f"Config not found: {path}"
    with open(path) as f:
        return yaml.safe_load(f)


def test_demo_socket_config_selects_socket_backend():
    """demo_socket.yaml sets runtime.backend to socket."""
    cfg = _load_yaml("configs/system/demo_socket.yaml")
    assert cfg["runtime"]["backend"] == "socket"
    assert cfg["runtime"]["socket_port"] == 15555
    assert cfg["runtime"]["socket_host"] == "127.0.0.1"


def test_demo_socket_config_has_required_fields():
    """demo_socket.yaml contains all required top-level keys."""
    cfg = _load_yaml("configs/system/demo_socket.yaml")
    for key in ("root_path", "artifacts_root", "host", "port", "planner", "retarget", "runtime"):
        assert key in cfg, f"Missing key: {key}"


def test_demo_fixed_config_selects_floodnet_file_backend():
    """demo_fixed.yaml sets runtime.backend to floodnet_file."""
    cfg = _load_yaml("configs/system/demo_fixed.yaml")
    assert cfg["runtime"]["backend"] == "floodnet_file"
    assert "floodnet_output_dir" in cfg["runtime"]


def test_demo_fixed_config_has_required_fields():
    """demo_fixed.yaml contains all required top-level keys."""
    cfg = _load_yaml("configs/system/demo_fixed.yaml")
    for key in ("root_path", "artifacts_root", "host", "port", "planner", "retarget", "runtime"):
        assert key in cfg, f"Missing key: {key}"


def test_tracking_socket_floodnet_config():
    """tracking_socket_floodnet.yaml selects socket_floodnet motion source."""
    cfg = _load_yaml("../motion_tracking/sim2real/config/tracking_socket_floodnet.yaml")
    assert cfg["motion_source"] == "socket_floodnet"
    assert cfg["socket_host"] == "127.0.0.1"
    assert cfg["socket_port"] == 15555


def test_tracking_floodnet_config_unchanged():
    """tracking_floodnet.yaml fallback config still uses floodnet motion source."""
    cfg = _load_yaml("../motion_tracking/sim2real/config/tracking_floodnet.yaml")
    assert cfg["motion_source"] == "floodnet"
    assert "floodnet_clip_path" in cfg


def test_socket_and_file_backends_distinct():
    """Socket mainline and file fallback configs use different backends."""
    socket_cfg = _load_yaml("configs/system/demo_socket.yaml")
    file_cfg = _load_yaml("configs/system/demo_fixed.yaml")
    assert socket_cfg["runtime"]["backend"] != file_cfg["runtime"]["backend"]
    assert socket_cfg["runtime"]["backend"] == "socket"
    assert file_cfg["runtime"]["backend"] == "floodnet_file"


def test_socket_mainline_paired_configs_aligned():
    """Text2Humanoid demo_socket and motion_tracking tracking_socket_floodnet
    use the same host:port pair."""
    t2h = _load_yaml("configs/system/demo_socket.yaml")
    mt = _load_yaml("../motion_tracking/sim2real/config/tracking_socket_floodnet.yaml")
    assert t2h["runtime"]["socket_host"] == mt["socket_host"]
    assert t2h["runtime"]["socket_port"] == mt["socket_port"]
