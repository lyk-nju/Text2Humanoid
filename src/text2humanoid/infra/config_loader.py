from __future__ import annotations

import os
from pathlib import Path

import yaml

from text2humanoid.infra.artifact_store import ArtifactStore
from text2humanoid.infra.paths import get_root, set_root
from text2humanoid.orchestrator.pipeline_coordinator import PipelineCoordinator
from text2humanoid.orchestrator.session_manager import SessionManager
from text2humanoid.planner.floodnet_service import FloodNetPlannerService
from text2humanoid.retarget.g1_reference_adapter import G1ReferenceAdapter
from text2humanoid.retarget.nmr_service import NMRRetargetService
from text2humanoid.runtime.fallback_policy import FallbackPolicy
from text2humanoid.runtime.motion_tracking_client import (
    FloodNetFileBackend,
    MotionTrackingClient,
)
from text2humanoid.runtime.socket_backend import SocketBackend

_PROJECT_DIR = Path(__file__).resolve().parent.parent.parent.parent


def resolve_config_path(value: str) -> Path:
    p = Path(os.path.expandvars(value)).expanduser()
    if p.is_absolute():
        return p
    return (_PROJECT_DIR / p).resolve()


def resolve_root_path(cfg: dict) -> Path:
    raw = cfg.get("root_path", "auto")
    if raw == "auto" or raw is None:
        root = get_root()
    else:
        root = Path(os.path.expandvars(raw)).expanduser().resolve()
    set_root(root)
    return root


def resolve_path(root: Path, value: str) -> Path:
    p = Path(os.path.expandvars(value)).expanduser()
    if p.is_absolute():
        return p
    return root / p


def build_components(cfg: dict):
    root = resolve_root_path(cfg)

    artifacts_root = str(resolve_path(root, cfg["artifacts_root"]))
    artifact_store = ArtifactStore(artifacts_root)

    planner_cfg = cfg.get("planner", {})
    planner_config = str(resolve_path(root, planner_cfg["config_path"]))
    planner = FloodNetPlannerService(
        config_path=planner_config,
        chunk_frames=planner_cfg.get("chunk_frames", 40),
    )

    retarget_cfg = cfg.get("retarget", {})
    retarget = NMRRetargetService(
        apply_filter=retarget_cfg.get("apply_filter", True),
        tgt_fps=int(retarget_cfg.get("tgt_fps", 30)),
    )

    runtime_cfg = cfg.get("runtime", {})

    adapter_xml_path: str | None = None
    if retarget_cfg.get("xml_path"):
        adapter_xml_path = str(resolve_path(root, retarget_cfg["xml_path"]))

    adapter_tracking_config: str | None = None
    if runtime_cfg.get("tracking_config"):
        adapter_tracking_config = str(resolve_path(root, runtime_cfg["tracking_config"]))

    adapter = G1ReferenceAdapter(
        xml_path=adapter_xml_path,
        tracking_config_path=adapter_tracking_config,
    )

    backend_name = str(runtime_cfg.get("backend", "shim")).strip().lower()
    if backend_name == "floodnet_file":
        floodnet_output_dir = str(
            resolve_path(root, runtime_cfg.get("floodnet_output_dir", cfg.get("artifacts_root", "./artifacts/floodnet")))
        )
        backend = FloodNetFileBackend(
            output_dir=floodnet_output_dir,
            control_hz=int(runtime_cfg.get("control_hz", 50)),
        )
        runtime = MotionTrackingClient(backend=backend)
    elif backend_name == "socket":
        backend = SocketBackend(
            host=str(runtime_cfg.get("socket_host", "127.0.0.1")),
            port=int(runtime_cfg.get("socket_port", 15555)),
            control_hz=int(runtime_cfg.get("control_hz", 50)),
        )
        runtime = MotionTrackingClient(backend=backend)
    else:
        runtime = MotionTrackingClient(
            control_hz=int(runtime_cfg.get("control_hz", 50)),
            future_horizon_frames=int(runtime_cfg.get("future_horizon_frames", 16)),
            xml_path=adapter_xml_path,
        )
    fallback = FallbackPolicy(
        low_watermark_frames=int(runtime_cfg.get("low_watermark_frames", 20)),
        high_watermark_frames=int(runtime_cfg.get("high_watermark_frames", 60)),
    )
    coordinator = PipelineCoordinator(
        planner=planner,
        retarget=retarget,
        adapter=adapter,
        runtime=runtime,
        fallback=fallback,
    )
    session_manager = SessionManager(coordinator=coordinator)
    return session_manager, artifact_store
