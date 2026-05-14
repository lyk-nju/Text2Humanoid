from __future__ import annotations

import argparse
from pathlib import Path

import uvicorn
import yaml

from text2humanoid.api import create_app
from text2humanoid.infra.artifact_store import ArtifactStore
from text2humanoid.orchestrator.pipeline_coordinator import PipelineCoordinator
from text2humanoid.orchestrator.session_manager import SessionManager
from text2humanoid.planner.floodnet_service import FloodNetPlannerService
from text2humanoid.retarget.g1_reference_adapter import G1ReferenceAdapter
from text2humanoid.retarget.nmr_service import NMRRetargetService
from text2humanoid.runtime.fallback_policy import FallbackPolicy
from text2humanoid.runtime.motion_tracking_client import MotionTrackingClient


def build_components(cfg: dict):
    artifact_store = ArtifactStore(cfg["artifacts_root"])
    planner = FloodNetPlannerService(
        config_path=cfg["planner"]["config_path"],
        chunk_frames=cfg["planner"]["chunk_frames"],
    )
    retarget = NMRRetargetService(apply_filter=cfg["retarget"]["apply_filter"])
    adapter = G1ReferenceAdapter()
    runtime = MotionTrackingClient(
        control_hz=cfg["runtime"]["control_hz"],
        future_horizon_frames=cfg["runtime"]["future_horizon_frames"],
    )
    fallback = FallbackPolicy(
        low_watermark_frames=cfg["runtime"]["low_watermark_frames"],
        high_watermark_frames=cfg["runtime"]["high_watermark_frames"],
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Text2Humanoid API server")
    parser.add_argument("--config", type=str, default="configs/system/local_dev.yaml")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    session_manager, artifact_store = build_components(cfg)
    app = create_app(session_manager, artifact_store)
    uvicorn.run(app, host=cfg["host"], port=int(cfg["port"]))


if __name__ == "__main__":
    main()
