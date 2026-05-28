"""Offline reference replay — fixed text + manual trajectory → artifact bundle.

Runs the full pipeline (FloodNet → bridge → MakeTrackingEasy → G1ReferenceChunk)
without starting an HTTP server, and exports an inspectable artifact bundle.

Usage:
  PYTHONPATH=src python tools/replay/replay_trajectory.py \
    --config configs/system/local_dev.yaml \
    --text "walk forward slowly" \
    --waypoints '[[0,0,0,0],[2,3,0,4]]' \
    --replay-id demo_walk
"""

from __future__ import annotations

import argparse
import json
import time
import uuid
from pathlib import Path

import yaml

from text2humanoid.contracts.commands import PromptCommand, TrajectoryCondition, TrajectoryPoint
from text2humanoid.infra.artifact_store import ArtifactStore
from text2humanoid.infra.config_loader import build_components, resolve_config_path
from text2humanoid.orchestrator.pipeline_coordinator import PipelineCoordinator
from text2humanoid.retarget.bridge_263_to_140 import human_chunk_to_nmr_input


def _parse_waypoints(raw: str) -> list[TrajectoryPoint]:
    if raw.startswith("@"):
        path = Path(raw[1:])
        data = json.loads(path.read_text())
        return [TrajectoryPoint(t=p[0], x=p[1], y=p[2], z=p[3]) for p in data]
    data = json.loads(raw)
    return [TrajectoryPoint(t=p[0], x=p[1], y=p[2], z=p[3]) for p in data]


def run_replay_pipeline(
    command: PromptCommand,
    coordinator: PipelineCoordinator,
    artifact_store: ArtifactStore,
    replay_id: str | None = None,
) -> Path:
    """Run full offline replay pipeline and export an artifact bundle.

    Returns the export directory path.
    """
    rid = replay_id or uuid.uuid4().hex[:12]

    t_total_start = time.perf_counter()

    t0 = time.perf_counter()
    human_chunk = coordinator.planner.generate_chunk(command, start_time=0.0)
    t1 = time.perf_counter()

    nmr_chunk = human_chunk_to_nmr_input(human_chunk, tgt_fps=coordinator.retarget.output_fps)
    result = coordinator.retarget.retarget_chunk(nmr_chunk)
    t2 = time.perf_counter()

    ref_chunk = coordinator.adapter.from_nmr_result(
        chunk_id=human_chunk.chunk_id,
        start_time=nmr_chunk.start_time,
        fps=coordinator.retarget.output_fps,
        result=result,
    )
    t3 = time.perf_counter()

    timing = {
        "planner_ms": (t1 - t0) * 1000,
        "retarget_ms": (t2 - t1) * 1000,
        "adapter_ms": (t3 - t2) * 1000,
        "total_ms": (t3 - t_total_start) * 1000,
    }

    source_info: dict = {}
    if command.trajectory is not None:
        src = command.trajectory.to_source()
        source_info = {
            "source_type": src.source_type,
        }
        if src.source_type == "waypoints":
            source_info["num_waypoints"] = len(src.waypoints)
        elif src.source_type == "token_aligned" and src.token_aligned_traj is not None:
            source_info["token_count"] = len(src.token_aligned_traj)

    export_dir = artifact_store.export_replay_bundle(
        replay_id=rid,
        command=command.to_dict(),
        human_motion_263=human_chunk.motion_263,
        human_fps=human_chunk.fps,
        reference_chunk=ref_chunk,
        pipeline_timing=timing,
        metadata={
            "text": command.text,
            "planner_device": human_chunk.metadata.get("device", "unknown"),
            "trajectory_source": source_info,
        },
    )
    return export_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Offline reference replay")
    parser.add_argument("--config", type=str, default="configs/system/local_dev.yaml")
    parser.add_argument("--text", type=str, required=True)
    parser.add_argument("--waypoints", type=str, default=None,
                        help='JSON array [[t,x,y,z],...] or @path/to/waypoints.json')
    parser.add_argument("--token-trajectory", type=str, default=None,
                        help='JSON file with token_aligned_traj array')
    parser.add_argument("--replay-id", type=str, default=None)
    parser.add_argument("--chunk-frames", type=int, default=None)
    args = parser.parse_args()

    config_path = resolve_config_path(args.config)
    cfg = yaml.safe_load(open(config_path, encoding="utf-8"))

    session_manager, artifact_store = build_components(cfg)
    coordinator = session_manager._coordinator

    # Build trajectory condition
    traj = None
    if args.waypoints:
        waypoints = _parse_waypoints(args.waypoints)
        traj = TrajectoryCondition(waypoints=waypoints)
    elif args.token_trajectory:
        data = json.loads(Path(args.token_trajectory).read_text())
        traj = TrajectoryCondition(
            token_aligned_traj=data.get("traj"),
            token_mask=data.get("mask"),
        )

    command = PromptCommand(
        text=args.text,
        trajectory=traj,
        command_id=args.replay_id or uuid.uuid4().hex[:12],
    )
    if args.chunk_frames is not None:
        command.metadata["chunk_frames"] = args.chunk_frames

    export_dir = run_replay_pipeline(
        command=command,
        coordinator=coordinator,
        artifact_store=artifact_store,
        replay_id=args.replay_id,
    )

    # Load metadata for display
    meta = json.loads(open(export_dir / "metadata.json", encoding="utf-8").read())
    print(f"Replay ID:  {meta['replay_id']}")
    print(f"Export dir: {export_dir}")
    print(f"Human chunk:  {meta['human_shape'][0]} frames @ {meta['human_fps']}fps")
    print(f"Reference:    {meta['reference_frames']} frames @ {meta['reference_fps']}fps")
    print(f"DOF shape:    {meta['dof_shape']}")
    print(f"Local body:   {meta['local_body_shape']}")
    print(f"Joint names:  {len(meta['joint_names'])} joints")
    print(f"Body names:   {len(meta['body_names'])} bodies")
    print(f"Timing:       planner={meta['timing']['planner_ms']:.0f}ms  retarget={meta['timing']['retarget_ms']:.0f}ms  adapter={meta['timing']['adapter_ms']:.0f}ms")


if __name__ == "__main__":
    main()
