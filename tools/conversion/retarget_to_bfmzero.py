from __future__ import annotations

import argparse
import os
import time
import uuid
from pathlib import Path
from collections.abc import Sequence
from typing import Any

import numpy as np
import yaml

from text2humanoid.contracts.chunks import NMRInputChunk
from text2humanoid.runtime.bfmzero_chunk_npz import save_bfmzero_chunk_npz


_TEXT2HUMANOID_DIR = Path(__file__).resolve().parents[2]
_TEXT2MOTION_ROOT = _TEXT2HUMANOID_DIR.parent


def resolve_config_path(value: str) -> Path:
    path = Path(os.path.expandvars(value)).expanduser()
    if path.is_absolute():
        return path
    return (_TEXT2HUMANOID_DIR / path).resolve()


def resolve_root_path(cfg: dict[str, Any]) -> Path:
    raw = cfg.get("root_path", "auto")
    if raw == "auto" or raw is None:
        return Path(os.environ.get("TEXT2MOTION_ROOT", _TEXT2MOTION_ROOT)).expanduser().resolve()
    return Path(os.path.expandvars(raw)).expanduser().resolve()


def resolve_path(root: Path, value: str) -> Path:
    path = Path(os.path.expandvars(value)).expanduser()
    if path.is_absolute():
        return path
    return root / path


def floodnet_263_to_nmr_140(*args, **kwargs):
    from text2humanoid.retarget.bridge_263_to_140 import floodnet_263_to_nmr_140 as convert

    return convert(*args, **kwargs)


def NMRRetargetService(*args, **kwargs):
    from text2humanoid.retarget.nmr_service import NMRRetargetService as service_cls

    return service_cls(*args, **kwargs)


def make_tracking_easy_result_to_bfmzero_motion(*args, **kwargs):
    from text2humanoid.retarget.bfmzero_adapter import (
        make_tracking_easy_result_to_bfmzero_motion as convert,
    )

    return convert(*args, **kwargs)


def BFMZeroZmqSink(*args, **kwargs):
    from text2humanoid.runtime.bfmzero_zmq_sink import BFMZeroZmqSink as sink_cls

    return sink_cls(*args, **kwargs)


def _load_motion_263(path: str | Path) -> np.ndarray:
    path = Path(path)
    if path.suffix == ".npy":
        data = np.load(path)
    elif path.suffix == ".npz":
        npz = np.load(path)
        data = None
        for key in ("motion", "motion_263", "arr_0", "features"):
            if key in npz:
                data = npz[key]
                break
        if data is None:
            raise ValueError(f"NPZ does not contain a 263D motion key: {sorted(npz.files)}")
    else:
        raise ValueError(f"Unsupported motion format: {path.suffix}")

    data = np.asarray(data, dtype=np.float32)
    if data.ndim == 3:
        data = data[0]
    if data.ndim != 2 or data.shape[1] != 263:
        raise ValueError(f"Expected motion shape (T, 263), got {data.shape}")
    return data


def _save_mte_result(path: str | Path, result: dict[str, np.ndarray]) -> None:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        out_path,
        dof=np.asarray(result["dof"], dtype=np.float32),
        root_trans=np.asarray(result["root_trans"], dtype=np.float32),
        root_rot_quat=np.asarray(result["root_rot_quat"], dtype=np.float32),
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Retarget HumanML3D 263D motion to BFM-Zero tracking_online ZMQ format."
    )
    parser.add_argument("--motion", required=True, help="Path to (T,263) .npy or .npz motion")
    parser.add_argument("--config", default="configs/system/local_dev.yaml")
    parser.add_argument("--src-fps", type=float, default=20.0, help="Input 263D motion FPS")
    parser.add_argument("--retarget-fps", type=float, default=None, help="MTE/NMR FPS; defaults to config retarget.tgt_fps")
    parser.add_argument("--output-fps", type=float, default=50.0, help="BFM-Zero stream FPS")
    parser.add_argument("--chunk-id", default=None)
    parser.add_argument("--output", default=None, help="Optional path to save BFMZeroMotionChunk NPZ")
    parser.add_argument("--save-140", default=None, help="Optional path to save the intermediate (T,140) NMR input")
    parser.add_argument("--save-retarget", default=None, help="Optional path to save the raw MakeTrackingEasy retarget NPZ")
    parser.add_argument("--device", default="cpu", help="Device for MakeTrackingEasy FK conversion")
    parser.add_argument("--frame-start", type=int, default=0)
    parser.add_argument("--publish", action="store_true", help="Publish chunk over ZMQ")
    parser.add_argument("--dry-run", action="store_true", help="Build payload but do not publish")
    parser.add_argument("--host", default="*", help="ZMQ bind host")
    parser.add_argument("--port", type=int, default=5592, help="ZMQ bind port")
    parser.add_argument("--mark-end", action="store_true", help="Mark final frame with END")
    parser.add_argument("--no-realtime", action="store_true", help="Publish without frame sleeps")
    parser.add_argument("--startup-delay-sec", type=float, default=0.5)
    args = parser.parse_args(argv)

    cfg_path = resolve_config_path(args.config)
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    root = resolve_root_path(cfg)
    retarget_cfg = cfg.get("retarget", {})
    retarget_fps = float(args.retarget_fps or retarget_cfg.get("tgt_fps", 30))
    xml_path = str(
        resolve_path(
            root,
            retarget_cfg.get("xml_path", "MakeTrackingEasy/assets/g1_mocap_29dof.xml"),
        )
    )

    motion_263 = _load_motion_263(args.motion)
    print(f"Loaded motion_263: {motion_263.shape}")

    t0 = time.perf_counter()
    motion_140 = floodnet_263_to_nmr_140(
        motion_263,
        src_fps=float(args.src_fps),
        tgt_fps=retarget_fps,
    )
    motion_140_np = motion_140.cpu().numpy()
    t1 = time.perf_counter()
    print(f"Bridge 263->140: {motion_140_np.shape} ({(t1 - t0) * 1000:.1f} ms)")
    if args.save_140:
        save_140_path = Path(args.save_140)
        save_140_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(save_140_path, motion_140_np.astype(np.float32))
        print(f"Saved 140D motion: {save_140_path}")

    chunk_id = args.chunk_id or uuid.uuid4().hex[:12]
    nmr_chunk = NMRInputChunk(
        chunk_id=chunk_id,
        start_time=0.0,
        fps=int(round(retarget_fps)),
        motion_140=motion_140_np,
        metadata={"source": str(args.motion), "src_fps": float(args.src_fps)},
    )

    retarget = NMRRetargetService(
        apply_filter=bool(retarget_cfg.get("apply_filter", True)),
        tgt_fps=int(round(retarget_fps)),
    )
    result = retarget.retarget_chunk(nmr_chunk)
    t2 = time.perf_counter()
    print(f"NMR retarget: {result['dof'].shape} ({(t2 - t1) * 1000:.1f} ms)")
    if args.save_retarget:
        _save_mte_result(args.save_retarget, result)
        print(f"Saved MTE retarget: {args.save_retarget}")

    chunk = make_tracking_easy_result_to_bfmzero_motion(
        result,
        xml_path=xml_path,
        device=args.device,
        chunk_id=chunk_id,
        frame_start=args.frame_start,
        src_fps=retarget_fps,
        tgt_fps=float(args.output_fps),
    )
    t3 = time.perf_counter()
    print(
        f"BFM-Zero chunk: frames={chunk.num_frames} fps={chunk.fps} "
        f"chunk_id={chunk.chunk_id} ({(t3 - t2) * 1000:.1f} ms)"
    )
    if args.output:
        save_bfmzero_chunk_npz(args.output, chunk)
        print(f"Saved BFM-Zero chunk: {args.output}")

    if args.dry_run or not args.publish:
        return 0

    sink = BFMZeroZmqSink(host=args.host, port=args.port)
    try:
        sent = sink.publish_chunk(
            chunk,
            realtime=not args.no_realtime,
            mark_end=args.mark_end,
            startup_delay_sec=args.startup_delay_sec,
        )
    finally:
        sink.close()
    print(f"published={sent} endpoint=tcp://{args.host}:{args.port}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
