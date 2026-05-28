"""Retarget a raw HumanML3D motion file into a G1 reference chunk NPZ.

Bypasses FloodNet text generation — directly takes a (T, 263) .npy file,
runs the 263→140 bridge + MakeTrackingEasy NMR retarget + G1 adapter,
and saves a motion_tracking-compatible reference NPZ.

Usage:
  PYTHONPATH=src python tools/conversion/retarget_motion.py \\
    --motion path/to/motion.npy \\
    --config configs/system/local_dev.yaml \\
    --output artifacts/retargeted/
"""

from __future__ import annotations

import argparse
import json
import time
import uuid
from pathlib import Path

import numpy as np
import yaml

from text2humanoid.contracts.chunks import NMRInputChunk
from text2humanoid.infra.config_loader import resolve_config_path, resolve_root_path, resolve_path
from text2humanoid.retarget.bridge_263_to_140 import floodnet_263_to_nmr_140
from text2humanoid.retarget.g1_reference_adapter import G1ReferenceAdapter
from text2humanoid.retarget.nmr_service import NMRRetargetService
from text2humanoid.runtime.source_protocol import chunk_to_runtime_dict, validate_clip_payload


def _load_motion(path: str) -> np.ndarray:
    p = Path(path)
    if p.suffix == ".npy":
        data = np.load(p)
    elif p.suffix == ".npz":
        data = np.load(p)
        for key in ("motion", "motion_263", "arr_0", "features"):
            if key in data:
                data = data[key]
                break
    else:
        raise ValueError(f"Unsupported format: {p.suffix}")

    data = np.asarray(data, dtype=np.float32)
    if data.ndim == 3:
        data = data[0]  # (1, T, 263) → (T, 263)
    if data.ndim != 2 or data.shape[1] != 263:
        raise ValueError(f"Expected (T, 263), got {data.shape}")
    return data


def main():
    parser = argparse.ArgumentParser(description="Retarget HumanML3D motion to G1 reference")
    parser.add_argument("--motion", required=True, help="Path to (T,263) .npy or .npz file")
    parser.add_argument("--config", default="configs/system/local_dev.yaml",
                        help="Text2Humanoid system config YAML")
    parser.add_argument("--src-fps", type=float, default=20.0,
                        help="Source FPS (default 20)")
    parser.add_argument("--output", default="artifacts/retargeted",
                        help="Output directory for reference NPZ")
    parser.add_argument("--chunk-id", default=None,
                        help="Chunk ID prefix (auto-generated if omitted)")
    args = parser.parse_args()

    # Load config
    config_path = resolve_config_path(args.config)
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    root = resolve_root_path(cfg)

    retarget_cfg = cfg.get("retarget", {})
    runtime_cfg = cfg.get("runtime", {})
    tgt_fps = int(retarget_cfg.get("tgt_fps", 30))

    # Resolve adapter paths
    adapter_xml_path: str | None = None
    if retarget_cfg.get("xml_path"):
        adapter_xml_path = str(resolve_path(root, retarget_cfg["xml_path"]))
    adapter_tracking_config: str | None = None
    if runtime_cfg.get("tracking_config"):
        adapter_tracking_config = str(resolve_path(root, runtime_cfg["tracking_config"]))

    print(f"Root: {root}")
    print(f"Source FPS: {args.src_fps}  Target FPS: {tgt_fps}")

    # Load motion
    motion_263 = _load_motion(args.motion)
    print(f"Loaded motion: {motion_263.shape}")

    # Step 1: 263D → 140D bridge
    t0 = time.perf_counter()
    motion_140 = floodnet_263_to_nmr_140(
        motion_263, src_fps=args.src_fps, tgt_fps=float(tgt_fps),
    )
    t1 = time.perf_counter()

    chunk_id = args.chunk_id or uuid.uuid4().hex[:12]
    nmr_chunk = NMRInputChunk(
        chunk_id=chunk_id,
        start_time=0.0,
        fps=tgt_fps,
        motion_140=motion_140.cpu().numpy(),
        metadata={"source": args.motion, "src_fps": args.src_fps},
    )
    print(f"Bridge 263→140: {motion_140.shape[0]} frames, "
          f"{(t1 - t0) * 1000:.1f} ms")

    # Step 2: NMR retarget
    retarget = NMRRetargetService(
        apply_filter=bool(retarget_cfg.get("apply_filter", True)),
        tgt_fps=tgt_fps,
    )
    result = retarget.retarget_chunk(nmr_chunk)
    t2 = time.perf_counter()
    print(f"Retarget: {result['dof'].shape[0]} frames, "
          f"{(t2 - t1) * 1000:.1f} ms")

    # Step 3: adapter → G1ReferenceChunk
    adapter = G1ReferenceAdapter(
        xml_path=adapter_xml_path,
        tracking_config_path=adapter_tracking_config,
    )
    ref_chunk = adapter.from_nmr_result(
        chunk_id=chunk_id,
        start_time=0.0,
        fps=tgt_fps,
        result=result,
    )
    t3 = time.perf_counter()
    print(f"Adapter: {(t3 - t2) * 1000:.1f} ms")

    # Step 4: validate and save
    payload = chunk_to_runtime_dict(ref_chunk)
    errors = validate_clip_payload(payload)
    if errors:
        print("WARNING: validation errors:", errors)
    else:
        print("Clip payload validation: OK")

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    npz_path = out / f"reference_{chunk_id}.npz"
    np.savez(npz_path, **{k: v for k, v in payload.items() if isinstance(v, np.ndarray)})

    # Also save metadata
    meta = {
        "chunk_id": chunk_id,
        "source": args.motion,
        "src_fps": args.src_fps,
        "tgt_fps": tgt_fps,
        "num_frames": int(ref_chunk.num_frames),
        "joint_names": ref_chunk.joint_names,
        "body_names": ref_chunk.body_names,
        "timing_ms": {
            "bridge": round((t1 - t0) * 1000, 1),
            "retarget": round((t2 - t1) * 1000, 1),
            "adapter": round((t3 - t2) * 1000, 1),
            "total": round((t3 - t0) * 1000, 1),
        },
    }
    meta_path = out / f"meta_{chunk_id}.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, default=str)

    print(f"\nSaved: {npz_path} ({ref_chunk.num_frames} frames)")
    print(f"Meta:  {meta_path}")
    print(f"Total: {(t3 - t0) * 1000:.1f} ms")

    print(f"\nTo consume in motion_tracking:")
    print(f"  Set tracking.yaml:  floodnet_clip_path: \"{npz_path.resolve()}\"")
    print(f"  Or:  PYTHONPATH=sim2real/src python sim2real/src/deploy.py --sim2sim \\")
    print(f"         --tracking-config config/tracking_floodnet.yaml")
    print(f"       with floodnet_clip_path set to the NPZ above.")


if __name__ == "__main__":
    main()
