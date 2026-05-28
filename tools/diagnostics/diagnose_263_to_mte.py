from __future__ import annotations

import argparse
import json
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


def _array_summary(array: np.ndarray) -> dict[str, Any]:
    arr = np.asarray(array, dtype=np.float32)
    return {
        "shape": list(arr.shape),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "mean": float(np.mean(arr)),
        "mean_abs": float(np.mean(np.abs(arr))),
    }


def _norm_summary(array: np.ndarray) -> dict[str, float]:
    norm = np.linalg.norm(np.asarray(array, dtype=np.float32), axis=-1).reshape(-1)
    p50, p95, p100 = np.percentile(norm, [50, 95, 100])
    return {"p50": float(p50), "p95": float(p95), "max": float(p100)}


def _quat_summary(quat_wxyz: np.ndarray) -> dict[str, Any]:
    quat = np.asarray(quat_wxyz, dtype=np.float32)
    dots = np.sum(quat[1:] * quat[:-1], axis=1) if quat.shape[0] > 1 else np.asarray([], dtype=np.float32)
    if dots.size:
        min_dot = float(np.min(dots))
        argmin = int(np.argmin(dots) + 1)
        negative = int(np.count_nonzero(dots < 0.0))
    else:
        min_dot = 1.0
        argmin = 0
        negative = 0
    return {
        "shape": list(quat.shape),
        "min_consecutive_dot": min_dot,
        "min_dot_frame": argmin,
        "negative_dot_count": negative,
        "norm_min": float(np.min(np.linalg.norm(quat, axis=1))),
        "norm_max": float(np.max(np.linalg.norm(quat, axis=1))),
    }


def _save_mte_result(path: Path, result: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        dof=np.asarray(result["dof"], dtype=np.float32),
        root_trans=np.asarray(result["root_trans"], dtype=np.float32),
        root_rot_quat=np.asarray(result["root_rot_quat"], dtype=np.float32),
    )


def _mte_summary(result: dict[str, np.ndarray]) -> dict[str, Any]:
    return {
        "dof": _array_summary(result["dof"]),
        "root_trans": _array_summary(result["root_trans"]),
        "root_quat": _quat_summary(result["root_rot_quat"]),
    }


def _bfmzero_summary(chunk) -> dict[str, Any]:
    return {
        "frames": chunk.num_frames,
        "fps": chunk.fps,
        "joint_pos": _array_summary(chunk.joint_pos),
        "joint_vel_norm": _norm_summary(chunk.joint_vel),
        "root_pos": _array_summary(chunk.root_pos),
        "root_quat": _quat_summary(chunk.root_quat),
        "root_lin_vel_norm": _norm_summary(chunk.root_lin_vel_w),
        "root_ang_vel_norm": _norm_summary(chunk.root_ang_vel_w),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Diagnose 263D -> 140D -> MTE/NMR -> BFM-Zero quality.")
    parser.add_argument("--motion", required=True, help="Path to (T,263) .npy or .npz motion")
    parser.add_argument("--config", default="configs/system/local_dev.yaml")
    parser.add_argument("--src-fps", type=float, default=20.0)
    parser.add_argument("--retarget-fps", type=float, default=None)
    parser.add_argument("--output-fps", type=float, default=50.0)
    parser.add_argument("--output-dir", default="assets/saved")
    parser.add_argument("--chunk-id", default=None)
    parser.add_argument("--device", default="cpu")
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

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    chunk_id = args.chunk_id or uuid.uuid4().hex[:12]

    report: dict[str, Any] = {
        "chunk_id": chunk_id,
        "motion_path": str(args.motion),
        "src_fps": float(args.src_fps),
        "retarget_fps": retarget_fps,
        "output_fps": float(args.output_fps),
    }

    motion_263 = _load_motion_263(args.motion)
    report["motion_263"] = _array_summary(motion_263)

    t0 = time.perf_counter()
    motion_140 = floodnet_263_to_nmr_140(
        motion_263,
        src_fps=float(args.src_fps),
        tgt_fps=retarget_fps,
    ).cpu().numpy()
    report["motion_140"] = _array_summary(motion_140)
    report["motion_140"]["root_velocity_norm"] = _norm_summary(motion_140[:, 0:2])
    np.save(output_dir / f"{chunk_id}_motion_140.npy", motion_140)

    nmr_chunk = NMRInputChunk(
        chunk_id=chunk_id,
        start_time=0.0,
        fps=int(round(retarget_fps)),
        motion_140=motion_140,
        metadata={"source": str(args.motion), "src_fps": float(args.src_fps)},
    )

    for apply_filter in (True, False):
        label = "filter_true" if apply_filter else "filter_false"
        retarget = NMRRetargetService(
            apply_filter=apply_filter,
            tgt_fps=int(round(retarget_fps)),
        )
        result = retarget.retarget_chunk(nmr_chunk)
        _save_mte_result(output_dir / f"{chunk_id}_mte_{label}.npz", result)
        report[f"mte_{label}"] = _mte_summary(result)

        bfm_chunk = make_tracking_easy_result_to_bfmzero_motion(
            result,
            xml_path=xml_path,
            device=args.device,
            chunk_id=f"{chunk_id}_{label}",
            frame_start=0,
            src_fps=retarget_fps,
            tgt_fps=float(args.output_fps),
        )
        save_bfmzero_chunk_npz(output_dir / f"{chunk_id}_bfmzero_{label}.npz", bfm_chunk)
        report[f"bfmzero_{label}"] = _bfmzero_summary(bfm_chunk)

    report["timing_sec"] = {"total": round(time.perf_counter() - t0, 3)}
    report_path = output_dir / f"{chunk_id}_report.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote diagnostics: {report_path}")
    for key in ("mte_filter_true", "mte_filter_false", "bfmzero_filter_true", "bfmzero_filter_false"):
        item = report[key]
        if key.startswith("mte"):
            print(
                f"{key}: min_quat_dot={item['root_quat']['min_consecutive_dot']:.4f} "
                f"neg_dots={item['root_quat']['negative_dot_count']}"
            )
        else:
            print(
                f"{key}: root_ang_p95={item['root_ang_vel_norm']['p95']:.3f} "
                f"root_ang_max={item['root_ang_vel_norm']['max']:.3f}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
