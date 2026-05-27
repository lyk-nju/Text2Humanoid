from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


HUMANML3D_CHAINS: tuple[tuple[int, ...], ...] = (
    (0, 2, 5, 8, 11),
    (0, 1, 4, 7, 10),
    (0, 3, 6, 9, 12, 15),
    (9, 14, 17, 19, 21),
    (9, 13, 16, 18, 20),
)


G1_SIMPLE_CHAINS: tuple[tuple[int, ...], ...] = (
    (0, 1, 4, 7, 10, 14, 18),
    (0, 2, 5, 8, 11, 15, 19),
    (0, 3, 6, 9),
    (9, 12, 16, 20, 22, 24, 26, 28),
    (9, 13, 17, 21, 23, 25, 27, 29),
)


def load_motion_array(path: str | Path, *, expected_dim: int | None = None) -> np.ndarray:
    path = Path(path)
    if path.suffix == ".npy":
        data = np.load(path)
    elif path.suffix == ".npz":
        npz = np.load(path, allow_pickle=True)
        data = None
        candidate_keys = (
            f"motion_{expected_dim}" if expected_dim is not None else "",
            "motion",
            "features",
            "arr_0",
        )
        for key in candidate_keys:
            if key and key in npz:
                data = npz[key]
                break
        if data is None:
            raise ValueError(f"NPZ does not contain a motion array key: {sorted(npz.files)}")
    else:
        raise ValueError(f"Unsupported motion format: {path.suffix}")

    arr = np.asarray(data, dtype=np.float32)
    if arr.ndim == 3 and arr.shape[0] == 1:
        arr = arr[0]
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D motion array, got {arr.shape}")
    if expected_dim is not None and arr.shape[1] != expected_dim:
        raise ValueError(f"Expected motion shape (T, {expected_dim}), got {arr.shape}")
    return arr


def load_retarget_npz(path: str | Path) -> dict[str, np.ndarray]:
    npz = np.load(path, allow_pickle=True)
    required = ("dof", "root_trans", "root_rot_quat")
    missing = [key for key in required if key not in npz]
    if missing:
        raise ValueError(f"Retarget npz missing keys {missing}; available keys: {sorted(npz.files)}")
    result = {key: np.asarray(npz[key], dtype=np.float32) for key in required}
    if result["dof"].ndim != 2 or result["dof"].shape[1] != 29:
        raise ValueError(f"dof must have shape (T, 29), got {result['dof'].shape}")
    n = result["dof"].shape[0]
    if result["root_trans"].shape != (n, 3):
        raise ValueError(f"root_trans must have shape {(n, 3)}, got {result['root_trans'].shape}")
    if result["root_rot_quat"].shape != (n, 4):
        raise ValueError(f"root_rot_quat must have shape {(n, 4)}, got {result['root_rot_quat'].shape}")
    return result


def select_frame_indices(num_frames: int, count: int = 12) -> np.ndarray:
    if num_frames <= 0:
        raise ValueError("num_frames must be positive")
    count = max(1, min(int(count), int(num_frames)))
    return np.rint(np.linspace(0, num_frames - 1, count)).astype(np.int64)


def world_joints_from_motion_140(motion_140: np.ndarray, *, integrate_root: bool = True) -> np.ndarray:
    arr = np.asarray(motion_140, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[1] != 140:
        raise ValueError(f"motion_140 must have shape (T, 140), got {arr.shape}")
    joints = arr[:, 8 : 8 + 22 * 3].reshape(arr.shape[0], 22, 3).copy()
    if integrate_root:
        root_xz = np.cumsum(arr[:, 0:2], axis=0)
        joints[:, :, 0] += root_xz[:, 0:1]
        joints[:, :, 2] += root_xz[:, 1:2]
    return joints


def quaternion_continuity(quat_wxyz: np.ndarray) -> dict[str, Any]:
    quat = np.asarray(quat_wxyz, dtype=np.float32)
    if quat.ndim != 2 or quat.shape[1] != 4:
        raise ValueError(f"quat_wxyz must have shape (T, 4), got {quat.shape}")
    norms = np.linalg.norm(quat, axis=1)
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
        "norm_min": float(np.min(norms)) if norms.size else 0.0,
        "norm_max": float(np.max(norms)) if norms.size else 0.0,
    }


def array_summary(array: np.ndarray) -> dict[str, Any]:
    arr = np.asarray(array, dtype=np.float32)
    return {
        "shape": list(arr.shape),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "mean": float(np.mean(arr)),
        "mean_abs": float(np.mean(np.abs(arr))),
    }


def norm_summary(array: np.ndarray) -> dict[str, float]:
    norm = np.linalg.norm(np.asarray(array, dtype=np.float32), axis=-1).reshape(-1)
    p50, p95, p100 = np.percentile(norm, [50, 95, 100])
    return {"p50": float(p50), "p95": float(p95), "max": float(p100)}


def summarize_retarget_motion(result: dict[str, np.ndarray]) -> dict[str, Any]:
    dof = np.asarray(result["dof"], dtype=np.float32)
    root_trans = np.asarray(result["root_trans"], dtype=np.float32)
    root_quat = np.asarray(result["root_rot_quat"], dtype=np.float32)
    return {
        "frames": int(dof.shape[0]),
        "dof": array_summary(dof),
        "dof_velocity_norm": norm_summary(np.diff(dof, axis=0)) if dof.shape[0] > 1 else {"p50": 0.0, "p95": 0.0, "max": 0.0},
        "root_trans": array_summary(root_trans),
        "root_velocity_norm": norm_summary(np.diff(root_trans, axis=0)) if root_trans.shape[0] > 1 else {"p50": 0.0, "p95": 0.0, "max": 0.0},
        "root_quat": quaternion_continuity(root_quat),
    }


def write_json(path: str | Path, data: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")


def display_points_for_up_axis(points: np.ndarray, *, up_axis: str) -> tuple[np.ndarray, tuple[str, str, str]]:
    arr = np.asarray(points, dtype=np.float32)
    if arr.ndim != 3 or arr.shape[2] != 3:
        raise ValueError(f"points must have shape (T, J, 3), got {arr.shape}")
    if up_axis == "y":
        return arr[:, :, [0, 2, 1]], ("x", "z", "y")
    if up_axis == "z":
        return arr.copy(), ("x", "y", "z")
    raise ValueError("up_axis must be 'y' or 'z'")


def _set_equal_3d_axes(ax, points: np.ndarray) -> None:
    mins = np.min(points, axis=(0, 1))
    maxs = np.max(points, axis=(0, 1))
    center = (mins + maxs) / 2.0
    radius = float(np.max(maxs - mins) / 2.0)
    radius = max(radius, 0.5)
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)


def render_skeleton_video(
    joints: np.ndarray,
    *,
    output_path: str | Path,
    fps: float,
    chains: tuple[tuple[int, ...], ...] = HUMANML3D_CHAINS,
    title: str = "",
    stride: int = 1,
    up_axis: str = "y",
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FFMpegWriter, FuncAnimation, PillowWriter

    points = np.asarray(joints, dtype=np.float32)
    if points.ndim != 3 or points.shape[2] != 3:
        raise ValueError(f"joints must have shape (T, J, 3), got {points.shape}")
    points, labels = display_points_for_up_axis(points, up_axis=up_axis)
    stride = max(1, int(stride))
    frame_ids = np.arange(0, points.shape[0], stride, dtype=np.int64)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_subplot(111, projection="3d")
    lines = [ax.plot([], [], [], linewidth=2.0)[0] for _ in chains]
    root_trace = ax.plot([], [], [], color="black", linewidth=1.0, alpha=0.45)[0]
    frame_text = ax.text2D(0.02, 0.95, "", transform=ax.transAxes)

    _set_equal_3d_axes(ax, points)
    ax.set_xlabel(labels[0])
    ax.set_ylabel(labels[1])
    ax.set_zlabel(labels[2])
    ax.view_init(elev=18, azim=-70)
    if title:
        ax.set_title(title)

    def update(i: int):
        frame = int(frame_ids[i])
        pose = points[frame]
        for line, chain in zip(lines, chains):
            chain_points = pose[np.asarray(chain, dtype=np.int64)]
            line.set_data(chain_points[:, 0], chain_points[:, 1])
            line.set_3d_properties(chain_points[:, 2])
        trace = points[: frame + 1, 0, :]
        root_trace.set_data(trace[:, 0], trace[:, 1])
        root_trace.set_3d_properties(trace[:, 2])
        frame_text.set_text(f"frame {frame}/{points.shape[0] - 1}")
        return [*lines, root_trace, frame_text]

    interval = 1000.0 / max(float(fps) / stride, 1e-6)
    anim = FuncAnimation(fig, update, frames=len(frame_ids), interval=interval, blit=False)
    suffix = output_path.suffix.lower()
    try:
        if suffix == ".gif":
            anim.save(output_path, writer=PillowWriter(fps=max(1, int(round(float(fps) / stride)))))
        else:
            anim.save(output_path, writer=FFMpegWriter(fps=max(1, int(round(float(fps) / stride)))))
    finally:
        plt.close(fig)


def render_retarget_diagnostics_video(
    result: dict[str, np.ndarray],
    *,
    output_path: str | Path,
    fps: float,
    stride: int = 1,
    title: str = "",
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FFMpegWriter, FuncAnimation, PillowWriter

    dof = np.asarray(result["dof"], dtype=np.float32)
    root = np.asarray(result["root_trans"], dtype=np.float32)
    quat = np.asarray(result["root_rot_quat"], dtype=np.float32)
    dots = np.sum(quat[1:] * quat[:-1], axis=1) if quat.shape[0] > 1 else np.ones(1, dtype=np.float32)
    stride = max(1, int(stride))
    frame_ids = np.arange(0, dof.shape[0], stride, dtype=np.int64)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    fig.suptitle(title or "Retarget diagnostics")
    t = np.arange(dof.shape[0])
    axes[0, 0].plot(root[:, 0], root[:, 2], color="black")
    marker_root = axes[0, 0].scatter([], [], color="red")
    axes[0, 0].set_title("root x/z trajectory")
    axes[0, 0].set_aspect("equal", adjustable="datalim")

    axes[0, 1].plot(t[1:], dots, color="tab:blue")
    marker_dot = axes[0, 1].axvline(0, color="red")
    axes[0, 1].axhline(0.0, color="black", linewidth=0.8)
    axes[0, 1].set_ylim(min(-1.05, float(np.min(dots)) - 0.05), 1.05)
    axes[0, 1].set_title("root quat consecutive dot")

    show_dofs = min(8, dof.shape[1])
    axes[1, 0].plot(t, dof[:, :show_dofs])
    marker_dof = axes[1, 0].axvline(0, color="red")
    axes[1, 0].set_title(f"first {show_dofs} dof")

    dof_delta = np.linalg.norm(np.diff(dof, axis=0), axis=1) if dof.shape[0] > 1 else np.zeros(1, dtype=np.float32)
    axes[1, 1].plot(t[1:], dof_delta, color="tab:orange")
    marker_delta = axes[1, 1].axvline(0, color="red")
    axes[1, 1].set_title("dof delta norm")

    for ax in axes.reshape(-1):
        ax.grid(True, alpha=0.25)

    def update(i: int):
        frame = int(frame_ids[i])
        marker_root.set_offsets([[root[frame, 0], root[frame, 2]]])
        marker_dot.set_xdata([frame, frame])
        marker_dof.set_xdata([frame, frame])
        marker_delta.set_xdata([frame, frame])
        return [marker_root, marker_dot, marker_dof, marker_delta]

    interval = 1000.0 / max(float(fps) / stride, 1e-6)
    anim = FuncAnimation(fig, update, frames=len(frame_ids), interval=interval, blit=False)
    suffix = output_path.suffix.lower()
    try:
        if suffix == ".gif":
            anim.save(output_path, writer=PillowWriter(fps=max(1, int(round(float(fps) / stride)))))
        else:
            anim.save(output_path, writer=FFMpegWriter(fps=max(1, int(round(float(fps) / stride)))))
    finally:
        plt.close(fig)
