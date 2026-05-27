from __future__ import annotations

from pathlib import Path

import numpy as np

from text2humanoid.contracts.bfmzero import BFMZeroMotionChunk, G1_ISAAC_JOINT_NAMES
from text2humanoid.contracts.pipeline import RobotMotion, TrackerInput


def load_g1_reference_npz(
    path: str | Path,
    *,
    fps: int,
    motion_id: str | None = None,
    source_input_id: str = "",
) -> RobotMotion:
    """Load the current Text2Humanoid G1 reference artifact as RobotMotion.

    Reference artifacts store root quaternions as xyzw and joints in BFM-Zero
    Isaac order.
    """

    path = Path(path)
    data = np.load(path, allow_pickle=False)
    joint_names = [str(name) for name in data["joint_names"].tolist()]
    if joint_names != G1_ISAAC_JOINT_NAMES:
        raise ValueError("reference joint_names must match BFM-Zero G1 Isaac joint order")

    return RobotMotion(
        motion_id=motion_id or path.stem,
        source_input_id=source_input_id,
        robot="unitree_g1",
        representation="g1_root_dof",
        root_pos=np.asarray(data["root_pos"], dtype=np.float32),
        root_quat=np.asarray(data["root_rot"], dtype=np.float32),
        dof_pos=np.asarray(data["dof_pos"], dtype=np.float32),
        fps=fps,
        joint_names=joint_names,
        quat_order="xyzw",
        metadata={
            "source_format": "text2humanoid_g1_reference_npz",
            "source_path": str(path),
            "root_quat_order": "xyzw",
            "joint_order": "isaac",
        },
    )


def robot_motion_to_bfmzero_motion(
    motion: RobotMotion,
    *,
    chunk_id: str | None = None,
    frame_start: int = 0,
    target_fps: int | None = None,
) -> BFMZeroMotionChunk:
    """Convert canonical G1 RobotMotion into BFM-Zero direct ZMQ motion."""

    if motion.robot != "unitree_g1":
        raise ValueError(f"expected unitree_g1 robot motion, got {motion.robot!r}")
    if motion.representation != "g1_root_dof":
        raise ValueError(f"expected g1_root_dof representation, got {motion.representation!r}")
    if motion.joint_names != G1_ISAAC_JOINT_NAMES:
        raise ValueError("RobotMotion joint_names must be BFM-Zero G1 Isaac order")

    fps = int(target_fps or motion.fps)
    root_pos = motion.root_pos
    dof_pos = motion.dof_pos
    root_quat_wxyz = _root_quat_wxyz(motion)
    if fps != motion.fps:
        root_pos, root_quat_wxyz, dof_pos = _resample_motion_arrays(
            root_pos=root_pos,
            root_quat=root_quat_wxyz,
            dof_pos=dof_pos,
            source_fps=motion.fps,
            target_fps=fps,
        )

    return BFMZeroMotionChunk(
        chunk_id=chunk_id or motion.motion_id,
        fps=fps,
        frame_start=frame_start,
        joint_pos=dof_pos,
        joint_vel=_finite_difference(dof_pos, fps),
        root_pos=root_pos,
        root_quat=root_quat_wxyz,
        root_lin_vel_w=_finite_difference(root_pos, fps),
        root_ang_vel_w=_angular_velocity_wxyz(root_quat_wxyz, fps),
        metadata={
            "source_motion_id": motion.motion_id,
            "source_representation": motion.representation,
            "source_fps": motion.fps,
            "runtime_joint_order": "isaac",
            "root_quat_order": "wxyz",
        },
    )


class G1ReferenceToBFMZeroInputBridge:
    """TrackerInputBridge for already-retargeted Unitree G1 reference motion."""

    def __init__(self, *, frame_start: int = 0, target_fps: int | None = None) -> None:
        self.frame_start = int(frame_start)
        self.target_fps = None if target_fps is None else int(target_fps)

    def convert(self, motion: RobotMotion) -> TrackerInput:
        chunk = robot_motion_to_bfmzero_motion(
            motion,
            frame_start=self.frame_start,
            target_fps=self.target_fps,
        )
        return TrackerInput(
            input_id=f"tracker_{motion.motion_id}",
            source_motion_id=motion.motion_id,
            tracker="bfm_zero",
            representation="bfmzero_motion_frame_stream",
            payload=chunk,
            fps=chunk.fps,
            metadata={
                "bridge": type(self).__name__,
                "payload_type": "BFMZeroMotionChunk",
            },
        )


def _root_quat_wxyz(motion: RobotMotion) -> np.ndarray:
    quat = np.asarray(motion.root_quat, dtype=np.float32)
    if motion.quat_order == "wxyz":
        return _normalize_quat(quat)
    if motion.quat_order == "xyzw":
        return _normalize_quat(quat[:, [3, 0, 1, 2]])
    raise ValueError("RobotMotion.quat_order must be 'wxyz' or 'xyzw'")


def _finite_difference(values: np.ndarray, fps: int) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    velocity = np.zeros_like(values, dtype=np.float32)
    if values.shape[0] <= 1:
        return velocity
    velocity[1:] = (values[1:] - values[:-1]) * float(fps)
    return velocity


def _resample_motion_arrays(
    *,
    root_pos: np.ndarray,
    root_quat: np.ndarray,
    dof_pos: np.ndarray,
    source_fps: int,
    target_fps: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if target_fps <= 0:
        raise ValueError(f"target_fps must be positive, got {target_fps}")
    n = dof_pos.shape[0]
    if n <= 1:
        return root_pos.copy(), root_quat.copy(), dof_pos.copy()

    source_t = np.arange(n, dtype=np.float64) / float(source_fps)
    duration = source_t[-1]
    target_n = int(round(duration * float(target_fps))) + 1
    target_t = np.arange(target_n, dtype=np.float64) / float(target_fps)
    target_t[-1] = duration
    return (
        _interp_columns(source_t, root_pos, target_t),
        _interp_quat_wxyz(source_t, root_quat, target_t),
        _interp_columns(source_t, dof_pos, target_t),
    )


def _interp_columns(source_t: np.ndarray, values: np.ndarray, target_t: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    out = np.empty((target_t.shape[0], values.shape[1]), dtype=np.float32)
    for i in range(values.shape[1]):
        out[:, i] = np.interp(target_t, source_t, values[:, i])
    return out


def _interp_quat_wxyz(source_t: np.ndarray, quat: np.ndarray, target_t: np.ndarray) -> np.ndarray:
    quat = _normalize_quat(quat).copy()
    for i in range(1, quat.shape[0]):
        if np.dot(quat[i - 1], quat[i]) < 0.0:
            quat[i] *= -1.0
    return _normalize_quat(_interp_columns(source_t, quat, target_t))


def _angular_velocity_wxyz(quat: np.ndarray, fps: int) -> np.ndarray:
    quat = _normalize_quat(quat)
    out = np.zeros((quat.shape[0], 3), dtype=np.float32)
    if quat.shape[0] <= 1:
        return out

    for i in range(1, quat.shape[0]):
        delta = _quat_mul_wxyz(_quat_inverse_wxyz(quat[i - 1]), quat[i])
        delta = delta / np.clip(np.linalg.norm(delta), 1e-8, None)
        if delta[0] < 0.0:
            delta = -delta
        angle = 2.0 * np.arctan2(np.linalg.norm(delta[1:]), np.clip(delta[0], -1.0, 1.0))
        if angle > np.pi:
            angle -= 2.0 * np.pi
        axis_norm = np.linalg.norm(delta[1:])
        if axis_norm > 1e-8:
            out[i] = (delta[1:] / axis_norm) * angle * float(fps)
    return out


def _normalize_quat(quat: np.ndarray) -> np.ndarray:
    quat = np.asarray(quat, dtype=np.float32)
    norm = np.linalg.norm(quat, axis=1, keepdims=True)
    if np.any(norm < 1e-8):
        raise ValueError("root_quat contains zero-norm quaternion")
    return quat / norm


def _quat_inverse_wxyz(q: np.ndarray) -> np.ndarray:
    out = q.copy()
    out[1:] *= -1.0
    return out / np.clip(np.dot(q, q), 1e-8, None)


def _quat_mul_wxyz(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return np.asarray(
        [
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ],
        dtype=np.float32,
    )
